import numpy as np
from tqdm import tqdm
import osqp
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
import os
import time


# -- helpers --- 
def _precompute_gmm(w, mus, Sigmas, jitter=0.0, max_tries=5):
    """
    Stable precompute for SPD Gaussians:
      - normalize weights
      - SPD check via Cholesky (+tiny jitter if needed)
      - constants c_k via logdet(Σ_k)
      - A_k = Σ_k^{-1}, symmetrized
      - Lipschitz constants L_k = 2*λ_max(A_k)
    """
    K, D = mus.shape
    ws = np.asarray(w, float)
    if np.any(ws < 0) or ws.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    ws = ws / ws.sum()

    invS = np.empty_like(Sigmas)
    consts = np.empty(K, float)
    L_lip = np.empty(K, float)

    for k in range(K):
        S = 0.5 * (Sigmas[k] + Sigmas[k].T)
        added = 0.0
        for _ in range(max_tries):
            try:
                L = np.linalg.cholesky(S)
                break
            except np.linalg.LinAlgError:
                added = max(jitter, 10*(added if added else jitter))
                S = S + added * np.eye(D)
        else:
            raise np.linalg.LinAlgError("Covariance not SPD after jitter")

        # inverse & constant
        invS[k] = np.linalg.inv(S)
        invS[k] = 0.5 * (invS[k] + invS[k].T)  # enforce symmetry
        logdetS = 2.0 * np.sum(np.log(np.diag(L)))
        consts[k] = np.exp(-0.5*logdetS) / ((2.0*np.pi)**(0.5*D))

        # L_lip = 2*λ_max(A)
        evals = np.linalg.eigvalsh(invS[k])
        L_lip[k] = 2.0 * float(np.max(evals))

    return dict(ws=ws, mus=mus, invS=invS, consts=consts, L_lip=L_lip)

def _gray_max_on_box(mu, A, lo, hi, work):
    """
    Exact M = max_{vertex v} (v-mu)^T A (v-mu) via binary-reflected Gray code.
    O(D * 2^D) updates; fast for D <= ~12.
    """
    x, diff, Adiff = work
    D = lo.size
    n = 1 << D
    x[:] = lo
    np.subtract(x, mu, out=diff)
    Adiff[:] = A @ diff
    val = float(diff @ Adiff)
    best = val
    if n == 1:
        return best
    prev_gray = 0
    for mask in range(1, n):
        gray = mask ^ (mask >> 1)
        flip = prev_gray ^ gray
        j = (flip & -flip).bit_length() - 1
        going_up = (gray >> j) & 1
        delta = (hi[j] - lo[j]) * (1.0 if going_up else -1.0)
        old_j = Adiff[j]
        diff[j] += delta
        Adiff[:] += delta * A[:, j]
        val = val + 2.0 * delta * old_j + (delta * delta) * A[j, j]
        x[j] = hi[j] if going_up else lo[j]
        if val > best:
            best = val
        prev_gray = gray
    return best

# --- worker: compute bounds for a block of cells [start:end) ---
def _gmm_bounds_block(start, end, mids, wid, widths_is_per_cell,
                      ws, mus, invS, consts, L_lip, cheap=False):
    """
    Worker process for tight bounds on a chunk of cells.
    Reuses per-component OSQP across the chunk by only updating l/u.
    (No verbose prints here; parent handles progress.)
    """
    K, D = mus.shape
    Mblk = end - start
    p_min_blk = np.empty(Mblk, float)
    p_max_blk = np.empty(Mblk, float)

    # Gray-code buffers (for D <= 12)
    xcorner = np.empty(D, float); diff = np.empty(D, float); Adiff = np.empty(D, float)
    use_gray = (D <= 12)

    if cheap:
        # (Normally handled vectorized in the main function; this is a fallback.)
        for i_local, i in enumerate(range(start, end)):
            logs = np.empty(K, float)
            x = mids[i]
            for k in range(K):
                A = invS[k]
                d = x - mus[k]
                d2 = float(d @ (A @ d))
                logs[k] = np.log(ws[k]*consts[k]) - 0.5*d2
            a = np.max(logs)
            val = np.exp(a) * np.sum(np.exp(logs - a))
            p_min_blk[i_local] = val
            p_max_blk[i_local] = val
        return p_min_blk, p_max_blk

    solvers = []
    As = []
    for k in range(K):
        A = 0.5 * (invS[k] + invS[k].T)
        P = sp.csc_matrix(2.0 * A)
        q = -2.0 * (A @ mus[k])
        I = sp.eye(D, format='csc')
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=I, l=np.zeros(D), u=np.zeros(D),
                     polishing=False, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
        solvers.append((solver, q))  # q unused after setup
        As.append(A)

    xwarm = [None] * K  # warm-starts per component (local to worker)

    for i_local, i in enumerate(range(start, end)):
        wi = wid[i] if widths_is_per_cell else wid
        lo = mids[i] - 0.5 * wi
        hi = mids[i] + 0.5 * wi

        logs_max = np.empty(K, float)  # for p_max (uses min q)
        logs_min = np.empty(K, float)  # for p_min (uses max q)

        for k in range(K):
            solver, _ = solvers[k]
            A = As[k]

            # min q (-> p_max): update l/u, warm start, solve
            solver.update(l=lo, u=hi)
            if xwarm[k] is not None:
                solver.warm_start(x=xwarm[k])
            res = solver.solve(raise_error=False)
            xk = res.x
            xwarm[k] = xk
            d = xk - mus[k]
            d2_min = float(d @ (A @ d))

            # max q (-> p_min)
            if use_gray:
                d2_max = _gray_max_on_box(mus[k], A, lo, hi, (xcorner, diff, Adiff))
            else:
                # Spectral (Rayleigh) bound: λ_max(A) * ||r||^2
                r = np.maximum(np.abs(lo - mus[k]), np.abs(hi - mus[k]))
                d2_max = 0.5 * L_lip[k] * float(r @ r)  # L_lip = 2*λ_max(A)

            logs_max[k] = np.log(ws[k]*consts[k]) - 0.5*d2_min
            logs_min[k] = np.log(ws[k]*consts[k]) - 0.5*d2_max

        # mixture log-sum-exp
        a = np.max(logs_max); b = np.max(logs_min)
        p_max_blk[i_local] = np.exp(a) * np.sum(np.exp(logs_max - a))
        p_min_blk[i_local] = np.exp(b) * np.sum(np.exp(logs_min - b))

        if p_min_blk[i_local] > p_max_blk[i_local] + 1e-12 * max(1.0, p_max_blk[i_local]):
            raise RuntimeError("Numerical issue: lower bound exceeded upper bound.")

    return p_min_blk, p_max_blk

# --- end of helpers ---


def gmm_bounds_over_cells(weights, means, covariances, midpoints, widths,
                          verbose=False, precomp=None, cheap=False,
                          parallel=False, n_workers=None, chunk_size=512,
                          cheap_block_size=None):
    """
    Parallel & vectorized version (drop-in):

      cheap=True  → extremely fast, fully vectorized:
        - One-shot vectorization via einsum.
        - If memory could be large (M*K*D), you can set 'cheap_block_size'
          to do memory-safe chunked vectorization (progress bar only if verbose=True).

      cheap=False → tight bounds:
        - Serial or ProcessPool chunking (each worker reuses OSQP and does Gray-code or spectral max).
        - Progress bars only if verbose=True.

    Extra args:
      parallel: enable ProcessPool for tight path only
      n_workers: number of processes (defaults to os.cpu_count())
      chunk_size: cells per worker chunk for tight path
      cheap_block_size: cells per chunk for cheap path vectorization (None = auto)
    """
    t0 = time.perf_counter()

    # --- inputs ---
    w  = np.asarray(weights, float)
    m  = np.asarray(means,   float)
    S  = np.asarray(covariances, float)
    mids = np.asarray(midpoints, float)
    wid  = np.asarray(widths,    float)

    K, D = m.shape
    if mids.ndim != 2 or mids.shape[1] != D:
        raise ValueError("midpoints must be (M,D)")
    M = mids.shape[0]

    # --- cheap midpoint path (fully vectorized) ---
    if cheap:
        pc = precomp if precomp is not None else _precompute_gmm(w, m, S)
        invS, consts, ws = pc["invS"], pc["consts"], pc["ws"]

        if verbose:
            print(f"[gmm][cheap] vectorized midpoint density | M={M}, K={K}, D={D}")

        # If user didn't set a cheap_block_size, pick one that keeps
        # the intermediate (Mblk,K,D) under ~1e8 floats (~800MB) as a soft target.
        if cheap_block_size is None:
            target = max(1, int(1e8 // max(1, K * D)))  # ≈ 800MB soft cap
            cheap_block_size = min(M, max(4096, target))

        if cheap_block_size >= M:
            # One-shot fast path
            xm = mids[:, None, :] - m[None, :, :]                   # (M,K,D)
            Ax = np.einsum('mkd,kde->mke', xm, invS, optimize=True)  # (M,K,D)
            d2 = np.einsum('mkd,mkd->mk', xm, Ax, optimize=True)     # (M,K)
            logs = np.log(ws[None, :] * consts[None, :]) - 0.5 * d2  # (M,K)
            a = np.max(logs, axis=1, keepdims=True)                  # (M,1)
            p = np.exp(a) * np.sum(np.exp(logs - a), axis=1, keepdims=True)
            p = p.ravel()
            if verbose:
                dt = (time.perf_counter() - t0) * 1e3
                print(f"[gmm][cheap] done one-shot in {dt:.1f} ms")
            return p.copy(), p
        else:
            # Memory-safe chunked vectorization
            if verbose:
                print(f"[gmm][cheap] chunked: block={cheap_block_size}, "
                      f"chunks={(M + cheap_block_size - 1)//cheap_block_size}")

            out = np.empty(M, float)
            it = range(0, M, cheap_block_size)
            if verbose:
                it = tqdm(it, desc="GMM cheap bounds")
            for s0 in it:
                s1 = min(s0 + cheap_block_size, M)
                xm = mids[s0:s1, None, :] - m[None, :, :]                 # (Mb,K,D)
                Ax = np.einsum('mkd,kde->mke', xm, invS, optimize=True)   # (Mb,K,D)
                d2 = np.einsum('mkd,mkd->mk', xm, Ax, optimize=True)      # (Mb,K)
                logs = np.log(ws[None, :] * consts[None, :]) - 0.5 * d2   # (Mb,K)
                a = np.max(logs, axis=1, keepdims=True)                   # (Mb,1)
                p_blk = np.exp(a) * np.sum(np.exp(logs - a), axis=1, keepdims=True)
                out[s0:s1] = p_blk.ravel()
            if verbose:
                dt = (time.perf_counter() - t0) * 1e3
                print(f"[gmm][cheap] done in {dt:.1f} ms (chunked)")
            return out.copy(), out

    # --- validate widths (tight path only) ---
    if not ((wid.ndim == 1 and wid.shape[0] == D) or (wid.ndim == 2 and wid.shape == (M, D))):
        raise ValueError("widths must be (D,) or (M,D)")
    widths_is_per_cell = (wid.ndim == 2)
    if np.any((wid if widths_is_per_cell else wid) <= 0):
        raise ValueError("widths must be positive")

    # --- precompute constants and A_k ---
    pc = precomp if precomp is not None else _precompute_gmm(w, m, S)
    invS, consts, L_lip, ws = pc["invS"], pc["consts"], pc["L_lip"], pc["ws"]

    # --- tight path ---
    p_min = np.empty(M, float)
    p_max = np.empty(M, float)

    # Prepare chunks
    n_chunks = max(1, (M + chunk_size - 1) // chunk_size)
    starts = [i * chunk_size for i in range(n_chunks)]
    ends = [min((i + 1) * chunk_size, M) for i in range(n_chunks)]

    max_method = "Gray-code" if m.shape[1] <= 12 else "Spectral"
    if verbose:
        mode = "parallel" if parallel and M > chunk_size else "serial"
        nw = (os.cpu_count() or 1) if n_workers is None else n_workers
        print(f"[gmm][tight] mode={mode} | M={M}, K={K}, D={m.shape[1]}, chunks={n_chunks}, "
              f"chunk_size={chunk_size}, workers={nw if mode=='parallel' else 1} | max={max_method}")

    # SERIAL (no ProcessPool) — still chunked
    if (not parallel) or (M <= chunk_size):
        it = range(n_chunks)
        if verbose:
            it = tqdm(it, total=n_chunks, desc="GMM bounds (tight, serial)")
        for ci in it:
            s0, s1 = starts[ci], ends[ci]
            blk_min, blk_max = _gmm_bounds_block(
                s0, s1, mids, wid, widths_is_per_cell, ws, m, invS, consts, L_lip, cheap=False
            )
            p_min[s0:s1] = blk_min
            p_max[s0:s1] = blk_max

        if verbose:
            dt = (time.perf_counter() - t0) * 1e3
            print(f"[gmm][tight] done in {dt:.1f} ms")
        return p_min, p_max

    # PARALLEL with ProcessPool
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _gmm_bounds_block, s0, s1, mids, wid, widths_is_per_cell,
                ws, m, invS, consts, L_lip, False
            )
            for s0, s1 in zip(starts, ends)
        ]

        it = enumerate(futures)
        # if verbose:
        #     it = tqdm(it, total=n_chunks, desc="GMM bounds (tight, parallel)")
        for idx, fut in it:
            s0, s1 = starts[idx], ends[idx]
            blk_min, blk_max = fut.result()
            p_min[s0:s1] = blk_min
            p_max[s0:s1] = blk_max

    if verbose:
        dt = (time.perf_counter() - t0) * 1e3
        print(f"[gmm][tight] done in {dt:.1f} ms (parallel)")

    return p_min, p_max


def build_partition_and_bounds(
    problem, ws, mus, covs, *,
    N_degree=2,
    refine_factor=2,
    eps=0.0,
    verbose=True,
    dtype=np.float32,
    cheap=False,
    use_event_guidance=False,
    use_gmm_guidance=False, # keep for API compatability
    cap_per_degree=10,
    parallel=True,
):
    """
    Same behavior as before, but every call to gmm_bounds_over_cells()
    is automatically configured for max throughput given `cheap`.
    """

    # ---------- FAST SETTINGS PICKER ----------
    def _fast_args(M: int):
        """Select fastest gmm_bounds_over_cells settings for this call size M."""
        if cheap:
            return dict(
                verbose=verbose,
                precomp=pc_bounds,
                cheap=True,
                parallel=False,          # ignored in cheap mode
                cheap_block_size=None,   # one-shot vectorized path
            )
        else:
            if(parallel):
                n_workers = os.cpu_count() or 1
                # ~one large chunk per worker to amortize OSQP setup
                chunk = int(np.ceil(max(1, M) / n_workers))
                return dict(
                    verbose=verbose,
                    precomp=pc_bounds,
                    cheap=False,
                    parallel=True,
                    n_workers=n_workers,
                    chunk_size=chunk,
                )
            else:
                n_workers = 1
                # ~one large chunk per worker to amortize OSQP setup
                chunk = 1024
                return dict(
                    verbose=verbose,
                    precomp=pc_bounds,
                    cheap=False,
                    parallel=False,
                    n_workers=n_workers,
                    chunk_size=chunk,
                )

    # -------------------
    # Unpack & sanity
    # -------------------
    X_dom   = np.asarray(problem["X_dom"],   float)  # (D,2)
    X_event = np.asarray(problem["X_event"], float)  # (D,2)
    N_x     = problem["N_x"]
    D = X_dom.shape[0]
    B1 = float(problem.get("B1", 0.0))
    if B1 < 0:
        raise ValueError("B1 must be nonnegative.")
    if int(refine_factor) < 2:
        raise ValueError("refine_factor must be >= 2")
    r = int(refine_factor)

    # -------------------
    # Initial grid (uniform N_x^D)
    # -------------------
    if np.isscalar(N_x):
        Nvec = np.full(D, int(N_x), dtype=int)
    else:
        Nvec = np.asarray(N_x, int)
        if Nvec.shape != (D,):
            raise ValueError("N_x must be int or shape (D,)")

    base_w = (X_dom[:,1] - X_dom[:,0]) / Nvec
    mids_1d = [X_dom[d,0] + (0.5 + np.arange(Nvec[d])) * base_w[d] for d in range(D)]
    grids = np.meshgrid(*mids_1d, indexing='ij')
    midpts = np.stack(grids, axis=-1).reshape(-1, D).astype(dtype)
    widths = np.broadcast_to(base_w.astype(dtype), (midpts.shape[0], D)).copy()

    # -------------------
    # Small utilities
    # -------------------
    def _boxes(mid, wid):
        half = 0.5 * wid
        return (mid - half), (mid + half)

    def _event_masks(mid, wid):
        lo, hi = _boxes(mid.astype(float), wid.astype(float))
        ev_lo, ev_hi = X_event[:,0], X_event[:,1]
        intersects = np.all((hi >= ev_lo) & (lo <= ev_hi), axis=1)
        inside     = np.all((lo >= (ev_lo - eps)) & (hi <= (ev_hi + eps)), axis=1)
        return intersects, inside

    def _volumes(wid):
        return np.prod(wid.astype(np.float64), axis=1).astype(dtype)

    # Refine selected parents -> append children; compute bounds ONLY for children
    def _refine_selected_and_update_bounds(midpts, widths, p_min, p_max, sel_mask, pc, tag):
        keep = ~sel_mask
        kept_mid, kept_wid = midpts[keep], widths[keep]
        kept_pmin, kept_pmax = p_min[keep], p_max[keep]

        lo_parent, _ = _boxes(midpts.astype(float), widths.astype(float))
        parent_w = widths.astype(float)

        cmid_list = []; cwid_list = []
        n_parents = int(sel_mask.sum())
        if n_parents == 0:
            if verbose:
                print(f"    {tag} parents=0 → children=0 | bounds_propagated=0 | cells_after={midpts.shape[0]}")
            return midpts, widths, p_min, p_max

        for i in np.flatnonzero(sel_mask):
            lo = lo_parent[i]; pw = parent_w[i]
            cw = (pw / r).astype(dtype)
            child_mids_1d = [(lo[d] + (0.5 + np.arange(r)) * cw[d]).astype(dtype) for d in range(D)]
            cgrids = np.meshgrid(*child_mids_1d, indexing='ij')
            cmid = np.stack(cgrids, axis=-1).reshape(-1, D).astype(dtype)
            cwid = np.broadcast_to(cw, (cmid.shape[0], D)).astype(dtype)
            cmid_list.append(cmid); cwid_list.append(cwid)

        cmid = np.concatenate(cmid_list, axis=0)
        cwid = np.concatenate(cwid_list, axis=0)

        # compute bounds ONLY for children (MAXED-OUT SETTINGS)
        M_children = cmid.shape[0]
        c_pmin, c_pmax = gmm_bounds_over_cells(
            ws, mus, covs,
            cmid.astype(float), cwid.astype(float),
            **_fast_args(M_children)
        )

        # stitch results
        mid_new  = np.concatenate([kept_mid, cmid], axis=0)
        wid_new  = np.concatenate([kept_wid, cwid], axis=0)
        pmin_new = np.concatenate([kept_pmin, c_pmin], axis=0)
        pmax_new = np.concatenate([kept_pmax, c_pmax], axis=0)

        if verbose:
            print(f"    {tag} parents={n_parents} → children={cmid.shape[0]} "
                  f"| bounds_propagated={cmid.shape[0]} | reused(kept)={kept_mid.shape[0]} "
                  f"| cells_after={mid_new.shape[0]}")

        return mid_new, wid_new, pmin_new, pmax_new

    # -------------------
    # Precompute GMM constants (once) & initial full bounds
    # -------------------
    pc_bounds = _precompute_gmm(ws, mus, covs)
    if verbose:
        print(f"[init] grid cells={midpts.shape[0]} | D={D} | levels={max(0, int(N_degree)-1)} | r={r}")
        print(f"[bounds] initial full propagation: {midpts.shape[0]} cells")

    # INITIAL full bounds (MAXED-OUT SETTINGS)
    M0 = midpts.shape[0]
    p_min, p_max = gmm_bounds_over_cells(
        ws, mus, covs,
        midpts.astype(float), widths.astype(float),
        **_fast_args(M0)
    )

    # -------------------
    # Refinement loop
    # -------------------
    levels = max(0, int(N_degree) - 1)
    for level in range(levels):
        if verbose:
            print(f"[level {level+1}/{levels}] cells_before={midpts.shape[0]}")

        # ---- Compute gap mass on the CURRENT grid (before event selection) ----
        volumes = _volumes(widths)
        p_minus = np.maximum(0.0, p_min - B1)
        p_plus  = p_max + B1
        gap_mass = (p_plus - p_minus).astype(np.float64) * volumes.astype(np.float64)

        # 2.1 EVENT refinement (pick top 'cap_per_degree' among those intersecting X_event)
        if use_event_guidance:
            inter_event, _ = _event_masks(midpts, widths)
            idx_evt = np.flatnonzero(inter_event)
            n_evt = idx_evt.size
            if n_evt > 0:
                M = midpts.shape[0]
                budget_evt = min(int(cap_per_degree), n_evt) if cap_per_degree is not None else n_evt

                if budget_evt > 0:
                    evt_gaps = gap_mass[idx_evt]
                    if budget_evt >= n_evt:
                        sel_evt = idx_evt
                        topg = evt_gaps
                    else:
                        part = np.argpartition(evt_gaps, n_evt - budget_evt)[n_evt - budget_evt:]
                        order = np.argsort(evt_gaps[part])[::-1]
                        sel_evt = idx_evt[part[order]]
                        topg = evt_gaps[part][order]

                    refine_mask_event = np.zeros(M, dtype=bool)
                    refine_mask_event[sel_evt] = True

                    if verbose:
                        print(f"    [event] parents={len(sel_evt)} (subset of {n_evt} intersecting X_event, cap={cap_per_degree}) "
                              f"| top_evt_gap_mass min/med/max="
                              f"{float(np.min(topg)):.3e}/{float(np.median(topg)):.3e}/{float(np.max(topg)):.3e}")

                    # refine and recompute bounds ONLY for new children (MAXED-OUT SETTINGS)
                    midpts, widths, p_min, p_max = _refine_selected_and_update_bounds(
                        midpts, widths, p_min, p_max, refine_mask_event, pc_bounds, tag="[event]"
                    )
                else:
                    if verbose:
                        print(f"    [event] parents=0 (cap={cap_per_degree})")
            else:
                if verbose:
                    print("    [event] no cells intersect X_event")
        else:
            if verbose:
                print("    [event] skipped (use_event_guidance=False)")

        # ---- Recompute gap mass AFTER event refinement (grid may have changed) ----
        volumes = _volumes(widths)
        p_minus = np.maximum(0.0, p_min - B1)
        p_plus  = p_max + B1
        gap_mass = (p_plus - p_minus).astype(np.float64) * volumes.astype(np.float64)

        # 2.2 GAP refinement (top 'cap_per_degree' globally), only if use_gmm_guidance=True
        if use_gmm_guidance:
            M = midpts.shape[0]
            budget = min(int(cap_per_degree), M) if cap_per_degree is not None else M
            if budget > 0:
                if budget >= M:
                    sel_idx = np.arange(M, dtype=int)
                else:
                    part = np.argpartition(gap_mass, M - budget)[M - budget:]
                    sel_idx = part[np.argsort(gap_mass[part])[::-1]]
                refine_mask_gap = np.zeros(M, dtype=bool)
                refine_mask_gap[sel_idx] = True
            else:
                sel_idx = np.array([], dtype=int)
                refine_mask_gap = np.zeros(M, dtype=bool)

            if verbose:
                if sel_idx.size:
                    topg = gap_mass[sel_idx]
                    print(f"    [gap] parents={sel_idx.size} (cap={cap_per_degree}) "
                          f"| top_gap_mass min/med/max="
                          f"{float(np.min(topg)):.3e}/{float(np.median(topg)):.3e}/{float(np.max(topg)):.3e}")
                else:
                    print(f"    [gap] parents=0 (cap={cap_per_degree})")

            # refine and recompute bounds ONLY for new children (MAXED-OUT SETTINGS)
            midpts, widths, p_min, p_max = _refine_selected_and_update_bounds(
                midpts, widths, p_min, p_max, refine_mask_gap, pc_bounds, tag="[gap]"
            )
        else:
            if verbose:
                print("    [gap] skipped (use_gmm_guidance=False)")

    # -------------------
    # Finalize & return
    # -------------------
    mask_intersects_event, mask_inside_event = _event_masks(midpts, widths)
    volumes = np.prod(widths.astype(np.float64), axis=1).astype(dtype)
    p_minus = np.maximum(0.0, p_min - B1)
    p_plus  = p_max + B1

    base_mass = float(np.dot(p_minus.astype(np.float64), volumes.astype(np.float64)))
    if base_mass > 1.0 + 1e-9:
        raise RuntimeError(
            f"Infeasible baseline from p^-: {base_mass:.6f} > 1.0. "
            "Likely overlap/duplication or incorrect bounds/B1."
        )

    if verbose:
        gap = (p_plus - p_minus).astype(np.float64) * volumes.astype(np.float64)
        max_gap = float(np.max(gap)) if gap.size else 0.0
        print(f"[final] cells={midpts.shape[0]} | max_gap={max_gap:.6e}")

    return {
        "midpts":     midpts.astype(dtype),
        "widths":     widths.astype(dtype),
        "volumes":    volumes.astype(dtype),
        "mask_upper": mask_intersects_event,   # intersects event
        "mask_lower": mask_inside_event,       # fully inside event (with eps)
        "p_minus":    p_minus.astype(dtype),
        "p_plus":     p_plus.astype(dtype),
    }


def waterfill_upper(problem, total_mass=1.0, objective="mass"):
    """
    Upper bound LP (max event probability):
        maximize   sum_{i in mask} p_i * vol_i   if objective="mass" (default)
                   sum_{i in mask} p_i           if objective="count"
        subject to sum_i p_i * vol_i <= total_mass
                   p_i in [pmin_i, pmax_i],  p_i >= 0
    """
    pmin = np.asarray(problem["p_minus"], float).reshape(-1)
    pmax = np.asarray(problem["p_plus"],  float).reshape(-1)
    mask = np.asarray(problem["mask_upper"], bool).reshape(-1)
    N = pmin.size
    if pmax.shape != (N,) or mask.shape != (N,):
        raise ValueError("Shape mismatch among p_minus/p_plus/mask_upper.")

    volumes = np.asarray(problem["volumes"], float)
    volumes = np.broadcast_to(volumes, (N,)).copy()
    if np.any(~np.isfinite(pmin) | ~np.isfinite(pmax) | ~np.isfinite(volumes)):
        raise ValueError("Non-finite values in inputs.")
    if np.any(volumes <= 0):
        raise ValueError("All volumes must be > 0.")

    # Feasible baseline: clamp to [0, pmax]
    p = np.maximum(pmin, 0.0)
    p = np.minimum(p, pmax)

    base_mass = float(np.dot(p, volumes))
    if base_mass > total_mass + 1e-12:
        raise ValueError(f"Infeasible: baseline mass {base_mass:.6g} exceeds total_mass={total_mass}.")

    budget = total_mass - base_mass
    gap = np.maximum(0.0, pmax - p)  # how much each p_i can still increase

    if objective == "mass":
        # Same marginal gain per unit *mass* (1.0) on all event cells → any order works.
        for i in np.flatnonzero(mask):
            if budget <= 0.0:
                break
            gi = gap[i]
            if gi <= 0.0:
                continue
            need = gi * volumes[i]
            if need <= budget + 1e-18:
                p[i] += gi
                budget -= need
            else:
                p[i] += budget / volumes[i]  # partial fill
                budget = 0.0
                break
        Pr_upper = float(np.dot(p[mask], volumes[mask]))

    elif objective == "count":
        # Gain per unit mass is 1/vol_i → fill smaller volumes first.
        idxs = np.flatnonzero(mask)
        order = idxs[np.argsort(volumes[idxs])]
        for i in order:
            if budget <= 0.0:
                break
            gi = gap[i]
            if gi <= 0.0:
                continue
            need = gi * volumes[i]
            if need <= budget + 1e-18:
                p[i] += gi
                budget -= need
            else:
                p[i] += budget / volumes[i]
                budget = 0.0
                break
        Pr_upper = float(np.sum(p[mask]))

    else:
        raise ValueError("objective must be 'mass' or 'count'.")

    # Numerical tidy (guard tiny negatives due to FP)
    p = np.clip(p, 0.0, pmax)
    return Pr_upper, p


def waterfill_lower(problem, total_mass=1.0, objective="mass"):
    """
    Lower bound LP (min event probability):
        minimize   sum_{i in mask} p_i * vol_i   if objective="mass" (default)
                   sum_{i in mask} p_i           if objective="count"
        subject to sum_i p_i * vol_i <= total_mass
                   p_i in [pmin_i, pmax_i],  p_i >= 0

    Strategy:
      - Start from p = clamp(pmin, 0, pmax).
      - Allocate any remaining mass ONLY to non-event cells (~mask), up to pmax.
      - Stop early; the constraint is ≤, so using all mass is not required.
    """
    pmin = np.asarray(problem["p_minus"], float).reshape(-1)
    pmax = np.asarray(problem["p_plus"],  float).reshape(-1)
    mask = np.asarray(problem["mask_lower"], bool).reshape(-1)
    N = pmin.size
    if pmax.shape != (N,) or mask.shape != (N,):
        raise ValueError("Shape mismatch among p_minus/p_plus/mask_lower.")

    volumes = np.asarray(problem["volumes"], float)
    volumes = np.broadcast_to(volumes, (N,)).copy()
    if np.any(~np.isfinite(pmin) | ~np.isfinite(pmax) | ~np.isfinite(volumes)):
        raise ValueError("Non-finite values in inputs.")
    if np.any(volumes <= 0):
        raise ValueError("All volumes must be > 0.")

    # Feasible baseline
    p = np.maximum(pmin, 0.0)
    p = np.minimum(p, pmax)

    base_mass = float(np.dot(p, volumes))
    if base_mass > total_mass + 1e-12:
        raise ValueError(f"Infeasible: baseline mass {base_mass:.6g} exceeds total_mass={total_mass}.")

    budget = total_mass - base_mass
    gap = np.maximum(0.0, pmax - p)

    if budget > 0.0:
        non_event = np.flatnonzero(~mask)

        if objective == "mass":
            # Same marginal gain per unit *mass* on all non-event cells → any order works.
            for i in non_event:
                if budget <= 0.0:
                    break
                gi = gap[i]
                if gi <= 0.0:
                    continue
                need = gi * volumes[i]
                if need <= budget + 1e-18:
                    p[i] += gi
                    budget -= need
                else:
                    p[i] += budget / volumes[i]
                    budget = 0.0
                    break

        elif objective == "count":
            # Gain of sum p_i per unit mass is 1/vol_i → fill smaller volumes first.
            order = non_event[np.argsort(volumes[non_event])]
            for i in order:
                if budget <= 0.0:
                    break
                gi = gap[i]
                if gi <= 0.0:
                    continue
                need = gi * volumes[i]
                if need <= budget + 1e-18:
                    p[i] += gi
                    budget -= need
                else:
                    p[i] += budget / volumes[i]
                    budget = 0.0
                    break
        else:
            raise ValueError("objective must be 'mass' or 'count'.")

    # Objective value on event cells
    if objective == "mass":
        Pr_lower = float(np.dot(p[mask], volumes[mask]))
    else:
        Pr_lower = float(np.sum(p[mask]))

    p = np.clip(p, 0.0, pmax)
    return Pr_lower, p


def print_list(ls):
    formatted_numbers = [f"{num:.4f}" for num in ls]
    # Use str.join() to combine the formatted strings with a comma
    output_string = ', '.join(formatted_numbers)
    print(output_string)


def save_problem_result_npz(path, data, 
    keys = ("D","N_x","X_dom", "X_event","time_points_ref","time_points","Pr_ref","Pr_opt_upper", "Pr_opt_lower",
            "N_degree", "use_gmm_guidance", "cheap"),
    dtype=None, compress=False):
    """Save only selected keys from `data` as arrays to an NPZ."""
    out = {}
    for k in keys:
        if k in data:
            out[k] = np.asarray(data[k], dtype=dtype) if dtype is not None else np.asarray(data[k])
    if not out:
        raise ValueError("None of the selected keys were found in `data`.")
    (np.savez_compressed if compress else np.savez)(path, **out)


def load_problem_result_npz(path, scalarize=True):
    """
    Load an .npz into a dict. If scalarize=True, convert 0-D arrays to Python scalars.
    """
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    if scalarize:
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray) and v.shape == ():
                # Try to convert 0-D array to a Python scalar
                try:
                    out[k] = v.item()
                except Exception:
                    pass
    return out