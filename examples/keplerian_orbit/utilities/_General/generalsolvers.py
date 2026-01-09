import numpy as np
from tqdm import tqdm
import osqp
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
import os
import time


# ============================================================
# Helpers
# ============================================================
def _precompute_gmm(w, mus, Sigmas, jitter=0.0, max_tries=5):
    """
    Stable precompute for SPD Gaussians:
      - normalize weights
      - SPD check via Cholesky (+tiny jitter if needed)
      - constants c_k from logdet(Σ_k)
      - A_k = Σ_k^{-1}, symmetrized
      - Lipschitz constants L_k = 2*λ_max(A_k)
    """
    K, D = mus.shape
    ws = np.asarray(w, float)
    if np.any(ws < 0) or ws.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    ws = ws / ws.sum()

    invS   = np.empty_like(Sigmas)
    consts = np.empty(K, float)
    L_lip  = np.empty(K, float)

    for k in range(K):
        S = 0.5 * (Sigmas[k] + Sigmas[k].T)
        added = 0.0
        for _ in range(max_tries):
            try:
                L = np.linalg.cholesky(S)
                break
            except np.linalg.LinAlgError:
                added = max(jitter, 10 * (added if added else jitter))
                S = S + added * np.eye(D)
        else:
            raise np.linalg.LinAlgError("Covariance not SPD after jitter")

        invS[k] = np.linalg.inv(S)
        invS[k] = 0.5 * (invS[k] + invS[k].T)

        logdetS = 2.0 * np.sum(np.log(np.diag(L)))
        consts[k] = np.exp(-0.5 * logdetS) / ((2.0 * np.pi) ** (0.5 * D))

        evals = np.linalg.eigvalsh(invS[k])
        L_lip[k] = 2.0 * float(np.max(evals))

    return dict(ws=ws, mus=mus, invS=invS, consts=consts, L_lip=L_lip)


def _gray_max_on_box(mu, A, lo, hi, work):
    """
    Exact M = max_{vertex v} (v-mu)^T A (v-mu) using Gray code.
    O(D*2^D), quite fast for D <= ~12.
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


def _effective_mean_cov(ws, mus, covs):
    """
    Effective mean/cov of a Gaussian mixture:
      mu_eff = Σ w_k mu_k
      Sigma_eff = Σ w_k [Sigma_k + (mu_k-mu_eff)(mu_k-mu_eff)^T]
    """
    ws = np.asarray(ws, float)
    ws = ws / ws.sum()

    mu_eff = np.sum(ws[:, None] * mus, axis=0)

    D = mus.shape[1]
    Sigma_eff = np.zeros((D, D), float)
    for k in range(ws.size):
        dm = (mus[k] - mu_eff).reshape(D, 1)
        Sigma_eff += ws[k] * (covs[k] + dm @ dm.T)

    Sigma_eff = 0.5 * (Sigma_eff + Sigma_eff.T)
    return mu_eff, Sigma_eff


def _safe_inv_spd(S, jitter=1e-12, max_tries=6):
    """Cholesky-based safe inverse for SPD-ish matrices."""
    D = S.shape[0]
    added = 0.0
    Stry = 0.5 * (S + S.T)

    for _ in range(max_tries):
        try:
            np.linalg.cholesky(Stry)
            invS = np.linalg.inv(Stry)
            return 0.5 * (invS + invS.T)
        except np.linalg.LinAlgError:
            added = max(jitter, 10.0 * (added if added else jitter))
            Stry = Stry + added * np.eye(D)

    raise np.linalg.LinAlgError("Effective covariance not SPD after jitter.")


def _mask_far_cells(midpts, mu_eff, invSigma_eff, far_N):
    """mask_far[i]=True if Mahalanobis distance >= far_N."""
    if far_N is None:
        return np.zeros(midpts.shape[0], dtype=bool)
    far_N = float(far_N)
    if far_N <= 0:
        return np.zeros(midpts.shape[0], dtype=bool)

    diff = midpts - mu_eff[None, :]
    Ad   = diff @ invSigma_eff
    d2   = np.einsum("md,md->m", Ad, diff)
    return d2 >= far_N * far_N


# ============================================================
# Tight worker over a block
# ============================================================
def _gmm_bounds_block(start, end, mids, wid, widths_is_per_cell,
                      ws, mus, invS, consts, L_lip):
    """
    Tight bounds on cells [start:end).
    Reuses OSQP solvers per component; Gray-code max if D<=12 else spectral max.
    """
    K, D = mus.shape
    Mblk = end - start

    p_min_blk = np.empty(Mblk, float)
    p_max_blk = np.empty(Mblk, float)

    # gray-code buffers
    xcorner = np.empty(D, float)
    diff    = np.empty(D, float)
    Adiff   = np.empty(D, float)
    use_gray = (D <= 12)

    # Precompute solvers + symmetric A_k
    solvers = []
    As = []
    for k in range(K):
        A = 0.5 * (invS[k] + invS[k].T)
        P = sp.csc_matrix(2.0 * A)
        q = -2.0 * (A @ mus[k])
        I = sp.eye(D, format="csc")

        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=I, l=np.zeros(D), u=np.zeros(D),
                     polishing=False, eps_abs=1e-10, eps_rel=1e-10, verbose=False)

        solvers.append(solver)
        As.append(A)

    # Precompute log(ws*consts)
    logwconst = np.log(ws * consts)

    xwarm = [None] * K

    for i_local, i in enumerate(range(start, end)):
        wi = wid[i] if widths_is_per_cell else wid
        lo = mids[i] - 0.5 * wi
        hi = mids[i] + 0.5 * wi

        logs_max = np.empty(K, float)
        logs_min = np.empty(K, float)

        for k in range(K):
            solver = solvers[k]
            A = As[k]

            # min q -> p_max
            solver.update(l=lo, u=hi)
            if xwarm[k] is not None:
                solver.warm_start(x=xwarm[k])
            res = solver.solve(raise_error=False)
            xk = res.x
            xwarm[k] = xk

            d = xk - mus[k]
            d2_min = float(d @ (A @ d))

            # max q -> p_min
            if use_gray:
                d2_max = _gray_max_on_box(mus[k], A, lo, hi, (xcorner, diff, Adiff))
            else:
                r = np.maximum(np.abs(lo - mus[k]), np.abs(hi - mus[k]))
                d2_max = 0.5 * L_lip[k] * float(r @ r)

            logs_max[k] = logwconst[k] - 0.5 * d2_min
            logs_min[k] = logwconst[k] - 0.5 * d2_max

        a = np.max(logs_max)
        b = np.max(logs_min)
        p_max_blk[i_local] = np.exp(a) * np.sum(np.exp(logs_max - a))
        p_min_blk[i_local] = np.exp(b) * np.sum(np.exp(logs_min - b))

        if p_min_blk[i_local] > p_max_blk[i_local] + 1e-12 * max(1.0, p_max_blk[i_local]):
            raise RuntimeError("Numerical issue: lower bound exceeded upper bound.")

    return p_min_blk, p_max_blk


# ============================================================
# Main bounds function
# ============================================================
def gmm_bounds_over_cells(weights, means, covariances, midpoints, widths,
                          verbose=False, precomp=None, cheap=False,
                          parallel=False, n_workers=None, chunk_size=512,
                          cheap_block_size=None,
                          far_N=None):
    """
    Compute bounds for a GMM over axis-aligned cells.

    cheap=True:
        midpoint evaluation only (p_min=p_max=pdf(midpoint)), vectorized.

    cheap=False:
        tight bounds by per-cell quadratic min/max.

    far_N:
        Used ONLY for reporting (and for B1 zeroing elsewhere).
        Even if Mahalanobis distance >= far_N, we still compute tight/cheap
        bounds over that cell here.
    """
    t0 = time.perf_counter()

    w    = np.asarray(weights, float)
    m    = np.asarray(means, float)
    S    = np.asarray(covariances, float)
    mids = np.asarray(midpoints, float)
    wid  = np.asarray(widths, float)

    K, D = m.shape
    if mids.ndim != 2 or mids.shape[1] != D:
        raise ValueError("midpoints must be (M,D)")
    M = mids.shape[0]

    # ---------- far mask (for logging only) ----------
    if far_N is not None:
        ws_eff = precomp["ws"] if (precomp is not None and "ws" in precomp) else (w / w.sum())
        mu_eff, Sigma_eff = _effective_mean_cov(ws_eff, m, S)
        invSigma_eff = _safe_inv_spd(Sigma_eff)
        mask_far = _mask_far_cells(mids, mu_eff, invSigma_eff, far_N)
        n_far = int(mask_far.sum())
    else:
        n_far = 0

    # Helper: vectorized pdf at centers (supports blocking)
    def _pdf_at_points(points, invS, consts, ws, block_size=None):
        Pn = points.shape[0]
        if Pn == 0:
            return np.empty(0, float)

        if block_size is None:
            target = max(1, int(1e8 // max(1, K * D)))
            block_size = min(Pn, max(4096, target))

        out = np.empty(Pn, float)
        if block_size >= Pn:
            xm   = points[:, None, :] - m[None, :, :]
            Ax   = np.einsum("pkd,kde->pke", xm, invS, optimize=True)
            d2   = np.einsum("pkd,pkd->pk", xm, Ax, optimize=True)
            logs = np.log(ws[None, :] * consts[None, :]) - 0.5 * d2
            a    = np.max(logs, axis=1, keepdims=True)
            out[:] = (np.exp(a) * np.sum(np.exp(logs - a), axis=1, keepdims=True)).ravel()
            return out

        for s0 in range(0, Pn, block_size):
            s1 = min(s0 + block_size, Pn)
            xm   = points[s0:s1, None, :] - m[None, :, :]
            Ax   = np.einsum("pkd,kde->pke", xm, invS, optimize=True)
            d2   = np.einsum("pkd,pkd->pk", xm, Ax, optimize=True)
            logs = np.log(ws[None, :] * consts[None, :]) - 0.5 * d2
            a    = np.max(logs, axis=1, keepdims=True)
            out[s0:s1] = (np.exp(a) * np.sum(np.exp(logs - a), axis=1, keepdims=True)).ravel()

        return out

    # ---------- cheap path ----------
    if cheap:
        pc = precomp if precomp is not None else _precompute_gmm(w, m, S)
        invS, consts, ws = pc["invS"], pc["consts"], pc["ws"]

        if cheap_block_size is None:
            target = max(1, int(1e8 // max(1, K * D)))
            cheap_block_size = min(M, max(4096, target))

        out = _pdf_at_points(mids, invS, consts, ws, block_size=cheap_block_size)

        if verbose:
            dt = (time.perf_counter() - t0) * 1e3
            msg_far = f" | far={n_far}/{M}" if far_N is not None else ""
            print(f"[gmm][cheap] done in {dt:.1f} ms{msg_far}")

        return out.copy(), out

    # ---------- tight path validation ----------
    if not ((wid.ndim == 1 and wid.shape[0] == D) or (wid.ndim == 2 and wid.shape == (M, D))):
        raise ValueError("widths must be (D,) or (M,D)")
    widths_is_per_cell = (wid.ndim == 2)
    if np.any((wid if widths_is_per_cell else wid) <= 0):
        raise ValueError("widths must be positive")

    pc = precomp if precomp is not None else _precompute_gmm(w, m, S)
    invS, consts, L_lip, ws = pc["invS"], pc["consts"], pc["L_lip"], pc["ws"]

    p_min = np.empty(M, float)
    p_max = np.empty(M, float)

    def _run_tight(mids_run, wid_run, out_min, out_max, tag=""):
        Mrun = mids_run.shape[0]
        n_chunks = max(1, (Mrun + chunk_size - 1) // chunk_size)
        starts = [i * chunk_size for i in range(n_chunks)]
        ends   = [min((i + 1) * chunk_size, Mrun) for i in range(n_chunks)]

        if verbose:
            mode = "parallel" if parallel and Mrun > chunk_size else "serial"
            nw = (os.cpu_count() or 1) if n_workers is None else n_workers
            print(f"[gmm][tight]{tag} mode={mode} | M={Mrun}, "
                  f"K={K}, D={D}, chunks={n_chunks}, chunk_size={chunk_size}, "
                  f"workers={nw if mode=='parallel' else 1}")

        # serial
        if (not parallel) or (Mrun <= chunk_size):
            it = range(n_chunks)
            if verbose:
                it = tqdm(it, total=n_chunks, desc=f"GMM bounds (tight, serial{tag})")
            for ci in it:
                s0, s1 = starts[ci], ends[ci]
                blk_min, blk_max = _gmm_bounds_block(
                    s0, s1, mids_run, wid_run, widths_is_per_cell,
                    ws, m, invS, consts, L_lip
                )
                out_min[s0:s1] = blk_min
                out_max[s0:s1] = blk_max
            return

        # parallel
        nwrk = (os.cpu_count() or 1) if n_workers is None else n_workers
        with ProcessPoolExecutor(max_workers=nwrk) as ex:
            futures = [
                ex.submit(_gmm_bounds_block, s0, s1,
                          mids_run, wid_run, widths_is_per_cell,
                          ws, m, invS, consts, L_lip)
                for s0, s1 in zip(starts, ends)
            ]
            it = enumerate(futures)
            if verbose:
                it = tqdm(it, total=n_chunks, desc=f"GMM bounds (tight, parallel{tag})")
            for idx, fut in it:
                s0, s1 = starts[idx], ends[idx]
                blk_min, blk_max = fut.result()
                out_min[s0:s1] = blk_min
                out_max[s0:s1] = blk_max

    # tight on ALL cells (no far shortcut)
    _run_tight(mids, wid, p_min, p_max)

    if verbose:
        dt = (time.perf_counter() - t0) * 1e3
        msg_far = f" | far={n_far}/{M}" if far_N is not None else ""
        print(f"[gmm][tight] done in {dt:.1f} ms{msg_far}")

    return p_min, p_max


# ============================================================
# Partition + refinement
# ============================================================
def build_partition_and_bounds(
    problem, ws, mus, covs, *,
    N_degree=2,
    refine_factor=2,
    eps=0.0,
    verbose=True,
    dtype=np.float32,
    cheap=False,
    use_event_guidance=False,
    use_gmm_guidance=False,
    cap_per_degree=10,
    parallel=True,
    far_N=10,
):
    """
    Vectorized and flat. Core process:
      (1) Event-face alignment (optional, one-shot).
      (2) Gap-based refinement: global pass.
    """

    X_dom   = np.asarray(problem["X_dom"], float)
    X_event = np.asarray(problem["X_event"], float)
    N_x     = problem["N_x"]
    D       = int(X_dom.shape[0])
    B1      = float(problem.get("B1", 0.0))
    if B1 < 0:
        raise ValueError("B1 must be nonnegative.")
    r = int(refine_factor)
    if r < 2:
        raise ValueError("refine_factor must be >= 2")

    # -------------------
    # Helpers
    # -------------------
    pc_bounds = _precompute_gmm(ws, mus, covs)

    def _fast_args(M: int):
        if cheap:
            return dict(verbose=verbose, precomp=pc_bounds, cheap=True,
                        parallel=False, cheap_block_size=None, far_N=far_N)
        if parallel:
            n_workers = 4
            chunk = int(np.ceil(max(1, M) / n_workers))
            return dict(verbose=verbose, precomp=pc_bounds, cheap=False,
                        parallel=True, n_workers=n_workers, chunk_size=chunk,
                        far_N=far_N)
        return dict(verbose=verbose, precomp=pc_bounds, cheap=False,
                    parallel=False, n_workers=1, chunk_size=1024,
                    far_N=far_N)

    def _as_Nvec(N_x):
        if np.isscalar(N_x):
            return np.full(D, int(N_x), dtype=int)
        Nvec = np.asarray(N_x, int)
        if Nvec.shape != (D,):
            raise ValueError("N_x must be int or shape (D,)")
        return Nvec

    def _make_uniform_grid(X_dom, Nvec):
        w = (X_dom[:, 1] - X_dom[:, 0]) / Nvec
        mids_1d = [X_dom[d, 0] + (0.5 + np.arange(Nvec[d])) * w[d] for d in range(D)]
        grids = np.meshgrid(*mids_1d, indexing="ij")
        midpts = np.stack(grids, axis=-1).reshape(-1, D)
        widths = np.broadcast_to(w, (midpts.shape[0], D)).copy()
        return midpts, widths

    def _boxes(mid, wid):
        half = 0.5 * wid
        return mid - half, mid + half

    ev_lo, ev_hi = X_event[:, 0], X_event[:, 1]
    def _event_masks(midpts, widths):
        half = 0.5 * widths
        lo   = midpts - half
        hi   = midpts + half
        intersects = np.all((hi > ev_lo) & (lo < ev_hi), axis=1)
        inside     = np.all((lo >= ev_lo) & (hi <= ev_hi), axis=1)
        return intersects, inside

    def _volumes(wid, out_dtype=dtype):
        return np.prod(wid.astype(np.float64), axis=1).astype(out_dtype)

    def _gap_mass(p_min, p_max, widths):
        return (np.log10((p_max - p_min).astype(np.float64)) +
                np.log10(_volumes(widths, out_dtype=np.float64)))

    def _top_k_desc(scores, k):
        n = scores.size
        if k <= 0:
            return np.empty(0, dtype=int)
        if k >= n:
            return np.argsort(scores)[::-1]
        part = np.argpartition(scores, n - k)[n - k:]
        return part[np.argsort(scores[part])[::-1]]

    def _budget(cap, total):
        return total if cap is None else min(int(cap), total)

    def _split_along_face(midpts, widths, dim, face, tiny=0.0):
        mid = midpts; wid = widths
        half = 0.5 * wid
        lo = mid - half; hi = mid + half

        straddle = (lo[:, dim] < face) & (hi[:, dim] > face)
        if not np.any(straddle):
            return mid, wid

        keep = ~straddle
        mid_keep, wid_keep = mid[keep], wid[keep]

        mid_s, wid_s = mid[straddle], wid[straddle]
        lo_s, hi_s = lo[straddle], hi[straddle]

        wid_low = wid_s.copy()
        wid_low[:, dim] = np.maximum(face - lo_s[:, dim], tiny)
        mid_low = mid_s.copy()
        mid_low[:, dim] = lo_s[:, dim] + 0.5 * wid_low[:, dim]

        wid_up = wid_s.copy()
        wid_up[:, dim] = np.maximum(hi_s[:, dim] - face, tiny)
        mid_up = mid_s.copy()
        mid_up[:, dim] = face + 0.5 * wid_up[:, dim]

        mid_new = np.concatenate([mid_keep, mid_low, mid_up], axis=0)
        wid_new = np.concatenate([wid_keep, wid_low, wid_up], axis=0)
        return mid_new, wid_new

    def align_grid_to_event_faces(midpts, widths, X_event, verbose=False):
        ev_lo = X_event[:, 0]; ev_hi = X_event[:, 1]
        for d in range(X_event.shape[0]):
            midpts, widths = _split_along_face(midpts, widths, d, float(ev_lo[d]))
            if verbose:
                print(f"    [align] split dim {d} low -> cells={midpts.shape[0]}")
            midpts, widths = _split_along_face(midpts, widths, d, float(ev_hi[d]))
            if verbose:
                print(f"    [align] split dim {d} high -> cells={midpts.shape[0]}")
        return midpts, widths

    # child placement grid
    s_1d  = [(0.5 + np.arange(r)) / r for _ in range(D)]
    sgrid = np.stack(np.meshgrid(*s_1d, indexing="ij"), axis=-1).reshape(-1, D)
    C     = int(sgrid.shape[0])

    def _refine(midpts, widths, p_min, p_max, sel_mask, tag):
        if not np.any(sel_mask):
            return midpts, widths, p_min, p_max

        keep_mask  = ~sel_mask
        kept_mid   = midpts[keep_mask]
        kept_wid   = widths[keep_mask]
        kept_pmin  = p_min[keep_mask]
        kept_pmax  = p_max[keep_mask]

        parent_mid = midpts[sel_mask]
        parent_wid = widths[sel_mask]
        parent_lo, _ = _boxes(parent_mid, parent_wid)

        Pn = parent_lo.shape[0]
        child_mid = parent_lo[:, None, :] + parent_wid[:, None, :] * sgrid[None, :, :]
        child_wid = (parent_wid / r)[:, None, :]

        child_mid = child_mid.reshape(Pn * C, D)
        child_wid = np.broadcast_to(child_wid, (Pn, C, D)).reshape(Pn * C, D)

        c_pmin, c_pmax = gmm_bounds_over_cells(
            ws, mus, covs,
            child_mid.astype(np.float64), child_wid.astype(np.float64),
            **_fast_args(child_mid.shape[0])
        )

        mid_new  = np.concatenate([kept_mid,  child_mid], axis=0)
        wid_new  = np.concatenate([kept_wid,  child_wid], axis=0)
        pmin_new = np.concatenate([kept_pmin, c_pmin],   axis=0)
        pmax_new = np.concatenate([kept_pmax, c_pmax],   axis=0)

        if verbose:
            print(f"    {tag} parents={Pn} -> children={child_mid.shape[0]} | "
                  f"cells_after={mid_new.shape[0]}")

        return mid_new, wid_new, pmin_new, pmax_new

    # -------------------
    # Initial grid & bounds
    # -------------------
    Nvec = _as_Nvec(N_x)
    midpts, widths = _make_uniform_grid(X_dom, Nvec)

    levels = max(0, int(N_degree) - 1)
    if verbose:
        print(f"[init] cells={midpts.shape[0]} | D={D} | levels={levels} | r={r}")

    p_min, p_max = gmm_bounds_over_cells(
        ws, mus, covs,
        midpts.astype(np.float64), widths.astype(np.float64),
        **_fast_args(midpts.shape[0])
    )

    if use_event_guidance:
        if verbose:
            print("[event-align] aligning grid to X_event faces (one shot)")
        midpts, widths = align_grid_to_event_faces(midpts, widths, X_event, verbose=verbose)
        p_min, p_max = gmm_bounds_over_cells(
            ws, mus, covs,
            midpts.astype(np.float64), widths.astype(np.float64),
            **_fast_args(midpts.shape[0])
        )
    else:
        if verbose:
            print("[event-align] skipped")

    # -------------------
    # Gap-based refinement
    # -------------------
    def _select_and_refine(pool_idx, tag):
        nonlocal midpts, widths, p_min, p_max, gap_mass
        if pool_idx.size == 0:
            if verbose:
                print(f"    {tag} parents=0")
            return False

        k = _budget(cap_per_degree, pool_idx.size)
        if k <= 0:
            if verbose:
                print(f"    {tag} parents=0 (cap={cap_per_degree})")
            return False

        sel_local = _top_k_desc(gap_mass[pool_idx], k)
        sel = pool_idx[sel_local]

        if verbose:
            print(f"    {tag} parents={sel.size} | top_gap_mass max={float(np.max(gap_mass[sel])):.3e}")

        mask = np.zeros(midpts.shape[0], dtype=bool)
        mask[sel] = True
        midpts, widths, p_min, p_max = _refine(midpts, widths, p_min, p_max, mask, tag)
        return True

    if use_gmm_guidance:
        for level in range(levels):
            if verbose:
                print(f"[gap {level+1}/{levels}] cells_before={midpts.shape[0]}")
            gap_mass = _gap_mass(p_min, p_max, widths)
            pool_idx = np.arange(midpts.shape[0])
            _select_and_refine(pool_idx, "[gap:global]")
    else:
        if verbose:
            print("[gap] skipped")

    # -------------------
    # Finalize
    # -------------------
    mask_intersects_event, mask_inside_event = _event_masks(midpts, widths)
    volumes = _volumes(widths)

    # cell-wise B1, zeroed for far cells
    if far_N is not None:
        ws_eff = np.asarray(ws, float); ws_eff /= ws_eff.sum()
        mu_eff, Sigma_eff = _effective_mean_cov(ws_eff, mus, covs)
        invSigma_eff = _safe_inv_spd(Sigma_eff)
        mask_far_final = _mask_far_cells(midpts, mu_eff, invSigma_eff, far_N)
    else:
        mask_far_final = np.zeros(midpts.shape[0], dtype=bool)

    B1_vec = np.full(midpts.shape[0], B1, dtype=np.float64)
    B1_vec[mask_far_final] = 0.0

    p_minus = np.maximum(0.0, p_min - B1_vec)
    p_plus  = p_max + B1_vec

    base_mass = float(np.dot(p_minus.astype(np.float64), volumes.astype(np.float64)))
    if base_mass > 1.0 + 1e-9:
        raise RuntimeError(
            f"Infeasible baseline from p^-: {base_mass:.6f} > 1.0."
        )

    if verbose:
        gap_abs = (p_max - p_min).astype(np.float64) * volumes.astype(np.float64)
        print(f"[final] cells={midpts.shape[0]} | max_gap_mass={float(np.max(gap_abs)):.6e}")

    return {
        "midpts":     midpts.astype(dtype, copy=False),
        "widths":     widths.astype(dtype, copy=False),
        "volumes":    volumes.astype(dtype, copy=False),
        "mask_upper": mask_intersects_event,
        "mask_lower": mask_inside_event,
        "p_minus":    p_minus.astype(dtype, copy=False),
        "p_plus":     p_plus.astype(dtype, copy=False),
    }


# ============================================================
# LP algorithms
# ============================================================
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
    formatted_numbers = [f"{num:.4e}" for num in ls]
    # Use str.join() to combine the formatted strings with a comma
    output_string = ', '.join(formatted_numbers)
    print(output_string)


def save_problem_result_npz(path, data, 
    keys=("D", "N_x", "X_dom", "X_event", "time_points_ref", "time_points",
          "Pr_ref", "Pr_keys", "Pr_keys_labels", "Pr_opt_upper", "Pr_opt_lower",
          "Pr_lp", "Pr_ut", "Pr_gmm", "Pr_pinngmm"),
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