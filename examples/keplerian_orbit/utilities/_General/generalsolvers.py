import numpy as np
from tqdm import tqdm
# Only dependency for the minimization QP:
import osqp
import scipy.sparse as sp


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


def _solve_min_box_osqp(A, mu, lo, hi, x0=None, eps_abs=1e-10, eps_rel=1e-10):
    """
    Solve min_{lo <= x <= hi} (x-mu)^T A (x-mu)  with A = A^T ⪰ 0
    as:   min 0.5 x^T P x + q^T x  s.t. lo <= x <= hi
          P = 2A, q = -2A mu
    """
    n = A.shape[0]
    P = sp.csc_matrix(2.0 * A)
    q = -2.0 * (A @ mu)
    I = sp.eye(n, format='csc')

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=I, l=lo, u=hi,
                 polishing=False,  # polishing is unnecessary for bound-only QPs when interior
                 eps_abs=eps_abs, eps_rel=eps_rel,
                 verbose=False)
    if x0 is not None:
        solver.warm_start(x=x0)

    res = solver.solve(raise_error=False)
    x = res.x
    d = x - mu
    return float(d @ (A @ d)), x


def gmm_bounds_over_cells(weights, means, covariances, midpoints, widths,
                          verbose=False, precomp=None, cheap=False):
    """
    Clean, single-solver version:
      - min q_k on box: OSQP (QP with simple bounds)
      - max q_k on box: exact Gray code if D<=12, else Rayleigh spectral bound
    Returns:
      p_min, p_max (both shape (M,))
    """
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

    # --- cheap midpoint path ---
    if cheap:
        pc = precomp if precomp is not None else _precompute_gmm(w, m, S)
        invS, consts, ws = pc["invS"], pc["consts"], pc["ws"]
        p = np.empty(M, float)
        for i in range(M):
            x = mids[i]
            logs = np.empty(K, float)
            for k in range(K):
                A = invS[k]
                diff = x - m[k]
                d2 = float(diff @ (A @ diff))
                logs[k] = np.log(ws[k]*consts[k]) - 0.5*d2
            a = np.max(logs)
            p[i] = np.exp(a) * np.sum(np.exp(logs - a))
        return p.copy(), p

    # --- validate widths ---
    if not ((wid.ndim == 1 and wid.shape[0] == D) or (wid.ndim == 2 and wid.shape == (M, D))):
        raise ValueError("widths must be (D,) or (M,D)")
    widths_is_per_cell = (wid.ndim == 2)
    if np.any((wid if widths_is_per_cell else wid) <= 0):
        raise ValueError("widths must be positive")

    # --- precompute constants and A_k ---
    pc = precomp if precomp is not None else _precompute_gmm(w, m, S)
    invS, consts, L_lip, ws = pc["invS"], pc["consts"], pc["L_lip"], pc["ws"]

    # --- outputs ---
    p_min = np.empty(M, float); p_max = np.empty(M, float)

    # gray-code work buffers
    xcorner = np.empty(D, float); diff = np.empty(D, float); Adiff = np.empty(D, float)

    # warm-starts per component
    xwarm = [None] * K

    rng = range(M)
    if verbose:
        try:
            from tqdm import tqdm
            rng = tqdm(range(M), desc="GMM bounds")
        except Exception:
            pass

    for i in rng:
        wi = wid[i] if widths_is_per_cell else wid
        lo = mids[i] - 0.5 * wi
        hi = mids[i] + 0.5 * wi

        logs_max = np.empty(K, float)  # log φ_k^max (uses d2_min)
        logs_min = np.empty(K, float)  # log φ_k^min (uses d2_max)

        for k in range(K):
            A = 0.5 * (invS[k] + invS[k].T)  # ensure symmetry

            # --- min q_k (for p_max): OSQP ---
            d2_min, xwarm[k] = _solve_min_box_osqp(A, m[k], lo, hi, x0=xwarm[k])

            # --- max q_k (for p_min): exact or spectral bound ---
            if D <= 12:
                d2_max = _gray_max_on_box(m[k], A, lo, hi, (xcorner, diff, Adiff))
            else:
                # Rayleigh bound: max_x (x-mu)^T A (x-mu) <= λ_max(A) * ||r||^2
                r = np.maximum(np.abs(lo - m[k]), np.abs(hi - m[k]))
                d2_max = 0.5 * L_lip[k] * float(r @ r)  # L_lip = 2*λ_max(A)

            logs_max[k] = np.log(ws[k]*consts[k]) - 0.5*d2_min
            logs_min[k] = np.log(ws[k]*consts[k]) - 0.5*d2_max

        # --- mixture bounds (log-sum-exp) ---
        a = np.max(logs_max); b = np.max(logs_min)
        p_max[i] = np.exp(a) * np.sum(np.exp(logs_max - a))
        p_min[i] = np.exp(b) * np.sum(np.exp(logs_min - b))

        # sanity
        if p_min[i] > p_max[i] + 1e-12 * max(1.0, p_max[i]):
            raise RuntimeError("Numerical issue: lower bound exceeded upper bound.")

    return p_min, p_max


# ---------------------------------------------

def build_partition_and_bounds(
    problem, ws, mus, covs, *,
    N_degree=2,
    refine_factor=2,
    eps=0.0,
    # NEW: simple Mahalanobis-based guidance
    use_gmm_guidance=True,
    z_thresh=3.0,          # 3-sigma shell
    w_min=0.01,            # only components with weight >= w_min
    verbose=True,
    dtype=np.float32,
    cheap=False,
):
    """
    Non-overlapping, level-by-level refinement.
    At each level, refine parents that (a) intersect X_event, OR (b) whose *midpoint*
    lies within Mahalanobis radius z_thresh of ANY "heavy" Gaussian (w_k >= w_min).

    Then compute tight GMM bounds on each final cell and apply B1:
      p_minus = max(0, p_min - B1),  p_plus = p_max + B1
    """
    X_dom   = np.asarray(problem["X_dom"],   float)
    X_event = np.asarray(problem["X_event"], float)
    N_x     = problem["N_x"]
    D = X_dom.shape[0]
    B1 = float(problem.get("B1", 0.0))
    if B1 < 0:
        raise ValueError("B1 must be nonnegative.")

    # base grid over full X_dom (true partition)
    if np.isscalar(N_x):
        Nvec = np.full(D, int(N_x), dtype=int)
    else:
        Nvec = np.asarray(N_x, int)
        if Nvec.shape != (D,):
            raise ValueError("N_x must be int or shape (D,)")

    if int(refine_factor) < 2:
        raise ValueError("refine_factor must be >= 2")
    r = int(refine_factor)

    base_w = (X_dom[:,1] - X_dom[:,0]) / Nvec
    mids_1d = [X_dom[d,0] + (0.5 + np.arange(Nvec[d])) * base_w[d] for d in range(D)]
    grids = np.meshgrid(*mids_1d, indexing='ij')
    midpts = np.stack(grids, axis=-1).reshape(-1, D).astype(dtype)
    widths = np.broadcast_to(base_w.astype(dtype), (midpts.shape[0], D)).copy()

    def _boxes(mid, wid):
        half = 0.5 * wid
        return (mid - half), (mid + half)

    def _event_masks(mid, wid):
        lo, hi = _boxes(mid.astype(float), wid.astype(float))
        ev_lo, ev_hi = X_event[:,0], X_event[:,1]
        inter = np.all((hi >= ev_lo) & (lo <= ev_hi), axis=1)
        inside = np.all((lo >= (ev_lo - eps)) & (hi <= (ev_hi + eps)), axis=1)
        return inter, inside

    # --- precompute for Mahalanobis-guided selection ---
    if use_gmm_guidance:
        pc = _precompute_gmm(ws, mus, covs)
        invS = pc["invS"]
        weights = pc["ws"]
        mus_np = mus.astype(float)
        heavy = np.flatnonzero(weights >= float(w_min))
        z2 = float(z_thresh) ** 2

    # refinement passes
    levels = max(0, int(N_degree) - 1)
    for level in range(levels):
        inter_event, _ = _event_masks(midpts, widths)
        refine_mask = inter_event.copy()

        if use_gmm_guidance and heavy.size > 0:
            M = midpts.shape[0]
            mids_f = midpts.astype(float)

            # Vectorized over cells for each heavy component:
            # d2_k(i) = (m_i - mu_k)^T A_k (m_i - mu_k)
            d2_any = np.full(M, np.inf)
            for k in heavy:
                A = 0.5 * (invS[k] + invS[k].T)  # ensure symmetry
                diff = mids_f - mus_np[k]        # (M, D)
                Ad   = diff @ A                  # (M, D)
                d2_k = np.einsum('ij,ij->i', diff, Ad)  # (M,)
                d2_any = np.minimum(d2_any, d2_k)

            refine_mask |= (d2_any <= z2)

        n_parents = int(np.count_nonzero(refine_mask))
        if verbose:
            print(f"[level {level+1}/{levels}] parents_to_refine = {n_parents}, cells_before = {midpts.shape[0]}")
        if n_parents == 0:
            continue

        # non-overlapping replacement: remove parents, append children
        keep = ~refine_mask
        kept_mid = midpts[keep]; kept_wid = widths[keep]

        lo_parent, _ = _boxes(midpts.astype(float), widths.astype(float))
        parent_w = widths.astype(float)

        new_mid_list = [kept_mid]; new_wid_list = [kept_wid]
        for i in np.flatnonzero(refine_mask):
            lo = lo_parent[i]; pw = parent_w[i]
            cw = (pw / r).astype(dtype)
            child_mids_1d = [(lo[d] + (0.5 + np.arange(r)) * cw[d]).astype(dtype) for d in range(D)]
            cgrids = np.meshgrid(*child_mids_1d, indexing='ij')
            cmid = np.stack(cgrids, axis=-1).reshape(-1, D).astype(dtype)
            cwid = np.broadcast_to(cw, (cmid.shape[0], D)).astype(dtype)
            new_mid_list.append(cmid); new_wid_list.append(cwid)

        midpts = np.concatenate(new_mid_list, axis=0)
        widths = np.concatenate(new_wid_list, axis=0)
        if verbose:
            print(f"               cells_after  = {midpts.shape[0]}")

    # final masks & volumes
    mask_upper, mask_lower = _event_masks(midpts, widths)
    volumes = np.prod(widths.astype(np.float64), axis=1).astype(dtype)

    # duplicate check (partition sanity)
    def _key(m, w):
        return (tuple(np.asarray(m, float).round(14)),
                tuple(np.asarray(w, float).round(14)))
    seen = set(); dups = 0
    for m,w in zip(midpts, widths):
        k = _key(m,w); dups += (k in seen); seen.add(k)
    if dups:
        raise RuntimeError(f"Grid has {dups} duplicate cells (should be 0).")

    # per-cell GMM bounds (tight), then apply B1
    pc_bounds = _precompute_gmm(ws, mus, covs)
    p_min, p_max = gmm_bounds_over_cells(ws, mus, covs,
                                         midpts.astype(float),
                                         widths.astype(float),
                                         verbose=verbose,
                                         precomp=pc_bounds, cheap=cheap)
    p_minus = np.maximum(0.0, p_min - B1)
    p_plus  = p_max + B1

    base_mass = float(np.dot(p_minus, volumes.astype(float)))
    if base_mass > 1.0 + 1e-9:
        raise RuntimeError(
            f"Infeasible baseline from p^-: {base_mass:.6f} > 1.0. "
            "Likely overlap/duplication or incorrect bounds/B1."
        )

    return {
        "midpts":     midpts.astype(dtype),
        "widths":     widths.astype(dtype),
        "volumes":    volumes.astype(dtype),
        "mask_upper": mask_upper,
        "mask_lower": mask_lower,
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
    with np.load(path, allow_pickle=False) as z:
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