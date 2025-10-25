import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.generalsolvers import gmm_bounds_over_cells


import numpy as np
import pytest


def test_gmm_bounds_over_cells_one_cell_any_dim():
    rng = np.random.default_rng(1234)  # reproducible
    Ds = [1, 2, 3, 4, 5, 6]                  # test multiple D in one function

    for D in Ds:
        # --- Random D-D GMM (K components, SPD covariances) ---
        K = 11
        weights = rng.random(K); weights /= weights.sum()
        means = rng.normal(0.0, 2., size=(K, D))

        def _rand_spd_DxD():
            A = rng.normal(size=(D, D))
            S = A @ A.T
            # jitter for conditioning; scale lightly with D
            S += 0.2 * np.eye(D)
            return S

        covariances = np.stack([_rand_spd_DxD() for _ in range(K)], axis=0)  # (K,D,D)

        # --- One random axis-aligned cell (midpoint + per-dim widths) ---
        dom_lo = np.full(D, -3.0)
        dom_hi = np.full(D,  3.0)
        w = rng.uniform(0.4, 1.2, size=D)
        lo = rng.uniform(dom_lo, dom_hi - w)   # ensures [lo, lo+w] ⊆ domain
        hi = lo + w
        mid = 0.5 * (lo + hi)

        midpoints = mid.reshape(1, D)   # (1,D)
        widths    = w.reshape(1, D)     # (1,D) per-cell widths

        # --- Bounds for this cell (your function must be in scope) ---
        # from your_module import gmm_bounds_over_cells
        p_min, p_max = gmm_bounds_over_cells(
            weights, means, covariances, midpoints, widths
        )
        pmin, pmax = float(p_min[0]), float(p_max[0])

        # --- Vectorized D-D GMM pdf ---
        # phi(x) = (2π)^(-D/2) |Σ|^(-1/2) exp(-1/2 (x-μ)^T Σ^{-1} (x-μ))
        def gmm_pdf_batch(X):
            X = np.asarray(X, float)              # (N,D)
            K = weights.shape[0]
            inv = np.empty((K, D, D), float)
            logdet = np.empty(K, float)
            for k in range(K):
                L = np.linalg.cholesky(covariances[k])
                inv[k] = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D)))
                logdet[k] = 2.0 * np.log(np.diag(L)).sum()
            const = (2.0 * np.pi) ** (D / 2.0)
            out = np.zeros(X.shape[0], float)
            for k in range(K):
                d = X - means[k]
                q = np.einsum("ni,ij,nj->n", d, inv[k], d, optimize=True)
                Nk = np.exp(-0.5 * q) / (const * np.exp(0.5 * logdet[k]))
                out += weights[k] * Nk
            return out

        # --- Monte Carlo inside the cell & assertions ---
        N = 20000 if D <= 3 else 10000  # keep runtime sane in higher D
        eps = 1e-12
        # Uniform samples over [lo, hi) per dimension
        X = rng.uniform(lo, hi, size=(N, D))
        vals = gmm_pdf_batch(X)

        # Optional debug prints if a failure occurs
        print("\n")
        print(f"D={D}  bounds: [{pmin:.6g}, {pmax:.6g}]  samples: [{vals.min():.6g}, {vals.max():.6g}]")

        assert np.all(vals >= pmin - eps), f"value below lower bound for the cell (D={D})"
        assert np.all(vals <= pmax + eps), f"value above upper bound for the cell (D={D})"


# --------------------------- Test Helpers ---------------------------

def _rand_spd(rng, D):
    """
    Draw a random symmetric positive-definite (SPD) matrix via
    Wishart-like construction plus diagonal jitter for conditioning.
    """
    A = rng.normal(size=(D, D))
    S = A @ A.T
    S += 0.2 * np.eye(D)
    return S

def _rand_means(rng, K, D, spread=3.5, cluster_frac=0.35, heavy_frac=0.2):
    """
    Generate 'more random' GMM means to stress-test bounds.
    Modes used:
      - Uniform in [-spread, spread]^D
      - Gaussian with std ≈ spread
      - Heavy-tailed (Student-t, df=3)
      - Clustered around random centers with local Gaussian noise
    """
    means = np.empty((K, D), float)
    C = max(1, int(0.2 * K))
    centers = rng.uniform(-spread, spread, size=(C, D))

    p_uniform = 0.25
    p_gauss   = 0.25
    p_heavy   = heavy_frac
    p_cluster = cluster_frac
    ps = np.array([p_uniform, p_gauss, p_heavy, p_cluster], float)
    ps = ps / ps.sum()

    modes = rng.choice(4, size=K, p=ps)
    for k, mode in enumerate(modes):
        if mode == 0:  # uniform
            means[k] = rng.uniform(-spread, spread, size=D)
        elif mode == 1:  # broad Gaussian
            means[k] = rng.normal(0.0, spread, size=D)
        elif mode == 2:  # heavy-tailed Student-t
            means[k] = (spread / 1.5) * rng.standard_t(df=3, size=D)
        else:  # clustered
            c = centers[rng.integers(0, C)]
            means[k] = c + rng.normal(0.0, spread * 0.25, size=D)
    return means

def _precompute_inv_logdet(covariances):
    """
    Precompute inverses and log-determinants (via Cholesky) for speed and stability.
    Returns:
      invs: (K,D,D), logdets: (K,)
    """
    K, D, _ = covariances.shape
    invs = np.empty((K, D, D), float)
    logdets = np.empty(K, float)
    for k in range(K):
        L = np.linalg.cholesky(covariances[k])
        invs[k] = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(D)))
        logdets[k] = 2.0 * np.log(np.diag(L)).sum()
    return invs, logdets

def _gmm_pdf_batch_precomp(X, weights, means, invs, logdets):
    """
    Evaluate a GMM pdf for a batch of points X (N,D) using precomputed
    inverses and log-determinants. Memory-friendly: only holds (N,) and (D,) scale arrays.
    """
    X = np.asarray(X, float)
    K, D = means.shape
    const = (2.0 * np.pi) ** (D / 2.0)
    out = np.zeros(X.shape[0], float)
    for k in range(K):
        d = X - means[k]                     # (N,D)
        q = np.einsum("ni,ij,nj->n", d, invs[k], d, optimize=True)  # (N,)
        Nk = np.exp(-0.5 * q) / (const * np.exp(0.5 * logdets[k]))  # (N,)
        out += weights[k] * Nk
    return out

def _streaming_cell_check(rng, lo, hi, weights, means, invs, logdets,
                          pmin, pmax, N_total, batch_cap_bytes=32_000_000):
    """
    Memory-efficient Monte Carlo validation:
      - Uniformly sample points in [lo,hi] in batches.
      - Evaluate GMM pdf on each batch.
      - Check bounds incrementally (avoid keeping all N_total samples in RAM).
    batch_cap_bytes controls approximate memory for X (not counting temporaries).
    """
    D = lo.size
    eps = 1e-12

    # Choose batch size so that X ~ (batch_size * D * 8 bytes) <= batch_cap_bytes
    max_batch = max(10_000, int(batch_cap_bytes // (max(D,1) * 8)))
    remaining = N_total

    # Keep track of sample extrema for optional debugging
    observed_min = np.inf
    observed_max = -np.inf

    while remaining > 0:
        B = min(remaining, max_batch)
        X = rng.uniform(lo, hi, size=(B, D))
        vals = _gmm_pdf_batch_precomp(X, weights, means, invs, logdets)

        # Update running min/max (for optional debug / diagnostics)
        vmin = float(vals.min()); vmax = float(vals.max())
        if vmin < observed_min: observed_min = vmin
        if vmax > observed_max: observed_max = vmax

        # Incremental assertions (fast short-circuit on failure)
        if not np.all(vals >= pmin - eps):
            raise AssertionError(f"Below lower bound: min(vals)={vmin:.4e} vs pmin={pmin:.4e}")
        if not np.all(vals <= pmax + eps):
            raise AssertionError(f"Above upper bound: max(vals)={vmax:.4e} vs pmax={pmax:.4e}")

        remaining -= B

    # Return extrema in case the caller wants to log them
    return observed_min, observed_max


# ------------------------ The Parameterized Test ------------------------

@pytest.mark.parametrize("D", [1, 4, 6])
@pytest.mark.parametrize("K", [5, 11])
def test_gmm_bounds_over_cells_many_cells_any_dim(D, K):
    """
    Property-based test (Monte Carlo):
      For a random D-dimensional GMM with K components and many random cells,
      verify that every Monte Carlo sample inside each cell lies within the
      bounds returned by `gmm_bounds_over_cells`.

    Strategy:
      1) Draw a random full-covariance GMM with 'diverse' (heavy-tailed / clustered)
         means to create challenging shapes.
      2) Generate M random axis-aligned cells (boxes) via random widths and locations.
      3) Compute (p_min, p_max) for each cell using the provided bounder.
      4) For each cell, perform a large uniform Monte Carlo inside the box and
         check all pdf values lie within [p_min, p_max], using **batching** to
         keep memory bounded.

    Notes:
      - We precompute inverses & log-determinants once per GMM.
      - Batch size is auto-tuned to keep arrays under ~32 MB by default.
      - This test focuses on **validity** (conservatism) of bounds rather than tightness.
    """
    # Reproducible RNG per (D, K)
    seed = (D * 73856093) ^ (K * 19349663)
    rng = np.random.default_rng(seed)

    # ----- Random GMM with 'more random' means -----
    weights = rng.random(K); weights /= weights.sum()
    means   = _rand_means(rng, K, D, spread=3.5, cluster_frac=0.35, heavy_frac=0.2)
    covariances = np.stack([_rand_spd(rng, D) for _ in range(K)], axis=0)

    # Precompute inverses & log-dets (shared across all cells and batches)
    invs, logdets = _precompute_inv_logdet(covariances)

    # ----- Random cells (many) -----
    # High N_samples per cell to stress the bounds; M chosen for coverage.
    M = 100
    N_samples = 1_000_000

    dom_lo = np.full(D, -5.0)
    dom_hi = np.full(D,  5.0)

    widths = rng.uniform(0.4, 1.8, size=(M, D))
    los = np.empty((M, D)); his = np.empty((M, D)); mids = np.empty((M, D))
    for i in range(M):
        lo = rng.uniform(dom_lo, dom_hi - widths[i])
        hi = lo + widths[i]
        los[i], his[i] = lo, hi
        mids[i] = 0.5 * (lo + hi)

    # ----- Compute bounds for all cells (function under test) -----
    # Ensure gmm_bounds_over_cells is imported or defined in scope.
    p_min, p_max = gmm_bounds_over_cells(weights, means, covariances, mids, widths)

    # ----- Monte Carlo checks with batching (memory efficient) -----
    for i in range(M):
        lo, hi = los[i], his[i]
        pmin_i = float(p_min[i]); pmax_i = float(p_max[i])

        _streaming_cell_check(
            rng=rng,
            lo=lo,
            hi=hi,
            weights=weights,
            means=means,
            invs=invs,
            logdets=logdets,
            pmin=pmin_i,
            pmax=pmax_i,
            N_total=N_samples,
            batch_cap_bytes=32_000_000,  # ≈32MB cap for X per batch
        )
