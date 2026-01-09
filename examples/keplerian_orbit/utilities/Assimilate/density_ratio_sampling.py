import numpy as np


def sample_density_ratio_class_gmm_1d(
    weights,
    means,
    covs,
    B1,
    M,
    x_min,
    x_max,
    rng=None,
    grid_for_u_max=10001,
    max_trials_factor=1000,
):
    """
    Draw weighted samples from a 1D density-ratio class defined by

        l(x) = max(GMM(x) - B1, 0)
        u(x) = GMM(x) + B1,

    where GMM(x) is a 1D Gaussian mixture density.

    This function:
      1. Draws M samples {x_i^u} from the normalized upper function u(x)
         on [x_min, x_max] via rejection sampling with a uniform proposal.
      2. Computes normalized importance weights for the *lower-bound* density

             f_low(x) ∝ l(x),

         using Eq. (10)-style weights

             w_i ∝ l(x_i^u) / u(x_i^u).

    Parameters
    ----------
    weights : np.ndarray, shape (K,)
        GMM mixture weights (must sum to 1).
    means : np.ndarray, shape (K, 1)
        Component means for the 1D GMM.
    covs : np.ndarray, shape (K, 1, 1)
        Component covariance matrices for the 1D GMM.
    B1 : float
        Half-width of the PDF band.
    M : int
        Number of samples to draw from the upper density u(x).
    x_min, x_max : float
        Bounds of the domain on which the density-ratio class is defined.
    rng : np.random.Generator or None, optional
        Random number generator. If None, use np.random.default_rng().
    grid_for_u_max : int, optional
        Number of grid points used to approximate max_x u(x) for rejection sampling.
    max_trials_factor : int, optional
        Safety factor: maximum number of proposal draws is max_trials_factor * M.

    Returns
    -------
    x_u : np.ndarray, shape (M,)
        Samples {x_i^u} drawn from the normalized upper function u(x)
        on [x_min, x_max].
    w : np.ndarray, shape (M,)
        Normalized importance weights corresponding to the lower-bound
        density f_low(x) ∝ l(x).
    """
    if B1 < 0.0:
        raise ValueError("B1 must be non-negative.")

    # ---- basic validation & shapes ----
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1, 1)   # (K,1)
    covs = np.asarray(covs, dtype=float).reshape(-1, 1, 1)  # (K,1,1)

    K = weights.shape[0]
    if means.shape != (K, 1):
        raise ValueError(f"means must have shape (K,1), got {means.shape}")
    if covs.shape != (K, 1, 1):
        raise ValueError(f"covs must have shape (K,1,1), got {covs.shape}")

    vars_ = covs[:, 0, 0]
    if np.any(vars_ <= 0):
        raise ValueError("All covariance entries must be positive in 1D.")

    x_min = float(x_min)
    x_max = float(x_max)
    if x_max <= x_min:
        raise ValueError("Require x_max > x_min.")

    if rng is None:
        rng = np.random.default_rng()

    # ---- helper: GMM pdf in 1D ----
    def gmm_pdf_1d(x):
        """
        Evaluate 1D GMM pdf at x, where x is array-like of shape (N,).
        Returns array of shape (N,).
        """
        x = np.asarray(x, dtype=float)
        x_col = x[None, :]           # (1,N)
        vars_col = vars_[:, None]    # (K,1)

        norm_consts = 1.0 / np.sqrt(2.0 * np.pi * vars_col)   # (K,1)
        exponents = -0.5 * ((x_col - means) ** 2 / vars_col)  # (K,N)
        comp_pdfs = norm_consts * np.exp(exponents)           # (K,N)
        return np.dot(weights, comp_pdfs)                     # (N,)

    # ---- approximate max u(x) on the domain for rejection sampling ----
    grid_x = np.linspace(x_min, x_max, grid_for_u_max)
    gmm_grid = gmm_pdf_1d(grid_x)
    u_grid = gmm_grid + B1
    u_max = float(u_grid.max())
    if u_max <= 0.0:
        raise ValueError("Upper band u(x) is non-positive on the domain; "
                         "check B1, x_min, x_max, or the GMM parameters.")

    # ---- rejection sample from unnormalized u(x) on [x_min, x_max] ----
    x_u = np.empty(M, dtype=float)
    n_accepted = 0
    n_trials = 0
    max_trials = max_trials_factor * M

    while n_accepted < M:
        if n_trials >= max_trials:
            raise RuntimeError(
                "Rejection sampling from u(x) did not converge fast enough. "
                "Consider increasing max_trials_factor or adjusting the domain."
            )

        # Proposal: uniform on [x_min, x_max]
        x_prop = rng.uniform(x_min, x_max)
        u_prop = gmm_pdf_1d([x_prop])[0] + B1

        # Acceptance probability: u(x_prop) / u_max (standard rejection sampling
        # with uniform proposal and envelope height u_max)
        if u_prop > 0.0:
            if rng.random() < (u_prop / u_max):
                x_u[n_accepted] = x_prop
                n_accepted += 1

        n_trials += 1

    # ---- compute importance weights for lower-bound density f_low ∝ l(x) ----
    gmm_vals = gmm_pdf_1d(x_u)
    l_vals = np.clip(gmm_vals - B1, 0.0, None)
    u_vals = gmm_vals + B1

    # Unnormalized importance weights for f_low(x) ∝ l(x):
    #   w_i ∝ l(x_i^u) / u(x_i^u)
    # per Eq. (10) with f = l / ∫ l.
    ratio = l_vals / u_vals  # in [0,1]
    if not np.any(ratio > 0.0):
        raise ValueError(
            "Lower bound l(x) is zero at all sampled points. "
            "B1 may be too large, or the domain too narrow."
        )

    w_unnorm = ratio
    w = w_unnorm / np.sum(w_unnorm)

    return x_u, w
