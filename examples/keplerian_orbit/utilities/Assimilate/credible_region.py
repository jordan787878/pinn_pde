"""
credible_region.py

Tools for computing outer credible regions for 1D density-ratio classes
defined via a PDF band

    l(x) = max(GMM(x) - B1, 0)
    u(x) = GMM(x) + B1,

where GMM(x) is a 1D Gaussian mixture density.
"""

import numpy as np


def make_gmm_pdf_1d(weights, means, covs):
    """
    Return a callable gmm_pdf(x) for a 1D Gaussian mixture.

    Parameters
    ----------
    weights : (K,)
    means   : (K,) or (K,1)
    covs    : (K,) or (K,1,1)  (variances in 1D)

    Returns
    -------
    gmm_pdf : callable
        Function mapping x -> mixture density at x.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)          # (K,)
    m = np.asarray(means, dtype=float).reshape(-1, 1)         # (K,1)
    c = np.asarray(covs, dtype=float)
    if c.ndim == 1:
        var = c.reshape(-1)                                   # (K,)
    else:
        var = c.reshape(-1, 1, 1)[:, 0, 0]                    # (K,)
    if np.any(var <= 0):
        raise ValueError("All variances must be positive in 1D.")
    std = np.sqrt(var).reshape(-1, 1)                         # (K,1)

    # precompute normalizing constants
    norm_consts = 1.0 / np.sqrt(2.0 * np.pi * std**2)         # (K,1)

    def gmm_pdf(x):
        x = np.asarray(x, dtype=float)
        x_flat = x.reshape(1, -1)                             # (1,N)
        # (K,N): each row = component pdf over x
        exponents = -0.5 * ((x_flat - m)**2 / (std**2))
        comp_pdfs = norm_consts * np.exp(exponents)           # (K,N)
        mixture = np.dot(w, comp_pdfs)                        # (N,)
        return mixture.reshape(x.shape)                       # match input shape

    return gmm_pdf


def _construct_quantile_bound(x, l, u, conf_level):
    if l.shape != x.shape:
        raise ValueError(f"l_func(x) must return shape {x.shape}, got {l.shape}")
    if u.shape != x.shape:
        raise ValueError(f"u_func(x) must return shape {x.shape}, got {u.shape}")

    if np.any(l < 0) or np.any(u < 0):
        raise ValueError("l(x) and u(x) must be nonnegative everywhere on the grid.")

    if np.any(l > u + 1e-14):
        raise ValueError("l(x) must not exceed u(x) at any grid point.")

    if np.all(l == 0):
        raise ValueError("Lower bound l(x) is identically zero on the grid.")

    dx = np.diff(x)

    # ---- numerical integrals for extremal CDFs ----
    def _cumulative_trapz(f):
        """
        Left cumulative integral of f over x via trapezoidal rule:
        F_left[i] = ∫_{x_min}^{x[i]} f(t) dt
        """
        return np.concatenate(
            ([0.0], np.cumsum(0.5 * (f[:-1] + f[1:]) * dx))
        )

    int_u_left = _cumulative_trapz(u)
    int_l_left = _cumulative_trapz(l)

    total_u = float(int_u_left[-1])
    total_l = float(int_l_left[-1])

    int_u_right = total_u - int_u_left
    int_l_right = total_l - int_l_left

    eps = 1e-15

    # Upper CDF bound
    denom_upper = int_u_left + int_l_right
    F_upper = int_u_left / np.maximum(denom_upper, eps)

    # Lower CDF bound
    denom_lower = int_l_left + int_u_right
    F_lower = int_l_left / np.maximum(denom_lower, eps)

    # enforce monotonicity & [0,1] range for numerical robustness
    F_upper = np.maximum.accumulate(F_upper)
    F_lower = np.maximum.accumulate(F_lower)
    F_upper = np.clip(F_upper, 0.0, 1.0)
    F_lower = np.clip(F_lower, 0.0, 1.0)

    # ---- quantile bounds ----
    tail = 0.5 * (1.0 - conf_level)
    p_left = tail
    p_right = 1.0 - tail

    q_min_left = float(np.interp(p_left, F_upper, x))
    q_max_right = float(np.interp(p_right, F_lower, x))

    return q_min_left, q_max_right, x, F_upper, F_lower


def outer_credible_region_1d_from_bounds(
    l_func,
    u_func,
    conf_level=0.95,
    num_grid=4001,
    x_min=-10.0,
    x_max=10.0,
):
    """
    Compute the outer (conservative) 100*conf_level % credible region for a 1D
    Density Ratio Class defined by (possibly unnormalized) PDF bounds l(x), u(x).

    The class is
        Γ = { f(x) / ∫ f : l(x) <= f(x) <= u(x) }.

    For this class, extremal CDF bounds are

        F_upper(theta) = ∫_{-∞}^theta u /
                         ( ∫_{-∞}^theta u + ∫_theta^∞ l )

        F_lower(theta) = ∫_{-∞}^theta l /
                         ( ∫_{-∞}^theta l + ∫_theta^∞ u )

    Then, for p in (0,1),

        q_min(p) solves F_upper(q_min(p)) = p   (smallest possible p-quantile)
        q_max(p) solves F_lower(q_max(p)) = p   (largest  possible p-quantile)

    The outer conf_level credible interval is

        [ q_min(alpha), q_max(1 - alpha) ],

    where alpha = (1 - conf_level) / 2.

    Parameters
    ----------
    l_func, u_func : callables
        Functions mapping ndarray x -> ndarray of same shape with
        nonnegative values, representing lower and upper bounds l(x), u(x).
        They need not be normalized PDFs.
    conf_level : float, optional
        Desired credible mass, e.g. 0.95 for a 95% outer credible interval.
    num_grid : int, optional
        Number of grid points for x.
    x_min, x_max : float, optional
        Grid bounds.

    Returns
    -------
    q_left : float
        Left endpoint of the outer credible region.
    q_right : float
        Right endpoint of the outer credible region.
    x_grid : np.ndarray, shape (N,)
        Grid of x-values used for the numerical approximation.
    F_upper : np.ndarray, shape (N,)
        Upper bound CDF evaluated on x_grid.
    F_lower : np.ndarray, shape (N,)
        Lower bound CDF evaluated on x_grid.
    """
    # ---- construct x-grid ----
    x = np.linspace(x_min, x_max, num_grid)  # (N,)
    # ---- evaluate l(x), u(x) on grid ----
    l = np.asarray(l_func(x), dtype=float)
    u = np.asarray(u_func(x), dtype=float)
    return _construct_quantile_bound(x, l, u, conf_level=conf_level)


def outer_credible_region_gmm_1d(
    weights,
    means,
    covs,
    B1,
    conf_level=0.95,
    num_grid=4001,
    # grid_margin=7.0,
    x_min=-6,
    x_max=6,
):
    """
    Compute the outer (conservative) 100*(1-alpha)% credible region for a 1D
    density-ratio class defined by PDF bounds

        l(x) = max(GMM(x) - B1, 0)
        u(x) = GMM(x) + B1

    where GMM is a 1D Gaussian mixture density.

    For (possibly unnormalized) l(x), u(x), approximate extremal CDFs as

        F_upper(theta) = ∫_{-∞}^theta u /
                         ( ∫_{-∞}^theta u + ∫_theta^∞ l )

        F_lower(theta) = ∫_{-∞}^theta l /
                         ( ∫_{-∞}^theta l + ∫_theta^∞ u )

    Then, for p in (0,1),

        q_min(p)  solves F_upper(q_min(p)) = p  (smallest possible p-quantile)
        q_max(p)  solves F_lower(q_max(p)) = p  (largest possible p-quantile)

    The outer (1-alpha) credible interval is

        [ q_min(alpha), q_max(1 - alpha) ].

    Parameters
    ----------
    weights : np.ndarray, shape (K,)
        GMM mixture weights (must sum to 1).
    means : np.ndarray, shape (K, 1)
        Component means for the 1D GMM.
    covs : np.ndarray, shape (K, 1, 1)
        Component covariance matrices for the 1D GMM.
    B1 : float
        Half-width of the PDF interval: l(x) = max(GMM(x) - B1, 0),
        u(x) = GMM(x) + B1.
    alpha : float, optional
        Tail probability for the credible interval. Default 0.05 (95%).
    num_grid : int, optional
        Number of grid points in x for numerical integration and inversion.
    grid_margin : float, optional
        Number of standard deviations beyond the extreme means to extend
        the grid, if x_min/x_max are not provided.
    x_min, x_max : float or None, optional
        Optional explicit grid bounds. If None, they are chosen from
        GMM means ± grid_margin * max_std.

    Returns
    -------
    q_left : float
        Left endpoint of the outer 100*(1-alpha)% credible region.
    q_right : float
        Right endpoint of the outer 100*(1-alpha)% credible region.
    x_grid : np.ndarray
        Grid of x-values used for the numerical approximation.
    F_upper : np.ndarray
        Upper bound CDF evaluated on x_grid.
    F_lower : np.ndarray
        Lower bound CDF evaluated on x_grid.
    """
    # ---- basic validation & shapes ----
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1, 1)      # (K,1)
    covs = np.asarray(covs, dtype=float).reshape(-1, 1, 1)     # (K,1,1)

    K = weights.shape[0]
    if means.shape != (K, 1):
        raise ValueError(f"means must have shape (K,1), got {means.shape}")
    if covs.shape != (K, 1, 1):
        raise ValueError(f"covs must have shape (K,1,1), got {covs.shape}")

    vars_ = covs[:, 0, 0]
    if np.any(vars_ <= 0):
        raise ValueError("All covariance entries must be positive in 1D.")

    stds = np.sqrt(vars_)

    # ---- construct x-grid ----
    if x_min is None:
        x_min = float(np.min(means) - grid_margin * np.max(stds))
    if x_max is None:
        x_max = float(np.max(means) + grid_margin * np.max(stds))

    x = np.linspace(x_min, x_max, num_grid)  # (N,)

    # ---- evaluate GMM pdf ----
    gmm_pdf_func = make_gmm_pdf_1d(weights, means, covs)
    gmm_pdf = gmm_pdf_func(x)

    # ---- bounding functions l(x), u(x) ----
    l = np.clip(gmm_pdf - B1, 0.0, None)
    u = gmm_pdf + B1
    return _construct_quantile_bound(x, l, u, conf_level=conf_level)


def outer_credible_region_gmm_1d_with_measurement(
    weights,
    means,
    covs,
    B1,
    y,
    like_func,
    conf_level=0.95,
    num_grid=4001,
    grid_margin=6.0,
    x_min=None,
    x_max=None,
):
    """
    Compute the outer (conservative) 100*(1-alpha)% credible region for a 1D
    density-ratio class defined by PDF bounds

        l(x) = max(GMM(x) - B1, 0)
        u(x) = GMM(x) + B1

    where GMM is a 1D Gaussian mixture density.

    For (possibly unnormalized) l(x), u(x), approximate extremal CDFs as

        F_upper(theta) = ∫_{-∞}^theta u /
                         ( ∫_{-∞}^theta u + ∫_theta^∞ l )

        F_lower(theta) = ∫_{-∞}^theta l /
                         ( ∫_{-∞}^theta l + ∫_theta^∞ u )

    Then, for p in (0,1),

        q_min(p)  solves F_upper(q_min(p)) = p  (smallest possible p-quantile)
        q_max(p)  solves F_lower(q_max(p)) = p  (largest possible p-quantile)

    The outer (1-alpha) credible interval is

        [ q_min(alpha), q_max(1 - alpha) ].

    Parameters
    ----------
    weights : np.ndarray, shape (K,)
        GMM mixture weights (must sum to 1).
    means : np.ndarray, shape (K, 1)
        Component means for the 1D GMM.
    covs : np.ndarray, shape (K, 1, 1)
        Component covariance matrices for the 1D GMM.
    B1 : float
        Half-width of the PDF interval: l(x) = max(GMM(x) - B1, 0),
        u(x) = GMM(x) + B1.
    alpha : float, optional
        Tail probability for the credible interval. Default 0.05 (95%).
    num_grid : int, optional
        Number of grid points in x for numerical integration and inversion.
    grid_margin : float, optional
        Number of standard deviations beyond the extreme means to extend
        the grid, if x_min/x_max are not provided.
    x_min, x_max : float or None, optional
        Optional explicit grid bounds. If None, they are chosen from
        GMM means ± grid_margin * max_std.

    Returns
    -------
    q_left : float
        Left endpoint of the outer 100*(1-alpha)% credible region.
    q_right : float
        Right endpoint of the outer 100*(1-alpha)% credible region.
    x_grid : np.ndarray
        Grid of x-values used for the numerical approximation.
    F_upper : np.ndarray
        Upper bound CDF evaluated on x_grid.
    F_lower : np.ndarray
        Lower bound CDF evaluated on x_grid.
    """
    # ---- basic validation & shapes ----
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1, 1)      # (K,1)
    covs = np.asarray(covs, dtype=float).reshape(-1, 1, 1)     # (K,1,1)

    K = weights.shape[0]
    if means.shape != (K, 1):
        raise ValueError(f"means must have shape (K,1), got {means.shape}")
    if covs.shape != (K, 1, 1):
        raise ValueError(f"covs must have shape (K,1,1), got {covs.shape}")

    vars_ = covs[:, 0, 0]
    if np.any(vars_ <= 0):
        raise ValueError("All covariance entries must be positive in 1D.")

    stds = np.sqrt(vars_)

    # ---- construct x-grid ----
    if x_min is None:
        x_min = float(np.min(means) - grid_margin * np.max(stds))
    if x_max is None:
        x_max = float(np.max(means) + grid_margin * np.max(stds))

    x = np.linspace(x_min, x_max, num_grid)  # (N,)

    # ---- evaluate GMM pdf ----
    gmm_pdf_func = make_gmm_pdf_1d(weights, means, covs)
    gmm_pdf = gmm_pdf_func(x)

    likelihoods = like_func(y, x)

    # ---- bounding functions l(x), u(x) ----
    l = np.clip(gmm_pdf - B1, 0.0, None)
    u = gmm_pdf + B1
    l = l*likelihoods
    u = u*likelihoods

    return _construct_quantile_bound(x, l, u, conf_level=conf_level)


# --- Old Versions ---

def gmm_pdf(x, weights, means, covars):
    weights_flat = weights.flatten()
    means_flat = means.flatten()
    covars_flat = covars.flatten()
    stds = np.sqrt(covars_flat)
    # vectorized mixture pdf at scalar x
    return np.sum(weights_flat * norm.pdf(x, loc=means_flat, scale=stds))


def gmm_cdf(x, weights, means, covars):
    weights_flat = weights.flatten()
    means_flat = means.flatten()
    covars_flat = covars.flatten()
    cdf_val = 0.0
    for weight, mean, covar in zip(weights_flat, means_flat, covars_flat):
        std = np.sqrt(covar)
        cdf_val += weight * norm.cdf(x, loc=mean, scale=std)
    return cdf_val


def find_gmm_percentile(target_p, weights, means, covars):
    means_flat = means.flatten()
    covars_flat = covars.flatten()
    func = lambda x: gmm_cdf(x, weights, means, covars) - target_p
    min_mean = np.min(means_flat)
    max_mean = np.max(means_flat)
    max_std = np.max(np.sqrt(covars_flat))

    lower_bound = min_mean - 10 * max_std
    upper_bound = max_mean + 10 * max_std

    if func(lower_bound) > 0:
        while func(lower_bound) > 0:
            lower_bound -= max_std
    if func(upper_bound) < 0:
        while func(upper_bound) < 0:
            upper_bound += max_std

    percentile_value = brentq(func, lower_bound, upper_bound)
    return percentile_value


def find_gmm_mode_in_interval(weights, means, covars, lower_bound, upper_bound):
    """
    Find argmax_x GMM(x) for x in [lower_bound, upper_bound].
    """

    def neg_pdf(x):
        return -gmm_pdf(x, weights, means, covars)

    # bounded 1D optimization
    res = minimize_scalar(neg_pdf, bounds=(lower_bound, upper_bound), method="bounded")
    mode_x = res.x

    # (optional) you can also return the pdf value at the mode if useful:
    mode_pdf = gmm_pdf(mode_x, weights, means, covars)
    return mode_x, mode_pdf


def calculate_gmm_prediction_interval(weights, means, covars, confidence_level=0.95):
    alpha = 1.0 - confidence_level
    lower_p = alpha / 2.0
    upper_p = 1.0 - alpha / 2.0

    lower_bound = find_gmm_percentile(lower_p, weights, means, covars)
    upper_bound = find_gmm_percentile(upper_p, weights, means, covars)

    # find maximum likelihood point (mode) within this CI
    ml_point, ml_pdf_val = find_gmm_mode_in_interval(
        weights, means, covars, lower_bound, upper_bound
    )

    # # You can choose whatever return format you like; e.g.:
    # return {
    #     "interval": [lower_bound, upper_bound],
    #     "ml_point": ml_point,
    #     "ml_pdf": ml_pdf_val,   # optional
    # }
    return [[lower_bound, upper_bound]], ml_point # Return as list of intervals


def get_confidence_interval_core(ws, mus, covs, confidence_level=0.95):
    ci, ml_point = calculate_gmm_prediction_interval(ws, mus, covs, confidence_level=confidence_level)
    return np.array(ci[0]).reshape((1, -1)), ml_point