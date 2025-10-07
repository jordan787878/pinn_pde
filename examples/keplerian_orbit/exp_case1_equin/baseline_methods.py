import sympy as sp
import numpy as np
from tqdm import tqdm
from scipy.linalg import expm, cholesky
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture
from exp_utilities.constants import Case1_6D_Constants_Equin
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import load_trained_model
from _General.neuralnetworks import TimeToGMM6D_V0

constants = Case1_6D_Constants_Equin()
SAVE_PATH_LINEAR_PROPAGATE = "output/baseline_methods/linear_propagation.npz"
SAVE_PATH_UNSCENT_PROPAGATE = "output/baseline_methods/unscent_propagation.npz"
SAVE_PATH_GMM_PROPAGATE = "output/baseline_methods/gmm_propagation.npz"


class PropagationData:
    def __init__(self, path: str):
        self.data = np.load(path)
    def get(self, time):
        times = self.data["times"]
        means = self.data["means"]
        covs = self.data["covs"]
        # dt_precision = self.data["dt_precision"]
        # time_threshold = 10. *10**(-1. *dt_precision)
        time = np.round(time, 3)
        idx = np.where(abs(time-times)< 1e-3)[0]
        if len(idx) == 0:
            assert("The propagation result does not have data at this time")
        idx = idx[0]
        if self.data["method"] == "gmm":
            weights = self.data["weights"]
            return times[idx], weights, means[idx, :, :], covs[idx, :, :, :]
        return times[idx], np.array([1.]), means[idx, :], covs[idx, :, :]


def p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    scales = np.sqrt(np.diag(cov))  # std per dimension
    cov_scaled = cov / np.outer(scales, scales)
    x_scaled = (x - mean) / scales
    rv = multivariate_normal(mean=np.zeros(len(mean)), cov=cov_scaled)
    pdf_eval = rv.pdf(x_scaled) / np.prod(scales)  # back-transform
    return pdf_eval.reshape(-1,)


def _get_Jacobian_expression():
    print("")
    x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6', real=True)
    T, MU_EARTH = sp.symbols('T MU_EARTH', real=True, positive=True)
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 0
    f5 = 0
    f6 = ((MU_EARTH/ x1**3)**0.5) * T
    f = sp.Matrix([f1, f2, f3, f4, f5, f6])
    x = sp.Matrix([x1, x2, x3, x4, x5, x6])

    # symbolic Jacobian
    J = sp.simplify(f.jacobian(x))
    print("[info] Jacobian matrix: ", J)


def get_Jacobian(constants, x, dtype=np.float64):
    # force 64-bit everywhere
    x = np.asarray(x, dtype=dtype)
    mu = np.array(constants.MU_EARTH, dtype=dtype)
    T  = np.array(constants.T,        dtype=dtype)

    J = np.zeros((6, 6), dtype=dtype)
    x1 = x[0]

    # -1.5 * sqrt(mu) * T * x1^(-5/2)
    J[5, 0] = -1.5 * np.sqrt(mu) * T / (x1 ** (dtype(2.5)))
    return J


def orbit_dyn(constants, x, dtype=np.float64):
    # force 64-bit everywhere
    x  = np.asarray(x, dtype=dtype)
    mu = np.array(constants.MU_EARTH, dtype=dtype)
    T  = np.array(constants.T,        dtype=dtype)

    f6 = np.sqrt(mu / (x[0] ** dtype(3.0))) * T

    out = np.zeros(6, dtype=dtype)
    out[5] = f6
    return out


def linear_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=None):
    # --- initialization ---
    x = constants.MEAN_I.copy()     # initial mean
    Px = constants.COV_I.copy()     # initial covariance
    x = np.float64(x)
    Px = np.float64(Px)

    dtt = np.float64(10**(-1*dt_precision))
    tf = constants.T_PRIME_SPAN[-1]
    kf = int(np.ceil(tf / dtt))
    current_time = np.float64(constants.T_PRIME_SPAN[0])

    # --- storage arrays ---
    times = [current_time]
    means = [x.copy()]
    covs = [Px.copy()]
    t_to_save = current_time + dt_save

    # --- time stepping loop ---
    for k in tqdm(range(kf), desc="propagting over time"):
        # compute dynamics and Jacobian
        fx = orbit_dyn(constants, x)
        Jx = get_Jacobian(constants, x)
        # propagate mean and covariance: P_dot = Jx * P + P * Jx.T
        x = x + fx * dtt
        Px = Px + (Jx @ Px + Px @ Jx.T)*dtt

        # update time
        current_time += dtt
        # current_time = np.round(current_time, dt_precision)
        if(abs(current_time-t_to_save) < dtt/2):
            # store results
            times.append(np.round(current_time,3))
            means.append(x.copy())
            covs.append(Px.copy())
            t_to_save += dt_save

    # --- convert lists to arrays ---
    times = np.array(times)
    means = np.array(means)       # shape: (kf+1, n)
    covs = np.array(covs)         # shape: (kf+1, n, n)
    np.savez(save_path,
             times=times,
             means=means,
             covs=covs,
             dt_precision=dt_precision,
             method="lp")


# --- New Integrator ---

def linear_propagation_new(constants, dt_save=1e-2, save_path=None):
    """
    Linear (time-varying) propagation using solve_ivp with outputs saved at dt_save.
    Integrates:
        ẋ = f(x)
        Φ̇ = A(x) Φ,  Φ(t0)=I
    and saves:
        times (T,), means (T,6), covs (T,6,6), method="lp"
    """
    # --- init mean/cov (float64) ---
    x0 = np.asarray(constants.MEAN_I, dtype=np.float64)
    P0 = np.asarray(constants.COV_I,  dtype=np.float64)

    t0 = float(constants.T_PRIME_SPAN[0])
    tf = float(constants.T_PRIME_SPAN[-1])

    # save grid
    t_eval = np.arange(t0, tf + 0.5*dt_save, dt_save, dtype=float)

    # augmented initial condition: [x; vec(Φ)], Φ=I
    Phi0 = np.eye(6, dtype=np.float64).ravel()
    y0_aug = np.concatenate([x0, Phi0])

    # augmented RHS: mean dynamics + variational equation
    def rhs_aug(t, y):
        x = y[:6]
        Phi = y[6:].reshape(6, 6)
        fx = orbit_dyn(constants, x)        # (6,)
        A  = get_Jacobian(constants, x)     # (6,6)
        dPhi = A @ Phi
        return np.concatenate([fx, dPhi.ravel()])

    # integrate once on the save grid
    sol = solve_ivp(rhs_aug, (t0, tf), y0_aug,
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, method="RK45")
    if not sol.success:
        raise RuntimeError(f"Variational integration failed: {sol.message}")

    # unpack mean and build covariance at each saved time
    times = sol.t
    means = sol.y[:6, :].T                                # (T,6)
    covs  = np.empty((times.size, 6, 6), dtype=np.float64)
    for k in range(times.size):
        Phi_k = sol.y[6:, k].reshape(6, 6)
        Pk = Phi_k @ P0 @ Phi_k.T
        covs[k] = 0.5 * (Pk + Pk.T)                      # symmetrize

    # save payload (keeps your downstream loader happy)
    if save_path is not None:
        np.savez(save_path,
                 times=np.round(times, 3),
                 means=means,
                 covs=covs,
                 dt_precision=None,   # kept for compatibility; not used
                 method="lp")

    return times, means, covs


# def _unscented_sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0):
#     """
#     Generate Unscented Transform sigma points for N-dim state.

#     Parameters
#     ----------
#     mean : (N,) array_like
#         State mean vector.
#     cov : (N,N) array_like
#         State covariance (symmetric, PSD).
#     alpha : float, optional
#         Spread of the sigma set (small, e.g. 1e-3). Affects higher-order terms.
#     beta : float, optional
#         Prior knowledge about distribution; 2 is optimal for Gaussian.
#     kappa : float, optional
#         Secondary scaling (often 0 or 3-N).

#     Returns
#     -------
#     X : (2N+1, N) ndarray
#         Sigma points. X[0] is the mean, others are +/- columns of the scaled root.
#     Wm : (2N+1,) ndarray
#         Weights for computing the mean.
#     Wc : (2N+1,) ndarray
#         Weights for computing the covariance.

#     """
#     m = np.asarray(mean, dtype=np.float64).reshape(-1)
#     P = np.asarray(cov, dtype=np.float64)
#     N = m.size
#     # kappa = 3 - N
#     lam = alpha**2 * (N + kappa) - N
#     c = N + lam
#     if c <= 0:
#         raise ValueError("N + lambda must be positive; adjust alpha/kappa.")
#     S = cholesky(P, lower=True)
#     S *= np.sqrt(c)  # scale by sqrt(N+lambda)

#     # Sigma points
#     X = np.empty((2*N + 1, N), dtype=float)
#     X[0] = m
#     X[1:N+1]     = m + S.T   # columns of S
#     X[N+1:2*N+1] = m - S.T

#     # Weights
#     Wm = np.full(2*N + 1, 1.0/(2.0*c), dtype=float)
#     Wc = np.full(2*N + 1, 1.0/(2.0*c), dtype=float)
#     Wm[0] = lam / c
#     Wc[0] = lam / c + (1.0 - alpha**2 + beta)
#     return X, Wm, Wc


# def unscent_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=None):
#     """
#     TODO Validate this on Colab 1D OU process
#     """
#     # --- initialization ---
#     x = constants.MEAN_I.copy()     # initial mean
#     Px = constants.COV_I.copy()     # initial covariance
#     x = np.float64(x)    
#     Px = np.float64(Px) 

#     dtt = np.float64(10**(-1*dt_precision))
#     tf = constants.T_PRIME_SPAN[-1]
#     kf = int(np.ceil(tf / dtt))
#     current_time = np.float64(constants.T_PRIME_SPAN[0])

#     # --- storage arrays ---
#     times = [current_time]
#     means = [x.copy()]
#     covs = [Px.copy()]
#     t_to_save = current_time + dt_save

#     # --- time stepping loop ---
#     for k in tqdm(range(kf), desc="propagting over time"):
#         # sigma set (standard params)
#         sig, wm, wc = _unscented_sigma_points(x, Px)

#         # propagate each sigma via Euler
#         for i in range(sig.shape[0]):
#             fx = orbit_dyn(constants, sig[i])
#             sig[i] = sig[i] + fx * dtt

#         # recombine
#         x = wm @ sig
#         Xd = sig - x
#         Px = Xd.T @ (wc[:, None] * Xd)

#         # update time
#         current_time += dtt
#         current_time = np.round(current_time, dt_precision)
#         if(abs(current_time-t_to_save) < dtt/2):
#             # store results
#             times.append(np.round(current_time,3))
#             means.append(x.copy())
#             covs.append(Px.copy())
#             t_to_save += dt_save

#     # --- convert lists to arrays ---
#     times = np.array(times)
#     means = np.array(means)       # shape: (kf+1, n)
#     covs = np.array(covs)         # shape: (kf+1, n, n)
#     np.savez(save_path,
#              times=times,
#              means=means,
#              covs=covs,
#              dt_precision=dt_precision,
#              method="ut")


# --- New Integrator ---

def _unscented_sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0):
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    P = np.asarray(cov, dtype=np.float64)
    n = m.size
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    if c <= 0:
        raise ValueError("N + lambda must be positive; adjust alpha/kappa.")
    S = cholesky(P, lower=True) * np.sqrt(c)

    X = np.empty((2*n + 1, n), dtype=np.float64)
    X[0] = m
    X[1:n+1]     = m + S.T
    X[n+1:2*n+1] = m - S.T

    Wm = np.full(2*n + 1, 1.0/(2.0*c), dtype=np.float64)
    Wc = np.full(2*n + 1, 1.0/(2.0*c), dtype=np.float64)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return X, Wm, Wc


def unscent_propagation(constants,
                        dt_save=1e-2,
                        save_path=None,
                        rtol=1e-10,
                        atol=1e-12,
                        ):
    """
    Unscented propagation with high-accuracy ODE integration between save times.

    Saves mean/cov at t_k = t0 + k*dt_save for k=0..K.
    """
    # --- initial mean/cov ---
    x  = np.asarray(constants.MEAN_I, dtype=np.float64).copy()
    Px = np.asarray(constants.COV_I,  dtype=np.float64).copy()

    t0 = float(constants.T_PRIME_SPAN[0])
    tf = float(constants.T_PRIME_SPAN[-1])
    t_eval = np.arange(t0, tf + 0.5*dt_save, dt_save, dtype=np.float64)

    # storage
    times = []
    means = []
    covs  = []

    # save at t0
    times.append(np.round(t0, 3))
    means.append(x.copy())
    covs.append(Px.copy())

    # RHS wrapper for solve_ivp
    def rhs(_t, state):
        return orbit_dyn(constants, state)

    # march over save grid, propagating sigma points each interval
    for k in tqdm(range(1, len(t_eval)), desc="UT (solve_ivp)"):
        ta = t_eval[k-1]
        tb = t_eval[k]

        # sigma set at current (x, Px)
        sig, wm, wc = _unscented_sigma_points(x, Px)

        # propagate each sigma from ta -> tb using solve_ivp
        sig_next = np.empty_like(sig)
        for i in range(sig.shape[0]):
            sol = solve_ivp(rhs, (ta, tb), sig[i],
                            t_eval=[tb], rtol=rtol, atol=atol, method="RK45")
            if not sol.success:
                raise RuntimeError(f"sigma {i} propagation failed: {sol.message}")
            sig_next[i] = sol.y[:, -1]

        # recombine to mean/cov at tb
        x = wm @ sig_next
        Xd = sig_next - x  # (2n+1, n)
        Px = Xd.T @ (wc[:, None] * Xd)

        # save
        times.append(np.round(tb, 3))
        means.append(x.copy())
        covs.append(Px.copy())

    # pack + save
    times = np.array(times, dtype=np.float64)
    means = np.stack(means, axis=0)  # (K+1, n)
    covs  = np.stack(covs,  axis=0)  # (K+1, n, n)

    if save_path is not None:
        np.savez(save_path,
                 times=times,
                 means=means,
                 covs=covs,
                 method="ut")


def test_ut_vs_linear_on_linear_system():
    rng = np.random.default_rng(0)
    N = 6
    A = rng.normal(size=(N, N)) * 1e-2
    x  = rng.normal(size=N)
    P  = (rng.normal(size=(N, N))); P = P @ P.T + 1e-6*np.eye(N)

    dtt = 1e-3

    # one Euler step (truth for linear)
    x_true = x + (A @ x) * dtt
    P_true = P + (A @ P + P @ A.T) * dtt

    # run UT one step
    sig, wm, wc = _unscented_sigma_points(x, P)
    sig = sig + (sig @ A.T) * dtt
    x_ut = wm @ sig
    Xd   = sig - x_ut
    P_ut = Xd.T @ (wc[:, None] * Xd)

    print("||x_true - x_ut||:", np.linalg.norm(x_true - x_ut))
    print("||P_true - P_ut||_F:", np.linalg.norm(P_true - P_ut, 'fro'))


def fit_gmm_to_gaussian(mean6, cov6, K=1, *,
                        n_train=100_000,
                        n_val=20_000,
                        max_iter=10000,
                        tol=1e-5,
                        reg=1e-6,
                        random_state=None):
    """
    Fit a K-component GMM to a target 6D Normal N(mean6, cov6) using EM.
    Returns:
        weights: (K,)
        means:   (K, 6)
        covs:    (K, 6, 6)
        val_nll: float (mean negative log-likelihood on held-out samples)
    """
    rng = np.random.default_rng(random_state)
    D = 6
    mean6 = np.asarray(mean6, dtype=float).reshape(D)
    cov6  = np.asarray(cov6,  dtype=float).reshape(D, D)

    # --- draw training & validation samples from the target 6D normal ---
    X = rng.multivariate_normal(mean6, cov6, size=n_train)
    X_val = rng.multivariate_normal(mean6, cov6, size=n_val)

    # --- initialize GMM parameters ---
    # means: pick K random data points; covs: start as global cov; weights: uniform
    init_idx = rng.choice(n_train, size=K, replace=False)
    means = X[init_idx].copy()
    covs = np.repeat(cov6[None, :, :], K, axis=0).copy()
    weights = np.full(K, 1.0 / K)

    # precompute constants
    log2pi = np.log(2.0 * np.pi)

    def comp_log_gauss(X, mu, Sigma):
        """Log N(X | mu, Sigma) for all rows in X. Uses Cholesky + Mahalanobis."""
        L = np.linalg.cholesky(Sigma)
        # Solve L * y = (X - mu)^T  -> y = L^{-1}(X - mu)^T
        Xm = X - mu
        # solve_triangular (scipy.linalg) is a bit faster, but we'll keep pure NumPy
        y = np.linalg.solve(L, Xm.T)  # (D,N)
        maha = np.sum(y * y, axis=0)  # (N,)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (D * log2pi + logdet + maha)

    def e_step(X, weights, means, covs):
        """Responsibilities (N,K) and log-likelihood."""
        N = X.shape[0]
        log_prob = np.empty((N, K))
        for k in range(K):
            log_prob[:, k] = np.log(weights[k] + 1e-300) + comp_log_gauss(X, means[k], covs[k])
        log_p_x = logsumexp(log_prob, axis=1)           # (N,)
        resp = np.exp(log_prob - log_p_x[:, None])      # softmax
        ll = np.sum(log_p_x)
        return resp, ll

    def m_step(X, resp, reg):
        """Update weights, means, covs (with reg*I for numerical stability)."""
        N_k = resp.sum(axis=0) + 1e-300                 # (K,)
        weights = N_k / resp.shape[0]
        means = (resp.T @ X) / N_k[:, None]             # (K,D)
        covs = np.empty((K, D, D))
        for k in range(K):
            Xm = X - means[k]
            cov_k = (resp[:, [k]] * Xm).T @ Xm / N_k[k]
            covs[k] = cov_k + reg * np.eye(D)
        return weights, means, covs

    # --- EM loop ---
    prev_ll = -np.inf
    for it in range(max_iter):
        resp, ll = e_step(X, weights, means, covs)
        weights, means, covs = m_step(X, resp, reg=reg)
        if np.isfinite(prev_ll):
            if(it % 100 == 0):
                print(it, abs(ll-prev_ll))
            # if abs(ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            if abs(ll - prev_ll) < tol:
                break
        prev_ll = ll

    # --- validation NLL on fresh samples from the target Normal ---
    # (lower is better; for a perfect match to the true Normal with K=1,
    #  this should approach the entropy + cross-entropy with itself)
    def logprob_gmm(X, weights, means, covs):
        N = X.shape[0]
        log_prob = np.empty((N, K))
        for k in range(K):
            log_prob[:, k] = np.log(weights[k] + 1e-300) + comp_log_gauss(X, means[k], covs[k])
        return logsumexp(log_prob, axis=1)

    val_nll = -np.mean(logprob_gmm(X_val, weights, means, covs))
    p_true = p_normal(X_val, mean6, cov6).reshape(-1,)
    val_nll_true = -np.mean(np.log(p_true))
    print("[fitting]: ", val_nll, val_nll_true)

    return weights, means, covs, float(val_nll)


def fit_gmm_to_gaussian_new(mean, cov, K=11, n_samples=100000, random_state=0, return_kl=True):
    def _covs_to_full(gm: GaussianMixture) -> np.ndarray:
        """Always return covariances as (K, d, d) for any covariance_type."""
        K, d = gm.weights_.shape[0], gm.means_.shape[1]
        ct = gm.covariance_type
        C = gm.covariances_
        if ct == "full":
            return C
        if ct == "diag":
            return np.stack([np.diag(C[k]) for k in range(K)], axis=0)
        if ct == "tied":
            return np.repeat(C[None, :, :], K, axis=0)
        if ct == "spherical":
            return np.stack([np.eye(d) * C[k] for k in range(K)], axis=0)
        raise ValueError(f"Unknown covariance_type: {ct}")

    # sample from target N(mean, cov)
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    # fit GMM
    gm = GaussianMixture(n_components=K, verbose=2, tol=1e-6, covariance_type="diag").fit(X)
    covs_full = _covs_to_full(gm)                # <- ensures shape (K, d, d)
    out = {"weights": gm.weights_, "means": gm.means_, "covs": covs_full, "model": gm}
    # quick MC KL(target || GMM)
    if return_kl:
        pdf_true = p_normal(X, mean, cov)
        out["kl_est"] = np.mean(np.log(pdf_true) - gm.score_samples(X))
    return out


# def gmm_propagation(constants, dt_precision=6, dt_save=1e-2, Kc=11, save_path=None):
#     """
#     NOTE: The gmm fit to initial p0 has large error (tv, rel_error ...)
#     """
#     # --- initialization (fit a Kc-components GMM to p0) ---
#     # weights, means, covs, val_nll = fit_gmm_to_gaussian(
#     #     constants.MEAN_I.copy(), constants.COV_I.copy(), K=Kc
#     # )

#     gmm_data = np.load("data/pre_compute/gmm_x_params.npz")
#     weights = gmm_data["weights"].astype(np.float64)
#     means = gmm_data["means_x"].astype(np.float64)
#     covs = gmm_data["covs_x"].astype(np.float64)

#     # fit_result = fit_gmm_to_gaussian_new(constants.MEAN_I, constants.COV_I, K=Kc)
#     # weights = fit_result["weights"]
#     # means = fit_result["means"]
#     # covs = fit_result["covs"]
#     # print(constants.MEAN_I)
#     # print(means)
#     # print("\n")
#     # print(constants.COV_I)
#     # print(covs)
#     # print("[check] gmm fitting kl_est: ", fit_result["kl_est"])

#     print("[check] gmm fitting sum of weights: ", np.sum(weights), weights)

#     xs = means.copy()
#     Pxs = covs.copy()
#     xs = np.float64(xs)
#     Pxs = np.float64(Pxs)

#     dtt = np.float64(10**(-1*dt_precision))
#     tf = constants.T_PRIME_SPAN[-1]
#     kf = int(np.ceil(tf / dtt))
#     current_time = np.float64(constants.T_PRIME_SPAN[0])

#     # --- storage arrays ---
#     times = [current_time]
#     means = [xs.copy()] # (N, Kc, 6)
#     covs = [Pxs.copy()] # (N, Kc, 6, 6)
#     t_to_save = current_time + dt_save

#     # --- time stepping loop ---
#     for k in tqdm(range(kf), desc="propagting over time"):
#         for j in range(Kc):
#             x = xs[j, :]
#             Px = Pxs[j, :, :]
#             # compute dynamics and Jacobian
#             fx = orbit_dyn(constants, x)
#             Jx = get_Jacobian(constants, x)
#             # propagate mean and covariance: P_dot = Jx * P + P * Jx.T
#             x = x + fx * dtt
#             Px = Px + (Jx @ Px + Px @ Jx.T)*dtt
#             xs[j, :]  = x
#             Pxs[j, :, :] = Px

#         # update time
#         current_time += dtt
#         # current_time = np.round(current_time, dt_precision)
#         if(abs(current_time-t_to_save) < dtt/2):
#             # store results
#             times.append(np.round(current_time,3))
#             means.append(xs.copy())
#             covs.append(Pxs.copy())
#             t_to_save += dt_save

#     # --- convert lists to arrays ---
#     times = np.array(times)
#     means = np.array(means)       # shape: (kf+1, n)
#     covs = np.array(covs)         # shape: (kf+1, n, n)
#     # print(means.shape, means)
#     # print(covs.shape, covs)
#     np.savez(save_path,
#              times=times,
#              means=means,
#              covs=covs,
#              weights=weights,
#              dt_precision=dt_precision,
#              method="gmm")


def gmm_propagation(constants,
                    dt_save=1e-2,
                    Kc=11,
                    save_path=None,
                    rtol=1e-10,
                    atol=1e-12,
                    ):
    """
    GMM propagation using high-accuracy ODE integration between save times.
    - Each component j integrates mean x_j and covariance P_j via:
        dx/dt = orbit_dyn(constants, x)
        dP/dt = J(x) P + P J(x)^T
    - Saves at t_k = t0 + k*dt_save.

    Output NPZ matches your loader (method='gmm').
    """

    # --- load / initialize the initial GMM (means, covs, weights) ---
    gmm_data = np.load("data/pre_compute/gmm_x_params.npz")
    weights = gmm_data["weights"].astype(np.float64)
    xs      = gmm_data["means_x"].astype(np.float64)   # (Kc, 6)
    Pxs     = gmm_data["covs_x"].astype(np.float64)    # (Kc, 6, 6)
    assert xs.shape[0] == Kc and Pxs.shape[0] == Kc, "Kc must match precomputed GMM."

    print("[check] gmm fitting sum of weights: ", np.sum(weights), weights)

    # --- time grid (save-at times only) ---
    t0 = float(constants.T_PRIME_SPAN[0])
    tf = float(constants.T_PRIME_SPAN[-1])
    t_eval = np.arange(t0, tf + 0.5*dt_save, dt_save, dtype=np.float64)

    # --- storage (time-major, same shapes as your original code) ---
    times = [np.round(t0, 3)]
    means = [xs.copy()]           # list of (Kc, 6)
    covs  = [Pxs.copy()]          # list of (Kc, 6, 6)

    # --- RHS for augmented (x, vec(P)) system for one component ---
    def rhs_aug(_t, y_aug):
        x = y_aug[:6]
        P = y_aug[6:].reshape(6, 6)
        fx = orbit_dyn(constants, x)
        Jx = get_Jacobian(constants, x)
        Pdot = Jx @ P + P @ Jx.T
        return np.hstack([fx, Pdot.ravel()])

    # --- march over save intervals, propagate each component with solve_ivp ---
    for k in tqdm(range(1, len(t_eval)), desc="GMM (solve_ivp)"):
        ta, tb = t_eval[k-1], t_eval[k]

        for j in range(Kc):
            y0_aug = np.hstack([xs[j], Pxs[j].ravel()])
            sol = solve_ivp(rhs_aug, (ta, tb), y0_aug,
                            t_eval=[tb], rtol=rtol, atol=atol, method="RK45")
            if not sol.success:
                raise RuntimeError(f"GMM component {j} propagation failed: {sol.message}")
            yb = sol.y[:, -1]
            xs[j]  = yb[:6]
            Pxs[j] = yb[6:].reshape(6, 6)

        # save snapshot at tb
        times.append(np.round(tb, 3))
        means.append(xs.copy())
        covs.append(Pxs.copy())

    # --- stack & save ---
    times = np.asarray(times, dtype=np.float64)                # (T,)
    means = np.asarray(means, dtype=np.float64)                # (T, Kc, 6)
    covs  = np.asarray(covs,  dtype=np.float64)                # (T, Kc, 6, 6)

    if save_path is not None:
        np.savez(save_path,
                 times=times,
                 means=means,
                 covs=covs,
                 weights=weights,
                 method="gmm")
        

def main():
    # _get_Jacobian_expression()
    # test_ut_vs_linear_on_linear_system()
    # return
    
    global constants
    # constants.test_printout()

    # --- Do linear propagation and save result to SAVE_PATH_LINEAR_PROPAGATE ---
    # linear_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=SAVE_PATH_LINEAR_PROPAGATE)
    # linear_propagation_new(constants, dt_save=1e-2, save_path=SAVE_PATH_LINEAR_PROPAGATE)
    # # Example usage
    # data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    # print(data_lp.data["times"])
    # t, w, mu, cov = data_lp.get(constants.T_PRIME_SPAN[-1])
    # print(t, w, mu, cov)

    # # --- Do unscented propagation and save result to SAVE_PATH_UNSCENT_PROPAGATE ---
    # unscent_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=SAVE_PATH_UNSCENT_PROPAGATE)
    # unscent_propagation(constants, dt_save=1e-2, save_path=SAVE_PATH_UNSCENT_PROPAGATE)
    # # Example usage
    # data_us = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    # print(data_us.data["times"])
    # t, w, mu, cov = data_us.get(constants.T_PRIME_SPAN[-1])
    # print(t, w, mu, cov)

    # gmm_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=SAVE_PATH_GMM_PROPAGATE)
    gmm_propagation(constants, dt_save=1e-2, save_path=SAVE_PATH_GMM_PROPAGATE)
    data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)
    # Example usage
    print(data_gmm.data["times"])
    t, ws, mus, covs = data_gmm.get(constants.T_PRIME_SPAN[-1])
    print("gmm data at time: ", t)
    print("gmm weights: ", ws.shape)
    print(mus.shape)
    print(covs.shape)


if __name__ == "__main__":
    main()