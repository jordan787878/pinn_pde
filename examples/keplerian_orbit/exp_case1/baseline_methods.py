import sympy as sp
import numpy as np
from tqdm import tqdm
from scipy.linalg import expm, cholesky
from exp_utilities.constants import Case1_6D_Constants

constants = Case1_6D_Constants()
SAVE_PATH_LINEAR_PROPAGATE = "output/baseline_methods/linear_propagation.npz"
SAVE_PATH_UNSCENT_PROPAGATE = "output/baseline_methods/unscent_propagation.npz"


def _get_Jacobian_expression():
    print("")
    x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6', real=True)
    T, MU_EARTH, R, THETA, PHI, W = sp.symbols('T MU_EARTH R THETA PHI W', real=True, positive=True)
    f1 = x4
    f2 = x5
    f3 = x6
    f4 = x1 *THETA**2 *x5**2 \
           + T**2 *x1 * (sp.sin(THETA*x2))**2 *(W +PHI *x6/T)**2 \
           - T**2 *MU_EARTH/(R**3 * x1**2)
    f5 = -2. *x5*x4/x1 \
           + T**2 * sp.sin(2.*THETA*x2) *(W +PHI *x6/T)**2 /(2. *THETA)
    f6 = -2. *x5 *THETA*T *(W +PHI *x6/T) * (sp.tan(THETA*x2))**(-1) /PHI \
           -2. *T*x4*(W + PHI * x6/T)/(x1*PHI)
    
    f = sp.Matrix([f1, f2, f3, f4, f5, f6])
    x = sp.Matrix([x1, x2, x3, x4, x5, x6])

    # symbolic Jacobian
    J = sp.simplify(f.jacobian(x))
    print("[info] Jacobian matrix: ", J)


def get_Jacobian(constants, x):
    MU_EARTH = constants.MU_EARTH
    T = constants.T
    R = constants.R
    THETA = constants.THETA
    PHI = constants.PHI
    W = constants.W
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    J = np.zeros((6, 6))
    J[0, 3] = np.float64(1.)
    J[1, 4] = np.float64(1.)
    J[2, 5] = np.float64(1.)
    J[3, :] = np.array([2*MU_EARTH*T**2/(R**3*x1**3) + THETA**2*x5**2 + (PHI*x6 + T*W)**2* np.sin(THETA*x2)**2, 
                        THETA*x1*(PHI*x6 + T*W)**2* np.sin(2*THETA*x2), 
                        0, 
                        0, 
                        2*THETA**2*x1*x5, 
                        2*PHI*x1*(PHI*x6 + T*W)* np.sin(THETA*x2)**2], dtype=np.float64)
    J[4, :] = np.array([2.0*x4*x5/x1**2, 
                        1.0*(PHI*x6 + T*W)**2* np.cos(2.0*THETA*x2), 
                        0., 
                        -2.0*x5/x1, 
                        -2.0*x4/x1, 
                        1.0*PHI*(PHI*x6 + T*W)* np.sin(2.0*THETA*x2)/THETA], dtype=np.float64)
    J[5, :] = np.array([2.0*x4*(PHI*x6 + T*W)/(PHI*x1**2), 
                        2.0*THETA**2*x5*(PHI*x6 + T*W)/(PHI* np.sin(THETA*x2)**2), 
                        0., 
                        2.0*(-PHI*x6 - T*W)/(PHI*x1), 
                        -2.0*THETA*(PHI*x6 + T*W)/(PHI* np.tan(THETA*x2)), 
                        -2.0*THETA*x5/ np.tan(THETA*x2) - 2.0*x4/x1], dtype=np.float64)
    return J


def orbit_dyn(constants, x):
    MU_EARTH = constants.MU_EARTH
    T = constants.T
    R = constants.R
    THETA = constants.THETA
    PHI = constants.PHI
    W = constants.W
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    f1 = x4
    f2 = x5
    f3 = x6
    f4 = x1 *THETA**2 *x5**2 \
           + T**2 *x1 * (sp.sin(THETA*x2))**2 *(W +PHI *x6/T)**2 \
           - T**2 *MU_EARTH/(R**3 * x1**2)
    f5 = -2. *x5*x4/x1 \
           + T**2 * sp.sin(2.*THETA*x2) *(W +PHI *x6/T)**2 /(2. *THETA)
    f6 = -2. *x5 *THETA*T *(W +PHI *x6/T) * (sp.tan(THETA*x2))**(-1) /PHI \
           -2. *T*x4*(W + PHI * x6/T)/(x1*PHI)
    return np.array([f1, f2, f3, f4, f5, f6], dtype=np.float64)


def linear_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=None):
    # --- initialization ---
    x = constants.N_MEAN_I.copy()     # initial mean
    Px = constants.N_COV_I.copy()     # initial covariance
    x = np.float64(x)
    Px = np.float64(Px)

    dtt = np.float64(10**(-1*dt_precision))
    tf = constants.T_PRIME_SPAN[-1]
    kf = int(np.ceil(tf / dtt))
    current_time = constants.T_PRIME_SPAN[0]

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
             dt_precision=dt_precision)


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
            assert("The linear propagation result does not have data at this time")
        idx = idx[0]
        return times[idx], means[idx, :], covs[idx, :, :]


def _unscented_sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0):
    """
    Generate Unscented Transform sigma points for N-dim state.

    Parameters
    ----------
    mean : (N,) array_like
        State mean vector.
    cov : (N,N) array_like
        State covariance (symmetric, PSD).
    alpha : float, optional
        Spread of the sigma set (small, e.g. 1e-3). Affects higher-order terms.
    beta : float, optional
        Prior knowledge about distribution; 2 is optimal for Gaussian.
    kappa : float, optional
        Secondary scaling (often 0 or 3-N).

    Returns
    -------
    X : (2N+1, N) ndarray
        Sigma points. X[0] is the mean, others are +/- columns of the scaled root.
    Wm : (2N+1,) ndarray
        Weights for computing the mean.
    Wc : (2N+1,) ndarray
        Weights for computing the covariance.

    """
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    P = np.asarray(cov, dtype=np.float64)
    N = m.size
    lam = alpha**2 * (N + kappa) - N
    c = N + lam
    if c <= 0:
        raise ValueError("N + lambda must be positive; adjust alpha/kappa.")
    S = cholesky(P, lower=True)
    S *= np.sqrt(c)  # scale by sqrt(N+lambda)

    # Sigma points
    X = np.empty((2*N + 1, N), dtype=float)
    X[0] = m
    X[1:N+1]     = m + S.T   # columns of S
    X[N+1:2*N+1] = m - S.T

    # Weights
    Wm = np.full(2*N + 1, 1.0/(2.0*c), dtype=float)
    Wc = np.full(2*N + 1, 1.0/(2.0*c), dtype=float)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return X, Wm, Wc


def unscent_propagation(constants, dt_precision=7, dt_save=1e-2, save_path=None):
    """
    TODO Validate this on Colab 1D OU process
    """
    # --- initialization ---
    x = constants.MEAN_I.copy()     # initial mean
    Px = constants.COV_I.copy()     # initial covariance
    x = np.float64(x)    
    Px = np.float64(Px) 

    dtt = np.float64(10**(-1*dt_precision))
    tf = constants.T_PRIME_SPAN[-1]
    kf = int(np.ceil(tf / dtt))
    current_time = constants.T_PRIME_SPAN[0]

    # --- storage arrays ---
    times = [current_time]
    means = [x.copy()]
    covs = [Px.copy()]
    t_to_save = current_time + dt_save

    # --- time stepping loop ---
    for k in tqdm(range(kf), desc="propagting over time"):
        # compute sigma points
        _sigma_pts, _w_mean, _w_cov = _unscented_sigma_points(x, Px)

        for i in range(_sigma_pts.shape[0]):
            _x_i = _sigma_pts[i, :]
            fx_i = orbit_dyn(constants, _x_i)
            _sigma_pts[i, :] = _x_i + fx_i * dtt

        x = _w_mean @ _sigma_pts
        X_diff = _sigma_pts - x
        Px = X_diff.T @ (_w_cov[:, None] * X_diff)

        # update time
        current_time += dtt
        current_time = np.round(current_time, dt_precision)
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
             dt_precision=dt_precision)


def main():
    # _get_Jacobian_expression()
    
    global constants

    # --- Do linear propagation and save result to SAVE_PATH_LINEAR_PROPAGATE ---
    linear_propagation(constants, dt_precision=6, dt_save=5e-3, save_path=SAVE_PATH_LINEAR_PROPAGATE)
    # Example usage
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    print(data_lp.data["times"])
    t, mu, cov = data_lp.get(constants.T_PRIME_SPAN[-1])
    print(t, mu, cov)

    # # --- Do unscented propagation and save result to SAVE_PATH_UNSCENT_PROPAGATE ---
    # unscent_propagation(constants, dt_precision=6, dt_save=5e-3, save_path=SAVE_PATH_UNSCENT_PROPAGATE)
    # # Example usage
    # data_us = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    # print(data_us.data["times"])
    # t, mu, cov = data_us.get(constants.T_PRIME_SPAN[-1])
    # print(t, mu, cov)


if __name__ == "__main__":
    main()