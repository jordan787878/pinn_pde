import sympy as sp
import numpy as np
from exp_utilities.constants import Case1_6D_Constants

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.baseline_methods import (PropagationData, 
    linear_propagation_master, unscent_propagation_master, gmm_propagation_master)
from utilities._General.util import set_publication_plot_style
from utilities._General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, plot_1d_true_vs_gmm_marginals_model


constants = Case1_6D_Constants()
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
            assert("The linear propagation result does not have data at this time")
        idx = idx[0]
        if self.data["method"] == "gmm":
            weights = self.data["weights"]
            return times[idx], weights, means[idx, :, :], covs[idx, :, :, :]
        return times[idx], np.array([1.]), means[idx, :], covs[idx, :, :]


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


def get_Jacobian(constants, x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype)
    MU_EARTH = np.asarray(constants.MU_EARTH, dtype=dtype)
    T = np.asarray(constants.T, dtype=dtype)
    R = np.asarray(constants.R, dtype=dtype)
    THETA = np.asarray(constants.THETA, dtype=dtype)
    PHI = np.asarray(constants.PHI, dtype=dtype)
    W = np.asarray(constants.W, dtype=dtype)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    J = np.zeros((6, 6), dtype=dtype)
    J[0, 3] = np.float64(1.)
    J[1, 4] = np.float64(1.)
    J[2, 5] = np.float64(1.)
    J[3, :] = np.array([2*MU_EARTH*T**2/(R**3*x1**3) + THETA**2*x5**2 + (PHI*x6 + T*W)**2* np.sin(THETA*x2)**2, 
                        THETA*x1*(PHI*x6 + T*W)**2* np.sin(2*THETA*x2), 
                        0, 
                        0, 
                        2*THETA**2*x1*x5, 
                        2*PHI*x1*(PHI*x6 + T*W)* np.sin(THETA*x2)**2], dtype=dtype)
    J[4, :] = np.array([2.0*x4*x5/x1**2, 
                        1.0*(PHI*x6 + T*W)**2* np.cos(2.0*THETA*x2), 
                        0., 
                        -2.0*x5/x1, 
                        -2.0*x4/x1, 
                        1.0*PHI*(PHI*x6 + T*W)* np.sin(2.0*THETA*x2)/THETA], dtype=dtype)
    J[5, :] = np.array([2.0*x4*(PHI*x6 + T*W)/(PHI*x1**2), 
                        2.0*THETA**2*x5*(PHI*x6 + T*W)/(PHI* np.sin(THETA*x2)**2), 
                        0., 
                        2.0*(-PHI*x6 - T*W)/(PHI*x1), 
                        -2.0*THETA*(PHI*x6 + T*W)/(PHI* np.tan(THETA*x2)), 
                        -2.0*THETA*x5/ np.tan(THETA*x2) - 2.0*x4/x1], dtype=dtype)
    return J


def orbit_dyn(constants, x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype)
    MU_EARTH = np.asarray(constants.MU_EARTH, dtype=dtype)
    T = np.asarray(constants.T, dtype=dtype)
    R = np.asarray(constants.R, dtype=dtype)
    THETA = np.asarray(constants.THETA, dtype=dtype)
    PHI = np.asarray(constants.PHI, dtype=dtype)
    W = np.asarray(constants.W, dtype=dtype)
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
    return np.array([f1, f2, f3, f4, f5, f6], dtype=dtype)


def sample_p_init(N=1000):
    global constants
    _mu = np.asarray(constants.N_MEAN_I, dtype=np.float64)
    _Sigma = mu = np.asarray(constants.N_COV_I, dtype=np.float64)
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean=_mu, cov=_Sigma, size=N,
                                check_valid="raise", tol=1e-12)  # (N, D)
    return X


def main():
    # _get_Jacobian_expression()
    
    global constants
    # constants.test_printout()
    x0, P0 = constants.N_MEAN_I.copy(), constants.N_COV_I.copy()
    x_dim = x0.size
    dyn_fcn = lambda t, x, c: orbit_dyn(c, x)
    jac_fcn = lambda t, x, c: get_Jacobian(c, x)
    t_span = (np.float64(np.round(constants.T_PRIME_SPAN[0], 2)), np.float64(np.round(constants.T_PRIME_SPAN[-1], 2)))

    # --- Do linear propagation and save result to SAVE_PATH_LINEAR_PROPAGATE ---
    linear_propagation_master(
        x0=x0, P0=P0,
        dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
        constants=constants,
        t_span=t_span,
        dt_save=0.01,
        Q=None,
        save_path=SAVE_PATH_LINEAR_PROPAGATE
    )
    # linear_propagation(constants, dt_precision=6, dt_save=1e-2, save_path=SAVE_PATH_LINEAR_PROPAGATE)
    # Example usage
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    print(data_lp.data["times"])
    t, w, mu, cov = data_lp.get(constants.T_PRIME_SPAN[-1])
    print(t, w, mu, cov)

    # --- Do unscented propagation and save result to SAVE_PATH_UNSCENT_PROPAGATE ---
    unscent_propagation_master(
        x0=x0, P0=P0,
        dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
        constants=constants,
        t_span=t_span,
        dt_save=0.01,
        Q=None,
        G=np.zeros((6,6)),
        save_path=SAVE_PATH_UNSCENT_PROPAGATE,
        alpha=1.0,
    )
    # Example usage
    data_us = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    print(data_us.data["times"])
    t, w, mu, cov = data_us.get(constants.T_PRIME_SPAN[-1])
    print(t, w, mu, cov)

    # --- gmm propagation ---
    X_tr = sample_p_init(N=100000)
    t_prime = 0.0
    # fit_classic_gmm(t_prime, X_tr, np.asarray(x0, dtype=np.float64).reshape(-1,), 
    #                 np.asarray(P0, dtype=np.float64).reshape((x_dim, x_dim)), "output/baseline_methods/",
    #                 K_list=(10, 20, 30, 40))
    model = GMMWhitenedModel.load("output/baseline_methods/"+"gmm_whitened_t{:.2f}.npz".format(t_prime))
    set_publication_plot_style()
    plot_1d_true_vs_gmm_marginals_model(model, X_tr)
    gmm_params = model.print_x_params()
    gmm_propagation_master(
        weights = gmm_params["weights"],
        means0 = gmm_params["means_x"],
        covs0 = gmm_params["covs_x"],
        dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
        constants=constants,
        t_span=t_span,
        dt_save=0.01,
        Q=None,
        save_path=SAVE_PATH_GMM_PROPAGATE
    )
    # Example usage
    data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)
    print(data_gmm.data["times"])
    t, ws, mus, covs = data_gmm.get(constants.T_PRIME_SPAN[-1])
    print("gmm data at time: ", t)
    print("gmm weights: ", ws.shape)
    print(mus.shape)
    print(covs.shape)


if __name__ == "__main__":
    main()