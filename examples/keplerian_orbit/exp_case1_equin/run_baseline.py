import sympy as sp
import numpy as np
from exp_utilities.constants import Case1_6D_Constants_Equin
import sys
sys.path.insert(0, '../utilities/')
from _General.baseline_methods import (PropagationData, 
    linear_propagation_master, unscent_propagation_master, gmm_propagation_master)
from _General.util import set_publication_plot_style
from _General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, plot_1d_true_vs_gmm_marginals_model


constants = Case1_6D_Constants_Equin()
SAVE_PATH_LINEAR_PROPAGATE = "output/baseline_methods/linear_propagation.npz"
SAVE_PATH_UNSCENT_PROPAGATE = "output/baseline_methods/unscent_propagation.npz"
SAVE_PATH_GMM_PROPAGATE = "output/baseline_methods/gmm_propagation.npz"


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


def sample_p_init(N=1000):
    global constants
    _mu = np.asarray(constants.MEAN_I, dtype=np.float64)
    _Sigma = mu = np.asarray(constants.COV_I, dtype=np.float64)
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean=_mu, cov=_Sigma, size=N,
                                check_valid="raise", tol=1e-12)  # (N, D)
    return X
      

def main():
    # _get_Jacobian_expression()
    # test_ut_vs_linear_on_linear_system()
    # return
    
    global constants
    # constants.test_printout()
    x0, P0 = constants.MEAN_I.copy(), constants.COV_I.copy()
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
        G=np.zeros((6, 6)),
        save_path=SAVE_PATH_UNSCENT_PROPAGATE,
        alpha=1e-3,
    )
    # Example usage
    data_us = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    print(data_us.data["times"])
    t, w, mu, cov = data_us.get(constants.T_PRIME_SPAN[-1])
    print(t, w, mu, cov)

    # --- gmm propagation ---
    X_tr = sample_p_init(N=1000000)
    t_prime = 0.0
    fit_classic_gmm(t_prime, X_tr, np.asarray(x0, dtype=np.float64).reshape(-1,), 
                    np.asarray(P0, dtype=np.float64).reshape((x_dim, x_dim)), "output/baseline_methods/",
                    K_list=(10, 20, 30, 40))
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