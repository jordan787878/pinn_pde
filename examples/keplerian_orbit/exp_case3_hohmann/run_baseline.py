import sympy as sp
import numpy as np
from exp_utilities.constants import Case2_4D_Constants_Low_Thrust

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.baseline_methods import (PropagationData, 
    linear_propagation_master, unscent_propagation_master, gmm_propagation_master)
from utilities._General.util import set_publication_plot_style
from utilities._General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, plot_1d_true_vs_gmm_marginals_model


constants = Case2_4D_Constants_Low_Thrust()
SAVE_PATH_LINEAR_PROPAGATE = "output/baseline_methods/linear_propagation.npz"
SAVE_PATH_UNSCENT_PROPAGATE = "output/baseline_methods/unscent_propagation.npz"
SAVE_PATH_GMM_PROPAGATE = "output/baseline_methods/gmm_propagation.npz"


def _get_Jacobian_expression():
    print("")
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4', real=True)
    T, MU_EARTH, R, J2_VR, PHI, W, aT = sp.symbols('T MU_EARTH R J2_VR PHI W aT', real=True, positive=True)
    f1 = x3
    f2 = x4
    f3 = T**2 *x1 * (W +PHI *x4/T)**2 \
       - T**2 *MU_EARTH/(R**3 * x1**2) \
       + J2_VR/(x1**4)
    f4 = -2. *T*x3*(W + PHI * x4/T)/(x1*PHI)

    # low thrust
    r = x1
    vr  = x3
    vphi = x4
    rdot_phys = (R / T) * vr                
    r_phys   = R * r
    aux = W + (PHI / T) * vphi
    V_mag = (rdot_phys*rdot_phys + (r_phys*aux)**2)**0.5
    f3 += (T**2 / R) * aT * (rdot_phys / V_mag)
    f4 += (T**2 / PHI) * aT * (aux / V_mag)
    
    f = sp.Matrix([f1, f2, f3, f4])
    x = sp.Matrix([x1, x2, x3, x4])

    # symbolic Jacobian
    J = sp.simplify(f.jacobian(x))
    print("[info] Jacobian matrix: ", J)


def get_Jacobian(constants, x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype)
    MU_EARTH = np.asarray(constants.MU_EARTH, dtype=dtype)
    T = np.asarray(constants.T, dtype=dtype)
    R = np.asarray(constants.R, dtype=dtype)
    PHI = np.asarray(constants.PHI, dtype=dtype)
    W = np.asarray(constants.W, dtype=dtype)
    J2_VR = np.asarray(constants.J2_VR, dtype=dtype)
    aT = np.asanyarray(constants.A_THRUST, dtype=dtype)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    J = np.zeros((4, 4), dtype=dtype)
    J[0, 2] = np.float64(1.)
    J[1, 3] = np.float64(1.)
    J[2, :] = np.array([-4*J2_VR/x1**5 + 2*MU_EARTH*T**2/(R**3.0*x1**3) - 1.0*T**2.0*aT*x1*x3*(PHI*x4 + T*W)**2/(R**1.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5) + (PHI*x4 + T*W)**2, 
                        0., 
                        -1.0*T**2.0*aT*x3**2/(R**1.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5) + T**2.0*aT/(R**1.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**0.5), 
                        PHI*x1*(PHI*x4 + T*W)*(-1.0*R**2*T**2.0*aT*x1*x3 + 2*R**3.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5)/(R**3.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5)], 
            dtype=np.float64)
    J[3, :] = np.array([(PHI*x4 + T*W)*(-1.0*R**2*T**2.0*aT*x1**3*(PHI*x4 + T*W)**2 + 2.0*R**3.0*x3*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5)/(PHI*R**3.0*x1**2*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5), 
                        0., 
                        -(PHI*x4 + T*W)*(1.0*R**2*T**2.0*aT*x1*x3 + 2.0*R**3.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5)/(PHI*R**3.0*x1*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5), 
                        -1.0*T**2.0*aT*x1**2*(PHI*x4 + T*W)**2/(R**1.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**1.5) + T**2.0*aT/(R**1.0*(x1**2*(PHI*x4 + T*W)**2 + x3**2)**0.5) - 2.0*x3/x1], 
            dtype=np.float64)
    return J


def orbit_dyn(constants, x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype)
    MU_EARTH = np.asarray(constants.MU_EARTH, dtype=dtype)
    T = np.asarray(constants.T, dtype=dtype)
    R = np.asarray(constants.R, dtype=dtype)
    PHI = np.asarray(constants.PHI, dtype=dtype)
    W = np.asarray(constants.W, dtype=dtype)
    J2_VR = np.asarray(constants.J2_VR, dtype=dtype)
    aT = np.asanyarray(constants.A_THRUST, dtype=dtype)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    f1 = x3
    f2 = x4
    f3 = T**2 *x1 * (W +PHI *x4/T)**2 \
       - T**2 *MU_EARTH/(R**3 * x1**2 )\
       + J2_VR/(x1**4)
    f4 = -2. *T*x3*(W + PHI * x4/T)/(x1*PHI)

    # low thrust
    r = x1
    vr  = x3
    vphi = x4
    rdot_phys = (R / T) * vr                
    r_phys   = R * r
    aux = W + (PHI / T) * vphi
    V_mag = (rdot_phys*rdot_phys + (r_phys*aux)**2)**0.5
    f3 += (T**2 / R) * aT * (rdot_phys / V_mag)
    f4 += (T**2 / PHI) * aT * (aux / V_mag)

    return np.array([f1, f2, f3, f4], dtype=np.float64)


def sample_p_init(N=1000):
    global constants
    _mu = np.asarray(constants.N_MEAN_I, dtype=np.float64)
    _Sigma = mu = np.asarray(constants.N_COV_I, dtype=np.float64)
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean=_mu, cov=_Sigma, size=N,
                                check_valid="raise", tol=1e-12)  # (N, D)
    return X


def main():
    # _get_Jacobian_expression(); return
    global constants

    # constants.test_printout()
    x0, P0 = constants.N_MEAN_I.copy(), constants.N_COV_I.copy()
    x_dim = x0.size
    dyn_fcn = lambda t, x, c: orbit_dyn(c, x)
    jac_fcn = lambda t, x, c: get_Jacobian(c, x)
    Q = np.diag([0, 0, constants.N_Q_NOISE[0], constants.N_Q_NOISE[1]]).astype(np.float64)
    t_span = (np.float64(np.round(constants.T_PRIME_SPAN[0], 2)), np.float64(np.round(constants.T_PRIME_SPAN[-1], 2)))

    # --- Do linear propagation and save result to SAVE_PATH_LINEAR_PROPAGATE ---
    linear_propagation_master(
        x0=x0, P0=P0,
        dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
        constants=constants,
        t_span=t_span,
        dt_save=0.01,
        Q=Q,
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
        Q=Q,
        save_path=SAVE_PATH_UNSCENT_PROPAGATE,
        alpha=1.0,
    )
    # Example usage
    data_us = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    print(data_us.data["times"])
    t, w, mu, cov = data_us.get(constants.T_PRIME_SPAN[-1])
    print(t, w, mu, cov)

    # --- gmm propagation ---
    X_tr = sample_p_init(N=10000) # [test] default is 1E+6 samples
    t_prime = 0.0
    fit_classic_gmm(t_prime, X_tr, np.asarray(x0, dtype=np.float64).reshape(-1,), 
                    np.asarray(P0, dtype=np.float64).reshape((x_dim, x_dim)), "output/baseline_methods/",
                    K_list=(20, 30, 40))
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
        Q=Q,
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