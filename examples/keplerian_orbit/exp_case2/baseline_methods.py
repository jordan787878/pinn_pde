import sympy as sp
import numpy as np
from tqdm import tqdm
from scipy.linalg import expm
from exp_utilities.constants import Case2_4D_Constants

constants = Case2_4D_Constants()
SAVE_PATH_LINEAR_PROPAGATE = "output/baseline_methods/linear_propagation.npz"


def _get_Jacobian_expression():
    print("")
    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4', real=True)
    T, W, PHI, MU_EARTH, R = sp.symbols('T W PHI MU_EARTH R', real=True, positive=True)
    A = W + PHI*x4/T
    f1 = x3
    f2 = x4
    f3 = T**2 * x1 * A**2 - T**2 * MU_EARTH / (R**3 * x1**2)
    f4 = -2*T*x3*A/(x1*PHI)
    f = sp.Matrix([f1, f2, f3, f4])
    x = sp.Matrix([x1, x2, x3, x4])

    # symbolic Jacobian
    J = sp.simplify(f.jacobian(x))
    print("[info] Jacobian matrix: ", J)


def get_Jacobian(constants, x):
    x1 = x[0]; x2 = x[1]; x3 = x[2]; x4 = x[3]
    J = np.array([[0., 0., 1., 0.],
                  [0., 0., 0., 1.],
                  [2*constants.MU_EARTH*constants.T**2/(constants.R**3*x1**3) + (constants.PHI*x4 + constants.T*constants.W)**2, 
                   0., 
                   0.,
                   2*constants.PHI*x1*(constants.PHI*x4 + constants.T*constants.W)],
                  [2*x3*(constants.PHI*x4 + constants.T*constants.W)/(constants.PHI*x1**2), 
                   0., 
                   2*(-constants.PHI*x4 - constants.T*constants.W)/(constants.PHI*x1), 
                   -2*x3/x1]], dtype=np.float32)
    return J


def fourdorbit_dyn_normsph(constants, x, use_j2):
    # normalized spherical coordinate dynamics
    if(use_j2):
        J2 = constants.J2
    else:
        J2 = 0.0
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    W = constants.W; T=constants.T; PHI=constants.PHI; MU_EARTH=constants.MU_EARTH; R=constants.R
    A = W + PHI*x4/T
    f1 = x3
    f2 = x4
    f3 = T**2 * x1 * A**2 - T**2 * MU_EARTH / (R**3 * x1**2)
    f4 = -2*T*x3*A/(x1*PHI)
    return np.array([f1, f2, f3, f4])


def linear_propagation(constants, dtt=1e-7, dt_save=1e-2, save_path=None):
    # --- initialization ---
    x = constants.N_MEAN_I.copy()     # initial mean
    Px = constants.N_COV_I.copy()     # initial covariance

    tf = constants.T_PRIME_SPAN[-1]
    kf = int(tf / dtt)
    current_time = constants.T_PRIME_SPAN[0]

    # --- storage arrays ---
    times = [current_time]
    means = [x.copy()]
    covs = [Px.copy()]
    t_to_save = current_time + dt_save

    # --- time stepping loop ---
    for k in tqdm(range(kf), desc="propagting over time"):
        # compute dynamics and Jacobian
        fx = fourdorbit_dyn_normsph(constants, x, use_j2=False)
        Jx = get_Jacobian(constants, x)
        stm = expm(Jx * dtt)

        # propagate mean and covariance
        x = x + fx * dtt
        Px = stm @ Px @ stm.T

        # update time
        current_time += dtt
        current_time = np.round(current_time, 7)
        if(abs(current_time-t_to_save) < dtt/2):
            # store results
            times.append(current_time)
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
             covs=covs)


def get_linear_propagation_data(data, time):
    times = data["times"]
    means = data["means"]
    covs = data["covs"]
    time = np.round(time, 5)
    # print(time)
    # print(times)
    idx = np.where(abs(time-times)<1e-4)[0][0]
    return times[idx], means[idx, :], covs[idx, :, :]
    # find the index of time in times


def main():
    # _get_Jacobian_expression()
    global constants
    linear_propagation(constants, save_path=SAVE_PATH_LINEAR_PROPAGATE)

    # --- Example usage ---
    data = np.load(SAVE_PATH_LINEAR_PROPAGATE)
    print(data["times"])
    t, mu, cov = get_linear_propagation_data(data, 0.2)
    print(t, mu, cov)


if __name__ == "__main__":
    main()