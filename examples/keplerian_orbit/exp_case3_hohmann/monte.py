import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from exp_utilities.constants import Case2_4D_Constants_Low_Thrust
from exp_utilities.plot_util import plot_marginal_pdf_cart

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, plot_1d_true_vs_gmm_marginals_model
from utilities._General.astrodynamics import *

GRID_FOLDER = "data/grids/"
SAMPLES_FOLDER = "data/samples/"

# --- globals (module scope) ---
GENERATE: bool = True
SHOW_PLOT: bool = True
NSAMPLES: int = 1000000

constants = Case2_4D_Constants_Low_Thrust()
np.random.seed(0)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("y", "yes", "t", "true", "1", "on"):  return True
    if v in ("n", "no", "f", "false", "0", "off"): return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generate",   type=_str2bool, default=True,  help="Generate data (True/False)")
    p.add_argument("--showplot",   type=_str2bool, default=False, help="Show plots (True/False)")
    p.add_argument("--Nsamples",   type=int, default=NSAMPLES, help="Number of samples (int)")
    return p.parse_args()


def _f64(x):
    """Return x as np.float64 array/scalar (no-op if already float64)."""
    return np.asarray(x, dtype=np.float64)


def x_samples_monte_new(t_list, data_folder, stat_sample=10000):
    global constants

    # ---- cast all constants used here to float64 once ----
    N_MEAN_I   = _f64(constants.N_MEAN_I)
    N_COV_I    = _f64(constants.N_COV_I)
    THETA      = _f64(constants.THETA)
    N_Q_NOISE  = _f64(constants.N_Q_NOISE)
    J2         = _f64(constants.J2)
    W          = _f64(constants.W)
    PHI        = _f64(constants.PHI)
    T          = _f64(constants.T)
    MU_EARTH   = _f64(constants.MU_EARTH)
    R          = _f64(constants.R)
    R_EARTH    = _f64(constants.R_EARTH)
    A_THRUST   = _f64(constants.A_THRUST)

    # Pack the float64 constants we need into a simple namespace-like dict
    c64 = {
        "N_MEAN_I": N_MEAN_I, "N_COV_I": N_COV_I, "THETA": THETA,
        "N_Q_NOISE": N_Q_NOISE, "J2": J2, "W": W, "PHI": PHI, "T": T,
        "MU_EARTH": MU_EARTH, "R": R, "R_EARTH": R_EARTH,
        "A_THRUST": A_THRUST,
    }

    # Time-stepping parameters for propagation (float64)
    dtt = np.float64(1e-4)
    # steps = {int(round(t / dtt)): np.float64(t) for t in _f64(t_list)}  # use float64 keys/values
    steps = {int(round(t / dtt)): int(round(t / dtt)) * dtt for t in _f64(t_list)}

    # Initial samples (float64)
    X_samples = np.random.multivariate_normal(c64["N_MEAN_I"], c64["N_COV_I"], stat_sample).astype(np.float64)
    mc_time = []

    # time loop
    start_time = time.time()
    max_k = max(steps)  # cache once
    for k in tqdm(range(0, max_k + 1), total=max_k + 1, desc="Propagating", unit="step"):
        if k in steps:
            print(f"hit t ≈ {k*dtt:.8f} (requested {steps[k]:.8f})")
            np.save(data_folder + "X_t{:.2f}.npy".format(k*dtt), X_samples)  # saved as float64
            mc_time.append(time.time() - start_time)

        # process each sample
        for i in range(stat_sample):
            # state vector in float64
            x = np.zeros(6, dtype=np.float64)
            x[0] = X_samples[i, 0]
            x[1] = (np.float64(0.5) * np.float64(np.pi)) / c64["THETA"]
            x[2] = X_samples[i, 1]
            x[3] = X_samples[i, 2]
            x[5] = X_samples[i, 3]

            # Brownian increment (float64)
            dW2 = np.sqrt(c64["N_Q_NOISE"]) * np.sqrt(dtt) * np.random.randn(2).astype(np.float64)
            dW = np.zeros(6)
            dW[3] = dW2[0]
            dW[5] = dW2[1]

            # Dynamics + noise (float64)
            x = x + fourdorbit_dyn_normsph(x, c64, use_j2=True) * dtt + dW

            # Keep the 4-dim subset (float64)
            x_final = np.array([x[0], x[2], x[3], x[5]], dtype=np.float64)
            X_samples[i, :] = x_final  # already float64
    np.save(data_folder+"mc_time.npy", np.array(mc_time))


def fourdorbit_dyn_normsph(x, c64, use_j2):
    """Normalized spherical coordinate dynamics (all float64)."""
    J2 = c64["J2"] if use_j2 else np.float64(0.0)

    r   = np.float64(x[0])
    th  = np.float64(x[1])
    phi = np.float64(x[2])
    vr  = np.float64(x[3])
    vth = np.float64(0.0)
    vphi= np.float64(x[5])

    T    = c64["T"]
    PHI  = c64["PHI"]
    W    = c64["W"]
    MU   = c64["MU_EARTH"]
    R    = c64["R"]
    RE   = c64["R_EARTH"]
    aT   = c64["A_THRUST"]

    f1 = vr
    f2 = vth
    f3 = vphi
    aux = W + (PHI / T) * vphi

    f4 = (T**2) * r * (aux**2) \
         - (T**2) * MU / ((R**3) * (r**2)) \
         + np.float64(2.0) * (np.float64(3.0) * (T**2) * J2 * MU * (RE**2)) / (np.float64(2.0) * (R**5) * (r**4))

    f5 = np.float64(0.0)
    f6 = -np.float64(2.0) * T * vr * aux / (r * PHI)

    rdot_phys = (R / T) * vr                
    r_phys   = R * r
    V_mag = np.sqrt(rdot_phys*rdot_phys + (r_phys*aux)**2)
    f4 += (T**2 / R) * aT * (rdot_phys / V_mag)
    f6 += (T**2 / PHI) * aT * (aux / V_mag)

    return np.array([f1, f2, f3, f4, f5, f6], dtype=np.float64)


def p_init(constants, x):
    """
    x is numpy array of shape (N x 4), N is the sample size
    """
    pdf_func = multivariate_normal(mean=constants.N_MEAN_I, cov=constants.N_COV_I)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    return pdf_eval


def generate_data(constants, data_folder, N_samples):
    if(GENERATE):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
            
        # new method (more efficient)
        x_samples_monte_new(T_monte, data_folder, stat_sample=N_samples)


def generate_true_pdf_by_fitting_gmm(constants, data_folder):
    if(GENERATE):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
        mu_whiten = constants.N_MEAN_I
        cov_whiten = constants.N_COV_I
        for t_prime in T_monte:
            Xsamples = np.load(data_folder + "X_t{:.2f}.npy".format(t_prime))  # saved as float64
            fit_classic_gmm(t_prime, Xsamples, mu_whiten, cov_whiten, data_folder)


def show_fitted_gmm(constants, data_folder):
    if(SHOW_PLOT):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
        for t_prime in T_monte:
            Xsamples = np.load(data_folder + "X_t{:.2f}.npy".format(t_prime))  # saved as float64
            model = GMMWhitenedModel.load(data_folder+"gmm_whitened_t{:.2f}.npz".format(t_prime))
            title="True vs GMM (1D marginals) at t:{:.2f}".format(t_prime)
            plot_1d_true_vs_gmm_marginals_model(model, Xsamples, title=title)

        plt.show()


def print_mc_time(mc_folder):
    mc_time = np.load(mc_folder+"mc_time.npy") # print mc computation time
    print("[check] MC computation time: ", mc_time)


def main():
    global constants#; constants.test_printout()
    global GENERATE, SHOW_PLOT, NSAMPLES
    _sci = f"{NSAMPLES:.0e}".replace("e+0", "e+").replace("e-0", "e-")

    # --- Generate data ---
    data_folder = f"data/Xsamples_{_sci}_np64/"
    os.makedirs(data_folder, exist_ok=True)
    generate_data(constants, data_folder, NSAMPLES)
    print_mc_time(data_folder)

    generate_true_pdf_by_fitting_gmm(constants, data_folder)
    show_fitted_gmm(constants, data_folder)

    _data_folder_no_thrust = "../exp_case2_j2/data/Xsamples_1e+6_np64/"
    plot_marginal_pdf_cart(constants, data_folder, _data_folder_no_thrust)
    plt.show()
    

if __name__ == "__main__":
    args = parse_args()
    GENERATE  = bool(args.generate)
    SHOW_PLOT = bool(args.showplot)
    NSAMPLES  = int(args.Nsamples)
    main()