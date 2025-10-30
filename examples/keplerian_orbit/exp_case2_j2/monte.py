import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from exp_utilities.constants import Case2_4D_Constants

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

constants = Case2_4D_Constants()
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


def p_sol_monte(t=0.0, linespace_num=51, stat_sample=10000):
    """
    Propagate stat_sample samples from the initial 4D Gaussian distribution
    (augmented to a 6D state) using the dynamics function dyn_normsph, and directly
    bin the final 4D states (columns 0, 2, 3, 5) without storing all samples.
    
    Parameters:
        t            : time at which to propagate the samples.
        linespace_num: number of bins along each dimension.
        stat_sample  : number of samples to generate.
        
    Returns:
        midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4: midpoints for the bins.
        frequency_4d : the 4D probability density function (normalized).
    """
    global constants

    # Define bins for each dimension.
    bins_x1 = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x2 = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x3 = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x4 = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)

    # Initialize the frequency array for the 4D pdf.
    frequency_4d = np.zeros((len(bins_x1)-1, len(bins_x2)-1, len(bins_x3)-1, len(bins_x4)-1), dtype=np.float32)

    # Time-stepping parameters for propagation.
    dtt = 1e-4
    kf = int(t / dtt)

    # Process each sample individually.
    for i in tqdm(range(stat_sample), desc="Processing samples"):
        # Generate one sample from the 4D Gaussian distribution.
        sample_4d = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I).astype(np.float32)
        
        # Build the 6D state from the 4D sample.
        # X[0] = sample_4d[0], X[1] = constant, X[2] = sample_4d[1], X[3] = sample_4d[2], X[5] = sample_4d[3]
        x = np.zeros(6, dtype=np.float32)
        x[0] = sample_4d[0]
        x[1] = 0.5 * np.float32(np.pi) / constants.THETA
        x[2] = sample_4d[1]
        x[3] = sample_4d[2]
        x[5] = sample_4d[3]
        
        # Propagate the sample using the dynamics (e.g., RK4 with J2 dynamics).
        for k in range(kf):
            dW = np.sqrt(constants.N_Q_NOISE) * np.sqrt(dtt) * np.random.randn(3)
            dW = np.concatenate([np.zeros(3), dW]) # because g = diag([0,0,0,1,1,1])
            x = x + fourdorbit_dyn_normsph(x, constants, use_j2=True) * dtt + dW  # dyn_normsph(x) is assumed to be defined elsewhere.
        
        # Extract the 4D state (columns 0, 2, 3, 5).
        x_final = np.array([x[0], x[2], x[3], x[5]], dtype=np.float32)
        
        # Determine the bin indices for each dimension.
        idx1 = np.digitize(x_final[0], bins_x1) - 1
        idx2 = np.digitize(x_final[1], bins_x2) - 1
        idx3 = np.digitize(x_final[2], bins_x3) - 1
        idx4 = np.digitize(x_final[3], bins_x4) - 1
        
        # Only update the frequency if the indices are within the valid range.
        if (0 <= idx1 < frequency_4d.shape[0] and
            0 <= idx2 < frequency_4d.shape[1] and
            0 <= idx3 < frequency_4d.shape[2] and
            0 <= idx4 < frequency_4d.shape[3]):
            frequency_4d[idx1, idx2, idx3, idx4] += 1

    # Normalize the frequency array to get the probability density.
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    dx3 = bins_x3[1] - bins_x3[0]
    dx4 = bins_x4[1] - bins_x4[0]
    frequency_4d /= (dx1 * dx2 * dx3 * dx4 * stat_sample)

    # Check that the integrated pdf is approximately 1.
    print("[check] sum pdf(monte) =", np.sum(frequency_4d) * dx1 * dx2 * dx3 * dx4)

    # Calculate midpoints for each bin.
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
    midpoints_x4 = (bins_x4[:-1] + bins_x4[1:]) / 2

    return midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4, frequency_4d


def x_samples_monte(t=0.0, stat_sample=10000):
    """
    """
    global constants

    # Time-stepping parameters for propagation.
    dtt = 1e-4
    kf = int(t / dtt)
    X_samples = np.zeros((stat_sample, 4))

    # Process each sample individually.
    for i in tqdm(range(stat_sample), desc="Processing samples"):
        # Generate one sample from the 4D Gaussian distribution.
        sample_4d = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I).astype(np.float32)
        
        # Build the 6D state from the 4D sample.
        # X[0] = sample_4d[0], X[1] = constant, X[2] = sample_4d[1], X[3] = sample_4d[2], X[5] = sample_4d[3]
        x = np.zeros(6, dtype=np.float32)
        x[0] = sample_4d[0]
        x[1] = 0.5 * np.float32(np.pi) / constants.THETA
        x[2] = sample_4d[1]
        x[3] = sample_4d[2]
        x[5] = sample_4d[3]
        
        # Propagate the sample using the dynamics (e.g., RK4 with J2 dynamics).
        for k in range(kf):
            dW = np.sqrt(constants.N_Q_NOISE) * np.sqrt(dtt) * np.random.randn(3)
            dW = np.concatenate([np.zeros(3), dW]) # because g = diag([0,0,0,1,1,1])
            x = x + fourdorbit_dyn_normsph(x, constants, use_j2=True) * dtt + dW  # dyn_normsph(x) is assumed to be defined elsewhere.
        
        # Extract the 4D state (columns 0, 2, 3, 5).
        x_final = np.array([x[0], x[2], x[3], x[5]], dtype=np.float32)
        X_samples[i, :] = x_final.copy()

    print(X_samples.dtype)
    return X_samples


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

    # Pack the float64 constants we need into a simple namespace-like dict
    c64 = {
        "N_MEAN_I": N_MEAN_I, "N_COV_I": N_COV_I, "THETA": THETA,
        "N_Q_NOISE": N_Q_NOISE, "J2": J2, "W": W, "PHI": PHI, "T": T,
        "MU_EARTH": MU_EARTH, "R": R, "R_EARTH": R_EARTH,
    }

    # Time-stepping parameters for propagation (float64)
    dtt = np.float64(1e-4)
    steps = {int(round(t / dtt)): np.float64(t) for t in _f64(t_list)}  # use float64 keys/values

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
            dW3 = np.sqrt(c64["N_Q_NOISE"]) * np.sqrt(dtt) * np.random.randn(3).astype(np.float64)
            dW  = np.concatenate([np.zeros(3, dtype=np.float64), dW3])  # g = diag([0,0,0,1,1,1])

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

    f1 = vr
    f2 = vth
    f3 = vphi
    aux = W + (PHI / T) * vphi

    f4 = (T**2) * r * (aux**2) \
         - (T**2) * MU / ((R**3) * (r**2)) \
         + np.float64(2.0) * (np.float64(3.0) * (T**2) * J2 * MU * (RE**2)) / (np.float64(2.0) * (R**5) * (r**4))

    f5 = np.float64(0.0)
    f6 = -np.float64(2.0) * T * vr * aux / (r * PHI)

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
    # os.makedirs(data_folder, exist_ok=True)
    # generate_data(constants, data_folder, NSAMPLES)
    print_mc_time(data_folder)

    generate_true_pdf_by_fitting_gmm(constants, data_folder)
    show_fitted_gmm(constants, data_folder)
    

if __name__ == "__main__":
    args = parse_args()
    GENERATE  = bool(args.generate)
    SHOW_PLOT = bool(args.showplot)
    NSAMPLES  = int(args.Nsamples)
    main()