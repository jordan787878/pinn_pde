import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from typing import Sequence, Tuple, List
from scipy.stats import norm, multivariate_normal
from exp_utilities.constants import Case1_6D_Constants, Case1_6D_Constants_Equin
from exp_utilities.plot_util import compute_marginals_over_time, plot_time_curves_3d, precompute_pdfsol_streaming
import sys
sys.path.insert(0, '../utilities/')
from _General.astrodynamics import *

GRID_FOLDER = "data/grids/"
SAMPLES_FOLDER = "data/samples/"

constants = Case1_6D_Constants_Equin()
np.random.seed(0)


def p_sol_monte(t=0.0, linespace_num=31, stat_sample=10000000):
    global constants
    X = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=stat_sample).astype(np.float32)
    
    for i in tqdm(range(stat_sample), desc="Propagating samples"):
        x1, x2, x3, x4, x5, x6 = X[i, :]
        # update state
        x6 = x6 + np.sqrt(constants.MU_EARTH/x1**3)* constants.T * t
        X[i,:] = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float32)
    
    # Define bins for each dimension
    bins_x1 = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x2 = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x3 = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x4 = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x5 = np.linspace(constants.X5_RANGE[0], constants.X5_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x6 = np.linspace(constants.X6_RANGE[0], constants.X6_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)

    # Digitize to find bin indices for each dimension
    bin_indices_x1 = np.digitize(X[:, 0], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[:, 1], bins_x2) - 1
    bin_indices_x3 = np.digitize(X[:, 2], bins_x3) - 1
    bin_indices_x4 = np.digitize(X[:, 3], bins_x4) - 1  
    bin_indices_x5 = np.digitize(X[:, 4], bins_x5) - 1
    bin_indices_x6 = np.digitize(X[:, 5], bins_x6) - 1  

    # Initialize frequency
    # NOTE: for 6d PDF, N_d > 51 exceeds memory
    N_d = linespace_num-1
    frequency = np.zeros((N_d, N_d, N_d, N_d, N_d, N_d)).astype(np.float32)

    # Count occurrences in each bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        idx_x1 = bin_indices_x1[i]
        idx_x2 = bin_indices_x2[i]
        idx_x3 = bin_indices_x3[i]
        idx_x4 = bin_indices_x4[i]
        idx_x5 = bin_indices_x5[i]
        idx_x6 = bin_indices_x6[i]

        # Check if the indices are valid
        if (0 <= idx_x1 < frequency.shape[0] and
            0 <= idx_x2 < frequency.shape[1] and
            0 <= idx_x3 < frequency.shape[2] and
            0 <= idx_x4 < frequency.shape[3] and
            0 <= idx_x5 < frequency.shape[4] and
            0 <= idx_x6 < frequency.shape[5]
            ):
            frequency[idx_x1, idx_x2, idx_x3, idx_x4, idx_x5, idx_x6] += 1
            # print("bin into: ", idx_x1, idx_x2, idx_x3, idx_x4)

    # Normalize the frequency to get the probability density
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    dx3 = bins_x3[1] - bins_x3[0]
    dx4 = bins_x4[1] - bins_x4[0]
    dx5 = bins_x5[1] - bins_x5[0]
    dx6 = bins_x6[1] - bins_x6[0]
    frequency /= (dx1 * dx2 * dx3 * dx4 * dx5 * dx6 * stat_sample) 
    # # NOTE: I do not normalize the p with "r"*dr*dphi*dr_dot*dphi_dot. Instead, a simple version is used.

    # Check the sum of the probability density function
    print("[check] sum pdf(monte) = 1.0", np.sum(frequency) * dx1 * dx2 * dx3 * dx4 * dx5 * dx6)

    # Calculate the midpoints for bins (optional, depending on your needs)
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
    midpoints_x4 = (bins_x4[:-1] + bins_x4[1:]) / 2
    midpoints_x5 = (bins_x5[:-1] + bins_x5[1:]) / 2
    midpoints_x6 = (bins_x6[:-1] + bins_x6[1:]) / 2
    return midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4, midpoints_x5, midpoints_x6, frequency


def p_init(constants, x):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    # print(constants.COV_I)
    # pdf_func = multivariate_normal(mean=constants.MEAN_I, cov=constants.COV_I)
    # pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    # return pdf_eval
    # (the following is more stable due to decoupling)
    pdf_eval = np.float32(1.0)
    for i in range(6):
        pdf_func_i = multivariate_normal(mean=constants.MEAN_I[i], cov=constants.COV_I[i,i])
        x_i = x[:,i].astype(np.float32)
        pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,1).astype(x_i.dtype)
    return pdf_eval


def p_sol(constants, x, t_prime):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    # shift x
    x1 = x[:, 0]
    x6 = x[:, 5]
    x6 = x6 - (t_prime * constants.T) * np.sqrt(constants.MU_EARTH/ x1**3)
    x[:, 5] = x6
    pdf_eval = np.float32(1.0)
    for i in range(6):
        pdf_func_i = multivariate_normal(mean=constants.MEAN_I[i], cov=constants.COV_I[i,i])
        x_i = x[:,i].astype(np.float32)
        pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,1).astype(x_i.dtype)
    return pdf_eval


def wrap_to_pi(theta):
    """
    Wrap angles to [-pi, pi).
    """
    return (theta + np.pi) % (2*np.pi) - np.pi


def fit_p_init_Gaussian(stat_sample=1000000):
    constants = Case1_6D_Constants()
    X = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)
    
    # convert rns to cartesian
    X_cart = sphere_to_cartesian(rnsphere_to_sphere(X, constants.TI, constants))
    for i in tqdm(range(stat_sample), desc="Propagating samples"):
        _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(X_cart[i,:], constants.MU_EARTH)
        _m = true_to_mean_anomaly(_e, _nu)

        x1 = _a
        x2 = _e*np.sin(_w+_RAAN)
        x3 = _e*np.cos(_w+_RAAN)
        x4 = np.sin(_i)*np.sin(_RAAN)/(1.+np.cos(_i))
        x5 = np.sin(_i)*np.cos(_RAAN)/(1.+np.cos(_i))
        x6 = _m + _w + _RAAN
        x6 = wrap_to_pi(x6)
        X[i, :] = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float32)
    
    
    # now I want to fit a single Gaussian for each column of X, how can I do so? and return the mean and covariance of each fitting.
    mu_vec  = X.mean(axis=0)                         # shape (6,)
    var_vec = X.var(axis=0, ddof=0)                  # shape (6,)
    cov_diag = np.diag(var_vec)                      # 6x6 diagonal covariance
    min_vec = X.min(axis=0)              # shape (6,)
    max_vec = X.max(axis=0)              # shape (6,)

    for i in range(X.shape[1]):
        x_col = X[:, i]
        pdf_func = multivariate_normal(mean=mu_vec[i], cov=cov_diag[i,i])
        x_points = np.linspace(min_vec[i], max_vec[i], endpoint=True)
        pdf_values = pdf_func.pdf(x_points).reshape(-1,)
        plt.figure()
        plt.hist(x_col, 20, density=True) # this is a count of data --> needs to be normalized to a pdf
        plt.plot(x_points, pdf_values) # this is a fitted pdf
    plt.show()
    return mu_vec, cov_diag, min_vec, max_vec


def generate_data(data_folder, N_samples):
    t_span = constants.T_PRIME_SPAN
    mc_time = []
    for t_prime in t_span:
        start_time = time.time()
        
        x1s, x2s, x3s, x4s, x5s, x6s, pdf = p_sol_monte(t=t_prime, linespace_num=31, stat_sample=N_samples)   
        mc_time.append(time.time() - start_time)
        np.save(data_folder+"pdf_t{:.3f}.npy".format(t_prime), pdf)
        if t_prime == 0.0:
            np.save(GRID_FOLDER+"x1s.npy", x1s)
            np.save(GRID_FOLDER+"x2s.npy", x2s)
            np.save(GRID_FOLDER+"x3s.npy", x3s)
            np.save(GRID_FOLDER+"x4s.npy", x4s)
            np.save(GRID_FOLDER+"x5s.npy", x5s)
            np.save(GRID_FOLDER+"x6s.npy", x6s)

    np.save(data_folder+"mc_time.npy", np.array(mc_time))


def print_mc_time(mc_folder):
    mc_time = np.load(mc_folder+"mc_time.npy") # print mc computation time
    print("[check] MC computation time: ", mc_time)


def test_pfunc():
    global constants
    N_samples = 100
    _x = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], N_samples),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], N_samples),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], N_samples),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], N_samples),
        np.random.uniform(constants.X5_RANGE[0], constants.X5_RANGE[1], N_samples),
        np.random.uniform(constants.X6_RANGE[0], constants.X6_RANGE[1], N_samples),
    ])
    p_from_init = p_init(constants, _x).reshape(-1,)
    p_from_sol = p_sol(constants, _x, 0.0).reshape(-1,)
    print("[check] ", np.max(np.abs(p_from_init-p_from_sol)))
    

def main():
    # --- Fit p_init in Equinoctial element set ---
    # mu_vec, cov_diag, min_vec, max_vec = fit_p_init_Gaussian(stat_sample=1000000)
    # np.savez("data/constants.npz", mu_vec=mu_vec, 
    #          cov_diag=cov_diag,
    #          min_vec=min_vec,
    #          max_vec=max_vec)    

    # --- Generate data ---
    global constants
    print(constants.MEAN_I)
    print(constants.COV_I)
    print(constants.X1_RANGE)
    print(constants.X2_RANGE)
    print(constants.X3_RANGE)
    print(constants.X4_RANGE)
    print(constants.X5_RANGE)
    print(constants.X6_RANGE)
    data_folder = "data/1e+6"

    # generate_data(data_folder+"/", 100000)

    # --- Test MC results ---
    # print_mc_time(data_folder)

    # --- Pre-computation for plotting data ---
    # 1) save p(t0) max
    # p_max = p_sol(constants, constants.MEAN_I.reshape(1,-1), 0.0) # compute maximum PDF at t0 (directly evaluated at the mean of initial Gaussian)
    # print(p_max.item())
    # np.savez("data/pre_compute/p_init_max.npz", value=np.float32(p_max.item()), label="mean at Gaussian")

    # 2) save marginal pdf over time
    # for i in range(1,7): # compute marginalized PDF (to each dimension) over discrete time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir=data_folder,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdf_t{:.3f}.npy",
    #     )
    #     np.savez(data_folder+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    # precompute_pdf_init_streaming(constants, p_init, "data/1e+6")
    
    # --- Plots ---
    # for i in range(1, 7):
    #     _data = np.load(data_folder+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(i, constants, _data, title="Marginal p(x"+str(i)+",t)")

    # precompute_pdfsol_streaming(constants, p_sol, "data/pre_compute")

    # for i in range(1,7): # compute marginalized PDF (to each dimension) over discrete time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir="data/pre_compute",
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdfsol_t{:.3f}.npy",
    #     )
    #     np.savez("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)
    test_pfunc()

    # --- Plots ---
    # This plot validates the "analytical" joint PDF by p_sol() --> can used to validate error.
    # for i in range(1, 7):
    #     data_MC = np.load(data_folder+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     data_sol = np.load("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(i, constants, data_MC, data2=data_sol, p_init=p_init,
    #                         leg_txt = ["p MC", "p sol."], title="Marginal p(x"+str(i)+",t)")
    

if __name__ == "__main__":
    main()