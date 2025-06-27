import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from exp_utilities.constants import Case2_4D_Constants
from exp_utilities.plot_utilites import check_pdf_Nrphi, check_pdf_cartesian_wrt_samples
import sys
sys.path.insert(0, '../utilities/')
from _General.astrodynamics import *

GRID_FOLDER = "data/grids/"
SAMPLES_FOLDER = "data/samples/"

constants = Case2_4D_Constants()
np.random.seed(0)


# --- Refactor ---
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


def p_init(constants, x):
    """
    x is numpy array of shape (N x 4), N is the sample size
    """
    pdf_func = multivariate_normal(mean=constants.N_MEAN_I, cov=constants.N_COV_I)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    return pdf_eval


def get_p_init_max(constants):
    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    joint_pdf_true = p_init(constants, grid_points)
    p_init_max = np.max(joint_pdf_true)
    return p_init_max


def get_max_e1_init(constants, p_net):
    t = constants.TI
    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    pdf_true = p_init(constants, grid_points).reshape(x1_grid.shape) # obtain analytical p(true)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    e1 = pdf_true - pdf_nn
    e1_vec = e1.reshape(-1)
    max_e1 = np.max(np.abs(e1_vec))
    return max_e1


def fourdorbit_generate_samples(constants, use_j2=True, N_samples=100, dtt=1e-4):
    for t_prime in constants.T_PRIME_SPAN:
        X = fourdorbit_propagate_samples(constants, use_j2=use_j2, t=t_prime, stat_sample=N_samples, dtt=dtt)
        np.save(SAMPLES_FOLDER+"samples_t{:.3f}.npy".format(t_prime), X)


def fourdorbit_propagate_samples(constants, use_j2=False, t=0.2, stat_sample=1, dtt=1e-4):
    X_four_dim = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)
    # append constant theta' = 0.5pi/THETA, theta'_dot = 0.0
    X = np.zeros((stat_sample, 6))
    X[:,0] = X_four_dim[:,0]
    X[:,1] = np.ones((stat_sample,)) * 0.5 * np.float32(np.pi) / constants.THETA
    X[:,2] = X_four_dim[:,1]
    X[:,3] = X_four_dim[:,2]
    X[:,5] = X_four_dim[:,3]
    # convert rns to sphere
    kf = int(t/dtt)
    for i in range(stat_sample):
        x = X[i, :]
        # ode45 propogate the X_sph(t)
        for k in range(kf):
            dW = np.sqrt(constants.N_Q_NOISE) * np.sqrt(dtt) * np.random.randn(3)
            dW = np.concatenate([np.zeros(3), dW]) # because g = diag([0,0,0,1,1,1])
            x = x + fourdorbit_dyn_normsph(x, constants, use_j2) * dtt + dW
        X[i, :] = x
    return X


def fourdorbit_dyn_normsph(x, constants, use_j2):
    # normalized spherical coordinate dynamics
    if(use_j2):
        J2 = constants.J2
    else:
        J2 = 0.0
    r = x[0]
    th = x[1]
    phi = x[2]
    vr = x[3]
    vth = x[4]
    vphi = x[5]
    f1 = vr
    f2 = 0.0 # (constants)
    f3 = vphi
    aux = constants.W + constants.PHI/constants.T * vphi
    f4 = constants.T**2 * r * aux**2 - \
         constants.T**2 * constants.MU_EARTH /(constants.R**3 * r**2) + \
         2.0*(3*constants.T**2 * J2 * constants.MU_EARTH * constants.R_EARTH**2)/(2*constants.R**5 * r**4)
    f5 = 0.0 # (constants)
    f6 = -2*constants.T/(r * constants.PHI) * vr * aux
    return np.array([f1, f2, f3, f4, f5, f6])


def generate_data(constants, data_folder, N_samples):
    mc_time = []
    # dt = 0.01
    # t_span = np.arange(0.00, 0.20+dt, dt)
    # for t_prime in t_span:
    for t_prime in constants.T_PRIME_SPAN:
        start_time = time.time()
        x1s, x2s, x3s, x4s, pdf = p_sol_monte(t=t_prime, linespace_num=51, stat_sample=N_samples)   
        mc_time.append(time.time() - start_time)
        np.save(data_folder+"pdf_t{:.3f}.npy".format(t_prime), pdf)
        if t_prime == 0.0:
            np.save(GRID_FOLDER+"x1s.npy", x1s)
            np.save(GRID_FOLDER+"x2s.npy", x2s)
            np.save(GRID_FOLDER+"x3s.npy", x3s)
            np.save(GRID_FOLDER+"x4s.npy", x4s)
    np.save(data_folder+"mc_time.npy", np.array(mc_time))
    

def main():
    # --- Generate data ---
    data_folder = "data/1e+5/"
    generate_data(constants, data_folder, 100000)
    # fourdorbit_generate_samples(constants)

    # --- Test MC results ---
    # test_monte_accuracy()
    check_pdf_Nrphi(constants, mc_folder=data_folder)
    # check_pdf_cartesian_wrt_samples(constants, mc_folder=data_folder)
    # test_monte_cartesian_pdf_xy(constants, data_folder)
    


if __name__ == "__main__":
    main()