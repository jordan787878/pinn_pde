import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from typing import Sequence, Tuple, List
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from exp_utilities.constants import Case1_6D_Constants
from exp_utilities.plot_util import compute_marginals_over_time, plot_time_curves_3d, precompute_pdf_init_streaming
# from exp_utilities.plot_utilites import check_pdf_Nrphi, check_pdf_cartesian_wrt_samples
import sys
sys.path.insert(0, '../utilities/')
from _General.astrodynamics import *

GRID_FOLDER = "data/grids/"
SAMPLES_FOLDER = "data/samples/"

constants = Case1_6D_Constants()
np.random.seed(0)


def p_sol_monte(t=0.0, linespace_num=31, stat_sample=10000000):
    global constants
    X = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)
    
    # convert rns to cartesian
    X_cart = sphere_to_cartesian(rnsphere_to_sphere(X, constants.TI, constants))
    for i in tqdm(range(stat_sample), desc="Propagating samples"):
        _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(X_cart[i,:], constants.MU_EARTH)
        _m = true_to_mean_anomaly(_e, _nu)
        _m = _m + np.sqrt(constants.MU_EARTH/_a**3) * constants.T * t
        _nu = solve_kepler(_e, _m)
        x_cart_t = COE2RV(_a, _e, _i, _RAAN, _w, _nu, _lonper, constants.MU_EARTH)
        x_t = sphere_to_rnsphere(cartesian_to_sphere(x_cart_t.reshape(1,-1)), t, constants)
        # update state
        X[i,:] = x_t
    
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
    pdf_func = multivariate_normal(mean=constants.N_MEAN_I, cov=constants.N_COV_I)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    return pdf_eval


# def get_p_init_max(constants):
#     x1s = np.load("data/grids/x1s.npy")
#     x2s = np.load("data/grids/x2s.npy")
#     x3s = np.load("data/grids/x3s.npy")
#     x4s = np.load("data/grids/x4s.npy")
#     x5s = np.load("data/grids/x5s.npy")
#     x6s = np.load("data/grids/x6s.npy")
#     x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid = np.meshgrid(x1s, x2s, x3s, x4s, x5s, x6s, indexing="ij") # the indexing is very important
#     grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel(), x6_grid.ravel()]).T
#     print(grid_points.shape)
#     joint_pdf_true = p_init(constants, grid_points)
#     p_init_max = np.max(joint_pdf_true)
#     return p_init_max
# def get_p_init_max_streaming(
#     constants,
#     p_init,                                # callable: p_init(constants, pts[(K,6)]) -> (K,) or (K,1)
#     grid_dir: str = "data/grids",
#     axes_files: Sequence[str] = ("x1s.npy","x2s.npy","x3s.npy","x4s.npy","x5s.npy","x6s.npy"),
#     max_points_in_batch: int = 1_000_000,  # tune to your RAM/GPU
#     return_argmax_point: bool = True,
#     verbose: bool = True
# ):
#     # ---- Load per-axis grids ----
#     axes: List[np.ndarray] = [np.load(f"{grid_dir}/{fn}") for fn in axes_files]
#     sizes = [len(a) for a in axes]
#     if verbose:
#         total = 1
#         for n in sizes: total *= n
#         print(f"[info] grid sizes {sizes} -> ~{total:,} total points")

#     # ---- Choose block sizes so product <= max_points_in_batch ----
#     # Start near the 6th root and clamp by axis sizes.
#     root = max(1, round(max_points_in_batch ** (1/6)))
#     block = [min(n, root) for n in sizes]
#     # Nudge down if needed
#     def prod(v):
#         r = 1
#         for x in v: r *= x
#         return r
#     while prod(block) > max_points_in_batch:
#         i = int(np.argmax(block))
#         block[i] = max(1, block[i]-1)
#     if verbose:
#         print(f"[info] using block sizes {block} -> {prod(block):,} points/batch")

#     # Precompute index ranges per axis
#     index_blocks: List[List[Tuple[int,int]]] = []
#     for n, b in zip(sizes, block):
#         starts = list(range(0, n, b))
#         index_blocks.append([(s, min(s+b, n)) for s in starts])

#     p_max = -np.inf
#     argmax_point = None

#     # ---- Iterate over 6-D blocks; build *local* mesh only ----
#     for (a1,b1) in index_blocks[0]:
#         x1 = axes[0][a1:b1]
#         for (a2,b2) in index_blocks[1]:
#             x2 = axes[1][a2:b2]
#             for (a3,b3) in index_blocks[2]:
#                 x3 = axes[2][a3:b3]
#                 for (a4,b4) in index_blocks[3]:
#                     x4 = axes[3][a4:b4]
#                     for (a5,b5) in index_blocks[4]:
#                         x5 = axes[4][a5:b5]
#                         for (a6,b6) in index_blocks[5]:
#                             x6 = axes[5][a6:b6]

#                             # Local block mesh (sizes are small by construction)
#                             g = np.meshgrid(x1, x2, x3, x4, x5, x6, indexing="ij")
#                             pts = np.stack([gi.ravel() for gi in g], axis=1)  # (K, 6)

#                             vals = np.asarray(p_init(constants, pts)).reshape(-1)
#                             if vals.size == 0:
#                                 continue
#                             # Ignore NaNs; find local max
#                             lm = np.nanmax(vals)
#                             if lm > p_max:
#                                 p_max = lm
#                                 if return_argmax_point:
#                                     k = int(np.nanargmax(vals))
#                                     argmax_point = pts[k].copy()

#     if not np.isfinite(p_max):
#         raise RuntimeError("Failed to find a finite maximum over the grid.")

#     if return_argmax_point:
#         return float(p_max), argmax_point  # (max value, corresponding 6-D point)
#     return float(p_max)


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


def test_monte_accuracy(constants, mc_folder):
    """
    marginalize the joint pdf to a single coordinate, and compare it to the true analytical pdf at init time
    the true analytical pdf is obtained by 2 ways (but equivalent)
    1. using the univariate normal
    2. first compute the joint pdf (on the meshgrid) using multivariate normal, then marginalize
    """
    print("[test] pdf(monte) accuracy at t=0")

    x1s = np.load(GRID_FOLDER+"x1s.npy")
    x2s = np.load(GRID_FOLDER+"x2s.npy")
    x3s = np.load(GRID_FOLDER+"x3s.npy")
    x4s = np.load(GRID_FOLDER+"x4s.npy")
    pdf = np.load(mc_folder+"pdf_t{:.3f}.npy".format(0.0))
    print("x1s, pdf(monte) data type: ", x1s.dtype, pdf.dtype)

    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    joint_pdf_true = p_init(grid_points).reshape(x1_grid.shape)
    print("x1 ranges type: ", x1s.dtype)
    print("true joint pdf shape, type: ", joint_pdf_true.shape, joint_pdf_true.dtype)

    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]

    fig, axs = plt.subplots(2, 2, figsize=(8,6))
    for j in range(4):
        if(j == 0):
            marginalize_pdf = np.sum(pdf, axis=(1,2,3)) * dx2 * dx3 * dx4
            marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(1,2,3)) * dx2 * dx3 * dx4
            ax = axs[0,0]
            x_axis = x1s
            ax.set_xlabel(r"$r'$")
            ax.set_ylabel(r"$p(r',t_0)$")
        elif(j == 1):
            marginalize_pdf = np.sum(pdf, axis=(0,2,3)) * dx1 * dx3 * dx4
            marginalize_pdf_true = norm.pdf(x2s, loc=constants.N_MEAN_I[1], scale=constants._N_COV_I[1,1]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,2,3)) * dx1 * dx3 * dx4
            ax = axs[1,0]
            x_axis = x2s
            ax.set_xlabel(r"$\phi'$")
            ax.set_ylabel(r"$p(\phi',t_0)$")
        elif(j == 2):
            marginalize_pdf = np.sum(pdf, axis=(0,1,3)) * dx1 * dx2 * dx4
            marginalize_pdf_true = norm.pdf(x3s, loc=constants.N_MEAN_I[2], scale=constants._N_COV_I[2,2]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,1,3)) * dx1 * dx2 * dx4
            ax = axs[0,1]
            x_axis = x3s
            ax.set_xlabel(r"$v'_r$")
            ax.set_ylabel(r"$p(v'_r,t_0)$")
        else:
            marginalize_pdf = np.sum(pdf, axis=(0,1,2)) * dx1 * dx2 * dx3
            marginalize_pdf_true = norm.pdf(x4s, loc=constants.N_MEAN_I[3], scale=constants._N_COV_I[3,3]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,1,2)) * dx1 * dx2 * dx3
            ax = axs[1,1]
            x_axis = x4s
            ax.set_xlabel(r"$v_{\phi}'$")
            ax.set_xlabel(r"$p(v_{\phi}', t_0)$")
        # ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical (marginal)")
        ax.plot(x_axis, pdf_test, "k--", label="analytical", linewidth=2, alpha=0.7)
        ax.plot(x_axis, marginalize_pdf, "b:", label="monte", marker='o', markersize=4, linewidth=1, alpha=0.8)
        ax.legend()
    # plt.show()
    # Save the figure as a high-quality PDF file
    plt.savefig("figs/figure.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

"""
def fourdorbit_generate_samples(constants, use_j2=False, N_samples=100, dtt=1e-4):
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
            x = x + fourdorbit_dyn_normsph(x, constants, use_j2) * dtt # + g*dw
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
    f4 = constants.T**2 * r * aux**2 - constants.T**2 * constants.MU_EARTH /(constants.R**3 * r**2) + \
         2.0*(3*constants.T**2 * J2 * constants.MU_EARTH * constants.R_EARTH**2)/(2*constants.R**5 * r**4)
    f5 = 0.0 # (constants)
    f6 = -2*constants.T/(r * constants.PHI) * vr * aux
    return np.array([f1, f2, f3, f4, f5, f6])
"""

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

        # if(t_prime == constants.T_PRIME_SPAN[0] or t_prime == constants.T_PRIME_SPAN[-1]):
        #     fig, axs = plot_marginals_6d([x1s,x2s,x3s,x4s,x5s,x6s], pdf)
        #     plt.show()

    np.save(data_folder+"mc_time.npy", np.array(mc_time))


def print_mc_time(mc_folder):
    mc_time = np.load(mc_folder+"mc_time.npy") # print mc computation time
    print("[check] MC computation time: ", mc_time)

    
def main():
    global constants
    # --- Generate data ---
    data_folder = "data/1e+6/"
    # generate_data(data_folder, 1000000)
    # fourdorbit_generate_samples(constants, use_j2=False, N_samples=100, dtt=1e-4)

    # --- Test MC results ---
    print_mc_time(data_folder)

    # --- Pre-computation for plotting data ---
    # 1) save p(t0) max
    # p_max = p_init(constants, constants.N_MEAN_I) # compute maximum PDF at t0 (directly evaluated at the mean of initial Gaussian)
    # np.savez(data_folder+"pre_compute/p_init_max.npz", value=np.float32(p_max.item()), label="mean at Gaussian")

    # 2) save marginal pdf over time
    # for i in range(1,7): # compute marginalized PDF (to each dimension) over discrete time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir="data/1e+6",
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdf_t{:.3f}.npy",
    #     )
    #     np.savez(data_folder+"pre_compute/marginal_p_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    precompute_pdf_init_streaming(constants, p_init, "data/1e+6")
    
    # --- Plots ---
    # for i in range(1, 7):
    #     _data = np.load(data_folder+"pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(_data, title="Marginal p(x"+str(i)+"|t): curves (no interpolation)")

    # test_monte_accuracy()
    # check_pdf_Nrphi(constants, mc_folder=data_folder)
    # check_pdf_cartesian_wrt_samples(constants, mc_folder=data_folder)
    # test_monte_cartesian_pdf_xy(constants, data_folder)
    

if __name__ == "__main__":
    main()