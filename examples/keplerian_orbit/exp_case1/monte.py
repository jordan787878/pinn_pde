import numpy as np
import torch
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from exp_utilities.constants import Case1_6D_Constants

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.astrodynamics import *
from utilities._General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, plot_1d_true_vs_gmm_marginals_model


GENERATE = False
SHOW_PLOT = False


GRID_FOLDER = "data/grids/"
constants = Case1_6D_Constants()
np.random.seed(0)


def parse_args():
    def _str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ("y", "yes", "t", "true", "1", "on"):  return True
        if v in ("n", "no", "f", "false", "0", "off"): return False
        raise argparse.ArgumentTypeError("Expected a boolean value.")
    p = argparse.ArgumentParser()
    p.add_argument("--generate",   type=_str2bool, default=GENERATE,  help="Generate data (True/False)")
    p.add_argument("--showplot",   type=_str2bool, default=SHOW_PLOT, help="Show plots (True/False)")
    return p.parse_args()


def p_init(constants, x):
    """
    Evaluate the multivariate Gaussian PDF using NumPy or PyTorch depending on input type.

    Parameters
    ----------
    constants : object
        Must contain:
        - N_MEAN_I: (D,) mean vector
        - N_COV_I: (D,D) covariance matrix
    x : ndarray or torch.Tensor
        Input samples of shape (N, D).

    Returns
    -------
    pdf_eval : ndarray or torch.Tensor
        PDF values of shape (N, 1), matching the type of `x`.
    """
    # Check if input is torch tensor
    use_torch = torch.is_tensor(x)

    if use_torch:
        # Torch computation
        mu = torch.as_tensor(constants.N_MEAN_I, dtype=torch.float32, device=x.device)
        cov = torch.as_tensor(constants.N_COV_I, dtype=torch.float32, device=x.device)

        D = mu.shape[0]
        cov_inv = torch.linalg.inv(cov)
        cov_det = torch.linalg.det(cov)
        norm_const = 1.0 / torch.sqrt((2 * torch.pi) ** D * cov_det)

        diff = x.to(torch.float32) - mu
        mahal = torch.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        pdf_eval = norm_const * torch.exp(-0.5 * mahal)

        return pdf_eval.view(-1, 1)
    else:
        # NumPy computation
        mu = np.asarray(constants.N_MEAN_I, dtype=np.float32)
        cov = np.asarray(constants.N_COV_I, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)

        D = mu.shape[0]
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * cov_det)

        diff = x - mu
        mahal = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        pdf_eval = norm_const * np.exp(-0.5 * mahal)

        return pdf_eval.reshape(-1, 1).astype(np.float32)


def p_sol_monte(t=0.0, linespace_num=31, stat_sample=1000000):
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

    def _print_min_max_per_dim(X: np.ndarray):
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == 6, f"Expected (N,6), got {X.shape}"
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        for i, (mn, mx) in enumerate(zip(mins, maxs), start=1):
            print(f"x{i}: min={mn:.6g}  max={mx:.6g}")
    _print_min_max_per_dim(X)
    
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
    return midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4, midpoints_x5, midpoints_x6, frequency, X


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


def generate_data(data_folder, N_samples):
    t_span = constants.T_PRIME_SPAN
    for j in range(1, 1+1):
        mc_time = []
        for t_prime in t_span:
            start_time = time.time()
            
            x1s, x2s, x3s, x4s, x5s, x6s, pdf, X = p_sol_monte(t=t_prime, linespace_num=31, stat_sample=N_samples)   
            
            if j == 1:
                mc_time.append(time.time() - start_time)

            # saving main data
            d = os.path.join(data_folder, f"run{j}")
            os.makedirs(d, exist_ok=True)
            # np.save(d+"/pdf_t{:.3f}.npy".format(t_prime), pdf)
            np.save(d+"/xsamples_t{:.3f}.npy".format(t_prime), X)
            
            # if t_prime == 0.0 and j == 1:
            #     np.save(GRID_FOLDER+"x1s.npy", x1s)
            #     np.save(GRID_FOLDER+"x2s.npy", x2s)
            #     np.save(GRID_FOLDER+"x3s.npy", x3s)
            #     np.save(GRID_FOLDER+"x4s.npy", x4s)
            #     np.save(GRID_FOLDER+"x5s.npy", x5s)
            #     np.save(GRID_FOLDER+"x6s.npy", x6s)

        if j == 1:
            np.save(data_folder+"/mc_time.npy", np.array(mc_time))


def generate_true_pdf_by_fitting_gmm(constants, data_folder):
    if(GENERATE):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
        mu_whiten = constants.N_MEAN_I
        cov_whiten = constants.N_COV_I
        for t_prime in T_monte:
            Xsamples = np.load(data_folder + "xsamples_t{:.3f}.npy".format(t_prime))  # saved as float64
            fit_classic_gmm(t_prime, Xsamples, mu_whiten, cov_whiten, data_folder)


def show_fitted_gmm(constants, data_folder):
    if(SHOW_PLOT):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
        T_show = [T_monte[0], T_monte[-1]]
        for t_prime in T_show:
            Xsamples = np.load(data_folder + "xsamples_t{:.3f}.npy".format(t_prime))  # saved as float64
            model = GMMWhitenedModel.load(data_folder+"gmm_whitened_t{:.2f}.npz".format(t_prime))
            title="True vs GMM (1D marginals) at t:{:.2f}".format(t_prime)
            plot_1d_true_vs_gmm_marginals_model(model, Xsamples, title=title)
        plt.show()


def print_mc_time(mc_folder):
    mc_time = np.load(mc_folder+"mc_time.npy") # print mc computation time
    print("[check] MC computation time: ", mc_time)

    
def main():
    global constants

    # --- Generate data ---
    # data_folder = "data/1e+6/"
    # generate_data(data_folder, 1000000)

    # --- Test MC results ---
    # print_mc_time(data_folder)

    # --- Generate dataset: each consists of 10e+6 samples ---
    data_folder = "dataset"
    # generate_data(data_folder, 1000000)

    generate_true_pdf_by_fitting_gmm(constants, "dataset/run1/")
    show_fitted_gmm(constants, "dataset/run1/")
    

if __name__ == "__main__":
    args = parse_args()
    GENERATE  = bool(args.generate)
    SHOW_PLOT = bool(args.showplot)
    main()