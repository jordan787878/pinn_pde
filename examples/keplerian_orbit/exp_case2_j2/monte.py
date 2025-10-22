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
# from exp_utilities.plot_utilites import check_pdf_Nrphi, check_pdf_cartesian_wrt_samples, set_publication_plot_style
from exp_utilities.classic_gmm import GMMWhitenedModel
import sys
sys.path.insert(0, '../utilities/')
from _General.astrodynamics import *

GRID_FOLDER = "data/grids/"
SAMPLES_FOLDER = "data/samples/"

# --- globals (module scope) ---
GENERATE: bool = True
SHOW_PLOT: bool = True
NSAMPLES: int = 1000

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
    p.add_argument("--Nsamples",   type=int, default=10000, help="Number of samples (int)")
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
    """
    dx = f dt + g*dw
    g*dw, where g = [0, 0, 0;
                     0, 0, 0;
                     0, 0, 0;
                     1, 0, 0;
                     0, 1, 0;
                     0, 0, 1]
    dw in R^3 with noise intensity diag([constants.N_Q_NOISE])
    To simulate this SDE: the last three compoenent of dw (over a fixed dt) is
      sqrt(constants.N_Q_NOISE) * sqrt(dt) * N(0,1, size=3),
      and the first three components are zeros.
    """
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


def generate_data(constants, data_folder, N_samples):
    if(GENERATE):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
            
        # new method (more efficient)
        x_samples_monte_new(T_monte, data_folder, stat_sample=N_samples)


def fit_classic_gmm(t_prime, X_tr, mu_whiten, cov_whiten, data_folder):
    # Whiten statistics
    mu  = np.asarray(mu_whiten, dtype=np.float64)
    std = np.sqrt(np.asarray(np.diag(cov_whiten), dtype=np.float64))

    # Draw training data from the push-forward in x
    n_train = len(X_tr)
    print("data size: ", n_train)

    # 1) Fit
    model = GMMWhitenedModel(K_list=(1, 2, 4, 8, 16), n_init=3, tol=1e-5, max_iter=2000, reg_covar=1e-6, random_state=0)
    model.fit(X_tr, mu_x=mu, std_x=std)

    # 2) Save to disk
    model.save(data_folder+"gmm_whitened_t{:.2f}.npz".format(t_prime))


def generate_true_pdf_by_fitting_gmm(constants, data_folder):
    if(GENERATE):
        T_monte = np.round(constants.T_PRIME_SPAN, 2)
        mu_whiten = constants.N_MEAN_I
        cov_whiten = constants.N_COV_I
        for t_prime in T_monte:
            Xsamples = np.load(data_folder + "X_t{:.2f}.npy".format(t_prime))  # saved as float64
            fit_classic_gmm(t_prime, Xsamples, mu_whiten, cov_whiten, data_folder)


def plot_1d_true_vs_gmm_marginals_model(model,
                                        X_true,
                                        dims=None,
                                        bins="fd",
                                        qrange=(0.001, 0.999),
                                        figsize=(12, 6),
                                        title="True vs GMM (1D marginals)"):
    """
    Overlay histograms of true samples with analytical GMM 1D marginals (in x-space).

    model  : GMMWhitenedModel (must have model.params with weights, means_z, covs_z, mu_x, std_x)
    X_true : (N, D) true samples in x-space
    dims   : list of dims to plot (default: all)
    bins   : 'fd' | 'scott' | int
    qrange : robust plotting range from true-sample quantiles
    """
    assert hasattr(model, "params") and model.params is not None, "Model must be fitted/loaded."
    p = model.params

    X_true = np.asarray(X_true)
    N, D = X_true.shape
    if dims is None:
        dims = list(range(D))

    w     = p.weights              # (K,)
    m_z   = p.means_z              # (K,D)
    covs  = p.covs_z               # (K,D,D)
    mu_x  = p.mu_x                 # (D,)
    std_x = p.std_x                # (D,)

    # per-dim std in z (for analytic 1D marginals)
    var_z = np.stack([np.diag(C) for C in covs], axis=0)  # (K,D)
    sd_z  = np.sqrt(var_z + 0.0)                          # (K,D)

    ncols = min(3, len(dims))
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for idx, j in enumerate(dims):
        ax = axes[idx]
        # robust range from true data
        lo, hi = np.quantile(X_true[:, j], qrange[0]), np.quantile(X_true[:, j], qrange[1])
        xs = np.linspace(lo, hi, 600)

        # analytical 1D marginal in x:
        # z_j = (x_j - mu_x[j]) / std_x[j]
        # p_xj(x) = (1/std_x[j]) * Σ_k w_k * N(z_j | m_z[k,j], var_z[k,j])
        zj  = (xs - mu_x[j]) / std_x[j]
        mk  = m_z[:, j]          # (K,)
        sdk = sd_z[:, j]         # (K,)
        phi = (np.exp(-0.5 * ((zj[None, :] - mk[:, None]) / sdk[:, None])**2) /
               (sdk[:, None] * np.sqrt(2.0 * np.pi)))                 # (K, M)
        p_xj = (w[:, None] * phi).sum(axis=0) / std_x[j]              # (M,)

        ax.hist(X_true[:, j], bins=bins, range=(lo, hi), density=True, alpha=0.45, label="True")
        ax.plot(xs, p_xj, lw=2, label="GMM marginal")
        ax.set_title(f"x{j}")
        if idx == 0:
            ax.legend(frameon=False)

    # hide empty axes
    for k in range(len(dims), len(axes)):
        axes[k].axis("off")

    fig.suptitle(title)
    fig.tight_layout()


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


def test_J2effect_exp_case2():
    """
    show the effect of J2 & Brownian noise by samples
    """
    global constants
    set_publication_plot_style()

    exps = ["../exp_case2/", "../exp_case2_j2/"]
    colors = ["blue", "red"]
    labels = ["Case 1 (no J2)","Case 2 (J2 + Stochastic noise)"]

    for t_prime in constants.T_PRIME_SPAN:
        # print("[test] pdf(NN) marginalized to rphi at t=", t_prime)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(2):
            samples = np.load(exps[i]+"data/samples/samples_t{:.3f}.npy".format(t_prime))
            r_samples = samples[:,0]
            phi_samples = samples[:,2]
            # scatter samples of (r, phi) on to the plot
            ax.scatter(r_samples, phi_samples, s=30, c='white', linewidths=0.5, 
                        edgecolor=colors[i], alpha=1.0, label=labels[i])
        # Adding labels and title
        ax.set_xlabel(r"$r'$")
        ax.set_ylabel(r"$\phi'$")
        ax.text(
            0.01, 0.99,                   # near top-left
            f"t = {t_prime:.3f}",
            transform=ax.transAxes,       # use axes coords
            fontsize=32,                  # big text
            va='top', ha='left'           # align text box
        )
        ax.legend(loc='upper right', fontsize=18)
        fig.tight_layout(pad=0.2)
        fig.savefig("figs/j2_effect_t{:.3f}.pdf".format(t_prime), format='pdf'); plt.close()
        # plt.show()


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

    # fourdorbit_generate_samples(constants)

    # --- Test MC results ---
    X = np.load(data_folder+"X_t0.20.npy")
    print(X.shape, X.dtype)
    # test_monte_accuracy()
    # test_J2effect_exp_case2()
    # check_pdf_Nrphi(constants, mc_folder=data_folder)
    # check_pdf_cartesian_wrt_samples(constants, mc_folder=data_folder)
    # test_monte_cartesian_pdf_xy(constants, data_folder)
    

if __name__ == "__main__":
    args = parse_args()
    GENERATE  = bool(args.generate)
    SHOW_PLOT = bool(args.showplot)
    NSAMPLES  = int(args.Nsamples)
    main()