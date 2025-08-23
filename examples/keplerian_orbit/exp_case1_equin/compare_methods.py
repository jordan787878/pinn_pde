import numpy as np
import torch
from scipy.stats import multivariate_normal, norm
from monte import p_init, p_init_scaled, p_sol, print_mc_time
from exp_utilities.constants import Case1_6D_Constants_Equin
from exp_utilities.plot_util import plot_time_curves_3d, plot_pdf_metrics, corner_plot_single, corner_compare_overlay_lower, corner_plot_from_samples, plt
from baseline_methods import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, PropagationData
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, PNet_Scaled, load_trained_model
from _General.util import compute_volume, save_metrics_npz, load_metrics_npz
from functools import partial
constants = Case1_6D_Constants_Equin()


# -----------------------------
# Sampling from sol joint PDF
# -----------------------------
def sample_joint_pdf(constants, t_prime, n_samples=50_000, seed=0, dtype=np.float32):
    """
    Sample from the 6D joint implied by p_sol_plotting:
      - x1..x5 ~ N(mean[i], var[i]) independent
      - y6     ~ N(mean[5], var[5]) independent
      - x6     = y6 + (t' * T) * sqrt(MU_EARTH / x1^3)
    Returns: (N,6) float32 array
    """
    rng = np.random.default_rng(seed)
    means = np.asarray(constants.MEAN_I, dtype=dtype)          # shape (6,)
    vars_ = np.asarray(np.diag(constants.COV_I), dtype=dtype)  # shape (6,)
    stds  = np.sqrt(vars_).astype(dtype)

    # draw independent normal samples for x1..x5 and y6
    X = rng.normal(loc=means, scale=stds, size=(n_samples, 6)).astype(dtype)

    # add the nonlinear coupling into x6 using x1
    shift = (t_prime * dtype(constants.T)) * np.sqrt(dtype(constants.MU_EARTH) / (X[:, 0]**3))
    X[:, 5] = X[:, 5] + shift.astype(dtype)
    return X


# -----------------------------
# Sampling from baseline joint PDF
# -----------------------------
def sample_joint_pdf_baseline(data_lp, t_prime, n_samples=50_000, seed=0, dtype=np.float32, jitter=1e-8):
    """
    Sample from a 6D multivariate normal N(means(t'), cov(t')).
    Expects data_lp.get(t_prime) -> (means, cov) with shapes (6,), (6,6).
    Returns: (N, 6) float32 array.
    """
    rng = np.random.default_rng(seed)

    _, means, cov = data_lp.get(t_prime)            # means: (6,), cov: (6,6)
    means = np.asarray(means, dtype=dtype)
    cov   = np.asarray(cov,   dtype=dtype)
    X = rng.multivariate_normal(mean=means.astype(np.float64),
                                cov=cov.astype(np.float64),
                                size=n_samples).astype(dtype, copy=False)
    return X


def p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    scales = np.sqrt(np.diag(cov))  # std per dimension
    cov_scaled = cov / np.outer(scales, scales)
    x_scaled = (x - mean) / scales
    rv = multivariate_normal(mean=np.zeros(6), cov=cov_scaled)
    pdf_eval = rv.pdf(x_scaled) / np.prod(scales)  # back-transform
    return pdf_eval.reshape(-1,)
    # print(constants.COV_I)
    # pdf_func = multivariate_normal(mean=constants.MEAN_I, cov=constants.COV_I)
    # pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    # return pdf_eval
    # (the following is more stable due to decoupling)
    # pdf_eval = np.float32(1.0)
    # for i in range(6):
    #     pdf_func_i = multivariate_normal(mean=mean[i], cov=cov[i,i])
    #     x_i = x[:,i].astype(np.float32)
    #     pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,).astype(x_i.dtype)
    # return pdf_eval


def p_total_variation(constants, p1, p2, verbose=False, eps=np.finfo(np.float32).tiny):
    """
    NOTE: Inspect this by Corner Plot?
    Given the pdf(X) of shape (N_batch,), where X is a batch of (N_batch, 6) samples, how to show a 6 by 6 corner plot, that shows the covariance and correlation of each two dimension?
    """
    bounds = np.array([
        constants.X1_RANGE,
        constants.X2_RANGE,
        constants.X3_RANGE,
        constants.X4_RANGE,
        constants.X5_RANGE,
        constants.X6_RANGE,
    ])
    vol_est = compute_volume(bounds)
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    diff = np.abs(p1-p2)
    diff[diff < eps] = 0.0 # Set values below threshold to zero
    tv = 0.5 * np.mean(diff) * vol_est
    return 100. *tv.item()


def p_rel_worst_error(p1, p2):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    # print(np.max(p1), np.max(p2))
    max_p2 =  np.max(p2).item()
    if(max_p2 > 0):
        max_diff = np.max(np.abs(p1-p2)).item()
        # print(max_diff, max_p2)
        rel_error = max_diff / max_p2
        return 100.*rel_error
    else:
        return np.NaN


def compute_and_update_metrics(metrics, pdf_eval=None, pdf_ref=None, key=None):
    global constants
    rel_error= p_rel_worst_error(pdf_eval, pdf_ref)
    tv = p_total_variation(constants, pdf_eval, pdf_ref)
    metrics["rel_error_"+key].append(rel_error)
    metrics["tv_"+key].append(tv)
    print(key, " - rel_error: {:.2f} %, tv: {:.2f} %".format(rel_error, tv))


def compute_relKL_and_update_metrics(metrics, t, X, p_data_normal=None, p_net=None, key=None):
    global constants
    eps = np.finfo(np.float32).tiny   # machine precision of np.float32 ≈ 1.175e-38
    if(p_net is not None):
        X_scaled = constants.scaled_x(X)
        _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)
        _t = np.ones((len(_x_tensor), 1)) * t
        _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
        pdf_eval = constants.SCALING_PDF * p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
    if(p_data_normal is not None):
        _, _mu, _cov = p_data_normal.get(t)
        _mu = np.float32(_mu)
        _cov = np.float32(_cov)
        pdf_eval = p_normal(X, _mu, _cov).reshape(-1,)
    mask = pdf_eval >= eps
    if np.any(mask):
        KL = np.mean(-np.log(pdf_eval[mask])).item()
    else:
        print("All pdf_lp values are below machine precision!")
        KL = np.NaN
    metrics["rel_kl_"+key].append(KL)
    print(key, " - rel_kl: {:.2f}".format(KL))


def compute_pdf_variations(p_net=None, data_lp=None, data_ut=None, save_path=None):
    global constants
    metrics = {
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_pinn": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_pinn": [],
        "rel_kl_lp": [],
        "rel_kl_ut": [],
        "rel_kl_pinn": [],
        "t": np.array([0.0, 0.02, 0.05, 0.08, 0.1], dtype=np.float32) # data_lp.data["times"]
    } 

    N_samples = 10000000
    X = get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
    X_scaled = constants.scaled_x(X)
    _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

    for idx, t in enumerate(metrics["t"]):
        print("\ntime {:.4f}".format(t))

        print("[info] pdf ref")
        pdf_ref = p_sol(constants, X, t).reshape(-1,)
        X_mc = sample_joint_pdf(constants, t_prime=t, n_samples=N_samples, seed=123)

        print("[info] pdf PINN")
        _t = np.ones((len(_x_tensor), 1)) * t
        _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
        pdf_pinn = constants.SCALING_PDF * p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        compute_and_update_metrics(metrics, pdf_eval=pdf_pinn, pdf_ref=pdf_ref, key="pinn")
        compute_relKL_and_update_metrics(metrics, t, X_mc, p_net=p_net, key="pinn")

        print("[info] pdf LP")
        _, _mu_lp, _cov_lp = data_lp.get(t)
        _mu_lp = np.float32(_mu_lp)
        _cov_lp = np.float32(_cov_lp)
        pdf_lp = p_normal(X, _mu_lp, _cov_lp).reshape(-1,)
        compute_and_update_metrics(metrics, pdf_eval=pdf_lp, pdf_ref=pdf_ref, key="lp")
        compute_relKL_and_update_metrics(metrics, t, X_mc, p_data_normal=data_lp, key="lp")

        print("[info] pdf UT")
        _, _mu_ut, _cov_ut = data_ut.get(t)
        _mu_ut = np.float32(_mu_ut)
        _cov_ut = np.float32(_cov_ut)
        pdf_ut = p_normal(X, _mu_ut, _cov_ut).reshape(-1,)
        compute_and_update_metrics(metrics, pdf_eval=pdf_ut, pdf_ref=pdf_ref, key="ut")
        compute_relKL_and_update_metrics(metrics, t, X_mc, p_data_normal=data_ut, key="ut")
        
    save_metrics_npz(metrics, save_path)


def get_uniform_Xsamples_numpy(N_samples=None):
    global constants
    X = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], N_samples),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], N_samples),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], N_samples),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], N_samples),
        np.random.uniform(constants.X5_RANGE[0], constants.X5_RANGE[1], N_samples),
        np.random.uniform(constants.X6_RANGE[0], constants.X6_RANGE[1], N_samples),
    ])
    return X


def compare_corner_plots(p_net=None, data_lp=None, data_ut=None, save_path=None):
    global constants
    t_show = np.array([0.0, 0.05, 0.1], dtype=np.float32)

    N_samples = 1000000
    X = get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
    # X_scaled = constants.scaled_x(X)
    # _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

    for idx, t in enumerate(t_show):
        print("\ntime {:.4f}".format(t))

        print("[info] pdf ref")
        X_ref = sample_joint_pdf(constants, t, N_samples)
        corner_plot_from_samples(constants, X_ref)
        # pdf_ref = p_sol(constants, X, t).reshape(-1,)
        # corner_plot_single(constants,
        #     X, pdf_ref,
        #     labels=["x1","x2","x3","x4","x5","x6"],
        #     mode="scatter", cmap="Reds"
        # )

        # print("[info] pdf PINN")
        # _t = np.ones((len(_x_tensor), 1)) * t
        # _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
        # pdf_pinn = constants.SCALING_PDF * p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        # corner_plot_single(constants,
        #     X, pdf_pinn,
        #     labels=["x1","x2","x3","x4","x5","x6"],
        #     mode="heatmap", cmap="Reds"
        # )

        # # print("[info] pdf LP")
        # # _, _mu_lp, _cov_lp = data_lp.get(t)
        # # _mu_lp = np.float32(_mu_lp)
        # # _cov_lp = np.float32(_cov_lp)
        # # pdf_lp = p_normal(X, _mu_lp, _cov_lp).reshape(-1,)

        print("[info] pdf UT")
        _, _mu_ut, _cov_ut = data_ut.get(t)
        _mu_ut = np.float32(_mu_ut)
        _cov_ut = np.float32(_cov_ut)
        pdf_ut = p_normal(X, _mu_ut, _cov_ut).reshape(-1,)

        # corner_compare_overlay_lower(constants,
        #     X, pdf_ref, pdf_pinn,
        #     labels=["x1","x2","x3","x4","x5","x6"],)
        
        # corner_compare_overlay_lower(constants,
        #     X, pdf_ref, pdf_ut,
        #     labels=["x1","x2","x3","x4","x5","x6"],)

        plt.show()


def compare_methods():
    global constants

    # p_net = PNet(constants, input_feature=7)
    # scale = np.load("data/pre_compute/p_init_max.npz")["value"]
    # scale_torch = torch.tensor(scale, dtype=torch.float32)
    # p_net.scale = scale_torch

    p_net = PNet_Scaled(constants, input_feature=7)
    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale = p_max
    scale_torch = torch.tensor(scale, dtype=torch.float32)

    p_net.scale = scale_torch
    print("p net scale: ", p_net.scale)

    OUTPUT_PATH = "output/v0_scaled"
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    metrics_path = "output/baseline_methods/metrics.npz"

    # compute_pdf_variations(p_net=p_net, data_lp=data_lp, data_ut=data_ut, save_path=metrics_path)
    metrics = load_metrics_npz(metrics_path)

    # --- Visualize ---
    # data_normalize_e1_pinn_max = np.load(OUTPUT_PATH+"/data_e1_pinn_max.npz")
    # plot_pdf_metrics(metrics, data_normalize_e1_pinn_max=None)

    compare_corner_plots(p_net=p_net, data_lp=data_lp, data_ut=data_ut)
        


def main():
    compare_methods()


if __name__ == "__main__":
    main()