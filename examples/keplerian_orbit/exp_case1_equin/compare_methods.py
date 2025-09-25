import numpy as np
import torch
from tqdm import tqdm
import copy
from scipy.stats import multivariate_normal, norm
from monte import p_init, p_init_scaled, p_sol, print_mc_time
from exp_utilities.constants import Case1_6D_Constants_Equin
from exp_utilities.plot_util import (plot_time_curves_3d, plot_pdf_metrics, 
                                     plot_single_corner, plot_full_corner, plt)
from baseline_methods import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, PropagationData
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, PNet_Scaled, PNet_XL, E1Net_Scaled, E1Net_XL, load_trained_model
from _General.util import compute_volume, save_metrics_npz, load_metrics_npz
from _General.neuralnetworks import TimeToGMM6D, TimeToGMM6D_V0
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


def p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    scales = np.sqrt(np.diag(cov))  # std per dimension
    cov_scaled = cov / np.outer(scales, scales)
    x_scaled = (x - mean) / scales
    rv = multivariate_normal(mean=np.zeros(len(mean)), cov=cov_scaled)
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
    # diff[diff < eps] = 0.0 # Set values below threshold to zero
    tv = 0.5 * np.mean(diff) * vol_est
    return 100. *tv.item()


def p_normalize_constant(constants, p1):
    """
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
    return np.mean(p1) * vol_est


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


def compute_generalKL(t, X, p_data_normal=None, p_net=None, key=None):
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
    return KL


def compute_pdf_variations(N_batch = 10, p_net=None, p_net_gmm=None, data_lp=None, data_ut=None, e1_net=None, save_path=None):
    global constants
    metrics = {
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_pinn": [],
        "rel_error_pinngmm": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_pinn": [],
        "tv_pinngmm": [],
        "g_kl_lp": [],
        "g_kl_ut": [],
        "g_kl_pinn": [],
        "g_kl_pinngmm": [],
        "B1_pinn": [],
        # "t": constants.T_PRIME_SPAN,
        "t": np.round(np.arange(0.0, 0.3+0.05, 0.05, dtype=np.float32),2)
    } 
    print("evaluate metrics over times: ", metrics["t"])
    N_samples = int(1e+5)
    for idx, t in enumerate(metrics["t"]):
        print("\ntime {:.4f}".format(t))
        pdf_ref_max = 0.0
        delta_p_pinn_max = 0.0
        tv_pinn = 0.0
        gkl_pinn = 0.0
        delta_p_pinngmm_max = 0.0
        tv_pinngmm = 0.0
        gkl_pinngmm = 0.0
        delta_p_lp_max = 0.0
        tv_lp = 0.0
        gkl_lp = 0.0
        delta_p_ut_max = 0.0
        tv_ut = 0.0
        gkl_ut = 0.0
        # additional data for pinn
        e1_pinn_max = 0.0
        Z_pinn = 0.0
        for j in tqdm(range(1, N_batch+1), desc="Propagating batches"):
            X = get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
            # print(X[0:3, :])
            X_scaled = constants.scaled_x(X)
            _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

            # print("[info] pdf ref")
            pdf_ref = p_sol(constants, X, t).reshape(-1,)
            pdf_ref_max = max(pdf_ref_max, np.max(pdf_ref).item())
            X_mc = sample_joint_pdf(constants, t_prime=t, n_samples=N_samples, seed=j)

            # pinn-mlp
            _t = np.ones((len(_x_tensor), 1)) * t
            _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
            pdf_pinn = constants.SCALING_PDF * p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            _delta_p = np.max(np.abs(pdf_pinn - pdf_ref)).item()
            delta_p_pinn_max = max(delta_p_pinn_max, _delta_p)
            _tv = p_total_variation(constants, pdf_pinn, pdf_ref)
            tv_pinn += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_net=p_net)
            gkl_pinn += _gkl/N_batch
            _Z_pinn = p_normalize_constant(constants, pdf_pinn)
            Z_pinn += _Z_pinn/N_batch
            del pdf_pinn
            if(e1_net is not None):
                e1_pinn = constants.SCALING_PDF * e1_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                e1_pinn_max = max(e1_pinn_max, np.max(np.abs(e1_pinn)).item())
                del e1_pinn

            # pinn-gmm
            pdf_pinn = constants.SCALING_PDF * p_net_gmm(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            _delta_p = np.max(np.abs(pdf_pinn - pdf_ref)).item()
            delta_p_pinngmm_max = max(delta_p_pinngmm_max, _delta_p)
            _tv = p_total_variation(constants, pdf_pinn, pdf_ref)
            tv_pinngmm += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_net=p_net_gmm)
            gkl_pinngmm += _gkl/N_batch
            del pdf_pinn
            del _t, _t_tensor, X_scaled, _x_tensor

            # print("[info] pdf LP")
            _, _mu_lp, _cov_lp = data_lp.get(t)
            _mu_lp = np.float32(_mu_lp)
            _cov_lp = np.float32(_cov_lp)
            pdf_lp = p_normal(X, _mu_lp, _cov_lp).reshape(-1,)
            _delta_p = np.max(np.abs(pdf_lp - pdf_ref)).item()
            delta_p_lp_max = max(delta_p_lp_max, _delta_p)
            _tv = p_total_variation(constants, pdf_lp, pdf_ref)
            tv_lp += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_data_normal=data_lp)
            gkl_lp += _gkl/N_batch
            del pdf_lp

            # print("[info] pdf UT")
            _, _mu_ut, _cov_ut = data_ut.get(t)
            _mu_ut = np.float32(_mu_ut)
            _cov_ut = np.float32(_cov_ut)
            pdf_ut = p_normal(X, _mu_ut, _cov_ut).reshape(-1,)
            _delta_p = np.max(np.abs(pdf_ut - pdf_ref)).item()
            delta_p_ut_max = max(delta_p_ut_max, _delta_p)
            _tv = p_total_variation(constants, pdf_ut, pdf_ref)
            tv_ut += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_data_normal=data_ut)
            gkl_ut += _gkl/N_batch
            del pdf_ut
        
        # After all batch
        gkl_pinn += Z_pinn
        gkl_pinngmm += 1
        gkl_lp += 1
        gkl_ut += 1
        metrics["rel_error_pinn"].append(100.*delta_p_pinn_max/pdf_ref_max)
        metrics["tv_pinn"].append(tv_pinn)
        metrics["g_kl_pinn"].append(gkl_pinn)
        metrics["rel_error_pinngmm"].append(100.*delta_p_pinngmm_max/pdf_ref_max)
        metrics["tv_pinngmm"].append(tv_pinngmm)
        metrics["g_kl_pinngmm"].append(gkl_pinngmm)
        metrics["rel_error_lp"].append(100.*delta_p_lp_max/pdf_ref_max)
        metrics["tv_lp"].append(tv_lp)
        metrics["g_kl_lp"].append(gkl_lp)
        metrics["rel_error_ut"].append(100.*delta_p_ut_max/pdf_ref_max)
        metrics["tv_ut"].append(tv_ut)
        metrics["g_kl_ut"].append(gkl_ut)
        metrics["B1_pinn"].append(100. * 2. * e1_pinn_max/pdf_ref_max)
        
        print("[check]: max e1: {:.4f}, max e1 pinn: {:.4f}".format(
            delta_p_pinn_max, e1_pinn_max))
        print("[check] Z_pinn: {:.4f}".format(Z_pinn))
        print(metrics["rel_error_pinn"])
        print(metrics["rel_error_pinngmm"])
        print(metrics["tv_pinn"])
        print(metrics["tv_pinngmm"])
        print(metrics["g_kl_pinn"])
        print(metrics["g_kl_pinngmm"])

    # print(metrics["rel_error_pinn"])
    # print(metrics["tv_pinn"])
    # print(metrics["rel_error_lp"])
    # print(metrics["tv_lp"])
    # print(metrics["rel_error_ut"])
    # print(metrics["tv_ut"])
    # print(metrics["B1_pinn"])
    
    save_metrics_npz(metrics, save_path)


def compare_corner_plots(p_net=None, data_lp=None, data_ut=None, save_path=None, 
                         OUTPUT_PATH=None, p_net_gmm=None):
    global constants
    t_show = [constants.T_PRIME_SPAN[-1]]
    # t_show = np.array([0.4])

    N_samples = 1000000
    # X = get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
    # X_scaled = constants.scaled_x(X)
    # _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

    for idx, t in enumerate(t_show):
        print("\ntime {:.4f}".format(t))

        x_coords = (1, 6)

        print("[info] ref")
        X_ref = sample_joint_pdf(constants, t, N_samples)

        print("[info] pinn")
        data_marginal_pinn = np.load(
            f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz")

        print("[info] pdf LP")
        _, _mu_lp, _cov_lp = data_lp.get(t)
        _mu_lp = np.float32(_mu_lp)
        _cov_lp = np.float32(_cov_lp)
        gaussian_lp = (_mu_lp, _cov_lp)

        print("[info] pdf UT")
        _, _mu_ut, _cov_ut = data_ut.get(t)
        _mu_ut = np.float32(_mu_ut)
        _cov_ut = np.float32(_cov_ut)
        gaussian_ut = (_mu_ut, _cov_ut)

        # Singler corner plot as coordinate: x_coords
        ax = plot_single_corner(
            t, constants, x_coord1=x_coords[0], x_coord2=x_coords[1], X_samples=X_ref,
            data_marginal_pinn=data_marginal_pinn,
            gaussian_lp=gaussian_lp,
            gaussian_ut=None,#gaussian_ut,
            p_net_gmm=p_net_gmm,
        )

        # Full corner plot
        # plot_full_corner(constants, t, X_ref, OUTPUT_PATH=OUTPUT_PATH)

        plt.show()


def compare_methods():
    global constants

    # p_net = PNet(constants, input_feature=7)
    # scale = np.load("data/pre_compute/p_init_max.npz")["value"]
    # scale_torch = torch.tensor(scale, dtype=torch.float32)
    # p_net.scale = scale_torch
    # p_net = PNet_Scaled(constants, input_feature=7)

    p_net = PNet_XL(constants, input_feature=7)
    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale = p_max
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    print("p net scale: ", p_net.scale)
    OUTPUT_PATH = "output/v0_scaled_T0.3"
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    # --- "output/v0_scaled_T0.3(E1Net_XL) ---"
    # e1_net = E1Net_XL(constants, p_net=copy.deepcopy(p_net), scale=scale_torch, input_feature=7)

    # NOTE: test if e1_net needs p_net directly?
    e1_net = E1Net_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)

    # print("[check] e1_net scale: {:.5f}, normalize: {:.5f}".format(
    # e1_net.scale, e1_net.normalize))
    # e1_net.normalize = scale_torch
    # e1_net = E1Net_Scaled(constants)
    # e1_net.scale = scale_torch*0.05 # assuming 10% percent error
    # e1_net.set_p_net(p_net)
    
    e1_net = load_trained_model(e1_net, path=OUTPUT_PATH+"/e1_net.pth"); e1_net.eval()

    # --- pinn-gmm ---
    p_net_gmm = TimeToGMM6D_V0(constants, K=11)
    p_net_gmm = load_trained_model(p_net_gmm, path="output/pinn-gmm(V0)/p_net.pth"); p_net_gmm.eval()

    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)

    metrics_path = "output/baseline_methods/metrics.npz"
    # compute_pdf_variations(N_batch=1000, p_net=p_net, p_net_gmm=p_net_gmm, 
    #                        data_lp=data_lp, data_ut=data_ut, e1_net=e1_net, save_path=metrics_path)
    metrics = load_metrics_npz(metrics_path); plot_pdf_metrics(metrics)

    # Visualize marginal PDF
    compare_corner_plots(p_net=p_net, data_lp=data_lp, data_ut=data_ut, OUTPUT_PATH=OUTPUT_PATH,
                         p_net_gmm=p_net_gmm)
        

def main():
    compare_methods()


if __name__ == "__main__":
    main()