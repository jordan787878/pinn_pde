import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import multivariate_normal
from exp_utilities.constants import Case1_6D_Constants
from exp_utilities.plot_util import plt, plot_full_corner, plot_pdf_metrics
from baseline_methods import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, SAVE_PATH_GMM_PROPAGATE, PropagationData
import sys
sys.path.insert(0, '../utilities/')
from _General.astrodynamics import *
from _General.neuralnetworks import PNet, PNet_XL_Sphere, load_trained_model
from _General.neuralnetworks import TimeToGMM6D
from _General.util import compute_volume, save_metrics_npz, load_metrics_npz


constants = Case1_6D_Constants()


def _p_gaussian(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    scales = np.sqrt(np.diag(cov))  # std per dimension
    cov_scaled = cov / np.outer(scales, scales)
    x_scaled = (x - mean) / scales
    rv = multivariate_normal(mean=np.zeros(len(mean)), cov=cov_scaled)
    pdf_eval = rv.pdf(x_scaled) / np.prod(scales)  # back-transform
    return pdf_eval.reshape(-1,)


def _helper_p_normalize_const(constants, p1):
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


def _get_uniform_Xsamples_numpy(N_samples=None):
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


def compute_generalKL(t, X, p_data_normal=None, p_net=None, key=None):
    global constants
    eps = np.finfo(np.float32).tiny   # machine precision of np.float32 ≈ 1.175e-38
    if(p_net is not None):
        _x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        _t = np.ones((len(_x_tensor), 1)) * t
        _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
        pdf_eval = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        # [temp]
        Z_p = 1.
        # N_samples = 10000000
        # X_samples = _get_uniform_Xsamples_numpy(N_samples=N_samples)
        # _x_tensor = torch.tensor(X_samples, dtype=torch.float32, requires_grad=False)
        # _t = np.ones((len(_x_tensor), 1)) * t
        # _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
        # pdf_pinn = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        # Z_p = _helper_p_normalize_const(constants, pdf_pinn)
        # print(Z_p)

    if(p_data_normal is not None):
        _, _w, _mu, _cov = p_data_normal.get(t)
        if(_w.shape[0] > 1):
            pdf_eval = np.zeros(X.shape[0])
            for j in range(_w.shape[0]):
                pdf_func = multivariate_normal(mean=_mu[j,:], cov=_cov[j,:,:])
                p_k = pdf_func.pdf(X).reshape(-1,)
                pdf_eval += _w[j] * p_k
        else:
            pdf_func = multivariate_normal(mean=_mu, cov=_cov)
            pdf_eval = pdf_func.pdf(X).reshape(-1,)
            # _mu = np.float32(_mu)
            # _cov = np.float32(_cov)
            # pdf_eval = _p_gaussian(X, _mu, _cov).reshape(-1,)
        Z_p = 1.

    mask = pdf_eval >= eps
    if np.any(mask):
        KL = np.mean(-np.log(pdf_eval[mask])).item() + Z_p
    else:
        print("All pdf_lp values are below machine precision!")
        KL = np.NaN
    return KL


def compute_pdf_variations(data_mc=None, p_net=None, p_net_gmm=None,
                           p_net_gmm_noimp=None, p_net_gmm_uniform=None,
                           data_lp=None, data_ut=None, data_gmm=None,
                           e1_net=None, save_path=None):
    global constants
    metrics = {
        # "rel_error_lp": [],
        # "rel_error_ut": [],
        # "rel_error_pinn": [],
        # "tv_lp": [],
        # "tv_ut": [],
        # "tv_pinn": [],
        "g_kl_lp": [],
        "g_kl_ut": [],
        "g_kl_gmm": [],
        "g_kl_pinn": [],
        "g_kl_pinngmm": [],
        "g_kl_pinngmm(no-imp)": [],
        "g_kl_pinngmm(uniform)": [],
        # "B1_pinn": [],
        "t": constants.T_PRIME_SPAN,
        # "t": np.round(np.arange(0.0, 0.3+0.05, 0.05, dtype=np.float32),2)
    } 
    print("evaluate metrics over times: ", metrics["t"])
    N_batch = 1
    for idx, t in enumerate(metrics["t"]):
        print("\ntime {:.4f}".format(t))
        pdf_ref_max = 0.0
        delta_p_pinn_max = 0.0
        tv_pinn = 0.0
        g_kl_pinn = 0.0
        g_kl_pinngmm = 0.0
        g_kl_pinngmm_noimp = 0.0
        g_kl_pinngmm_uniform = 0.0
        delta_p_lp_max = 0.0
        tv_lp = 0.0
        g_kl_lp = 0.0
        delta_p_ut_max = 0.0
        tv_ut = 0.0
        g_kl_ut = 0.0
        g_kl_gmm = 0.0
        # additional data for pinn
        e1_pinn_max = 0.0
        Z_pinn = 0.0
        for j in tqdm(range(1, N_batch+1), desc="Propagating batches"):
            # X = _get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
            # X_scaled = constants.scaled_x(X)
            # _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)
            # # print("[info] pdf ref")
            # pdf_ref = p_sol(constants, X, t).reshape(-1,)
            # pdf_ref_max = max(pdf_ref_max, np.max(pdf_ref).item())
            # X_mc = sample_joint_pdf(constants, t_prime=t, n_samples=N_samples, seed=j)
            filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
            if(os.path.exists(filename_mc)):
                X_mc = np.load(filename_mc)
            else:
                continue

            if(p_net is not None):
                _gkl = compute_generalKL(t, X_mc, p_net=p_net)
                g_kl_pinn += _gkl/N_batch
                metrics["g_kl_pinn"].append(g_kl_pinn)

            if(p_net_gmm is not None):
                _gkl = compute_generalKL(t, X_mc, p_net=p_net_gmm)
                g_kl_pinngmm += _gkl/N_batch
                metrics["g_kl_pinngmm"].append(g_kl_pinngmm)

            if(p_net_gmm_noimp is not None):
                _gkl = compute_generalKL(t, X_mc, p_net=p_net_gmm_noimp)
                g_kl_pinngmm_noimp += _gkl/N_batch
                metrics["g_kl_pinngmm(no-imp)"].append(g_kl_pinngmm_noimp)

            if(p_net_gmm_uniform is not None):
                _gkl = compute_generalKL(t, X_mc, p_net=p_net_gmm_uniform)
                g_kl_pinngmm_uniform += _gkl/N_batch
                metrics["g_kl_pinngmm(uniform)"].append(g_kl_pinngmm_uniform)

            if(data_lp is not None):
                _gkl = compute_generalKL(t, X_mc, p_data_normal=data_lp)
                g_kl_lp += _gkl/N_batch
                metrics["g_kl_lp"].append(g_kl_lp)

            if(data_ut is not None):
                _gkl = compute_generalKL(t, X_mc, p_data_normal=data_ut)
                g_kl_ut += _gkl/N_batch
                metrics["g_kl_ut"].append(g_kl_ut)

            if(data_gmm is not None):
                _gkl = compute_generalKL(t, X_mc, p_data_normal=data_gmm)
                g_kl_gmm += _gkl/N_batch
                metrics["g_kl_gmm"].append(g_kl_gmm)

            # if(e1_net is not None):
            #     e1_pinn = constants.SCALING_PDF * e1_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            #     e1_pinn_max = max(e1_pinn_max, np.max(np.abs(e1_pinn)).item())
            #     del e1_pinn
            # del _t, _t_tensor, X_scaled, _x_tensor
            
        print(g_kl_pinn, g_kl_pinngmm, g_kl_lp, g_kl_ut, g_kl_gmm)

    # save metrics
    save_metrics_npz(metrics, save_path)


def compare_corner_plots(data_mc=None, data_lp=None, data_ut=None, data_gmm=None,
                         PNet_XL_PATH=None, p_net_gmm_N1=None, p_net_gmm=None, save_path=None):
    
    def _print_min_max_per_dim(X: np.ndarray):
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[1] == 6, f"Expected (N,6), got {X.shape}"
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        for i, (mn, mx) in enumerate(zip(mins, maxs), start=1):
            print(f"x{i}: min={mn:.6g}  max={mx:.6g}")

    global constants
    t_show = [constants.T_PRIME_SPAN[0], constants.T_PRIME_SPAN[-1]]
    for idx, t in enumerate(t_show):
        print("\ntime {:.4f}".format(t))
        if(data_mc is None):
            continue
        # print("[info] ref")
        # load samples from MC
        filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
        if(os.path.exists(filename_mc)):
            X_ref = np.load(filename_mc)
            # _print_min_max_per_dim(X_ref)
        else:
            continue
        # Full corner plot
        plot_full_corner(constants, t, X_ref, 
                         PNet_XL_PATH=PNet_XL_PATH,
                         p_net_gmm_N1=p_net_gmm_N1,
                         p_net_gmm=p_net_gmm,
                         data_lp=data_lp, 
                         data_gmm=data_gmm,
                         ranges="fixed"
                         )
        plt.show()


def compare_corner_plots_XYZ(data_mc=None, data_lp=None, data_ut=None, save_path=None, 
                             OUTPUT_PATH=None):
    global constants
    t_show = constants.T_PRIME_SPAN

    # N_samples = 1000000
    # X = _get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
    # X_scaled = constants.scaled_x(X)
    # _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

    # Compute XYZ ranges
    X_min = np.inf
    X_max = -np.inf
    Y_min = np.inf
    Y_max = -np.inf
    Z_min = np.inf
    Z_max = -np.inf
    for idx, t in enumerate(t_show):
        if(data_mc is None):
            continue

        # load samples from MC
        filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
        if(os.path.exists(filename_mc)):
            X_ref = np.load(filename_mc)
        else:
            continue

        X_cart_i = sphere_to_cartesian(rnsphere_to_sphere(X_ref, t, constants))
        XYZ = X_cart_i[:, 0:3]
        X_min = min(X_min, np.min(XYZ[:, 0]).item())
        X_max = max(X_max, np.max(XYZ[:, 0]).item())
        Y_min = min(Y_min, np.min(XYZ[:, 1]).item())
        Y_max = max(Y_max, np.max(XYZ[:, 1]).item())
        Z_min = min(Z_min, np.min(XYZ[:, 2]).item())
        Z_max = max(Z_max, np.max(XYZ[:, 2]).item())
    # print(X_min, X_max)
    # print(Y_min, Y_max)
    # print(Z_min, Z_max)
    ranges = [
        (X_min, X_max),
        (Y_min, Y_max),
        (Z_min, Z_max),
    ]

    # Plot with fixed XYZ ranges
    for idx, t in enumerate(t_show):
        print("\ntime {:.4f}".format(t))
        if(data_mc is None):
            continue

        print("[info] ref")
        # load samples from MC
        filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
        if(os.path.exists(filename_mc)):
            X_ref = np.load(filename_mc)
        else:
            continue

        X_cart_i = sphere_to_cartesian(rnsphere_to_sphere(X_ref, t, constants))
        XYZ = X_cart_i[:, 0:3]

        # Full corner plot
        plot_full_corner(constants, t, XYZ, OUTPUT_PATH=OUTPUT_PATH, ranges=ranges, 
                         labels=["X", "Y", "Z"])

        plt.show()


def main():
    data_mc = "dataset/run1/"
    global constants

    # mlp
    PNet_XL_PATH = "output/v0"
    p_net = PNet_XL_Sphere(constants)
    p_net = load_trained_model(p_net, path=PNet_XL_PATH+"/p_net.pth"); p_net.eval()

    # pinn-gmm
    PNet_GMM_PATH = "output/pinn-gmm"
    p_net_gmm = TimeToGMM6D(constants, K=11)
    p_net_gmm = load_trained_model(p_net_gmm, path=PNet_GMM_PATH+"/p_net.pth"); p_net_gmm.eval()

    PNet_GMM_PATH = "output/pinn-gmm(no-imp)"
    p_net_gmm_noimp = TimeToGMM6D(constants, K=11)
    p_net_gmm_noimp = load_trained_model(p_net_gmm_noimp, path=PNet_GMM_PATH+"/p_net.pth"); p_net_gmm_noimp.eval()

    PNet_GMM_PATH = "output/pinn-gmm(uniform)"
    p_net_gmm_uniform = TimeToGMM6D(constants, K=11)
    p_net_gmm_uniform = load_trained_model(p_net_gmm_uniform, path=PNet_GMM_PATH+"/p_net.pth"); p_net_gmm_uniform.eval()

    # baseline methods
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)

    # --- metrics ---
    metrics_path = "output/baseline_methods/metrics.npz"
    # compute_pdf_variations(data_mc=data_mc, 
    #                        p_net=p_net, p_net_gmm=p_net_gmm, 
    #                        p_net_gmm_noimp=p_net_gmm_noimp, p_net_gmm_uniform=p_net_gmm_uniform,
    #                        data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
    #                        save_path=metrics_path)
    metrics = load_metrics_npz(metrics_path); plot_pdf_metrics(metrics)
    
    # --- plots ---
    # compare_corner_plots(data_mc, 
    #                      data_lp=None, data_ut=None, data_gmm=None,
    #                      PNet_XL_PATH=None,
    #                      p_net_gmm_N1=None,
    #                      p_net_gmm=p_net_gmm,
    #                      )
    # compare_corner_plots_XYZ(MC_FOLDER)


if __name__ == "__main__":
    main()
