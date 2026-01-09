import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
from exp_utilities.constants import Case1_6D_Constants
from exp_utilities.plot_util import plt, plot_full_corner, plot_corner_elem, plot_pdf_metrics
from run_baseline import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, SAVE_PATH_GMM_PROPAGATE
from monte import p_init

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.astrodynamics import *
from utilities._General.neuralnetworks import PNet_XL, ENet_XL, TimeToGMM_V0, TimeToGMM_NoEncoder, load_trained_model
from utilities._General.util import (compute_volume, save_metrics_npz, load_metrics_npz, 
                                     p_total_variation, plot_training_history)
from utilities._General.baseline_methods import PropagationData
from utilities._General.classic_gmm import GMMWhitenedModel, make_gmm_pdf


COMPUTE = False
NBATCH: int = 10000
SAVEPLOT: bool = False
constants = Case1_6D_Constants()


def parse_args():
    def _str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ("y", "yes", "t", "true", "1", "on"):  return True
        if v in ("n", "no", "f", "false", "0", "off"): return False
        raise argparse.ArgumentTypeError("Expected a boolean value.")
    p = argparse.ArgumentParser()
    p.add_argument("--compute", type=_str2bool, default=COMPUTE,  help="Run compute stage (True/False)")
    p.add_argument("--Nbatch",  type=int, default=NBATCH,   help="Batch size (int)")
    p.add_argument("--saveplot", type=_str2bool, default=SAVEPLOT,   help="save plot (bool)")
    return p.parse_args()


def _helper_p_normalize_const(constants, p1):
    """
    """
    bounds = np.array([
        constants.N_X1_RANGE,
        constants.N_X2_RANGE,
        constants.N_X3_RANGE,
        constants.N_X4_RANGE,
        constants.N_X5_RANGE,
        constants.N_X6_RANGE,
    ])
    vol_est = compute_volume(bounds)
    p1 = p1.reshape(-1,)
    return np.mean(p1) * vol_est


def compute_generalKL(t, X, p_data_normal=None, p_net=None):
    global constants
    eps = np.finfo(np.float32).tiny   # machine precision of np.float32 ≈ 1.175e-38
    if(p_net is not None):
        _x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        _t = np.ones((len(_x_tensor), 1)) * t
        _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)
        pdf_eval = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)

    if(p_data_normal is not None):
        _, _w, _mu, _cov = p_data_normal.get(t)
        _, _w, _mu, _cov = p_data_normal.get(t)
        _mu = np.float32(_mu)
        _cov = np.float32(_cov)
        _pdf_func = make_gmm_pdf(_w, _mu, _cov)
        pdf_eval = _pdf_func(X).reshape(-1,)

    mask = pdf_eval >= eps
    if np.any(mask):
        KL = np.mean(-np.log(pdf_eval[mask])).item()
    else:
        print("All pdf values are below machine precision!")
        KL = np.inf
    return KL


def print_metrics_block(metrics, idx, colw=12, prec=6):
    """
    Print a fixed-width table block for time index `idx`.

    Layout (same as before):
      header: '=== t = ... ==='
      rows:   LP, UT, GMM, PINN, PINN-GMM
      cols:   rel(%), TV, gKL, B1(%)
    """
    t = float(metrics["t"][idx])

    def get(key):
        arr = metrics.get(key, None)
        return None if arr is None or idx >= len(arr) else arr[idx]

    def _fmt_num(x, width=colw, p=prec, suf=""):
        # em-dash for missing / non-finite
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—".rjust(width)
        # try decreasing precision until it fits
        pp = p
        while pp >= 1:
            s = f"{x:.{pp}g}{suf}"
            if len(s) <= width:
                return s.rjust(width)
            pp -= 1
        # last resort: hard cut (should rarely trigger)
        return (f"{x:.1e}{suf}")[-width:].rjust(width)

    # column headers (fixed widths)
    name_w = 20
    header = (
        f"\n=== t = {t:.3f} ===\n"
        f"{'Model':<{name_w}}"
        f"{'rel(%)':>{colw}}"
        f"{'TV (%)':>{colw}}"
        f"{'gKL':>{colw}}"
        f"{'B1 (%)':>{colw}}"
    )
    print(header)
    print("-" * (name_w + 4*colw))

    # rows in your preferred order
    rows = [
        ("LP",        get("rel_error_lp"),      get("tv_lp"),      get("g_kl_lp"),      None),
        ("UT",        get("rel_error_ut"),      get("tv_ut"),      get("g_kl_ut"),      None),
        ("GMM",       get("rel_error_gmm"),     get("tv_gmm"),     get("g_kl_gmm"),     None),
        ("PINN",      get("rel_error_pinn"),    get("tv_pinn"),    get("g_kl_pinn"),    get("B1_pinn")),
        ("PINN-GMM",  get("rel_error_pinngmm"), get("tv_pinngmm"), get("g_kl_pinngmm"), get("B1_pinngmm")),
        ("PINN-GMM(vanilla)",  get("rel_error_pinngmm_vanilla"), get("tv_pinngmm_vanilla"), get("g_kl_pinngmm_vanilla"), None),
        ("PINN-GMM(noencod)",  get("rel_error_pinngmm_noencoder"), get("tv_pinngmm_noencoder"), get("g_kl_pinngmm_noencoder"), None),
    ]

    for name, rel, tv, gkl, b1 in rows:
        print(
            f"{name:<{name_w}}"
            f"{_fmt_num(rel)}"
            f"{_fmt_num(tv)}"
            f"{_fmt_num(gkl)}"
            f"{_fmt_num(b1)}"
        )


def compute_pdf_variations(data_mc=None, p_net=None, p_net_gmm=None, 
                           p_net_gmm_vanilla= None, p_net_gmm_noencoder=None,
                           data_lp=None, data_ut=None, data_gmm=None,
                           e1_net=None, e1_net_gmm=None, save_path=None):
    global constants
    metrics = {
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_gmm": [],
        "rel_error_pinn": [],
        "rel_error_pinngmm": [],
        "rel_error_pinngmm_vanilla": [],
        "rel_error_pinngmm_noencoder": [],

        "tv_lp": [],
        "tv_ut": [],
        "tv_gmm": [],
        "tv_pinn": [],
        "tv_pinngmm": [],
        "tv_pinngmm_vanilla": [],
        "tv_pinngmm_noencoder": [],

        "g_kl_lp": [],
        "g_kl_ut": [],
        "g_kl_gmm": [],
        "g_kl_pinn": [],
        "g_kl_pinngmm": [],
        "g_kl_pinngmm_vanilla": [],
        "g_kl_pinngmm_noencoder": [],

        # "g_kl_pinngmm(no-imp)": [],
        # "g_kl_pinngmm(uniform)": [],

        "B1_pinn": [],
        "B1_pinngmm": [],
        "B1_pinngmm_raw": [],

        "t": constants.T_PRIME_SPAN,
    } 

    print("evaluate metrics over times: ", metrics["t"])
    N_batch = NBATCH
    bounds = np.array([
        constants.N_X1_RANGE,
        constants.N_X2_RANGE,
        constants.N_X3_RANGE,
        constants.N_X4_RANGE,
        constants.N_X5_RANGE,
        constants.N_X6_RANGE,
    ])
    vol_est = compute_volume(bounds)

    for idx, t in enumerate(metrics["t"]):
        pdf_ref_max = 0.0
       
        delta_p_pinn_max = 0.0
        tv_pinn = 0.0
        g_kl_pinn = 0.0

        delta_p_pinngmm_max = 0.0
        tv_pinngmm = 0.0
        g_kl_pinngmm = 0.0
        # g_kl_pinngmm_noimp = 0.0
        # g_kl_pinngmm_uniform = 0.0

        delta_p_pinngmm_vanilla_max = 0.0
        tv_pinngmm_vanilla = 0.0
        g_kl_pinngmm_vanilla = 0.0

        delta_p_pinngmm_noencoder_max = 0.0
        tv_pinngmm_noencoder = 0.0
        g_kl_pinngmm_noencoder = 0.0
        
        delta_p_lp_max = 0.0
        tv_lp = 0.0
        g_kl_lp = 0.0

        delta_p_ut_max = 0.0
        tv_ut = 0.0
        g_kl_ut = 0.0

        delta_p_gmm_max = 0.0
        tv_gmm = 0.0
        g_kl_gmm = 0.0

        # additional data for pinn
        e1_pinn_max = 0.0
        e1_pinngmm_max = 0.0
        Z_pinn = 0.0
        Z_pinn_gmm = 0.0
        
        # --- choose the source 'true' gmm pdf ---
        # gmm = GMMWhitenedModel.load(data_mc+"gmm_whitened_t{:.2f}.npz".format(t))
        data_path_list = []
        for data_idx in range(1, 1+1):
            data_path = data_mc+"run"+str(data_idx)+"/"+"gmm_whitened_t{:.2f}.npz".format(t)
            data_path_list.append(data_path)
        gmm = GMMWhitenedModel.load_multiple(data_path_list)

        # --- compute general KL without normalizing constant ---
        # filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
        N_datarun = 10
        for data_idx in range(1, 1+N_datarun):
            filename_mc = data_mc+"run"+str(data_idx)+"/"+"xsamples_t{:.3f}.npy".format(t) #[temp]
            X_mc = np.load(filename_mc)
            if(p_net is not None):
                g_kl_pinn += (compute_generalKL(t, X_mc, p_net=p_net))/N_datarun

            if(p_net_gmm is not None):
                g_kl_pinngmm += (compute_generalKL(t, X_mc, p_net=p_net_gmm))/N_datarun

            if(p_net_gmm_vanilla is not None):
                g_kl_pinngmm_vanilla += (compute_generalKL(t, X_mc, p_net=p_net_gmm_vanilla))/N_datarun

            if(p_net_gmm_noencoder is not None):
                g_kl_pinngmm_noencoder += (compute_generalKL(t, X_mc, p_net=p_net_gmm_noencoder))/N_datarun
                print("debug", g_kl_pinngmm_noencoder)

            if(data_lp is not None):
                g_kl_lp += (compute_generalKL(t, X_mc, p_data_normal=data_lp))/N_datarun
            
            if(data_ut is not None):
                g_kl_ut += (compute_generalKL(t, X_mc, p_data_normal=data_ut))/N_datarun

            if(data_gmm is not None):
                g_kl_gmm += (compute_generalKL(t, X_mc, p_data_normal=data_gmm))/N_datarun

        # --- compute rel. error and tv by batch-procesing ---
        for j in tqdm(range(1, N_batch+1), desc="Propagating batches"):
            X = constants.sample_x_uniform(N_samples=100000) # samples from random uniform
            pdf_ref = gmm.pdf(X).reshape(-1,)
            pdf_ref_max = max(pdf_ref_max, np.max(pdf_ref).item())
            
            _x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)
            _t = np.ones((len(_x_tensor), 1)) * t
            _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=False).view(-1,1)

            if(p_net is not None):
                pdf_pinn = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                _delta_p = np.max(np.abs(pdf_pinn - pdf_ref)).item()
                delta_p_pinn_max = max(delta_p_pinn_max, _delta_p)
                _tv = p_total_variation(pdf_pinn, pdf_ref, vol_est)
                tv_pinn += _tv/N_batch
                Z_pinn += (np.mean(pdf_pinn) * vol_est)/N_batch
                del pdf_pinn
            if(e1_net is not None):
                e1_pinn = e1_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                e1_pinn_max = max(e1_pinn_max, np.max(np.abs(e1_pinn)).item())
                del e1_pinn

            if(p_net_gmm is not None):
                pdf_pinngmm = p_net_gmm(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                _delta_p = np.max(np.abs(pdf_pinngmm - pdf_ref)).item()
                delta_p_pinngmm_max = max(delta_p_pinngmm_max, _delta_p)
                _tv = p_total_variation(pdf_pinngmm, pdf_ref, vol_est)
                tv_pinngmm += _tv/N_batch
                Z_pinn_gmm += (np.mean(pdf_pinngmm) * vol_est)/N_batch
                del pdf_pinngmm
            if(e1_net_gmm is not None):
                e1_pinngmm = e1_net_gmm(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                e1_pinngmm_max = max(e1_pinngmm_max, np.max(np.abs(e1_pinngmm)).item())
                del e1_pinngmm
            
            # pinn-gmm_vanilla
            if(p_net_gmm_vanilla is not None):
                pdf_pinn = p_net_gmm_vanilla(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                _delta_p = np.max(np.abs(pdf_pinn - pdf_ref)).item()
                delta_p_pinngmm_vanilla_max = max(delta_p_pinngmm_vanilla_max, _delta_p)
                _tv = p_total_variation(pdf_pinn, pdf_ref, vol_est)
                tv_pinngmm_vanilla += _tv/N_batch
                del pdf_pinn 

            if(p_net_gmm_noencoder is not None):
                pdf_pinn = p_net_gmm_noencoder(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                _delta_p = np.max(np.abs(pdf_pinn - pdf_ref)).item()
                delta_p_pinngmm_noencoder_max = max(delta_p_pinngmm_noencoder_max, _delta_p)
                _tv = p_total_variation(pdf_pinn, pdf_ref, vol_est)
                tv_pinngmm_noencoder += _tv/N_batch
                del pdf_pinn 

            del _t, _t_tensor, _x_tensor

            if(data_lp is not None):
                _, _, _mu_lp, _cov_lp = data_lp.get(t)
                _pdf_lp_func = make_gmm_pdf(1., _mu_lp, _cov_lp)
                pdf_lp = _pdf_lp_func(X).reshape(-1,)
                _delta_p = np.max(np.abs(pdf_lp - pdf_ref)).item()
                delta_p_lp_max = max(delta_p_lp_max, _delta_p)
                _tv = p_total_variation(pdf_lp, pdf_ref, vol_est)
                tv_lp += _tv/N_batch
                del pdf_lp, _pdf_lp_func

            if(data_ut is not None):
                _, _, _mu_ut, _cov_ut = data_ut.get(t)
                _pdf_ut_func = make_gmm_pdf(1., _mu_ut, _cov_ut)
                pdf_ut = _pdf_ut_func(X).reshape(-1,)
                _delta_p = np.max(np.abs(pdf_ut - pdf_ref)).item()
                delta_p_ut_max = max(delta_p_ut_max, _delta_p)
                _tv = p_total_variation(pdf_ut, pdf_ref, vol_est)
                tv_ut += _tv/N_batch
                del pdf_ut, _pdf_ut_func

            if(data_gmm is not None):
                _, _w_gmm, _mu_gmm, _cov_gmm = data_gmm.get(t)
                _pdf_gmm_func = make_gmm_pdf(_w_gmm, _mu_gmm, _cov_gmm)
                pdf_gmm = _pdf_gmm_func(X).reshape(-1,)
                _delta_p = np.max(np.abs(pdf_gmm - pdf_ref)).item()
                delta_p_gmm_max = max(delta_p_gmm_max, _delta_p)
                _tv = p_total_variation(pdf_gmm, pdf_ref, vol_est)
                tv_gmm += _tv/N_batch
                del pdf_gmm, _pdf_gmm_func

            # if(e1_net is not None):
            #     e1_pinn = constants.SCALING_PDF * e1_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            #     e1_pinn_max = max(e1_pinn_max, np.max(np.abs(e1_pinn)).item())
            #     del e1_pinn
            # del _t, _t_tensor, X_scaled, _x_tensor
        
        # After all batch
        g_kl_pinn += Z_pinn
        g_kl_pinngmm += 1; g_kl_pinngmm_vanilla += 1; g_kl_pinngmm_noencoder += 1
        g_kl_lp += 1
        g_kl_ut += 1
        g_kl_gmm += 1

        metrics["rel_error_lp"].append(100.*delta_p_lp_max/pdf_ref_max)
        metrics["tv_lp"].append(tv_lp)
        metrics["g_kl_lp"].append(g_kl_lp)

        metrics["rel_error_ut"].append(100.*delta_p_ut_max/pdf_ref_max)
        metrics["tv_ut"].append(tv_lp)
        metrics["g_kl_ut"].append(g_kl_ut)

        metrics["rel_error_gmm"].append(100.*delta_p_gmm_max/pdf_ref_max)
        metrics["tv_gmm"].append(tv_gmm)
        metrics["g_kl_gmm"].append(g_kl_gmm)

        metrics["rel_error_pinn"].append(100.*delta_p_pinn_max/pdf_ref_max)
        metrics["tv_pinn"].append(tv_pinn)
        metrics["g_kl_pinn"].append(g_kl_pinn)

        metrics["rel_error_pinngmm"].append(100.*delta_p_pinngmm_max/pdf_ref_max)
        metrics["tv_pinngmm"].append(tv_pinngmm)
        metrics["g_kl_pinngmm"].append(g_kl_pinngmm)

        metrics["rel_error_pinngmm_vanilla"].append(100.*delta_p_pinngmm_vanilla_max/pdf_ref_max)
        metrics["tv_pinngmm_vanilla"].append(tv_pinngmm_vanilla)
        metrics["g_kl_pinngmm_vanilla"].append(g_kl_pinngmm_vanilla)

        metrics["rel_error_pinngmm_noencoder"].append(100.*delta_p_pinngmm_noencoder_max/pdf_ref_max)
        metrics["tv_pinngmm_noencoder"].append(tv_pinngmm_noencoder)
        metrics["g_kl_pinngmm_noencoder"].append(g_kl_pinngmm_noencoder)

        metrics["B1_pinngmm"].append(100. * 2. * e1_pinngmm_max/pdf_ref_max)
        # print("[debug] B1_pinngmm: ", e1_pinngmm_max, pdf_ref_max)
        metrics["B1_pinn"].append(100. * 2. * e1_pinn_max/pdf_ref_max)
        # print("[debug] B1_pinn: ", e1_pinn_max, pdf_ref_max)
        metrics["B1_pinngmm_raw"].append(2. * e1_pinngmm_max)
            
        print_metrics_block(metrics, idx)
        print("[debug] B1_pinngmm_raw: ", 2. * e1_pinngmm_max)
        print("[debug] Z_pinn: ", Z_pinn, Z_pinn_gmm)

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

        # filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
        filename_mc = data_mc+"run1/"+"xsamples_t{:.3f}.npy".format(t)
        print(filename_mc)
        if(os.path.exists(filename_mc)):
            X_ref = np.load(filename_mc)
        else:
            print("[error] file ", filename_mc, " not found")
            return

        # Full corner plot
        plot_full_corner(constants, t, X_ref, 
                         PNet_XL_PATH=None,
                         p_net_gmm_N1=None,
                         p_net_gmm=p_net_gmm,
                        #  data_lp=data_lp, 
                        #  data_ut=data_ut,
                        #  data_gmm=data_gmm,
                         ranges="fixed",
                         save_plot=SAVEPLOT
                         )
        
        # # Single element plot 1D (x6,)
        # plot_corner_elem(constants, t, X_ref, (6,),
        #                  data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
        #                  p_net_gmm=p_net_gmm,
        #                  save_plot=SAVEPLOT)

        # Single element plot 2D (x1, x6)
        plot_corner_elem(constants, t, X_ref, (1, 6),
                         data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
                         p_net_gmm=p_net_gmm,
                         save_plot=SAVEPLOT)

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


def config_trained_models():
    trained_models = {
        "PINN-MLP_vanilla": "output/pinn-xl_vanilla",
        # "PINN-XL_bias": "output/pinn-xl_bias",
        # "PINN-XL_bias-test": "output/pinn-xl_bias-test",
        "PINN-GMM_vanilla" : "output/pinn-gmm_vanilla",
        "PINN-GMM-noencoder": "output/pinn-gmm-noencoder",
        "PINN-GMM_bias" : "output/pinn-gmm_bias",
    }
    return trained_models


def load_model_by_key(trained_models, key=""):
    if key not in trained_models:
        raise KeyError(f"Missing key {key!r}. Available: {list(trained_models.keys())}")

    # Scale of p0 max
    scale = p_init(constants, [constants.N_MEAN_I]).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32); print(scale_torch)

    KEY_PATH = trained_models[key]

    if(key == "PINN-MLP_vanilla"):
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = None
        rar_samples = None
        return p_net, e1_net, rar_samples

    if(key == "PINN-MLP_bias"):
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        rar_samples = None
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-GMM_vanilla"):
        p_net = TimeToGMM_V0(constants, K=11, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = None
        _p = Path(KEY_PATH) / "p_net-RARsamples.npz"
        if _p.is_file(): 
            rar_samples = np.load(_p)
        else:
            rar_samples = None
        del _p
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-GMM-noencoder"):
        p_net = TimeToGMM_NoEncoder(constants, K=11, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = None
        _p = Path(KEY_PATH) / "p_net-RARsamples.npz"
        if _p.is_file(): 
            rar_samples = np.load(_p)
        else:
            rar_samples = None
        del _p
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-GMM_bias"):
        p_net = TimeToGMM_V0(constants, K=11, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        _p = Path(KEY_PATH) / "p_net-RARsamples.npz"
        if _p.is_file(): 
            rar_samples = np.load(_p)
        else:
            rar_samples = None
        del _p
        return p_net, e1_net, rar_samples


def main():
    global constants
    # print(constants.N_MEAN_I)
    # print(constants.N_COV_I)
    # print(constants.NX_RANGE)
    # return

    # data_mc = "dataset/run1/"
    data_mc = "dataset/"

    # setup trained models dictionary
    trained_models = config_trained_models()

    # load pinn-mlp
    p_net, e1_net, _ = load_model_by_key(trained_models, key="PINN-MLP_vanilla")

    # load pinn-gmm
    p_net_gmm_vanilla, _, _ = load_model_by_key(trained_models, key="PINN-GMM_vanilla")
    p_net_gmm_noencoder, _, _ = load_model_by_key(trained_models, key="PINN-GMM-noencoder")
    p_net_gmm, e1_net_gmm, rar_samples = load_model_by_key(trained_models, key="PINN-GMM_bias")

    # baseline methods
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)

    # --- metrics ---
    metrics_path = "output/metric_NB="+str(NBATCH)+".npz"
    if(COMPUTE):
        compute_pdf_variations(data_mc=data_mc, 
            p_net=p_net, p_net_gmm=p_net_gmm,
            p_net_gmm_vanilla=p_net_gmm_vanilla, p_net_gmm_noencoder=p_net_gmm_noencoder,
            data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
            e1_net=e1_net, e1_net_gmm=e1_net_gmm,
            save_path=metrics_path)
    metrics = load_metrics_npz(metrics_path)
    print(metrics["B1_pinngmm_raw"])
    plot_pdf_metrics(metrics, save_plot=SAVEPLOT)
    
    # --- plots ---
    compare_corner_plots(data_mc, 
                         data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
                         PNet_XL_PATH=None,
                         p_net_gmm=p_net_gmm
                         )
    # compare_corner_plots_XYZ(MC_FOLDER)

    # --- plot training history ---
    plot_training_history(trained_models)


if __name__ == "__main__":
    args = parse_args()
    COMPUTE  = bool(args.compute)
    NBATCH    = int(args.Nbatch)
    SAVEPLOT = bool(args.saveplot)
    main()
