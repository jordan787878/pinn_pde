import numpy as np
import torch
import argparse
from tqdm import tqdm
from monte import p_init_scaled, p_sol, print_mc_time
from exp_utilities.constants import Case1_6D_Constants_Equin
from exp_utilities.plot_util import (plot_pdf_metrics, plot_corner_elem,
                                     plot_full_corner, plt)
from run_baseline import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, SAVE_PATH_GMM_PROPAGATE

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.neuralnetworks import PNet_XL, ENet_XL, TimeToGMM_V0, load_trained_model
from utilities._General.util import compute_volume, save_metrics_npz, load_metrics_npz, p_total_variation, plot_training_history
from utilities._General.baseline_methods import PropagationData
from utilities._General.classic_gmm import make_gmm_pdf
from utilities._General.generalsolvers import load_problem_result_npz


COMPUTE: bool = False
NBATCH: int = 10000
SAVEPLOT: bool = False
constants = Case1_6D_Constants_Equin()


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


# ---------------------------------------------
# Sampling from sol joint PDF (By pushforward)
# ---------------------------------------------
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
        _, _w, _mu, _cov = p_data_normal.get(t)
        _mu = np.float32(_mu)
        _cov = np.float32(_cov)
        _pdf_func = make_gmm_pdf(_w, _mu, _cov)
        pdf_eval = _pdf_func(X).reshape(-1,)
    mask = pdf_eval >= eps
    if np.any(mask):
        KL = np.mean(-np.log(pdf_eval[mask])).item()
    else:
        print("All pdf_lp values are below machine precision!")
        KL = np.NaN
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
    ]

    for name, rel, tv, gkl, b1 in rows:
        print(
            f"{name:<{name_w}}"
            f"{_fmt_num(rel)}"
            f"{_fmt_num(tv)}"
            f"{_fmt_num(gkl)}"
            f"{_fmt_num(b1)}"
        )


def compute_pdf_variations(N_batch=1, p_net=None, p_net_gmm=None, p_net_gmm_vanilla= None,
                           data_lp=None, data_ut=None, data_gmm=None,
                           e1_net=None,  e1_net_gmm=None, save_path=None):
    global constants
    metrics = {
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_gmm": [],
        "rel_error_pinn": [],
        "rel_error_pinngmm": [],
        "rel_error_pinngmm_vanilla": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_gmm": [],
        "tv_pinn": [],
        "tv_pinngmm": [],
        "tv_pinngmm_vanilla": [],
        "g_kl_lp": [],
        "g_kl_ut": [],
        "g_kl_gmm": [],
        "g_kl_pinn": [],
        "g_kl_pinngmm": [],
        "g_kl_pinngmm_vanilla": [],
        "B1_pinn": [],
        "B1_pinngmm": [],
        "B1_pinngmm_raw": [],
        "t": np.round(np.arange(0.0, 0.3+0.05, 0.05, dtype=np.float32),2)
    } 
    print("evaluate metrics over times: ", metrics["t"])
    N_samples = int(1e+5)

    bounds = np.array([
        constants.X1_RANGE,
        constants.X2_RANGE,
        constants.X3_RANGE,
        constants.X4_RANGE,
        constants.X5_RANGE,
        constants.X6_RANGE,
    ])
    vol_est = compute_volume(bounds)

    for idx, t in enumerate(metrics["t"]):
        print("\ntime {:.4f}".format(t))
        pdf_ref_max = 0.0

        delta_p_pinn_max = 0.0
        tv_pinn = 0.0
        gkl_pinn = 0.0

        delta_p_pinngmm_max = 0.0
        tv_pinngmm = 0.0
        gkl_pinngmm = 0.0

        delta_p_pinngmm_vanilla_max = 0.0
        tv_pinngmm_vanilla = 0.0
        gkl_pinngmm_vanilla = 0.0

        delta_p_lp_max = 0.0
        tv_lp = 0.0
        gkl_lp = 0.0

        delta_p_ut_max = 0.0
        tv_ut = 0.0
        gkl_ut = 0.0

        delta_p_gmm_max = 0.0
        tv_gmm = 0.0
        gkl_gmm = 0.0

        # additional data for pinn
        e1_pinn_max = 0.0
        Z_pinn = 0.0
        e1_pinngmm_max = 0.0

        # --- pinn-gmm verbose ---
        ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
        ws, mus, covs = (ws.detach().cpu().numpy(),
                            mus.detach().cpu().numpy(),
                            covs.detach().cpu().numpy())
        print("pinn-gmm params")
        print("weights")
        print(np.array2string(ws, formatter={'float_kind': lambda x: f"{float(x):.2f}"}))

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
            _tv = p_total_variation(pdf_pinn, pdf_ref, vol_est)
            tv_pinn += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_net=p_net)
            gkl_pinn += _gkl/N_batch
            _Z_pinn = np.mean(pdf_pinn) * vol_est
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
            _tv = p_total_variation(pdf_pinn, pdf_ref, vol_est)
            tv_pinngmm += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_net=p_net_gmm)
            gkl_pinngmm += _gkl/N_batch
            del pdf_pinn
            if(e1_net_gmm is not None):
                e1_pinngmm = constants.SCALING_PDF * e1_net_gmm(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
                e1_pinngmm_max = max(e1_pinngmm_max, np.max(np.abs(e1_pinngmm)).item())
                del e1_pinngmm

            # pinn-gmm_vanilla
            pdf_pinn = constants.SCALING_PDF * p_net_gmm_vanilla(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            _delta_p = np.max(np.abs(pdf_pinn - pdf_ref)).item()
            delta_p_pinngmm_vanilla_max = max(delta_p_pinngmm_vanilla_max, _delta_p)
            _tv = p_total_variation(pdf_pinn, pdf_ref, vol_est)
            tv_pinngmm_vanilla += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_net=p_net_gmm_vanilla)
            gkl_pinngmm_vanilla += _gkl/N_batch
            del pdf_pinn, _t, _t_tensor, X_scaled, _x_tensor  
            
            # LP
            _, _, _mu_lp, _cov_lp = data_lp.get(t)
            _mu_lp = np.float32(_mu_lp)
            _cov_lp = np.float32(_cov_lp)
            _pdf_lp_func = make_gmm_pdf(1., _mu_lp, _cov_lp)
            pdf_lp = _pdf_lp_func(X).reshape(-1,)
            _delta_p = np.max(np.abs(pdf_lp - pdf_ref)).item()
            delta_p_lp_max = max(delta_p_lp_max, _delta_p)
            _tv = p_total_variation(pdf_lp, pdf_ref, vol_est)
            tv_lp += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_data_normal=data_lp)
            gkl_lp += _gkl/N_batch
            del pdf_lp, _pdf_lp_func

            # UT
            _, _, _mu_ut, _cov_ut = data_ut.get(t)
            _mu_ut = np.float32(_mu_ut)
            _cov_ut = np.float32(_cov_ut)
            _pdf_ut_func = make_gmm_pdf(1., _mu_ut, _cov_ut)
            pdf_ut = _pdf_ut_func(X).reshape(-1,)
            _delta_p = np.max(np.abs(pdf_ut - pdf_ref)).item()
            delta_p_ut_max = max(delta_p_ut_max, _delta_p)
            _tv = p_total_variation(pdf_ut, pdf_ref, vol_est)
            tv_ut += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_data_normal=data_ut)
            gkl_ut += _gkl/N_batch
            del pdf_ut, _pdf_ut_func

            # GMM
            _, _w, _mu, _cov = data_gmm.get(t)
            _pdf_gmm_func = make_gmm_pdf(_w, _mu, _cov)
            pdf_gmm = _pdf_gmm_func(X).reshape(-1,)
            _delta_p = np.max(np.abs(pdf_gmm - pdf_ref)).item()
            delta_p_gmm_max = max(delta_p_gmm_max, _delta_p)
            _tv = p_total_variation(pdf_gmm, pdf_ref, vol_est)
            tv_gmm += _tv/N_batch
            _gkl = compute_generalKL(t, X_mc, p_data_normal=data_gmm)
            gkl_gmm += _gkl/N_batch
            del pdf_gmm
        
        # After all batch
        gkl_pinn += Z_pinn
        gkl_pinngmm += 1; gkl_pinngmm_vanilla += 1
        gkl_lp += 1
        gkl_ut += 1
        gkl_gmm += 1
        metrics["rel_error_pinn"].append(100.*delta_p_pinn_max/pdf_ref_max)
        metrics["tv_pinn"].append(tv_pinn)
        metrics["g_kl_pinn"].append(gkl_pinn)

        metrics["rel_error_pinngmm"].append(100.*delta_p_pinngmm_max/pdf_ref_max)
        metrics["tv_pinngmm"].append(tv_pinngmm)
        metrics["g_kl_pinngmm"].append(gkl_pinngmm)

        metrics["rel_error_pinngmm_vanilla"].append(100.*delta_p_pinngmm_vanilla_max/pdf_ref_max)
        metrics["tv_pinngmm_vanilla"].append(tv_pinngmm_vanilla)
        metrics["g_kl_pinngmm_vanilla"].append(gkl_pinngmm_vanilla)

        metrics["rel_error_lp"].append(100.*delta_p_lp_max/pdf_ref_max)
        metrics["tv_lp"].append(tv_lp)
        metrics["g_kl_lp"].append(gkl_lp)

        metrics["rel_error_ut"].append(100.*delta_p_ut_max/pdf_ref_max)
        metrics["tv_ut"].append(tv_ut)
        metrics["g_kl_ut"].append(gkl_ut)

        metrics["rel_error_gmm"].append(100.*delta_p_gmm_max/pdf_ref_max)
        metrics["tv_gmm"].append(tv_gmm)
        metrics["g_kl_gmm"].append(gkl_gmm)

        metrics["B1_pinn"].append(100. * 2. * e1_pinn_max/pdf_ref_max)
        metrics["B1_pinngmm"].append(100. * 2. * e1_pinngmm_max/pdf_ref_max)

        B1_pinngmm_raw = 2. * e1_pinngmm_max / constants.SCALING_PDF
        metrics["B1_pinngmm_raw"].append(B1_pinngmm_raw)
        print("[debug] B1_pinngmm_raw: ", B1_pinngmm_raw)
        print("[debug]: ", metrics["rel_error_pinngmm_vanilla"])
        
        print_metrics_block(metrics, idx)

    # save computed metrics
    save_metrics_npz(metrics, save_path)


def compare_corner_plots(OUTPUT_PATH=None, data_lp=None, data_ut=None, data_gmm=None,
                         p_net_gmm=None, rar_samples=None, save_path=None):
    global constants
    t_show = [constants.T_PRIME_SPAN[0],
              constants.T_PRIME_SPAN[3],
              constants.T_PRIME_SPAN[-1]]
    # t_show = np.array([0.4])

    N_samples = 1000000
    # X = get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
    # X_scaled = constants.scaled_x(X)
    # _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

    for idx, t in enumerate(t_show):
        print("\ntime {:.4f}".format(t))

        X_ref = sample_joint_pdf(constants, t, N_samples)

        # x_coords = (1, 6)
        # data_marginal_pinn = None
        # if(OUTPUT_PATH is not None):
        #     data_marginal_pinn = np.load(
        #         f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz")

        plot_full_corner(constants, t, X_ref, OUTPUT_PATH=OUTPUT_PATH, 
                         p_net_gmm=p_net_gmm,
                         save_plot=SAVEPLOT)

        # Single element plot 1D (x6,)
        # plot_corner_elem(constants, t, X_ref, (6,),
        #                  data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
        #                  PNet_XL_PATH=OUTPUT_PATH, p_net_gmm=p_net_gmm)

        # Single element plot 2D (x1, x6)
        # plot_corner_elem(constants, t, X_ref, (1, 6),
        #                  data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
        #                  PNet_XL_PATH=OUTPUT_PATH, p_net_gmm=p_net_gmm, rar_samples=rar_samples,
        #                  ranges="auto",
        #                  save_plot=SAVEPLOT)

        plt.show()


def config_trained_models():
    trained_models = {
        "PINN-MLP_vanilla": "output/pinn-xl_vanilla",
        # "PINN-MLP": "output/pinn-xl",
        # "PINN-MLP_bias": "output/pinn-xl_bias",
        # "PINN-MLP_bias_reg": "output/pinn-xl_bias_reg",
        "PINN-GMM_vanilla" : "output/pinn-gmm_vanilla",
        # "PINN-GMM" : "output/pinn-gmm",
        "PINN-GMM_bias" : "output/pinn-gmm_bias",
    }
    return trained_models


def load_model_by_key(trained_models, key=""):
    if key not in trained_models:
        raise KeyError(f"Missing key {key!r}. Available: {list(trained_models.keys())}")

    # Scale of p0 max
    _x_at_mean = constants.N_MEAN_I.copy()
    scale = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32)

    KEY_PATH = trained_models[key]

    if(key == "PINN-MLP_vanilla"):
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        rar_samples = None
        return p_net, e1_net, rar_samples

    if(key == "PINN-MLP"):
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        rar_samples = None
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-MLP_bias"):
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        rar_samples = None
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-MLP_bias_reg"):
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        rar_samples = None
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-GMM_vanilla"):
        p_net = TimeToGMM_V0(constants, K=5, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = None
        # e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        # e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        _p = Path(KEY_PATH) / "p_net-RARsamples.npz"
        if _p.is_file(): 
            rar_samples = np.load(_p)
        else:
            rar_samples = None
        del _p
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-GMM"):
        p_net = TimeToGMM_V0(constants, K=5, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        e1_net = None
        # e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        # e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        _p = Path(KEY_PATH) / "p_net-RARsamples.npz"
        if _p.is_file(): 
            rar_samples = np.load(_p)
        else:
            rar_samples = None
        del _p
        return p_net, e1_net, rar_samples
    
    if(key == "PINN-GMM_bias"):
        p_net = TimeToGMM_V0(constants, K=5, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=KEY_PATH+"/p_net.pth"); p_net.eval()
        # e1_net = None
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=scale_torch*0.02)
        e1_net = load_trained_model(e1_net, path=KEY_PATH+"/e1_net.pth"); e1_net.eval()
        # rar_samples = None
        _p = Path(KEY_PATH) / "p_net-RARsamples.npz"
        if _p.is_file(): 
            rar_samples = np.load(_p)
        else:
            rar_samples = None
        del _p
        return p_net, e1_net, rar_samples


def main():
    global constants
    print(SAVEPLOT)

    # setup trained models dictionary
    trained_models = config_trained_models()

    # load pinn-mlp
    p_net, e1_net, _ = load_model_by_key(trained_models, key="PINN-MLP_vanilla")

    # load pinn-gmm
    p_net_gmm_vanilla, _, _ = load_model_by_key(trained_models, key="PINN-GMM_vanilla")
    p_net_gmm, e1_net_gmm, rar_samples = load_model_by_key(trained_models, key="PINN-GMM_bias")

    # load baseline methods
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)

    metrics_path = "output/metric_NB="+str(NBATCH)+".npz"
    if(COMPUTE):
        compute_pdf_variations(N_batch=NBATCH, p_net=p_net, p_net_gmm=p_net_gmm, 
            p_net_gmm_vanilla=p_net_gmm_vanilla,
            data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
            e1_net=e1_net, e1_net_gmm=e1_net_gmm, save_path=metrics_path)
    metrics = load_metrics_npz(metrics_path); plot_pdf_metrics(metrics, save_plot=SAVEPLOT)

    # Visualize marginal PDF
    # [temp] don't view the adaptive pools
    rar_samples = None
    compare_corner_plots(
        OUTPUT_PATH=None,
        data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm,
        p_net_gmm=p_net_gmm, rar_samples=rar_samples,
    )
    
    # --- plot training history ---
    # plot_training_history(trained_models)
        

if __name__ == "__main__":
    args = parse_args()
    COMPUTE   = bool(args.compute)
    NBATCH    = int(args.Nbatch)
    SAVEPLOT = bool(args.saveplot)
    main()