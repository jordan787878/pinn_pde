import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sympy as sp
from tqdm import tqdm
import os
from scipy.stats import multivariate_normal
from scipy.linalg import expm, cholesky
import seaborn as sns
import time
from pinn_train import train_pnet, train_pnet_v0
from pinn_model import PNet, PNet_XL
from pinn_train import train_e1net_v0
from pinn_model import E1Net, E1Net_Prior
from test_gmmpnet import TimeToGMM1D
from test_flow import PDF_Flow, PDF_Flow_CNF
from types import SimpleNamespace
import sys
sys.path.insert(0, '../utilities/')
from _General.baseline_methods import (PropagationData, 
    linear_propagation_master, unscent_propagation_master, gmm_propagation_master)
from _General.util import set_publication_plot_style, custom_save_plot
from _General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, make_gmm_pdf

# --------------------------
# Device & dtype
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

const_mu = -2.0
const_std = 0.5
const_a = -0.1
const_b = 0.1
const_c = 0.5
const_d = 0.5
const_e = 0.8
x_low = -6
x_hig = 6
t0 = 0.0
T_end = 5.0
t1s = np.arange(0.0, 5.0 + 0.5, 0.5)
# t1s = np.arange(0.0, 5.0 + 1.0, 1.0)

# --- helper ---
const_mu_tensor = torch.tensor(const_mu, dtype=torch.float32)
const_std_tensor = torch.tensor(const_std, dtype=torch.float32)
const_a_tensor = torch.tensor(const_a, dtype=torch.float32)
const_b_tensor = torch.tensor(const_b, dtype=torch.float32)
const_c_tensor = torch.tensor(const_c, dtype=torch.float32)
const_d_tensor = torch.tensor(const_d, dtype=torch.float32)
const_e_tensor = torch.tensor(const_e, dtype=torch.float32)

colors_4set = sns.color_palette([
    "#FF008C",  # GA
    "#00FBFF",  # UT
    "#FF8400",  # GMM
    "#000000",  # PINN-GMM
])

# 7 high-contrast linestyle/marker pairs (index-bound)
linestyles_4set = [
    (0, (5, 2)),        # GA
    (0, (7, 2, 3, 2)),  # UT
    "--",               # GMM
    "-",                # PINN-GMM
]
markers_4set = ['o', # GA
                's', # UT
                '^', # GMM
                'None', # PINN-GMM
                ] 


def p_init(x):
    """
    taks numpy.array or torch.tensor as inputs
    """
    if isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        exponent = -0.5 * ((x - const_mu_tensor) / const_std_tensor) ** 2
        normalization = const_std_tensor * torch.sqrt(torch.tensor(2 * torch.pi))
        return torch.exp(exponent) / normalization
    elif isinstance(x, np.ndarray):
        return np.exp(-0.5*((x-const_mu)/const_std)**2) / (const_std*np.sqrt(2*np.pi))
    else:
        return "Neither PyTorch Tensor nor NumPy Array"


def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def get_e1_normalize(pnet):
    x = np.linspace(x_low, x_hig, num=200, endpoint=True).reshape(-1,1)
    p0_true = p_init(x)
    pt_x = torch.from_numpy(x).float()
    pt_t = pt_x*0.0 + t0
    p0_hat = pnet(pt_x, pt_t).data.cpu().numpy()
    e0_true = p0_true - p0_hat
    return np.max(np.abs(e0_true))


def res_func(x, t, pnet, verbose=False, beta=1.):
    p = pnet(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + beta*( (3*const_a_tensor *x*x + 2*const_b_tensor *x + const_c_tensor)*p \
                   + (const_a_tensor *x*x*x + const_b_tensor *x*x + const_c_tensor*x + const_d_tensor)*p_x \
                   - 0.5*const_e_tensor*const_e_tensor *p_xx )
    if(verbose):
        print(p_xx[0:10,:]) #; print(p_x.shape, p_t.shape, p_xx.shape, residual.shape)
    return residual
    

def load_train_model(net, PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # print("load pinn from: ", PATH)
    # print("best epoch: ", epoch, ", loss:", loss, "train time:", checkpoint['train_time'])
    net.eval()
    return net


def train_helper_sample_ic(batch_size):
    x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
    t_bc = (torch.ones(batch_size, 1) * t0).to(device)
    return x_bc, t_bc


def train_helper_sample_res(batch_size):
    x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
    return x, t

def train_helper_sample_ic_after_t(batch_size, t):
    x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
    t_bc = (torch.ones(batch_size, 1) * t).to(device)
    return x_bc, t_bc

def train_helper_sample_res_after_t(batch_size, t, t2=1):
    x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t = (torch.rand(batch_size, 1, requires_grad=True) * (t+t2 - t) + t).to(device)
    return x, t


def _get_Jacobian_expression():
    x1 = sp.symbols('x1', real=True)
    A, B, C, D = sp.symbols('A B C D' , real=True, positive=True)
    f1 = A*x1**3 + B*x1**2 + C*x1 + D
    f = sp.Matrix([f1])
    x = sp.Matrix([x1])
    # symbolic Jacobian
    J = sp.simplify(f.jacobian(x))
    print("[info] Jacobian matrix: ", J)


def get_Jacobian(c, x):
    x1 = x
    A, B, C, D = c.a, c.b, c.c, c.d
    J = 3*A*x1**2 + 2*B*x1 + C
    return J.reshape((1,1))
    

def nl_dyn(c, x):
    x1 = x
    A, B, C, D = c.a, c.b, c.c, c.d
    f1 = A*x1**3 + B*x1**2 + C*x1 + D
    return f1


def sample_p_init(N=1000, rng=None):
    x_dim = 1
    if rng is None:
        rng = np.random.default_rng(0)
    _mu  = np.asarray(const_mu, dtype=np.float64)
    _std = np.asarray(const_std, dtype=np.float64)
    Z = rng.normal(loc=_mu, scale=_std, size=(N, x_dim))
    return Z
        

def empirical_moments(x, p, N=1):
    # p = p/np.sum(p)
    dx = x[1]-x[0]
    mean = np.sum(p*x)*dx
    if(N == 1):
        return mean
    else:
        return np.sum(p*(x-mean)**2)*dx


def p_total_variation(p1, p2):
    vol_est = x_hig - x_low
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    tv = 0.5 * np.mean(np.abs(p1-p2)) * vol_est
    return 100.*np.clip(tv, 0., 1.)


def p_rel_worst_error(p1, p2):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    # print(np.max(p1), np.max(p2))
    norm_error = np.max(np.abs(p1-p2)) / np.max(p2)
    return 100.*norm_error.item()


def get_p_at_x_samples(x_mc_samples, pdf_func=None, pdf_pinn=None, t=None):
    if(pdf_func is not None):
        if hasattr(pdf_func, "pdf"):
            return pdf_func.pdf(x_mc_samples).reshape(-1,)
        else:
            return pdf_func(x_mc_samples).reshape(-1,)
    if(pdf_pinn is not None):
        x_mc_samples_tensor = torch.from_numpy(x_mc_samples.reshape(-1,1)).to(device)
        _t_mc_samples = np.ones(x_mc_samples_tensor.shape[0], dtype=x_mc_samples.dtype)*t
        t_mc_samples_tensor = torch.from_numpy(_t_mc_samples.reshape(-1,1)).to(device)
        return pdf_pinn(x_mc_samples_tensor, t_mc_samples_tensor).detach().cpu().numpy().reshape(-1,)


def p_g_kl(p_at_x, Z_p):
    return np.mean(-np.log(p_at_x)) + Z_p


def compute_metrics(t, x_mc_samples, pdf_mc, pdf_eval, pdf_func=None, pdf_pinn=None):
    norm_error = p_rel_worst_error(pdf_eval, pdf_mc)
    tv = p_total_variation(pdf_eval, pdf_mc)

    g_kl = 0.
    if(pdf_func is not None):
        Z_p = 1.
        g_kl = p_g_kl(get_p_at_x_samples(x_mc_samples, pdf_func=pdf_func), Z_p)

    if(pdf_pinn is not None):
        vol_est = x_hig - x_low
        Z_p = np.mean(pdf_eval) * vol_est
        print("[info] Z_p PINN: {:.5f}".format(Z_p))
        g_kl = p_g_kl(get_p_at_x_samples(x_mc_samples, pdf_pinn=pdf_pinn, t=t), Z_p)

    return norm_error, tv, g_kl


def test_MC_accuracy(x):
    pdf_true = p_init(x)
    pdf_mc = np.load("data/psim_t0.0.npy")
    norm_error = p_rel_worst_error(pdf_mc, pdf_true)
    tv = p_total_variation(pdf_mc, pdf_true)
    print("[test] Validate MC at t0 --- general error {:.5f} %, tv {:.5f} %".format(
        norm_error, tv))


def helper_save_metrics_npz(metrics, path):
    # convert lists to float arrays; keep 't' as-is if already np.ndarray
    out = {}
    for k, v in metrics.items():
        if k == "t":
            out[k] = np.asarray(v)
        else:
            out[k] = np.asarray(v, dtype=float)
    # sanity: all series (except t) should match len(t)
    n = len(out["t"])
    for k, v in out.items():
        if k != "t":
            assert len(v) == n, f"Length mismatch for {k}: {len(v)} vs t={n}"
    np.savez(path, **out)


def main(METHOD, TRAIN_FLAG=False, RUN_BASELINE=False):
    """
    LOAD_PRIOR (True): the best from prior work
    Base: the baseline MLP model
    Normal: PINN-Normal model
    """
    torch.manual_seed(0); np.random.seed(0)

    if(METHOD not in ["prior", "pinn", "pinn-gmm", "flow"]):
        print(METHOD, " not yet implemented")
        return

    if(METHOD == "prior"):
        # --- Base (prior)---
        p_net = PNet(scale=get_p_normalize()).to(device)
        TRAIN_FLAG = False

    if(METHOD == "pinn"):
        # --- Base ---
        p_net = PNet(scale=get_p_normalize()).to(device)

    if(METHOD == "pinn-gmm"):
        # --- PINN-GMM ---
        p_net = TimeToGMM1D().to(device)

    if(METHOD == "flow"):
        # --- FLOW ---
        p_net = PDF_Flow(scale=get_p_normalize()).to(device)
        # p_net = PDF_Flow_CNF(scale=get_p_normalize()).to(device)

    configuration = {
        "iterations": 10000,
        "sample_ic": train_helper_sample_ic,
        "sample_res": train_helper_sample_res,
        "p_ic": p_init,
        "res_func": res_func,
        "res_weight": 1.0,
    }
    if(METHOD == "pinn"):
        configuration["save_path"] = "data/p_net.pth"
    if(METHOD == "pinn-gmm"):
        configuration["save_path"] = "data/p_net(pinn-gmm).pth"
        configuration["iterations"] = 4000
    if(METHOD == "flow"):
        configuration["save_path"] = "data/p_net(flow).pth"

    if(TRAIN_FLAG):
        print("train pnet: ", configuration["save_path"])
        # train_pnet_model(p_net)
        train_pnet_v0(p_net, configuration)

    if(METHOD == "prior"):
        p_net = load_train_model(p_net, PATH="data/p_net(prior).pth")
    if(METHOD == "pinn" or METHOD == "pinn-gmm" or METHOD == "flow"):
        p_net = load_train_model(p_net, PATH=configuration["save_path"])

    # --- Error Neural Network ---
    if(METHOD == "prior"):
        e1_net = E1Net_Prior(scale=get_e1_normalize(p_net)).to(device)
    if(METHOD == "pinn"):
        e1_net = E1Net(scale=get_e1_normalize(p_net),
                   normalize=get_e1_normalize(p_net),
                   ).to(device)
    if(METHOD == "pinn-gmm"):
        e1_net = E1Net(scale=get_e1_normalize(p_net),
                   normalize=get_e1_normalize(p_net),
                   ).to(device)
    if(METHOD == "flow"):
        e1_net = E1Net(scale=get_e1_normalize(p_net),
                   normalize=get_e1_normalize(p_net),
                   ).to(device)
    print("[debug] e1_net scale: ", e1_net.scale)
        
    configuration_e1 = {
        "iterations": 20000,
        "sample_ic": train_helper_sample_ic,
        "sample_res": train_helper_sample_res,
        "p_ic": p_init,
        "res_func": res_func,
        "res_weight": 1.0,
        "RAR_eps": 0.05,
    }
    if(METHOD == "pinn"):
        configuration_e1["save_path"] = "data/e1_net.pth"
    if(METHOD == "pinn-gmm"):
        configuration_e1["save_path"] = "data/e1_net(pinn-gmm).pth"
        configuration_e1["iterations"] = 30000
    if(METHOD == "flow"):
        configuration_e1["save_path"] = "data/e1_net(flow).pth"

    if(TRAIN_FLAG):
        train_e1net_v0((p_net, e1_net), configuration_e1)

    if(METHOD == "prior"):
        e1_net = load_train_model(e1_net, PATH="data/e1_net(prior).pth")
    if(METHOD == "pinn" or METHOD == "pinn-gmm" or METHOD == "flow"):
        e1_net = load_train_model(e1_net, PATH=configuration_e1["save_path"])

    # --- Evaluation ---
    p_net.eval(); e1_net.eval()

    # --- Visual ---
    x = np.load("data/xsim.npy").astype(np.float32)
    # x = np.load("data/midpts_512.npy").astype(np.float32).reshape(-1,)

    # test_MC_accuracy(x)

    lp_path = os.path.join("data", "lp_np64_dt6.npz") # the last label specifies the precision used
    ut_path = os.path.join("data", "ut_np64_dt6.npz") # the last label specifies the precision used
    ut_path_alpha0_1 = os.path.join("data", "ut_np64_dt6_alpha0.1.npz") # the last label specifies the precision used
    gmm_path = os.path.join("data", "gmm_np64_dt6.npz")
    if(RUN_BASELINE):
        print("run baseline methods")
        x0, P0 = const_mu, const_std**2

        dyn_fcn = lambda t, x, c: nl_dyn(c, x)
        jac_fcn = lambda t, x, c: get_Jacobian(c, x)
        Q = np.asarray(const_e**2, dtype=np.float64).reshape((1,1))
        t_span = (np.float64(np.round(t0, 2)), np.float64(np.round(T_end, 2)))
        constants = SimpleNamespace(
            a=np.float64(const_a),
            b=np.float64(const_b),
            c=np.float64(const_c),
            d=np.float64(const_d),
        )

        linear_propagation_master(
            x0=x0, P0=P0,
            dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
            constants=constants,
            t_span=t_span,
            dt_save=0.01,
            Q=Q,
            save_path=lp_path
        )

        unscent_propagation_master(
            x0=x0, P0=P0,
            dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
            constants=constants,
            t_span=t_span,
            dt_save=0.01,
            Q=None,
            G=np.array([[const_e]]),
            save_path=ut_path,
            alpha=1e-3,
        )

        unscent_propagation_master(
            x0=x0, P0=P0,
            dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
            constants=constants,
            t_span=t_span,
            dt_save=0.01,
            Q=None,
            G=np.array([[const_e]]),
            save_path=ut_path_alpha0_1,
            alpha=1.0,
        )

        X_tr = sample_p_init(N=1000000)
        t_prime = 0.0
        fit_classic_gmm(t_prime, X_tr, np.asarray(x0, dtype=np.float64).reshape(-1,), 
                        np.asarray(P0, dtype=np.float64).reshape((1,1)), "data/",
                        K_list=(10, 20, 30, 40))
        model = GMMWhitenedModel.load("data/"+"gmm_whitened_t{:.2f}.npz".format(t_prime))
        gmm_params = model.print_x_params()
        gmm_propagation_master(
            weights = gmm_params["weights"],
            means0 = gmm_params["means_x"],
            covs0 = gmm_params["covs_x"],
            dyn_fcn=dyn_fcn, jac_fcn=jac_fcn,
            constants=constants,
            t_span=t_span,
            dt_save=0.01,
            Q=Q,
            save_path=gmm_path
        )
        
    data_lp = PropagationData(path=lp_path)
    data_ut = PropagationData(path=ut_path)
    data_ut_alpha0_1 = PropagationData(path=ut_path_alpha0_1)
    data_gmm = PropagationData(path=gmm_path)

    set_publication_plot_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = sns.color_palette("husl", 4)

    x_tensor = torch.from_numpy(x.reshape(-1,1)).to(device)
    t_span = np.array(t1s).astype(np.float32)
    metrics = {
        "norm_error_pinn": [],
        "norm_error_lp": [],
        "norm_error_ut": [],
        "norm_error_ut_alpha_0_1": [],
        "norm_error_gmm": [],
        "tv_pinn": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_ut_alpha_0_1": [],
        "tv_gmm": [],
        "g_kl_pinn": [],
        "g_kl_lp": [],
        "g_kl_ut": [],
        "g_kl_ut_alpha_0_1": [],
        "g_kl_gmm": [],
        "normalize_B1": [],
        "B1": [],
        "t": t_span
    }
    x_mc_samples = None
    B1 = None
    for t in t_span:
        x_mc_samples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32)
        x_mc_samples = x_mc_samples[0:1000000]
    
        pdf_mc = np.load("data/psim_t{:.1f}.npy".format(t)).astype(np.float32).reshape(-1,)
        # pdf_mc = np.load("data/p_512_t{:.1f}.npy".format(t)).astype(np.float32).reshape(-1,)
        if(t == 0):
            pdf_mc = p_init(x).reshape(-1,)

        _t = np.ones(x.shape[0], dtype=x.dtype)*t
        t_tensor = torch.from_numpy(_t.reshape(-1,1)).to(device)

        # --- LP ---
        _, _, _mu_lp, _cov_lp = data_lp.get(t)
        pdf_func = multivariate_normal(mean=_mu_lp, cov=_cov_lp)
        pdf_lp = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
        norm_error_lp, tv_lp, g_kl_lp = compute_metrics(t, x_mc_samples, pdf_mc, pdf_lp, pdf_func=pdf_func)
        # # compute coverage mass
        # idx_in_one_std = data_lp.indices_in_range(x, np.array([_mu_lp-(_cov_lp)**0.5, _mu_lp+(_cov_lp)**0.5]))
        # if(len(idx_in_one_std) > 0):
        #     x_in_one_std = x[idx_in_one_std]
        #     pdf_mass_lp = 100. *np.mean(pdf_mc[idx_in_one_std]) * (x_in_one_std[-1] - x_in_one_std[0])
        # else:
        #     pdf_mass_lp = 0.0

        # --- UT ---
        _, _, _mu_ut, _cov_ut = data_ut.get(t)
        pdf_func = multivariate_normal(mean=_mu_ut, cov=_cov_ut)
        pdf_ut = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
        norm_error_ut, tv_ut, g_kl_ut = compute_metrics(t, x_mc_samples, pdf_mc, pdf_ut, pdf_func=pdf_func)
        # # compute coverage mass
        # idx_in_one_std = data_ut.indices_in_range(x, np.array([_mu_ut-(_cov_ut)**0.5, _mu_ut+(_cov_ut)**0.5]))
        # if(len(idx_in_one_std) > 0):
        #     x_in_one_std = x[idx_in_one_std]
        #     pdf_mass_ut = 100. * np.mean(pdf_mc[idx_in_one_std]) * (x_in_one_std[-1] - x_in_one_std[0])
        # else:
        #     pdf_mass_ut = 0.0

        # --- UT (alpha=0.1) ---
        _, _, _mu_ut_alpha0_1, _cov_ut_alpha0_1 = data_ut_alpha0_1.get(t)
        pdf_func = multivariate_normal(mean=_mu_ut_alpha0_1, cov=_cov_ut_alpha0_1)
        pdf_ut_alpha0_1 = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
        norm_error_ut_alpha_0_1, tv_ut_alpha_0_1, g_kl_ut_alpha_0_1 = compute_metrics(t, x_mc_samples, pdf_mc, pdf_ut_alpha0_1, pdf_func=pdf_func)

        # --- GMM ---
        _, _w_gmm, _mu_gmm, _cov_gmm = data_gmm.get(t)
        print("[debug] gmm params shape: ", _w_gmm.shape, _mu_gmm.shape, _cov_gmm.shape)
        # print("[debug] gmm weights sum: ", np.sum(_w_gmm))
        pdf_func_gmm = make_gmm_pdf(_w_gmm, _mu_gmm, _cov_gmm)
        pdf_gmm = pdf_func_gmm(x).reshape(-1,).astype(x.dtype)
        norm_error_gmm, tv_gmm, g_kl_gmm = compute_metrics(t, x_mc_samples, pdf_mc, pdf_gmm, pdf_func=pdf_func_gmm)

        # --- PINN ---
        pdf_pinn_grid = p_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_mc.shape)
        e1_true = pdf_mc - pdf_pinn_grid
        norm_error_pinn, tv_pinn, g_kl_pinn = compute_metrics(t, x_mc_samples, pdf_mc, pdf_eval=pdf_pinn_grid, pdf_pinn=p_net)

        # --- PINN Error Bound ---
        e1_pinn = e1_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(-1,)
        B1 = 2.* np.max(np.abs(e1_pinn)).item()
        p_true_max = np.max(pdf_mc).item()
        normalize_B1 = 100.*B1/p_true_max
        alpha1 = np.max(np.abs(e1_true - e1_pinn)).item() / np.max(np.abs(e1_pinn)).item()

        print("time {:.2f} total variation , PINN: {:.5f} %,  LP: {:.5f} %,  UT: {:.5f} %,  GMM: {:.5f} %".format(
            t, tv_pinn, tv_lp, tv_ut, tv_gmm
        ))
        print("time {:.2f} worst general error, PINN: {:.5f} %,  LP: {:.5f} %,  UT: {:.5f} %,  GMM: {:.5f} %".format(
            t, norm_error_pinn, norm_error_lp, norm_error_ut, norm_error_gmm
        ))
        if(x_mc_samples is not None):
            print("time {:.2f} general KL, PINN: {:.5f},  LP: {:.5f},  GMM: {:.5f}".format(
                t, g_kl_pinn, g_kl_lp, g_kl_gmm
            ))
        if(B1 is not None):
            print("[debug] max error v.s. error est", np.max(np.abs(e1_true)).item(), np.max(np.abs(e1_pinn)).item())

        metrics["norm_error_pinn"].append(norm_error_pinn)
        metrics["norm_error_lp"].append(norm_error_lp)
        metrics["norm_error_ut"].append(norm_error_ut)
        metrics["norm_error_ut_alpha_0_1"].append(norm_error_ut_alpha_0_1)
        metrics["norm_error_gmm"].append(norm_error_gmm)

        metrics["tv_pinn"].append(tv_pinn)
        metrics["tv_lp"].append(tv_lp)
        metrics["tv_ut"].append(tv_ut)
        metrics["tv_ut_alpha_0_1"].append(tv_ut_alpha_0_1)
        metrics["tv_gmm"].append(tv_gmm)

        metrics["g_kl_pinn"].append(g_kl_pinn)
        metrics["g_kl_lp"].append(g_kl_lp)
        metrics["g_kl_ut"].append(g_kl_ut)
        metrics["g_kl_ut_alpha_0_1"].append(g_kl_ut_alpha_0_1)
        metrics["g_kl_gmm"].append(g_kl_gmm)

        metrics["B1"].append(B1)
        metrics["normalize_B1"].append(normalize_B1)

        mean_mc = empirical_moments(x, pdf_mc)
        cov_mc = empirical_moments(x, pdf_mc, N=2)
        print("Mean, MC: {:.3f}, LP: {:.3f}, UT: {:.3f}".format(
            mean_mc, _mu_lp[0], _mu_ut[0]
        ))
        print("Cov , MC: {:.3f}, LP: {:.3f}, UT: {:.3f}".format(
            cov_mc, _cov_lp[0,0], _cov_ut[0,0]
        ))
        # print("probability mass within +/- 1 std, LP: {:.3f} %, UT: {:.3f} %".format(
        #     pdf_mass_lp, pdf_mass_ut))

        # Skip times that aren't approximately integer seconds
        if not np.isclose(t, np.round(t), atol=1e-6):
            continue
        gap = 3
        ax.plot(np.full_like(x[::gap], t), x[::gap], pdf_mc[::gap],
                color="#0066FF", linestyle="-", marker="o")
        ax.plot(np.full_like(x, t), x, pdf_lp,
                color=colors_4set[0], linestyle=linestyles_4set[0])
        ax.plot(np.full_like(x, t), x, pdf_ut,
                color=colors_4set[1], linestyle=linestyles_4set[1])
        ax.plot(np.full_like(x, t), x, pdf_gmm,
                color=colors_4set[2], linestyle=linestyles_4set[2])
        ax.plot(np.full_like(x, t), x, pdf_pinn_grid,
                color=colors_4set[3], linestyle=linestyles_4set[3])

    legend_elements = [
        Line2D([0], [0], color="#0066FF", linestyle="-", marker="o", label=r"$p$"),
        Line2D([0], [0], color=colors_4set[0], linestyle=linestyles_4set[0], 
               label="GA"),
        Line2D([0], [0], color=colors_4set[1], linestyle=linestyles_4set[1], 
               label="UT"),
        Line2D([0], [0], color=colors_4set[2], linestyle=linestyles_4set[2], 
               label="GMM"),
        Line2D([0], [0], color=colors_4set[3], linestyle=linestyles_4set[3], 
               label="PINN-GMM"),
    ]
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.text2D(0.99, 0.8, "PDF", transform=ax.transAxes,
          ha="center", va="bottom")
    ax.legend(handles=legend_elements, 
        loc="upper left",             # corner inside the axes
        bbox_to_anchor=(0.4, 0.85),  # (x, y) in axes fraction coords
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )
    custom_save_plot(True, "figs/pdfs.pdf")

    # metric 1: worst relative error %
    set_publication_plot_style()
    plt.figure()
    plt.plot(metrics["t"], metrics["norm_error_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.fill_between(
        metrics["t"],
        metrics["norm_error_pinn"],
        metrics["normalize_B1"],
        color=colors[0],
        alpha=0.2,
        label="PINN Error Bound"
    )
    plt.plot(metrics["t"], metrics["norm_error_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["norm_error_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["norm_error_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=1.0)$")
    plt.plot(metrics["t"], metrics["norm_error_gmm"], color=colors[3],   label=r"$p$ GMM Linear Prop.")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("norm. worst error %")

    # metric 2: total variation %
    set_publication_plot_style()
    plt.figure()
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["tv_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=1.0)$")
    plt.plot(metrics["t"], metrics["tv_gmm"], color=colors[3],   label=r"$p$ GMM Linear Prop.")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("total variation %")

    # metric 3: negative log liklihood (relative KL)
    set_publication_plot_style()
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["g_kl_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["g_kl_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["g_kl_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=1.0)$")
    plt.plot(metrics["t"], metrics["g_kl_gmm"], color=colors[3],   label=r"$p$ GMM Linear Prop.")
    plt.grid(True)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("general KL")

    plt.show()

    # save metrics
    metrics_path = "data/metrics.npz"
    if(METHOD == "prior"):
        metrics_path = "data/metrics(prior).npz"
    if(configuration["save_path"] == "data/p_net(pinn-gmm).pth"):
        metrics_path = "data/metrics(pinn-gmm).npz"
    if(configuration["save_path"] == "data/p_net(flow).pth"):
        metrics_path = "data/metrics(flow).npz"
    helper_save_metrics_npz(metrics, metrics_path)


if __name__ == "__main__":
    main(METHOD = "pinn-gmm",
         TRAIN_FLAG=False, 
         RUN_BASELINE=False)