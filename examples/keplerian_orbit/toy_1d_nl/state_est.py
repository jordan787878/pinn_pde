import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
import seaborn as sns
from functools import partial
import copy
from test_gmmpnet import TimeToGMM1D
from pinn_model import E1Net
from main import (load_train_model, get_e1_normalize, t1s, PropagationData,
    train_helper_sample_ic_after_t, train_helper_sample_res_after_t, 
    train_helper_sample_ic, train_helper_sample_res, p_init, res_func,
    x_low, x_hig)
from pinn_train import train_pnet_v0
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.util import set_publication_plot_style, custom_save_plot
from utilities._General.classic_gmm import make_gmm_pdf


device = torch.device("cpu")


def load_samples_as_trajectories(t_span):
    """
    return: x_traj_true of shape (N_samples, N_time_points)
    """
    x_traj_true = []
    for t in t_span:
        x_mc_samples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32)
        x_traj_true.append(x_mc_samples)
    x_traj_true = np.array(x_traj_true).T
    # print(x_traj_true.shape)
    return x_traj_true


def gmm_pdf(x, weights, means, covars):
    weights_flat = weights.flatten()
    means_flat = means.flatten()
    covars_flat = covars.flatten()
    stds = np.sqrt(covars_flat)
    # vectorized mixture pdf at scalar x
    return np.sum(weights_flat * norm.pdf(x, loc=means_flat, scale=stds))


def gmm_cdf(x, weights, means, covars):
    weights_flat = weights.flatten()
    means_flat = means.flatten()
    covars_flat = covars.flatten()
    cdf_val = 0.0
    for weight, mean, covar in zip(weights_flat, means_flat, covars_flat):
        std = np.sqrt(covar)
        cdf_val += weight * norm.cdf(x, loc=mean, scale=std)
    return cdf_val


def find_gmm_percentile(target_p, weights, means, covars):
    means_flat = means.flatten()
    covars_flat = covars.flatten()
    func = lambda x: gmm_cdf(x, weights, means, covars) - target_p
    min_mean = np.min(means_flat)
    max_mean = np.max(means_flat)
    max_std = np.max(np.sqrt(covars_flat))

    lower_bound = min_mean - 10 * max_std
    upper_bound = max_mean + 10 * max_std

    if func(lower_bound) > 0:
        while func(lower_bound) > 0:
            lower_bound -= max_std
    if func(upper_bound) < 0:
        while func(upper_bound) < 0:
            upper_bound += max_std

    percentile_value = brentq(func, lower_bound, upper_bound)
    return percentile_value


def find_gmm_mode_in_interval(weights, means, covars, lower_bound, upper_bound):
    """
    Find argmax_x GMM(x) for x in [lower_bound, upper_bound].
    """

    def neg_pdf(x):
        return -gmm_pdf(x, weights, means, covars)

    # bounded 1D optimization
    res = minimize_scalar(neg_pdf, bounds=(lower_bound, upper_bound), method="bounded")
    mode_x = res.x

    # (optional) you can also return the pdf value at the mode if useful:
    mode_pdf = gmm_pdf(mode_x, weights, means, covars)
    return mode_x, mode_pdf


def calculate_gmm_prediction_interval(weights, means, covars, confidence_level=0.95):
    alpha = 1.0 - confidence_level
    lower_p = alpha / 2.0
    upper_p = 1.0 - alpha / 2.0

    lower_bound = find_gmm_percentile(lower_p, weights, means, covars)
    upper_bound = find_gmm_percentile(upper_p, weights, means, covars)

    # find maximum likelihood point (mode) within this CI
    ml_point, ml_pdf_val = find_gmm_mode_in_interval(
        weights, means, covars, lower_bound, upper_bound
    )

    # # You can choose whatever return format you like; e.g.:
    # return {
    #     "interval": [lower_bound, upper_bound],
    #     "ml_point": ml_point,
    #     "ml_pdf": ml_pdf_val,   # optional
    # }
    return [[lower_bound, upper_bound]], ml_point # Return as list of intervals


def get_confidence_interval(t_span, p_net_gmm, confidence_level=0.95):
    confidence_interval = []
    ml_points_tspan = []
    for t in t_span:
        ws, mus, covs = p_net_gmm.get_params(t)
        ci, ml_point = calculate_gmm_prediction_interval(ws, mus, covs, confidence_level=confidence_level)
        confidence_interval.append(ci)
        ml_points_tspan.append(ml_point)
    confidence_interval = np.array(confidence_interval)
    # print(confidence_interval.shape)
    return confidence_interval, ml_points_tspan


def get_confidence_interval_classic(t_span, classic_gmm, confidence_level=0.95):
    confidence_interval = []
    for t in t_span:
        _, ws, mus, covs = classic_gmm.get(t)
        ws = ws.reshape(-1,); mus = mus.reshape((-1, 1)); covs = covs.reshape((-1, 1, 1))
        ci, _ = calculate_gmm_prediction_interval(ws, mus, covs, confidence_level=confidence_level)
        confidence_interval.append(ci)
    confidence_interval = np.array(confidence_interval)
    # print(confidence_interval.shape)
    return confidence_interval


def get_confidence_interval_core(ws, mus, covs, confidence_level=0.95):
    ci, ml_point = calculate_gmm_prediction_interval(ws, mus, covs, confidence_level=confidence_level)
    return np.array(ci[0]).reshape((1, -1)), ml_point


def validate_confidence_interval(t_span, x_traj_true, conf_interval_pinn_gmm, label):
    """
    Validates confidence intervals by counting how many true trajectory 
    points fall within the intervals at each time step.

    Args:
        t_span (np.ndarray): The array of time points (unused in calculation, only for dimensions).
        x_traj_true (np.ndarray): True trajectories (shape: N_trajectories, T).
        conf_interval_pinn_gmm (list/np.ndarray): Confidence intervals 
                                        (shape: T, 1, 2) where the last dim is [min, max].

    Returns:
        overall_coverage_frequency (float): Average coverage across all time steps and trajectories.
        last_time_coverage_frequency (float): Coverage at the final time T[-1] across trajectories.
    """
    N_trajectories, T = x_traj_true.shape
    total_coverage_count = 0
    print(f"Validating coverage for {N_trajectories} trajectories over {T} time steps...")

    for idx in range(T):
        # CI at time t_idx, shape (1, 2): [min, max]
        _ci_t = conf_interval_pinn_gmm[idx]
        lower_bound = _ci_t[0, 0]
        upper_bound = _ci_t[0, 1]

        # True states at this time
        _x_traj_true_t = x_traj_true[:, idx]

        # Inside [lower_bound, upper_bound]
        is_inside = (_x_traj_true_t >= lower_bound) & (_x_traj_true_t <= upper_bound)

        # Accumulate coverage over all times and trajectories
        total_coverage_count += np.sum(is_inside)

    # Overall average coverage across all time points and trajectories
    overall_coverage_frequency = total_coverage_count / (N_trajectories * T)

    # --- Coverage at final time T[-1] ---
    ci_last = conf_interval_pinn_gmm[T - 1]   # or conf_interval_pinn_gmm[-1]
    lower_last = ci_last[0, 0]
    upper_last = ci_last[0, 1]

    x_last = x_traj_true[:, T - 1]            # or x_traj_true[:, -1]
    is_inside_last = (x_last >= lower_last) & (x_last <= upper_last)
    last_time_coverage_frequency = np.mean(is_inside_last)

    print(label, f"Overall Coverage Frequency: {overall_coverage_frequency*100:.4f}%")
    print(label, f"Final-Time Coverage Frequency: {last_time_coverage_frequency*100:.4f}%")

    return overall_coverage_frequency, last_time_coverage_frequency
        

def plot_x_traj_true(t_span, x_traj_true, conf_interval_list, label_list, 
    palette: str = "husl", subset=5000):
    x_traj_true_subset = x_traj_true[0:subset, :]

    set_publication_plot_style(font_size=24)
    fig, axs = plt.subplots()
    for traj in x_traj_true_subset:
        axs.plot(t_span, traj, "black", alpha=0.01, lw=0.5)

    colors = sns.color_palette(palette, n_colors=len(conf_interval_list))
    for idx_ci, ci in enumerate(conf_interval_list):
        for idx, t in enumerate(t_span):
            _ci_t = ci[idx] # of shape (D,2), D=1
            # plot a vertical bar for the confidence interval at time t
            lower_bound = _ci_t[0, 0]
            upper_bound = _ci_t[0, 1]
            
            # Use vlines to draw a simple vertical line from min to max
            show_label = None
            if(idx == 0):
                show_label = label_list[idx_ci]
            lw = 3
            if(idx_ci == 0):
                lw = 5
            
            axs.vlines(
                x=t,                     # X position is the time point
                ymin=lower_bound,        # Bottom of the vertical line
                ymax=upper_bound,        # Top of the vertical line
                colors=colors[idx_ci],
                lw=lw,                    # Line width
                alpha=0.5,
                label=show_label
            )
    axs.legend()
    axs.set_xlabel("t")
    axs.set_ylabel("x")
    custom_save_plot(True, "figs/confidence_interval(baseline).pdf")


def data_assimilation_gmm_prior(t, y_at_t, obs_model, ws_prior, mus_prior, covs_prior):
    H, R = obs_model
    K = len(ws_prior) # Number of components

    # Initialize updated parameters
    mus_posterior = np.zeros_like(mus_prior)
    covs_posterior = np.zeros_like(covs_prior)
    ws_posterior = np.zeros_like(ws_prior)
    betas = np.zeros_like(ws_prior)

    # --- Kalman update for each component ---
    for i in range(K):
        mu_i_prior = mus_prior[i]
        P_i_prior = covs_prior[i]

        # 1D implementation notes:
        # H*mu is H*mu
        # H*P*H^T is H*P*H
        # (H*P*H + R)^-1 is 1 / (H*P*H + R)

        # Eq (c): Kalman Gain K_i
        # K_i = P_i(t|y_{0:t-1}) * H * (H * P_i(t|y_{0:t-1}) * H^T + R)^{-1}
        denominator = H * P_i_prior * H + R
        K_i = P_i_prior * H / denominator # Shape (1, 1)

        # Eq (a): Update Mean mu_i
        # mu_i(t|y_t) = mu_i(t| y_{0:t-1}) + K_i * (y_t - H * mu_i(t| y_{0:t-1}) )
        mus_posterior[i] = mu_i_prior + K_i * (y_at_t - H * mu_i_prior)

        # Eq (b): Update Covariance P_i
        # P_i(t|y_t) = (I - K_i * H) * P_i(t|y_{0:t-1})
        I_minus_KH = 1.0 - K_i * H # In 1D, I is 1
        covs_posterior[i] = I_minus_KH * P_i_prior

        # Eq (e): Calculate Likelihood beta_i(t)
        # beta_i(t) = N(y_t; H*mu_i(t| y_{0:t-1}), H*P_i(t|y_{0:t-1})*H^T + R)
        # This is the likelihood of the observation given the component i's prior state
        prediction_mean = H * mu_i_prior
        prediction_cov = H * P_i_prior * H + R
        
        # Use scipy.stats.norm PDF since it's 1D
        betas[i] = norm.pdf(
            y_at_t, 
            loc=prediction_mean.item(), 
            scale=np.sqrt(prediction_cov.item())
        )
    
    # --- Update Weights (Normalization Step) ---

    # Eq (d): w_i(t|y_t) = (w_i(t|y_{0:t-1}) * beta_i(t)) / Sum(...)
    numerator_weights = ws_prior * betas
    denominator_weights = np.sum(numerator_weights)
    
    # Check for numerical stability issues (e.g., if denominator_weights is near zero)
    if denominator_weights < 1e-10:
        print(f"Warning: Low likelihoods at time {t}. Using uniform weights for posterior.")
        ws_posterior = np.ones_like(ws_prior) / K
    else:
        ws_posterior = numerator_weights / denominator_weights
    return ws_posterior, mus_posterior, covs_posterior


def p_init_gmm_pdf(x, ws_pos, mus_pos, covs_pos):
    """
    Evaluates the GMM PDF value at point(s) x given specific GMM parameters.
    Handles both numpy.array and torch.tensor inputs.
    """
    weights_flat = ws_pos.flatten()
    means_flat = mus_pos.flatten()
    covs_flat = covs_pos.flatten()

    if isinstance(x, torch.Tensor):
        x = x.to(dtype=torch.float32) 
        pdf_val = torch.zeros_like(x)
        for weight, mean, covar in zip(weights_flat, means_flat, covs_flat):
            std = torch.sqrt(torch.tensor(covar, dtype=torch.float32))
            exponent = -0.5 * ((x - mean) / std) ** 2
            normalization = std * torch.sqrt(torch.tensor(2 * torch.pi))
            pdf_val += weight * torch.exp(exponent) / normalization
        return pdf_val
        
    elif isinstance(x, np.ndarray):
        pdf_val = np.zeros_like(x, dtype=np.float32)
        for weight, mean, covar in zip(weights_flat, means_flat, covs_flat):
            std = np.sqrt(covar)
            pdf_val += weight * norm.pdf(x, loc=mean, scale=std)
        return pdf_val
            
    else:
        # Handle single scalar inputs if necessary
        return p_init_gmm_pdf(np.array([x]), ws_pos, mus_pos, covs_pos).item()
    

def get_conf_interval_size(conf_interval):
    return abs(conf_interval[0,1] - conf_interval[0,0])


def state_est_pinn_gmm(t_span, x_traj_true, p_net_gmm, traj_pick=0, confidence_level=0.95, seed=2):
    single_x_traj_true = x_traj_true[traj_pick, :]
    p_net_gmm_copy = copy.deepcopy(p_net_gmm)
    conf_interval_prior, ml_points_tspan_prior = get_confidence_interval(t_span, p_net_gmm, confidence_level=confidence_level)

    # compute measurements: y = H*x + e, H = 1, e~N(0, R)
    gap = 1
    t_obs = t_span[1::gap] # define the observation time points
    np.random.seed(seed)
    H = 1.0
    R = 0.5
    obs_model = (H, R)
    std_dev = np.sqrt(R)

    Y = []
    t_Y = []
    conf_interval_pos = [conf_interval_prior[0]]
    conf_interval_pos_size = [get_conf_interval_size(conf_interval_prior[0])]
    X_maxL = [ml_points_tspan_prior[0]]
    conf_interval_prior_retrain = []
    config = {
        "iterations": 4000,
        "sample_ic": train_helper_sample_ic,
        "sample_res": train_helper_sample_res,
        "p_ic": p_init,
        "res_func": res_func,
        "res_weight": 1.0,
        "save_path": None,
    }

    for idx_t in range(1, len(t_span)):
        t = t_span[idx_t]
        # get observation at time t
        y_at_t = None
        if(t in t_obs):
            y_at_t = single_x_traj_true[idx_t] + np.random.normal(loc=0., scale=std_dev)
            Y.append(y_at_t); t_Y.append(t)

        # get prior gmm
        if(idx_t == 1):
            ws, mus, covs = p_net_gmm.get_params(t)
        else:
            ws, mus, covs = p_net_gmm_copy.get_params(t)
        conf_interval_prior_t, _ = get_confidence_interval_core(ws, mus, covs, confidence_level=confidence_level)
        conf_interval_prior_retrain.append(conf_interval_prior_t)
        _gmm_prior = make_gmm_pdf(ws, mus, covs)

        # get the fixed prior (for visual)
        _ws, _mus, _covs = p_net_gmm.get_params(t)
        _gmm_fixed_prior = make_gmm_pdf(_ws, _mus, _covs)
        # conf_interval_fixed_prior_t = get_confidence_interval_core(_ws, _mus, _covs, confidence_level=confidence_level)
        # if(y_at_t < conf_interval_fixed_prior_t[0,0] or y_at_t > conf_interval_fixed_prior_t[0,1]):
        #     y_at_t = None
        #     print("reject y at t=", t)

        # data assimilation
        if(y_at_t is not None):
            ws_pos, mus_pos, covs_pos = data_assimilation_gmm_prior(t, y_at_t, obs_model, ws, mus, covs)
        else:
            ws_pos, mus_pos, covs_pos = ws, mus, covs
        conf_interval_pos_t, ml_point_t = get_confidence_interval_core(ws_pos, mus_pos, covs_pos, confidence_level=confidence_level)
        conf_interval_pos_size.append(get_conf_interval_size(conf_interval_pos_t))
        conf_interval_pos.append(conf_interval_pos_t)
        X_maxL.append(ml_point_t)
        _gmm_pos = make_gmm_pdf(ws_pos, mus_pos, covs_pos)

        # visualization
        plot_data_assimilation(t_span, single_x_traj_true, t_Y, Y, 
            conf_interval_prior, conf_interval_pos, conf_interval_prior_retrain, X_maxL)
        custom_save_plot(True, "figs/CI_t{:.2f}.pdf".format(t))

        plot_prior_and_post(_gmm_fixed_prior, _gmm_prior, _gmm_pos, single_x_traj_true[idx_t], y_at_t,
            conf_interval_pos_t)
        custom_save_plot(True, "figs/Bayes_t{:.2f}.pdf".format(t))
        plt.show()
        
        # # train p_net_gmm forward, p_init() becomes the gmm of ws_pos, mu_pos, cov_pos
        # if(idx_t < len(t_span)-1):
        #     print("forecasting from t=", t)
        #     config.update({"sample_ic": partial(train_helper_sample_ic_after_t, t=t)})
        #     config.update({"sample_res": partial(train_helper_sample_res_after_t, t=t)})
        #     config.update({"p_ic": partial(p_init_gmm_pdf, ws_pos=ws_pos, mus_pos=mus_pos, covs_pos=covs_pos)})
        #     config.update({"save_path": "data/p_net(pinn-gmm)_t0={:.2f}.pth".format(t)})
        #     p_net_gmm_copy = TimeToGMM1D().to(device)
        #     train_pnet_v0(p_net_gmm_copy, config)
        #     p_net_gmm_copy = load_train_model(p_net_gmm_copy, PATH=config["save_path"])

    # evaluate the frequency that the conf_interval_pos includes the single_x_traj_true
    coverage_frequency, last_time_coverage_frequency = validate_confidence_interval(t_span, single_x_traj_true.reshape((1,-1)), conf_interval_pos, "")
    return coverage_frequency, np.array(conf_interval_pos_size), last_time_coverage_frequency


def plot_data_assimilation(t_span, single_x_traj_true, t_Y, Y, 
    conf_interval_prior, conf_interval_pos, conf_interval_prior_retrain, X_maxL):
    set_publication_plot_style(font_size=22)
    fig, axs = plt.subplots()
    axs.plot(t_span, single_x_traj_true, "black", label=r"$X^*$")
    axs.scatter(t_Y, Y, marker="x", color="green", label=r"$Y$", s=100)
    
    # plot confidence interval from fixed prior
    for idx_t, t in enumerate(t_span):
        _ci_t = conf_interval_prior[idx_t]
        label = None
        if(idx_t) == 0: label=r"$\psi$ (baseline)"
        axs.vlines(
            x=t,                     # X position is the time point
            ymin=_ci_t[0,0],        # Bottom of the vertical line
            ymax=_ci_t[0,1],        # Top of the vertical line
            colors="blue",
            lw=6,                    # Line width
            alpha=0.3,
            label=label
        )

    # # plot confidence interval from posterior
    # for idx_t, ci in enumerate(conf_interval_prior_retrain):
    #     axs.vlines(
    #         x=t_Y[idx_t],                     # X position is the time point
    #         ymin=ci[0],        # Bottom of the vertical line
    #         ymax=ci[1],        # Top of the vertical line
    #         colors="green",
    #         lw=7,                    # Line width
    #         alpha=0.3,
    #     )

    # plot confidence interval from posterior
    # print(len(conf_interval_pos))
    for idx_t, ci in enumerate(conf_interval_pos):
        label = None
        if(idx_t) == 0: label=r"$\psi$ (with obs)"
        axs.vlines(
            x=t_span[idx_t],                     # X position is the time point
            ymin=ci[0,0],        # Bottom of the vertical line
            ymax=ci[0,1],        # Top of the vertical line
            colors="red",
            lw=3,                    # Line width
            alpha=0.8,
            label=label
        )
        label_x_maxL = None
        if(idx_t) == 0: label_x_maxL=r"$X$ (max-Like)"
        axs.scatter(t_span[idx_t], X_maxL[idx_t], marker="o", color="black", label=label_x_maxL, s=100)
    axs.set_xlabel("t")
    axs.set_ylabel("x")
    axs.legend()


def plot_prior_and_post(pdf_func_fixed_prior, pdf_func_prior, pdf_func_pos, x_true, y, conf_interval_pos_t):
    x_grid = np.linspace(x_low, x_hig, num=256, endpoint=True)
    pdf_fixed_prior = pdf_func_fixed_prior(x_grid)
    pdf_prior = pdf_func_prior(x_grid)
    pdf_pos = pdf_func_pos(x_grid)
    set_publication_plot_style(font_size=22)
    fig, axs = plt.subplots()
    axs.axvline(
        x=x_true,
        color="black",
        label="true state"
    )

    if(y is not None):
        axs.axvline(
        x=y,
        color="green",
        linestyle="--",
        label="observation"
    )
    axs.plot(x_grid, pdf_fixed_prior, "skyblue", label="fixed prior")
    axs.plot(x_grid, pdf_prior, "blue", label="prior")
    axs.plot(x_grid, pdf_pos, "red", label="pos")
    x_lo, x_hi = conf_interval_pos_t[0, 0], conf_interval_pos_t[0, 1]
    axs.axvspan(x_lo, x_hi, color="red", alpha=0.1, label=r"$\psi$")
    axs.set_xlabel("x")
    axs.set_ylabel("PDF")
    axs.legend()


def plot_coverage_size(t_span, base, mc_trials):
    set_publication_plot_style()
    fig, axs = plt.subplots()
    axs.plot(t_span, base, "black", label="baseline")
    for idx, trial in enumerate(mc_trials):
        label = None
        if idx == 0:
            label = "with obs"
        axs.plot(t_span, trial, "red", alpha=0.5, label=label)
    axs.set_xlabel("t")
    axs.set_ylabel("size of estimator")
    axs.legend()
    

def main():
    torch.manual_seed(0); np.random.seed(0)

    lp_path = os.path.join("data", "lp_np64_dt6.npz")
    data_lp = PropagationData(path=lp_path)

    p_net = TimeToGMM1D().to(device)
    p_net = load_train_model(p_net, PATH="data/p_net(pinn-gmm).pth")

    e1_net = E1Net(scale=get_e1_normalize(p_net),
                normalize=get_e1_normalize(p_net),
                ).to(device)
    e1_net = load_train_model(e1_net, PATH="data/e1_net(pinn-gmm).pth")

    t_span = np.array(t1s).astype(np.float32)
    x_traj_true = load_samples_as_trajectories(t_span)

    confidence_level = 0.95
    conf_interval_pinn_gmm, _ = get_confidence_interval(t_span, p_net, confidence_level=confidence_level)
    conf_interval_lp = get_confidence_interval_classic(t_span, data_lp, confidence_level=confidence_level)
    coverage_size_pinn_gmm_baseline = []
    for ci in conf_interval_pinn_gmm:
        coverage_size_pinn_gmm_baseline.append(get_conf_interval_size(ci))
    coverage_size_pinn_gmm_baseline = np.array(coverage_size_pinn_gmm_baseline)

    # conf_interval_list = [conf_interval_pinn_gmm, conf_interval_lp]
    # label_list = ["PINN-GMM", "GA"]

    # for idx, ci in enumerate(conf_interval_list):
    #     validate_confidence_interval(t_span, x_traj_true, ci, label_list[idx])
    # plot_x_traj_true(t_span, x_traj_true, conf_interval_list, label_list)
    # plt.show()

    # [Monte-Carlo testing]
    N_trials = 10
    coverage_freq = 0.
    coverage_freq_last_time = 0.
    coverage_size = []
    for i in range(0, N_trials+0):
        print("trial: ", i)
        _coverage_freq, _ci_size, _coverage_freq_last_time = state_est_pinn_gmm(t_span, x_traj_true, p_net, 
            traj_pick=i, confidence_level=confidence_level, seed=i)
        coverage_freq += _coverage_freq/N_trials
        coverage_freq_last_time += _coverage_freq_last_time/N_trials
        coverage_size.append(_ci_size)
    coverage_size = np.array(coverage_size)
    print("-" * 40)
    print("# MC overall   coverage frequency: {:.2f} %".format(100. * coverage_freq))
    print("# MC Last-time coverage frequency: {:.2f} %".format(100. * coverage_freq_last_time))
    print("-" * 40)
    # print("coverage size pinn-gmm baseline: ", coverage_size_pinn_gmm_baseline)
    # print("coverage size pinn-gmm with obs: ", coverage_size)
    plot_coverage_size(t_span, coverage_size_pinn_gmm_baseline, coverage_size)
    # custom_save_plot(True, "figs/confidence_interval(obs).pdf")

    plt.show()


if __name__ == "__main__":
    main()