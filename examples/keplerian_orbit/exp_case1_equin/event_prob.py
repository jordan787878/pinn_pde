import numpy as np
from train_p import constants
import argparse
from compare_methods import config_trained_models, load_model_by_key, sample_joint_pdf
import torch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.generalsolvers import *
from utilities._General.util import (plt, set_publication_plot_style, colors_6set, 
    lower_color, upper_color, custom_save_plot, load_metrics_npz)
from utilities._General.classic_gmm import make_gmm_pdf


COMPUTE: bool = False


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
    return p.parse_args()


def true_event_probability(problem, t):
    D = problem["D"]
    X_event = problem["X_event"]
    x_samples = sample_joint_pdf(constants, t, n_samples=1000000).reshape((-1, D))
    x_samples = constants.scaled_x(x_samples)
    _lo = X_event[:, 0]
    _hi = X_event[:, 1]
    mask = np.all((x_samples >= _lo) & (x_samples <= _hi), axis=1)
    Pr_ref = np.sum(mask)/ x_samples.shape[0]
    return Pr_ref


def get_error_bound_at_time(t, atol=1e-5):
    metrics = load_metrics_npz("output/metric_NB=100.npz")
    tspan = metrics["t"]
    B1s = metrics["B1_pinngmm_raw"]
    idx = np.where(np.isclose(tspan, t, atol=atol))[0]
    if idx.size == 0:
        raise ValueError(f"t={t} not found within atol={atol}. Available times include e.g. {tspan[:5]}...")
    return B1s[idx[0]]


def sample_event_box(X_dom, frac=0.1, *, bias=None, rng=None, seed=None, dtype=np.float32):
    """
    Sample an axis-aligned box X_event inside X_dom with per-dimension size
    = frac * (hi - lo). Supports scalar or per-dimension frac.

    Center control:
      - If `bias` (shape (D,)) is provided, X_event is centered at `bias`,
        projected into the feasible center region to keep the box inside X_dom.
      - Else the center is drawn uniformly from the feasible region (reproducible via seed/rng).

    Reproducibility:
      - If `rng` is provided (np.random.Generator), it is used as-is.
      - Else if `seed` is provided, uses np.random.default_rng(seed).
      - Else uses an unseeded Generator.

    Parameters
    ----------
    X_dom : (D,2) array-like
        Domain box per dimension.
    frac  : float or (D,) array-like in (0,1]
        Desired side-length fraction per dimension.
    bias  : (D,) array-like or None
        Desired center for X_event (projected into feasible center set).
    rng   : np.random.Generator or None
    seed  : int or array-like or None
    dtype : np.dtype

    Returns
    -------
    X_event : (D,2) ndarray
        Event box fully contained in X_dom.
    """
    dom = np.asarray(X_dom, float)
    if dom.ndim != 2 or dom.shape[1] != 2:
        raise ValueError("X_dom must be (D,2).")
    if np.any(dom[:, 1] <= dom[:, 0]):
        raise ValueError("X_dom must have hi > lo in every dimension.")
    D = dom.shape[0]

    frac = np.asarray(frac, float)
    if frac.ndim == 0:
        frac = np.full(D, float(frac))
    if frac.shape != (D,):
        raise ValueError("frac must be scalar or shape (D,).")
    if np.any(frac <= 0) or np.any(frac > 1):
        raise ValueError("frac entries must be in (0, 1].")

    lo, hi = dom[:, 0], dom[:, 1]
    span   = hi - lo
    w      = frac * span
    half_w = 0.5 * w

    # Feasible center region to keep the box inside X_dom
    ctr_lo = lo + half_w
    ctr_hi = hi - half_w

    # RNG setup (rng precedence, then seed)
    if rng is None:
        rng = np.random.default_rng(seed)

    if bias is not None:
        bias = np.asarray(bias, float)
        if bias.shape != (D,):
            raise ValueError("bias must have shape (D,).")
        center = np.clip(bias, ctr_lo, ctr_hi)
    else:
        # Uniform center in feasible region
        u = rng.random(D)
        center = ctr_lo + u * (ctr_hi - ctr_lo)

    start = center - half_w
    X_event = np.column_stack([start, start + w]).astype(dtype, copy=False)
    return X_event


def heaviest_component_mean(ws: np.ndarray, mus: np.ndarray):
    """
    ws  : (K,) weights
    mus : (K, D) means
    Returns (idx, mu_star, w_star)
    """
    ws = np.asarray(ws).reshape(-1)
    mus = np.asarray(mus)
    if mus.shape[0] != ws.shape[0]:
        raise ValueError(f"shape mismatch: ws has {ws.shape[0]} weights, mus has {mus.shape[0]} rows")
    if not np.all(np.isfinite(ws)):
        raise ValueError("ws contains non-finite values")
    idx = int(np.argmax(ws))  # if ties, picks the first
    return idx, mus[idx], ws[idx]


def set_static_X_event(problem):
    p_net, _ = problem["networks"]
    teval = 0.15
    ws, mus, covs = p_net.weights_means_covs_at(teval)
    ws = ws.detach().numpy()
    mus = mus.detach().numpy()
    covs = covs.detach().numpy()
    # Define the X_event as a random (0.1 - per dimension of the X_dom)
    _, bias_center, _ = heaviest_component_mean(ws, mus)
    print(bias_center)
    bias_center = bias_center*1.05
    # for i in range(5):
    #     bias_center[i] = bias_center[i]*0.9
    # bias_center[5] = bias_center[5]*0.9
    problem["X_event"] = sample_event_box(problem["X_dom"], seed=2, frac=0.16, bias=bias_center)
    # print(problem["X_event"])
    return problem
    # (Dynamic) set the X_event
    # _, bias_center, _ = heaviest_component_mean(ws, mus)
    # problem["X_event"] = sample_event_box(problem["X_dom"], seed=2, bias=bias_center)
    # print(problem["X_event"])
    # Compute the reference true Probability
    # Pr_ref = true_event_probability(problem, _t)
    # problem["Pr_ref"].append(Pr_ref)
    # print(_t, Pr_ref)


def run_solver(problem):
    problem["Pr_opt_upper"] = []
    problem["Pr_opt_lower"] = []
    for idx, _t in enumerate(problem["time_points"]):
        p_net, _ = problem["networks"]
        ws, mus, covs = p_net.weights_means_covs_at(_t)
        ws = ws.detach().numpy(); mus = mus.detach().numpy(); covs = covs.detach().numpy()

        # Access pinn error bound at time _t
        B1 = get_error_bound_at_time(_t); problem["B1"] =  B1

        # Config LP: discretization and compute pdf bounds over cells
        grid = build_partition_and_bounds(problem, ws, mus, covs, 
            N_degree=problem["N_degree"], 
            use_gmm_guidance=problem["use_gmm_guidance"], 
            cheap=problem["cheap"])
        problem.update(grid)
        print(problem["midpts"].shape[0], sum(problem["mask_lower"]), sum(problem["mask_upper"]))
    
        Pr_upper, _ = waterfill_upper(problem)
        Pr_lower, _ = waterfill_lower(problem)
        problem["Pr_opt_upper"].append(Pr_upper); problem["Pr_opt_lower"].append(Pr_lower)
        print("### [LP] Prob Lower: {:.4f}, Prob Ref: {:.4f}, Prob Upper: {:.4f} ###\n".format(
            Pr_lower, problem["Pr_ref"][idx], Pr_upper))

    return problem


def plot_summary(problem, result_label):
    set_publication_plot_style()
    print_list(problem["Pr_opt_lower"])
    print_list(problem["Pr_ref"])
    print_list(problem["Pr_opt_upper"])

    fig, axs = plt.subplots()
    tspan_ref = problem["time_points_ref"]
    Pr_ref = np.array(problem["Pr_ref"])

    tspan = problem["time_points"]
    Pr_lower = np.array(problem["Pr_opt_lower"])
    Pr_upper = np.array(problem["Pr_opt_upper"])

    axs.plot(tspan_ref, Pr_ref, label=r'$\mathbb{P}(X_{\text{event}})$', 
         linestyle='None', 
         marker='o', 
         markersize=10, 
         markeredgewidth=0.5,
         markerfacecolor='white',  # Sets the fill color to white
         markeredgecolor='black')  # Sets the edge color to white
    axs.fill_between(
        tspan,
        Pr_lower, Pr_upper,
        color='black',
        alpha=0.8,
        label=r"$\mathbb{P}^-, \mathbb{P}^+$"
    )
    axs.plot(tspan, Pr_lower, color=lower_color)
    axs.plot(tspan, Pr_upper, color=upper_color)
    axs.set_xlabel("t")
    axs.set_ylabel("Probability")
    axs.set_ylim([0, min(1., Pr_upper.max()*1.5)])
    plt.legend()
    custom_save_plot(True, "figs/"+result_label+".pdf")


def main():
    # Load PINNs
    trained_models = config_trained_models()
    # p_net, e1_net, _ = load_model_by_key(trained_models, key="PINN-XL_bias")
    p_net_gmm, e1_net_gmm, _ = load_model_by_key(trained_models, key="PINN-GMM_bias")
    networks = (p_net_gmm, e1_net_gmm)

    # Setup problem
    problem = {
        "D": 6,
        "N_x": 5, "N_degree": 4, "use_gmm_guidance": False, "cheap": False,
        "X_dom": constants._NX_RANGE_NP,
        "time_points_ref": np.round(np.arange(0.0, 0.3+0.1, 0.1, dtype=np.float32),2),
        "time_points": np.round(np.arange(0.0, 0.3+0.1, 0.1, dtype=np.float32),2),
        "networks": networks
    }
    result_label = "solver_Nx{:d}_Ndeg{:d}_GMMguide{:d}_Cheap{:d}".format(
        problem["N_x"], problem["N_degree"], problem["use_gmm_guidance"], problem["cheap"]
    )
    result_path = "output/"+result_label+".npz"

    # Define (static) X_event and compute ref true probability
    problem = set_static_X_event(problem)
    problem["Pr_ref"] = []
    for _t in problem["time_points_ref"]:
        Pr_ref = true_event_probability(problem, _t)
        problem["Pr_ref"].append(Pr_ref)
    print_list(problem["Pr_ref"])

    # Solving
    if(COMPUTE):
        problem = run_solver(problem); save_problem_result_npz(result_path, problem)
    
    # I/O
    problem_result = load_problem_result_npz(result_path, problem)
    plot_summary(problem_result, result_label); plt.show()



if __name__ == "__main__":
    args = parse_args()
    COMPUTE   = bool(args.compute)
    main()