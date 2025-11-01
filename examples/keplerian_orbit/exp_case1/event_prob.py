import numpy as np
from train_p import constants
from compare_methods import (config_trained_models, load_model_by_key,
    PropagationData, SAVE_PATH_GMM_PROPAGATE, SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE)
import argparse
import seaborn as sns
from exp_utilities.plot_util import plot_event_triptych_simple

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.generalsolvers import *
from utilities._General.util import (plt, set_publication_plot_style, 
    colors_4set, linestyles_4set, markers_4set,
    lower_color, upper_color, custom_save_plot, load_metrics_npz)
from utilities._General.classic_gmm import make_gmm_pdf, integrate_gmm_over_box_whitened


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
    X_event = problem["X_event"]
    _lo = X_event[:, 0]
    _hi = X_event[:, 1]
    N_run = 10
    Pr_ref = 0.0
    for i in range(1, N_run+1):
        x_samples = np.load("dataset/run"+str(i)+"/xsamples_t{:.3f}.npy".format(t))
        mask = np.all((x_samples >= _lo) & (x_samples <= _hi), axis=1)
        Pr_ref += (np.sum(mask)/ x_samples.shape[0])/N_run
    return Pr_ref


def get_error_bound_at_time(t, atol=1e-5):
    metrics = load_metrics_npz("output/metric_NB=10000.npz")
    tspan = metrics["t"]
    B1s = metrics["B1_pinngmm_raw"] #; print(B1s)
    idx = np.where(np.isclose(tspan, t, atol=atol))[0]
    if idx.size == 0:
        raise ValueError(f"t={t} not found within atol={atol}. Available times include e.g. {tspan[:5]}...")
    print("[debug] B1s: ", B1s)
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
    _, bias_center, _ = heaviest_component_mean(ws, mus)
    problem["X_event"] = sample_event_box(problem["X_dom"], seed=2, frac=0.05, bias=bias_center)
    return problem


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
            use_event_guidance=problem["use_event_guidance"], 
            use_gmm_guidance=problem["use_gmm_guidance"], 
            cap_per_degree = problem["cap_per_degree"],
            cheap=problem["cheap"],
            verbose=problem["verbose_refine"],
        )
        problem.update(grid)
        print("[debug] number of inside & intersect cells: ", sum(problem["mask_lower"]), sum(problem["mask_upper"]))
    
        Pr_upper, _ = waterfill_upper(problem)
        Pr_lower, _ = waterfill_lower(problem)
        problem["Pr_opt_upper"].append(Pr_upper); problem["Pr_opt_lower"].append(Pr_lower)
        print("### [LP] Prob Lower: {:.4f}, Prob Ref: {:.4f}, Prob Upper: {:.4f} ###\n".format(
            Pr_lower, problem["Pr_ref"][idx], Pr_upper))

    return problem


def estimate_event_probability(problem, Pr_key):
    problem[Pr_key] = []
    D = problem["D"]
    if(Pr_key == "Pr_pinngmm"):
        for idx, _t in enumerate(problem["time_points"]):
            p_net, _ = problem["networks"]
            ws, mus, covs = p_net.weights_means_covs_at(_t)
            ws = ws.detach().numpy(); mus = mus.detach().numpy(); covs = covs.detach().numpy()
            ws = ws.reshape(-1,)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _Pr = integrate_gmm_over_box_whitened(ws, mus, covs, problem["X_event"])
            problem[Pr_key].append(_Pr)

    if(Pr_key == "Pr_lp"):
        data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
        for idx, _t in enumerate(problem["time_points"]):
            _, ws, mus, covs = data_lp.get(_t)
            ws = ws.reshape(-1,)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _Pr = integrate_gmm_over_box_whitened(ws, mus, covs, problem["X_event"])
            problem[Pr_key].append(_Pr)
    
    if(Pr_key == "Pr_ut"):
        data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
        for idx, _t in enumerate(problem["time_points"]):
            _, ws, mus, covs = data_ut.get(_t)
            ws = ws.reshape(-1,)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _Pr = integrate_gmm_over_box_whitened(ws, mus, covs, problem["X_event"])
            problem[Pr_key].append(_Pr)

    if(Pr_key == "Pr_gmm"):
        data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)
        for idx, _t in enumerate(problem["time_points"]):
            _, ws, mus, covs = data_gmm.get(_t)
            ws = ws.reshape(-1,)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _Pr = integrate_gmm_over_box_whitened(ws, mus, covs, problem["X_event"])
            problem[Pr_key].append(_Pr)

    return problem


def plot_est_summary(problem):
    set_publication_plot_style()
    fig, axs = plt.subplots()
    tspan_ref = problem["time_points_ref"]
    Pr_ref = np.array(problem["Pr_ref"])
    tspan = problem["time_points"]

    axs.plot(tspan_ref, Pr_ref, label=r'$\mathbb{P}_{\text{ref}}(X)$', 
         linestyle='None', 
         marker='o', 
         markersize=12, 
         markeredgewidth=2.0,
         markerfacecolor='white',  # Sets the fill color to white
         markeredgecolor='black')  # Sets the edge color to white
    
    for idx, key in enumerate(problem["Pr_keys"]):
        axs.plot(tspan, problem[key], linestyle=linestyles_4set[idx],
                 color=colors_4set[idx], marker=markers_4set[idx], label=problem["Pr_keys_labels"][idx])

    axs.set_xlabel("t")
    axs.set_ylabel("Probability")
    plt.legend(ncol=1, loc="best")
    custom_save_plot(True, "figs/prob_est.pdf")


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

    axs.plot(tspan_ref, Pr_ref, label=r'$\mathbb{P}_{\text{ref}}(X)$', 
         linestyle='None', 
         marker='o', 
         markersize=9, 
         markeredgewidth=2.0,
         markerfacecolor='white',  # Sets the fill color to white
         markeredgecolor='black')  # Sets the edge color to white
    
    for idx, key in enumerate(problem["Pr_keys"]):
        axs.plot(tspan, problem[key], linestyle=linestyles_4set[idx],
                 color=colors_4set[idx], marker=markers_4set[idx], label=problem["Pr_keys_labels"][idx])

    # axs.fill_between(
    #     tspan,
    #     Pr_lower, Pr_upper,
    #     color='black',
    #     alpha=0.3,
    #     label=r"Bounds $\mathbb{P}^-, \mathbb{P}^+$"
    # )

    # problem_coarse = load_problem_result_npz("output/solver_Nx1_Ndeg10_GMMguide1_Cheap0.npz")
    # Pr_lower = np.array(problem_coarse["Pr_opt_lower"])
    # Pr_upper = np.array(problem_coarse["Pr_opt_upper"])
    # axs.fill_between(
    #     tspan,
    #     Pr_lower, Pr_upper,
    #     color='black',
    #     alpha=0.1,
    #     label=r"Bounds $\mathbb{P}^-, \mathbb{P}^+$ (Coarse)"
    # )

    # axs.plot(tspan, Pr_lower, color=lower_color)
    # axs.plot(tspan, Pr_upper, color=upper_color)

    axs.set_xlabel("t")
    axs.set_ylabel("Probability")
    axs.set_ylim([-0.01, min(1., Pr_upper.max()*1.5)])
    plt.legend(ncol=2)
    custom_save_plot(True, "figs/"+result_label+".pdf")


def main():
    # Load PINNs
    trained_models = config_trained_models()
    # p_net, e1_net, _ = load_model_by_key(trained_models, key="PINN-XL_bias")
    p_net_gmm, e1_net_gmm, _ = load_model_by_key(trained_models, key="PINN-GMM_bias")
    networks = (p_net_gmm, e1_net_gmm)
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)

    # Setup problem
    problem = {
        "D": 6,
        "N_x": 1, 
        "N_degree": 1, 
        "use_event_guidance": False, "use_gmm_guidance": False, 
        "cap_per_degree" : 256,
        "cheap": False,
        "verbose_refine": True,
        "X_dom": constants._NX_RANGE_NP,
        "time_points_ref": constants.T_PRIME_SPAN,
        "time_points": data_lp.data["times"],
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

    # Compute event probability estimates
    problem["Pr_keys"] = ["Pr_lp", "Pr_ut", "Pr_gmm", "Pr_pinngmm"]
    problem["Pr_keys_labels"] = [
        r'$\mathbb{P}_{\text{GA}}(X)$',
        r'$\mathbb{P}_{\text{UT}}(X)$',
        r'$\mathbb{P}_{\text{GMM}}(X)$',
        r'$\mathbb{P}_{\text{PINN-GMM}}(X)$'
    ]
    for key in problem["Pr_keys"]:
        problem = estimate_event_probability(problem, Pr_key=key)

    # logout
    print("time: ", end="\t"); print_list(problem["time_points"])
    print("Pr ref: ", end="\t")
    print_list(problem["Pr_ref"])
    for key in problem["Pr_keys"]:
        print(key, ": ", end="\t",)
        print_list(problem[key])

    # plot probability estimate only
    plot_est_summary(problem)
    plt.show()
    return

    # visualize X_event
    # data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    # data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    # data_gmm = PropagationData(SAVE_PATH_GMM_PROPAGATE)
    # plot_event_triptych_simple(constants, problem["X_event"], 0.3, 
    #         p_net_gmm=p_net_gmm, data_lp=data_lp, data_ut=data_ut, data_gmm=data_gmm)
    # plt.show()
    # return

    # Solving
    if(COMPUTE):
        problem = run_solver(problem)
        save_problem_result_npz(result_path, problem, 
                                keys=problem.keys())
    
    # I/O
    problem_result = load_problem_result_npz(result_path, problem)
    plot_summary(problem_result, result_label); plt.show()


if __name__ == "__main__":
    args = parse_args()
    COMPUTE   = bool(args.compute)
    main()