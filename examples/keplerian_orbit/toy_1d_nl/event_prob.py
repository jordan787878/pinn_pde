from test_gmmpnet import TimeToGMM1D
from pinn_model import E1Net
from main import load_train_model, get_e1_normalize, plt, p_init, t1s, x_low, x_hig
import torch
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.generalsolvers import *
from utilities._General.util import (set_publication_plot_style, colors_6set, lower_color, upper_color,
        custom_save_plot)
from utilities._General.classic_gmm import make_gmm_pdf


SAVEPLOT = True


def true_event_probability(problem, t):
    D = problem["D"]
    X_event = problem["X_event"]
    x_mc_samples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32).reshape((-1, D))
    _lo = X_event[:, 0]
    _hi = X_event[:, 1]
    mask = np.all((x_mc_samples >= _lo) & (x_mc_samples <= _hi), axis=1)
    Pr_ref = np.sum(mask)/ x_mc_samples.shape[0]
    return Pr_ref


def get_error_bound_at_time(t, atol=1e-5):
    metrics_pinngmm = np.load("data/metrics(pinn-gmm).npz")
    tspan = metrics_pinngmm["t"]
    B1s = metrics_pinngmm["B1"]
    idx = np.where(np.isclose(tspan, t, atol=atol))[0]
    if idx.size == 0:
        raise ValueError(f"t={t} not found within atol={atol}. Available times include e.g. {tspan[:5]}...")
    return B1s[idx[0]]


def run_solver(problem):
    Pr_upper, p_eval_upper = waterfill_upper(problem)
    Pr_lower, p_eval_lower = waterfill_lower(problem)
    return Pr_upper, Pr_lower, p_eval_upper, p_eval_lower


def plot_summary(problem):
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
    axs.set_ylim([0, 1])
    plt.legend()
    custom_save_plot(SAVEPLOT, "figs/solver_N{:d}_all_prob.pdf".format(problem["N_x"]))


def visualize_solver(problem, Pr_upper, Pr_lower, p_eval_upper, p_eval_lower):
    x = np.load("data/xsim.npy").astype(np.float32)
    pdf_mc = np.load("data/psim_t{:.1f}.npy".format(problem["t_eval"])).astype(np.float32).reshape(-1,)

    p_net = problem["networks"][0]
    mus, stds, ws = p_net.params(torch.tensor(problem["t_eval"]).reshape((1,1)))
    covs = stds**2
    ws = ws.detach().numpy().reshape(-1,)             # (K,)
    mus = mus.detach().numpy().reshape((-1, 1))       # (K, 1)
    covs = covs.detach().numpy().reshape((-1, 1, 1))  # (K, 1, 1)
    pdf_func = make_gmm_pdf(ws, mus, covs)
    p_net_x = pdf_func(x)

    # --- D=1 non-uniform cells: build edges from per-cell widths ---
    midpts = np.asarray(problem["midpts"]).reshape(-1)           # (M,)
    widths_arr = np.asarray(problem["widths"])
    if widths_arr.ndim == 2:
        widths_arr = widths_arr.reshape(-1)                      # (M,) for D=1
    elif widths_arr.ndim != 1:
        raise ValueError("problem['widths'] must be (M,) or (M,1) for D=1")

    cell_lo = midpts - 0.5 * widths_arr
    cell_hi = midpts + 0.5 * widths_arr

    # Sort by left edge to make a proper, non-overlapping stairs plot
    order = np.argsort(cell_lo)
    cell_lo = cell_lo[order]
    cell_hi = cell_hi[order]
    x_cells = midpts[order]

    # Stairs expects N+1 edges for N bins
    edges = np.concatenate([cell_lo, cell_hi[-1:]], axis=0)

    # Reorder LP evaluations accordingly
    y_lower = np.asarray(p_eval_lower).reshape(-1)[order]
    y_upper = np.asarray(p_eval_upper).reshape(-1)[order]
    y_minus= np.asarray(problem["p_minus"]).reshape(-1)[order]
    y_plus = np.asarray(problem["p_plus"]).reshape(-1)[order]

    xa = float(problem["X_event"][0,0])
    xb = float(problem["X_event"][0,1])

    set_publication_plot_style(font_size=24)
    fig, axs = plt.subplots()

    # Network band (optional visual context)
    axs.fill_between(
        x,
        p_net_x - problem["B1"], p_net_x + problem["B1"],
        color=colors_6set[0],
        alpha=0.2,
        label="PDF Bound"
    )

    axs.axvspan(xa, xb, alpha=0.1, color='tab:green', zorder=0,
                label=r'$X_{\text{event}}$')
    
    axs.stairs(
        y_minus, edges,
        fill=False, color="black", linewidth=2., linestyle="--",
    )
    axs.stairs(
        y_plus, edges,
        fill=False, color="black", linewidth=2., linestyle="--",
    )

    # Lower piecewise-constant (non-uniform bins)
    axs.stairs(
        y_lower, edges,
        fill=False, color=lower_color, linewidth=2.,
        label=r"lower $p_i$"
    )

    # Upper piecewise-constant (non-uniform bins)
    axs.stairs(
        y_upper, edges,
        fill=False, color=upper_color, linewidth=2.,
        label=r"upper $p_i$"
    )

    axs.set_xlabel("x")
    axs.set_ylabel("PDF")

    # Top-right annotation
    annot = r"$t=${:.2f}".format(problem["t_eval"])
    axs.text(
        0.98, 0.98, annot,
        transform=axs.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.5")
    )

    plt.legend()
    # N_x may be int or vector—use str() to be robust
    save_path = "figs/solver_N{}_pdf_t{:.2f}.pdf".format(problem["N_x"], problem["t_eval"])
    custom_save_plot(SAVEPLOT, save_path)
    plt.show()


def quick_check(networks):
    p_net, e1_net = networks
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    x = np.load("data/xsim.npy").astype(np.float32)
    x_tensor = torch.from_numpy(x.reshape(-1,1))
    t_span = np.array(t1s).astype(np.float32)
    for t in t_span:
        x_mc_samples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32)
        pdf_mc = np.load("data/psim_t{:.1f}.npy".format(t)).astype(np.float32).reshape(-1,)
        if(t == 0):
            pdf_mc = p_init(x).reshape(-1,)
        _t = np.ones(x.shape[0], dtype=x.dtype)*t
        t_tensor = torch.from_numpy(_t.reshape(-1,1))
        if not np.isclose(t, np.round(t), atol=1e-6):
            continue
        
        pdf_pinn = p_net(x_tensor, t_tensor).detach().numpy().reshape(pdf_mc.shape)
        e1_pinn = e1_net(x_tensor, t_tensor).detach().numpy().reshape(pdf_mc.shape)
        e1 = pdf_mc - pdf_pinn
        B1 = 2.* np.max(np.abs(e1_pinn)).item()

        # ax.plot(np.full_like(x, t), x, pdf_mc,
        #         color="black", linestyle="-")
        
        # ax.plot(np.full_like(x, t), x, pdf_pinn,
        #         color="blue", linestyle="-")

        ax.plot(np.full_like(x, t), x, e1,
                color="black", linestyle="-")
        
        ax.plot(np.full_like(x, t), x, e1_pinn,
                color="blue", linestyle="-")

    plt.show()


def main():
    p_net = TimeToGMM1D()
    p_net = load_train_model(p_net, PATH="data/p_net(pinn-gmm).pth")
    e1_net = E1Net(scale=get_e1_normalize(p_net),
                   normalize=get_e1_normalize(p_net))
    e1_net = load_train_model(e1_net, PATH="data/e1_net(pinn-gmm).pth")

    networks = (p_net, e1_net)
    # quick_check(networks)

    # Setup problem
    problem = {
        "D": 1,
        "N_x": 50,
        "X_dom": np.array([[x_low, x_hig]]),
        "time_points_ref": t1s,
        "time_points": t1s,
        # "time_points": np.linspace(t1s[0], t1s[-1], num=50, endpoint=True),
        "X_event": np.array([[-2. , 1.]]),
        "networks": networks
    }

    # Compute truth
    problem["Pr_ref"] = []
    for _t in problem["time_points_ref"]:
        problem["Pr_ref"].append(true_event_probability(problem, _t))

    # Solving (master)
    problem["Pr_opt_upper"] = []
    problem["Pr_opt_lower"] = []
    for idx, _t in enumerate(problem["time_points"]):
        p_net, _ = problem["networks"]
        mus, stds, ws = p_net.params(torch.tensor(_t).reshape((1,1)))
        covs = stds**2
        ws = ws.detach().numpy().reshape(-1,) # (K,)
        mus = mus.detach().numpy().reshape((-1, 1)) # (K, 1)
        covs = covs.detach().numpy().reshape((-1, 1, 1)) # (K, 1, 1)
        B1 = get_error_bound_at_time(_t)
        problem["B1"] =  B1

        # 1) Build grid that’s aware of the GMM bounds
        grid = build_partition_and_bounds(problem, ws, mus, covs, 
            N_degree=1, use_gmm_guidance=False, cheap=False)
        problem.update(grid)
        print(problem["midpts"].shape[0])
        print(sum(problem["mask_lower"]), sum(problem["mask_upper"]))

        Pr_upper, Pr_lower, _, _ = run_solver(problem)
        problem["Pr_opt_upper"].append(Pr_upper)
        problem["Pr_opt_lower"].append(Pr_lower)
        print(Pr_lower, problem["Pr_ref"][idx], Pr_upper)
        # np.testing.assert_array_less(Pr_lower, problem["Pr_ref"][idx])
        # np.testing.assert_array_less(problem["Pr_ref"][idx], Pr_upper)
    plot_summary(problem)
    plt.show()

    # Solving (single time for plot, input a valid time)
    _t_idx = -1
    problem["t_eval"] = t1s[_t_idx]
    Pr_upper, Pr_lower, p_eval_upper, p_eval_lower = run_solver(problem)
    problem["x_show"] = np.load("data/xsamples_t{:.1f}.npy".format(problem["t_eval"])).astype(np.float32)
    visualize_solver(problem, Pr_upper, Pr_lower, p_eval_upper, p_eval_lower)


if __name__ == "__main__":
    main()