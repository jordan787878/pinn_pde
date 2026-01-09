import sys
from pathlib import Path
import seaborn as sns
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.util import load_metrics_npz, plt, set_publication_plot_style, custom_save_plot


# method_list = ["lp", "ut", "gmm", "pinn", "pinngmm_vanilla", "pinngmm"]
# label_list = [
#     "GA",
#     "UT",
#     "GMM",
#     "PINN-vanilla",
#     "PINN-GMM-vanilla",
#     "PINN-GMM"
# ]

# colors_6set = sns.color_palette([
#     "#FF008C",  # GA
#     "#00FBFF",  # UT
#     "#FF8400",  # GMM
#     "#8C00FF",  # PINN-MLP
#     "#0066FF",  # PINN-GMM-vanilla
#     "#000000",  # PINN-GMM
# ])

# linestyles_6set = [
#     (0, (5, 2)),        # GA
#     (0, (7, 2, 3, 2)),  # UT
#     "--",               # GMM
#     "-.",               # PINN-MLP
#     ":",                # PINN-GMM-vanilla
#     "-",                # PINN-GMM
# ]
# markers_6set = [
#     'o', # GA
#     's', # UT
#     '^', # GMM
#     'D', # PINN-MLP
#     'X',  # PINN-GMM-vanilla
#     'None', # PINN-GMM
# ] 

method_list = [
    # "lp", 
    # "ut", 
    # "gmm", 
    "pinn", 
    "pinngmm_vanilla", 
    "pinngmm_noencoder",
    "pinngmm"
]

label_list = [
    # "GA",
    # "UT",
    # "GMM",
    "PINN-MLP (vanilla)",
    "PINN-GMM (vanilla)",
    "PINN-GMM (no-encoder)",
    "PINN-GMM",
]

colors_6set = sns.color_palette([
    # "#FF008C",  # GA
    # "#00FBFF",  # UT
    # "#FF8400",  # GMM
    "#8C00FF",  # PINN-MLP
    "#0066FF",  # PINN-GMM (vanilla)
    "#00FF1E",  # PINN-GMM (no-encoder)
    "#000000",  # PINN-GMM
])

linestyles_6set = [
    # (0, (5, 2)),        # GA
    # (0, (7, 2, 3, 2)),  # UT
    # "--",               # GMM
    "-.",               # PINN-MLP
    ":",                # PINN-GMM-vanilla
    "--",
    "-",                # PINN-GMM
]
markers_6set = [
    # 'o', # GA
    # 's', # UT
    # '^', # GMM
    'D', # PINN-MLP
    'X',  # PINN-GMM-vanilla
    "*",
    'None', # PINN-GMM
] 


def _format_mean_std(vals):
    """
    Format a list/array as 'mean\\pmstd' for LaTeX.

    - If vals is None or empty -> '-'
    - If all values are non-finite (NaN/inf) -> 'nan'
    - Otherwise -> compute mean/std over finite values only
                  and format 'm.s\\pms.s' with 2 decimals
    """
    if vals is None:
        return "-"

    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return "-"

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "nan"

    mean = float(finite.mean())
    std  = float(finite.std())  # population std over finite entries
    return f"{mean:.2f}$\\pm${std:.2f}"


def meta_table(exp_dict):
    """
    Print LaTeX table rows consistent with:

      Experiments | Methods | train loss \\hat p | TV | WNE | Error Bound | RD

    exp_dict is like:
        {
          "exp_case1_equin": { "rel_error_lp": [...], "tv_lp": [...], ... },
          "exp_case1":       { ... },
          "exp_case2_j2":    { ... },
        }
    """

    # method suffix in metric keys  -> LaTeX method name
    method_rows = [
        ("lp",              "GA"),
        ("ut",              "UT"),
        ("gmm",             "GMM"),
        ("pinn",            "PINN-MLP (vanilla)"),
        ("pinngmm_vanilla", "PINN-GMM (vanilla)"),
        ("pinngmm_noencoder", "PINN-GMM (no-encoder)"),  # not recorded yet
        ("pinngmm",         "PINN-GMM"),
    ]

    # optional pretty labels for experiments (you can edit this)
    pretty_exp_name = {
        "exp_case1_equin": r"\textbf{Case 1 Equin}",
        "exp_case1":       r"\textbf{Case 1}",
        "exp_case2_j2":    r"\textbf{Case 2 J2}",
    }

    for exp_name, metrics in exp_dict.items():
        # choose label for the "Experiments" column
        exp_label = pretty_exp_name.get(exp_name, fr"\textbf{{{exp_name}}}")

        print()  # blank line between experiment blocks for readability
        print(f"% {exp_name}")
        first_row = True

        for suffix, method_label in method_rows:
            # keys like 'rel_error_lp', 'tv_lp', 'g_kl_lp'
            rel_key = f"rel_error_{suffix}"
            tv_key  = f"tv_{suffix}"
            rd_key  = f"g_kl_{suffix}"

            rel_vals = metrics.get(rel_key, None)
            tv_vals  = metrics.get(tv_key, None)
            rd_vals  = metrics.get(rd_key, None)

            # Error Bound:
            #   - Use B1_pinngmm for PINN-GMM
            #   - Use B1_pinn for PINN-MLP (vanilla), if present
            if suffix == "pinngmm":
                eb_vals = metrics.get("B1_pinngmm", None)
            elif suffix == "pinn":
                eb_vals = metrics.get("B1_pinn", None)
            else:
                eb_vals = None

            train_loss_vals = None  # not in your current metrics; keep as '-'

            # train_loss_str = _format_mean_std(train_loss_vals)
            tv_str         = _format_mean_std(tv_vals)
            wne_str        = _format_mean_std(rel_vals)
            # eb_str         = _format_mean_std(eb_vals)
            rd_str         = _format_mean_std(rd_vals)

            if first_row:
                # first row uses multirow for the experiment name
                n_methods = len(method_rows)
                print(
                    f"      \\multirow{{{n_methods}}}{{2cm}}{{{exp_label}}} & "
                    f"{method_label} & {tv_str} & {wne_str} & {rd_str} \\\\"
                )
                first_row = False
            else:
                print(
                    f"      & {method_label} & "
                    f" {tv_str} & {wne_str} & {rd_str} \\\\"
                )


def plot_total_variation(exp_dict, n_grid=10):
    """
    For each method in `method_list`, normalize each experiment's time to [0,1],
    interpolate onto a common grid, then fill the band between per-grid min/max.
    Also plot the per-grid mean as a line.
    """
    set_publication_plot_style()
    fig, ax = plt.subplots()

    for idx_method, method in enumerate(method_list):
        t_common = np.linspace(0.0, 1.0, n_grid+idx_method)
        metric_label = f"tv_{method}"
        interp_stack = []
        print(metric_label)

        # collect all experiments for this method
        for key, metrics in exp_dict.items():
            t = np.asarray(metrics["t"], dtype=float)
            y = np.asarray(metrics[metric_label], dtype=float)

            # clean + sort
            m = np.isfinite(t) & np.isfinite(y)
            if not np.any(m):
                continue
            t, y = t[m], y[m]
            if t.size < 2:
                continue
            order = np.argsort(t)
            t, y = t[order], y[order]

            # normalize time to [0,1]
            t0, t1 = t[0], t[-1]
            if t1 == t0:
                continue
            t_norm = (t - t0) / (t1 - t0)

            # interpolate onto common normalized grid
            yi = np.interp(t_common, t_norm, y)
            interp_stack.append(yi)

        if not interp_stack:
            continue

        Y = np.vstack(interp_stack)  # shape: (n_experiments, n_grid)
        y_min = np.nanmin(Y, axis=0)
        y_max = np.nanmax(Y, axis=0)
        y_mean = np.nanmean(Y, axis=0)

        c = colors_6set[idx_method]
        alpha=0.2
        if(method == "pinngmm"): alpha=0.4
        show_metric_label = label_list[idx_method]
        ax.fill_between(t_common, y_min, y_max, alpha=alpha, color=c,
                        # zorder=zorder[idx_method]
                        # label=f"{show_metric_label} "
                        )
        # ax.plot(t_common, y_mean, lw=2, color=c, marker=markers_6set[idx_method],
        #         linestyle=linestyles_6set[idx_method],
        #         label=f"{show_metric_label}", zorder=10)
        ax.plot(
            t_common, y_min,
            linestyle=linestyles_6set[idx_method],
            marker=markers_6set[idx_method],
            # linewidth=1,
            color=c,
            label=f"{show_metric_label} "
        )
        ax.plot(
            t_common, y_max,
            linestyle=linestyles_6set[idx_method],
            marker=markers_6set[idx_method],
            # linewidth=1,
            color=c,
            label="_nolegend_",
        )

    # ax.set_yscale('log')  # uncomment if you want log scale
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 150)
    ax.set_xlabel("Normalized time $t'$")
    ax.set_ylabel("Total variation %")
    ax.legend()
    custom_save_plot(True, "meta_figs/TV_all.pdf")


def plot_worst_normalized_error(exp_dict, n_grid=10):
    """
    For each method in `method_list`, normalize each experiment's time to [0,1],
    interpolate onto a common grid, then fill the band between per-grid min/max.
    Also plot the per-grid mean as a line.
    """
    set_publication_plot_style()
    fig, ax = plt.subplots()
    for idx_method, method in enumerate(method_list):
        t_common = np.linspace(0.0, 1.0, n_grid+idx_method)
        metric_label = f"rel_error_{method}"
        interp_stack = []

        # collect all experiments for this method
        for key, metrics in exp_dict.items():
            t = np.asarray(metrics["t"], dtype=float)
            y = np.asarray(metrics[metric_label], dtype=float)

            # clean + sort
            m = np.isfinite(t) & np.isfinite(y)
            if not np.any(m):
                continue
            t, y = t[m], y[m]
            if t.size < 2:
                continue
            order = np.argsort(t)
            t, y = t[order], y[order]

            # normalize time to [0,1]
            t0, t1 = t[0], t[-1]
            if t1 == t0:
                continue
            t_norm = (t - t0) / (t1 - t0)

            # interpolate onto common normalized grid
            yi = np.interp(t_common, t_norm, y)
            interp_stack.append(yi)

        if not interp_stack:
            continue

        Y = np.vstack(interp_stack)  # shape: (n_experiments, n_grid)
        y_min = np.nanmin(Y, axis=0)
        y_max = np.nanmax(Y, axis=0)
        y_mean = np.nanmean(Y, axis=0)

        c = colors_6set[idx_method]
        alpha=0.2
        if(method == "pinngmm"): alpha=0.4
        show_metric_label = label_list[idx_method]
        ax.fill_between(t_common, y_min, y_max, alpha=alpha, color=c,
                        # zorder=zorder[idx_method]
                        # label=f"{show_metric_label} "
                        )
        # ax.plot(t_common, y_mean, lw=2, color=c, marker=markers_6set[idx_method],
        #         linestyle=linestyles_6set[idx_method],
        #         label=f"{show_metric_label}", zorder=10)
        ax.plot(
            t_common, y_min,
            linestyle=linestyles_6set[idx_method],
            marker=markers_6set[idx_method],
            # linewidth=1,
            color=c,
            label=f"{show_metric_label} "
            # zorder=zorder[idx_method] + 0.5,
        )
        ax.plot(
            t_common, y_max,
            linestyle=linestyles_6set[idx_method],
            marker=markers_6set[idx_method],
            # linewidth=1,
            color=c,
            label="_nolegend_",
            # zorder=zorder[idx_method] + 0.5,
        )

    # ax.set_yscale('log')  # uncomment if you want log scale
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 150)
    ax.set_xlabel("Normalized time $t'$")
    ax.set_ylabel("Worst Normalized Error %")
    ax.legend()
    custom_save_plot(True, "meta_figs/WNE_all.pdf")


def plot_relative_divergence(exp_dict, n_grid=10):
    set_publication_plot_style()
    fig, ax = plt.subplots()

    for idx_method, method in enumerate(method_list):
        t_common = np.linspace(0.0, 1.0, n_grid+idx_method)
        metric_label = f"g_kl_{method}"
        interp_stack = []

        # collect all experiments for this method
        for key, metrics in exp_dict.items():
            t = np.asarray(metrics["t"], dtype=float)
            y = np.asarray(metrics[metric_label], dtype=float)

            # clean + sort
            m = np.isfinite(t) & np.isfinite(y)
            if not np.any(m):
                continue
            t, y = t[m], y[m]
            if t.size < 2:
                continue
            order = np.argsort(t)
            t, y = t[order], y[order]

            # normalize time to [0,1]
            t0, t1 = t[0], t[-1]
            if t1 == t0:
                continue
            t_norm = (t - t0) / (t1 - t0)

            # interpolate onto common normalized grid
            yi = np.interp(t_common, t_norm, y)
            interp_stack.append(yi)

        if not interp_stack:
            continue

        Y = np.vstack(interp_stack)  # shape: (n_experiments, n_grid)
        y_min = np.nanmin(Y, axis=0)
        y_max = np.nanmax(Y, axis=0)
        y_mean = np.nanmean(Y, axis=0)

        c = colors_6set[idx_method]
        alpha=0.2
        if(method == "pinngmm"): alpha=0.4
        show_metric_label = label_list[idx_method]
        ax.fill_between(t_common, y_min, y_max, alpha=alpha, color=c,
                        # label=f"{show_metric_label} "
                        )
        # ax.plot(t_common, y_mean, lw=2, color=c, marker=markers_6set[idx_method],
        #         linestyle=linestyles_6set[idx_method],
        #         label=f"{show_metric_label}", zorder=10)
        # min and max-ish boundary lines
        ax.plot(
            t_common, y_min,
            linestyle=linestyles_6set[idx_method],
            marker=markers_6set[idx_method],
            # linewidth=1,
            color=c,
            label=f"{show_metric_label} "
        )
        ax.plot(
            t_common, y_max,
            linestyle=linestyles_6set[idx_method],
            marker=markers_6set[idx_method],
            # linewidth=1,
            color=c,
            label="_nolegend_",
        )

    # ax.set_yscale('log')  # uncomment if you want log scale
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 150)
    ax.set_xlabel("Normalized time $t'$")
    ax.set_ylabel("Relative Divergence")
    ax.legend()
    custom_save_plot(True, "meta_figs/RD_all.pdf")


def _bin_mean(y, w):
    n = len(y) // w
    if n == 0:
        return np.array([np.mean(y, dtype=float)])
    return y[:n*w].reshape(n, w).mean(axis=1)


def plot_training_history_meta(exp_list, window=100):
    colors = sns.color_palette([
        "#8C00FF",  # PINN-MLP
        "#0066FF",  # PINN-GMM-vanilla
        "#00FF1E",  # prior
        "#000000",  # PINN-GMM
    ])
    trained_models = {
        "PINN-MLP_vanilla": "output/pinn-xl_vanilla",
        "PINN-GMM_vanilla": "output/pinn-gmm_vanilla",
        "PINN-GMM-noencoder": "output/pinn-gmm-noencoder",
        "PINN-GMM_bias": "output/pinn-gmm_bias",
    }
    _exp_list = exp_list
    _label_list = ["PINN-MLP (vanilla)", 
                   "PINN-GMM (vanilla)", 
                   "PINN-GMM (no-encoder)", 
                   "PINN-GMM"]
    zorder = [2, 1, 3, 4]

    set_publication_plot_style()
    fig, ax = plt.subplots()
    i = 0
    for key in trained_models:
        # load & bin-average each run
        binned = []
        for exp in _exp_list:
            path = f"{exp}/{trained_models[key]}"
            y = np.asarray(torch.load(f"{path}/p_net.pth", map_location="cpu")["loss_history"], float)
            binned.append(_bin_mean(y, window))

        # pad to max number of bins with NaN
        B = max(map(len, binned))
        A = np.full((len(binned), B), np.nan, float)
        for r, yb in enumerate(binned):
            A[r, :len(yb)] = yb

        # envelope over bins
        x_bins = (np.arange(B) * window) + (window - 1) / 2.0
        lo = np.nanmin(A, axis=0)
        hi = np.nanmax(A, axis=0)
        m = np.isfinite(lo) & np.isfinite(hi)

        plt.fill_between(x_bins[m], lo[m], hi[m],
                         color=colors[i], alpha=0.7,
                         zorder=zorder[i], label=_label_list[i])
        i += 1

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(f"Loss (mean over {window}-iter bins)")
    plt.legend()
    custom_save_plot(True, "meta_figs/Loss_all.pdf")


def main():
    exp_list = ["exp_case1_equin",
                "exp_case1",
                "exp_case2_j2",]
    NBATCH = 10000
    exp_dict = dict.fromkeys(exp_list)
    for key in exp_dict.keys():
        metrics_path = key+"/output/metric_NB="+str(NBATCH)+".npz"
        metrics = load_metrics_npz(metrics_path)
        exp_dict[key] = metrics

    # meta_table(exp_dict)

    plot_total_variation(exp_dict)

    plot_worst_normalized_error(exp_dict)

    plot_relative_divergence(exp_dict)

    # plot_training_history_meta(exp_list)

    plt.show()


if __name__ == "__main__":
    main()