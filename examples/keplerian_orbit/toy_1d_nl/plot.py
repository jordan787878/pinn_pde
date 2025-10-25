import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib as mpl
from cycler import cycler

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.util import custom_save_plot


colors_7set = sns.color_palette([
    "#FF008C",  # GA
    "#00FBFF",  # UT
        "#00FBFF",  # UT
    "#FF8400",  # GMM
    "#00FF1E",  # prior
    "#8C00FF",  # PINN-MLP
    "#000000",  # PINN-GMM
    "#0066FF",  # PINN-Flow  ←
])

# 7 high-contrast linestyle/marker pairs (index-bound)
linestyles_7set = [
    (0, (5, 2)),        # GA
    (0, (7, 2, 3, 2)),  # UT
        (0, (7, 2, 3, 2)),  # UT
    "--",               # GMM
    (0, (9, 2, 1, 2)),  # prior  ←
    "-.",               # PINN-MLP
    "-",                # PINN-GMM
    ":",                # PINN-Flow
]
markers_7set = ['o', # GA
                's', # UT
                    's', # UT
                '^', # GMM
                'v',    # prior  ←
                'D', # PINN-MLP
                'None', # PINN-GMM
                'X'  # PINN-Flow
                ] 


def set_publication_plot_style(font_family='Times New Roman', font_size=18,
                               sci_power=(-2, 2), tick_pad=6,
                               legend_loc='upper left', legend_frame=True,
                               pair_line_marker_cycle=False):
    """
    Publication-ready Matplotlib defaults with consistent tick formatting.
    """
    mpl.rcdefaults()       # reset rcParams to built-in defaults
    plt.style.use('default')  # ensure default style (no external style lingering)
    
    plt.rcParams.update({
        # Typography
        'font.family': font_family,
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size,
        'lines.linewidth': 2,
        'lines.markersize': 7,
        'lines.markeredgewidth': 1.5,
        'lines.markerfacecolor': 'none',   # hollow markers = great contrast

        # Tick appearance & alignment
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.pad': tick_pad,
        'ytick.major.pad': tick_pad,
        'xtick.top': False,     # corner/pair plots usually cleaner without top/right ticks
        'ytick.right': False,

        # ---------- Legend defaults ----------
        'legend.loc': legend_loc,
        'legend.frameon': legend_frame,
        'legend.borderaxespad': 0.4,   # padding between legend and axes
        'legend.borderpad': 0.3,       # padding inside the legend box
        'legend.handlelength': 1.8,    # line handle length
        'legend.handletextpad': 0.6,   # space between handle and text
        'legend.columnspacing': 0.8,
        'legend.labelspacing': 0.4,
        'legend.markerscale': 2.0,     # scale marker size in legend
        # 'legend.numpoints': 1,        # uncomment for single-point line legends

        # Scientific notation like the helper used
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': sci_power,  # switch to sci notation outside these powers
        'axes.formatter.useoffset': False,   # avoid confusing 1eN offsets on axes
        'axes.unicode_minus': False,         # minus sign renders consistently with mathtext

        # Spacing / layout
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.05,
        'figure.constrained_layout.w_pad': 0.05,
        'figure.constrained_layout.hspace': 0.10,
        'figure.constrained_layout.wspace': 0.10,
        'axes.labelpad': 8,
        'axes.titlepad': 10,

        # ---------- Grid defaults ----------
        'axes.grid': True,            # turn grid on by default
        'axes.grid.axis': 'both',     # x and y
        'axes.grid.which': 'major',   # grid for major ticks (change to 'both' if desired)
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,

        # Save tight
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Figure size
        'figure.figsize': (10, 8),
    })
    if pair_line_marker_cycle:
        plt.rc('axes', prop_cycle=cycler(color=colors_7set) + cycler(linestyle=linestyles_7set) + cycler(marker=markers_7set))


def plot_metrics(show_flow=False):

    metrics = np.load("data/metrics.npz")
    metrics_pri = np.load("data/metrics(prior).npz")
    metrics_pinngmm = np.load("data/metrics(pinn-gmm).npz")
    metrics_flow = np.load("data/metrics(flow).npz")

    plot_labels = ["GA", "UT, "+r"$\alpha=10^{-3}$", "UT, "+r"$\alpha=1.0$", "GMM", 
        "PINN-MLP (prior)", "PINN-MLP", "PINN-GMM", "PINN-FLOW"]

    set_publication_plot_style(font_size=24, pair_line_marker_cycle=True)
    # metric 1: worst relative error %
    plt.figure()
    plt.plot(metrics["t"], metrics["norm_error_lp"], label=plot_labels[0])
    plt.plot(metrics["t"], metrics["norm_error_ut"], label=plot_labels[1])
    plt.plot(metrics["t"], metrics["norm_error_ut_alpha_0_1"], label=plot_labels[2])
    plt.plot(metrics["t"], metrics["norm_error_gmm"], label=plot_labels[3])
    plt.plot(metrics_pri["t"], metrics_pri["norm_error_pinn"], label=plot_labels[4])
    plt.fill_between(
        metrics_pri["t"],
        metrics_pri["norm_error_pinn"],
        metrics_pri["normalize_B1"],
        color=colors_7set[4],
        alpha=0.2,
        label="Error Bound"
    )

    plt.plot(metrics["t"], metrics["norm_error_pinn"], label=plot_labels[5])
    plt.fill_between(
        metrics["t"],
        metrics["norm_error_pinn"],
        metrics["normalize_B1"],
        color=colors_7set[5],
        alpha=0.2,
        label="Error Bound"
    )

    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["norm_error_pinn"], label=plot_labels[6])
    plt.fill_between(
        metrics_pinngmm["t"],
        metrics_pinngmm["norm_error_pinn"],
        metrics_pinngmm["normalize_B1"],
        color=colors_7set[6],
        alpha=0.2,
        hatch='//',                 # tilt/density: '/', '//', '///', etc.
        linewidth=0.0,               # hide polygon outline
        label="Error Bound"
    )
    if(show_flow):
        plt.plot(metrics_flow["t"], metrics_flow["norm_error_pinn"], label=plot_labels[7])
        plt.fill_between(
            metrics_flow["t"],
            metrics_flow["norm_error_pinn"],
            metrics_flow["normalize_B1"],
            color=colors_7set[7],
            alpha=0.2,
            label="Error Bound"
        )
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("Worst Normalized Error %")
    custom_save_plot(True, "figs/metric-WNE.pdf")


    # metric 2: total variation %
    set_publication_plot_style(font_size=24, pair_line_marker_cycle=True)
    plt.figure()
    plt.plot(metrics["t"], metrics["tv_lp"], label=plot_labels[0])
    plt.plot(metrics["t"], metrics["tv_ut"], label=plot_labels[1])
    plt.plot(metrics["t"], metrics["tv_ut_alpha_0_1"], label=plot_labels[2])
    plt.plot(metrics["t"], metrics["tv_gmm"], label=plot_labels[3])
    plt.plot(metrics_pri["t"], metrics_pri["tv_pinn"], label=plot_labels[4])
    plt.plot(metrics["t"], metrics["tv_pinn"], label=plot_labels[5])
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["tv_pinn"], label=plot_labels[6])
    if(show_flow):
        plt.plot(metrics_flow["t"], metrics_flow["tv_pinn"], label=plot_labels[7])
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("Total Variation %")
    custom_save_plot(True, "figs/metric-TV.pdf")

    # metric 3: negative log liklihood (relative KL)
    set_publication_plot_style(font_size=24, pair_line_marker_cycle=True)
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_lp"], label=plot_labels[0])
    plt.plot(metrics["t"], metrics["g_kl_ut"], label=plot_labels[1])
    plt.plot(metrics["t"], metrics["g_kl_ut_alpha_0_1"], label=plot_labels[2])
    plt.plot(metrics["t"], metrics["g_kl_gmm"], label=plot_labels[3])
    plt.plot(metrics_pri["t"], metrics_pri["g_kl_pinn"], label=plot_labels[4])
    plt.plot(metrics["t"], metrics["g_kl_pinn"], label=plot_labels[5])
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["g_kl_pinn"], label=plot_labels[6])
    if(show_flow):
        plt.plot(metrics_flow["t"], metrics_flow["g_kl_pinn"], label=plot_labels[7])
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("Relative Divergence")
    custom_save_plot(True, "figs/metric-RD.pdf")

    plt.show()


def main():
    plot_metrics()


if __name__ == "__main__":
    main()