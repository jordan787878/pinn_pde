import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import sys
sys.path.insert(0, '../utilities/')
from _General.util import set_publication_plot_style 


def plot_metrics():
    # colors = sns.color_palette(6)
    colors = sns.color_palette([
        "#000000",  
        "#8C00FF",  
        "#00FF1E",  # orange
        "#FF008C",  # purple
        "#00FBFF",  # green
        "#FF8400",  # brown
        "#999999",  # gray
    ])

    metrics = np.load("data/metrics.npz")
    metrics_pri = np.load("data/metrics(prior).npz")
    metrics_pinngmm = np.load("data/metrics(pinn-gmm).npz")

    set_publication_plot_style(font_size=14)

    plot_labels = ["LP", "UT, "+r"$\alpha=10^{-3}$", "UT, "+r"$\alpha=1.0$", "GMM", "PINN-MLP (prior)", "PINN-MLP", "PINN-GMM"]

    # metric 1: worst relative error %
    plt.figure()
    plt.plot(metrics["t"], metrics["norm_error_lp"], color=colors[3], label=plot_labels[0])
    plt.plot(metrics["t"], metrics["norm_error_ut"], color=colors[4], label=plot_labels[1])
    plt.plot(metrics["t"], metrics["norm_error_ut_alpha_0_1"], color=colors[4], marker="o", label=plot_labels[2])
    plt.plot(metrics["t"], metrics["norm_error_gmm"], color=colors[5], label=plot_labels[3])
    plt.plot(metrics_pri["t"], metrics_pri["norm_error_pinn"], color=colors[2], linestyle="-", label=plot_labels[4])
    plt.fill_between(
        metrics_pri["t"],
        metrics_pri["norm_error_pinn"],
        metrics_pri["normalize_B1"],
        color=colors[2],
        alpha=0.2,
        label="Error Bound"
    )

    plt.plot(metrics["t"], metrics["norm_error_pinn"], color=colors[1], label=plot_labels[5])
    plt.fill_between(
        metrics["t"],
        metrics["norm_error_pinn"],
        metrics["normalize_B1"],
        color=colors[1],
        alpha=0.2,
        label="Error Bound"
    )

    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["norm_error_pinn"], color=colors[0], linestyle="-", label=plot_labels[6])
    plt.fill_between(
        metrics_pinngmm["t"],
        metrics_pinngmm["norm_error_pinn"],
        metrics_pinngmm["normalize_B1"],
        color=colors[0],
        alpha=0.2,
        label="Error Bound"
    )

    plt.grid(True)
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("norm. worst error %")


    # metric 2: total variation %
    plt.figure()
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[3], label=plot_labels[0])
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[4], label=plot_labels[1])
    plt.plot(metrics["t"], metrics["tv_ut_alpha_0_1"], color=colors[4], marker="o", label=plot_labels[2])
    plt.plot(metrics["t"], metrics["tv_gmm"], color=colors[5], label=plot_labels[3])
    plt.plot(metrics_pri["t"], metrics_pri["tv_pinn"], color=colors[2], linestyle="-", label=plot_labels[4])
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[1], label=plot_labels[5])
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["tv_pinn"], color=colors[0], linestyle="-", label=plot_labels[6])

    plt.legend(ncol=2)
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("total variation %")

    # metric 3: negative log liklihood (relative KL)
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_lp"], color=colors[3], label=plot_labels[0])
    plt.plot(metrics["t"], metrics["g_kl_ut"], color=colors[4], label=plot_labels[1])
    plt.plot(metrics["t"], metrics["g_kl_ut_alpha_0_1"], color=colors[4], marker="o", label=plot_labels[2])
    plt.plot(metrics["t"], metrics["g_kl_gmm"], color=colors[5], label=plot_labels[3])
    plt.plot(metrics_pri["t"], metrics_pri["g_kl_pinn"], color=colors[2], linestyle="-", label=plot_labels[4])
    plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors[1], label=plot_labels[5])
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["g_kl_pinn"], color=colors[0], linestyle="-", label=plot_labels[6])
    plt.legend(ncol=2)
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("general KL")

    plt.show()


def main():
    plot_metrics()


if __name__ == "__main__":
    main()