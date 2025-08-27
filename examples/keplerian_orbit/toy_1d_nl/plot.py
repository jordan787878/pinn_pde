import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def plot_metrics():
    colors = sns.color_palette("husl", 3)
    metrics = np.load("data/metrics.npz")
    metrics_pri = np.load("data/metrics(prior).npz")

    # metric 1: worst relative error %
    plt.figure()
    plt.plot(metrics["t"], metrics["norm_error_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    # plt.plot(metrics["t"], metrics["normalize_B1"], color=colors[0], label=r"$Error Bound$ PINN")
    plt.fill_between(
        metrics["t"],
        metrics["norm_error_pinn"],
        metrics["normalize_B1"],
        color=colors[0],
        alpha=0.5,
        label="PINN Error Bound"
    )
    plt.plot(metrics_pri["t"], metrics_pri["norm_error_pinn"], color=colors[0], 
             linestyle="--",
             label=r"$\hat{p}$ PINN (prior)")
    plt.fill_between(
        metrics_pri["t"],
        metrics_pri["norm_error_pinn"],
        metrics_pri["normalize_B1"],
        color=colors[0],
        alpha=0.2,
        label="PINN Error Bound (prior)"
    )
    plt.plot(metrics["t"], metrics["norm_error_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["norm_error_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["norm_error_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("norm. worst error %")

    # metric 2: total variation %
    plt.figure()
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics_pri["t"], metrics_pri["tv_pinn"], color=colors[0], 
             linestyle="--",
             label=r"$\hat{p}$ PINN (prior)")
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["tv_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("total variation %")

    # metric 3: negative log liklihood (relative KL)
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics_pri["t"], metrics_pri["g_kl_pinn"], color=colors[0], 
            linestyle="--",
            label=r"$\hat{p}$ PINN (prior)")
    plt.plot(metrics["t"], metrics["g_kl_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["g_kl_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["g_kl_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("general KL")

    plt.show()


def main():
    plot_metrics()


if __name__ == "__main__":
    main()