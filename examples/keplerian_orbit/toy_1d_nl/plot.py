import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


def plot_metrics():
    colors = sns.color_palette("bright", 6)

    metrics = np.load("data/metrics.npz")
    metrics_pri = np.load("data/metrics(prior).npz")
    metrics_pinngmm = np.load("data/metrics(pinn-gmm).npz")

    # metric 1: worst relative error %
    plt.figure()
    plt.plot(metrics["t"], metrics["norm_error_pinn"], color=colors[0], 
             label=r"$\hat{p}$ PINN-MLP")
    plt.plot(metrics_pri["t"], metrics_pri["norm_error_pinn"], color=colors[1], 
             linestyle="-",
             label=r"$\hat{p}$ PINN (prior)")
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["norm_error_pinn"], color=colors[2], 
             linestyle="-",
             label=r"$\hat{p}$ PINN-GMM")
    plt.plot(metrics["t"], metrics["norm_error_lp"], color=colors[3],   label=r"$p$ LP")
    plt.plot(metrics["t"], metrics["norm_error_ut"], color=colors[4],   label=r"$p$ UT")
    plt.plot(metrics["t"], metrics["norm_error_ut_alpha_0_1"], color=colors[4], marker="o", label=r"$p$ UT $(\alpha=0.1)$")
    plt.plot(metrics["t"], metrics["norm_error_gmm"], color=colors[5],   label=r"$p$ LP GMM")
    plt.fill_between(
        metrics["t"],
        metrics["norm_error_pinn"],
        metrics["normalize_B1"],
        color=colors[0],
        alpha=0.2,
        label="PINN-MLP Error Bound"
    )
    plt.fill_between(
        metrics_pri["t"],
        metrics_pri["norm_error_pinn"],
        metrics_pri["normalize_B1"],
        color=colors[1],
        alpha=0.2,
        label="PINN (prior) Error Bound"
    )
    plt.fill_between(
        metrics_pinngmm["t"],
        metrics_pinngmm["norm_error_pinn"],
        metrics_pinngmm["normalize_B1"],
        color=colors[2],
        alpha=0.2,
        label="PINN-GMM Error Bound"
    )
    plt.grid(True)
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("norm. worst error %")

    # metric 2: total variation %
    plt.figure()
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics_pri["t"], metrics_pri["tv_pinn"], color=colors[1], 
             linestyle="-",
             label=r"$\hat{p}$ PINN (prior)")
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["tv_pinn"], color=colors[2], label=r"$\hat{p}$ PINN-GMM")
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[3],   label=r"$p$ LP")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[4],   label=r"$p$ UT")
    plt.plot(metrics["t"], metrics["tv_ut_alpha_0_1"], color=colors[4], marker="o", label=r"$p$ UT $(\alpha=0.1)$")
    plt.plot(metrics["t"], metrics["tv_gmm"], color=colors[5], label=r"$p$ LP GMM")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("total variation %")

    # metric 3: negative log liklihood (relative KL)
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics_pri["t"], metrics_pri["g_kl_pinn"], color=colors[1], 
            linestyle="-",
            label=r"$\hat{p}$ PINN (prior)")
    plt.plot(metrics_pinngmm["t"], metrics_pinngmm["g_kl_lp"], color=colors[2], label=r"$\hat{p}$ PINN-GMM")
    plt.plot(metrics["t"], metrics["g_kl_lp"], color=colors[3],   label=r"$p$ LP")
    plt.plot(metrics["t"], metrics["g_kl_ut"], color=colors[4],   label=r"$p$ UT")
    plt.plot(metrics["t"], metrics["g_kl_ut_alpha_0_1"], color=colors[4], marker="o", label=r"$p$ UT $(\alpha=0.1)$")
    plt.plot(metrics["t"], metrics["g_kl_gmm"], color=colors[5], label=r"$p$ LP GMM")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("general KL")

    plt.show()


def main():
    plot_metrics()


if __name__ == "__main__":
    main()