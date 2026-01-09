import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal

import sys
sys.path.insert(0, '../utilities/')
from _General.baseline_methods import PropagationData
from _General.classic_gmm import make_gmm_pdf
from _General.util import set_publication_plot_style, custom_save_plot

from pinn_model import PNet
from test_gmmpnet import TimeToGMM1D


# --------------------------
# User constants (keep consistent with your setup)
# --------------------------
const_mu = -2.0
const_std = 0.5
x_low, x_hig = -6.0, 6.0

def p_init(x_np: np.ndarray) -> np.ndarray:
    return np.exp(-0.5*((x_np-const_mu)/const_std)**2) / (const_std*np.sqrt(2*np.pi))

def get_p_normalize() -> float:
    x = np.linspace(x_low, x_hig, 200)
    return float(np.max(np.abs(p_init(x))))

def load_model(net, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net

@torch.no_grad()
def eval_pnet_1d(pnet, x_np: np.ndarray, t: float, device) -> np.ndarray:
    x = torch.from_numpy(x_np.reshape(-1, 1)).float().to(device)
    tt = torch.full_like(x, float(t))
    return pnet(x, tt).detach().cpu().numpy().reshape(-1,)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- style you requested ----
    MC_COLOR = "#0066FF"

    GA = dict(color="#FF008C", linestyle=(0, (5, 2)),        marker="o")
    UT = dict(color="#00FBFF", linestyle=(0, (7, 2, 3, 2)),  marker="s")
    GMM= dict(color="#FF8400", linestyle="--",               marker="^")

    PRIOR = dict(color="#00FF1E", linestyle=(0, (9, 2, 1, 2)), marker="v")
    PINN  = dict(color="#8C00FF", linestyle="-.",              marker="D")
    PGMM  = dict(color="#000000", linestyle="-",               marker=None)

    # ---- grids / times ----
    x = np.load("data/xsim.npy").astype(np.float32).reshape(-1,)
    t_plot = np.arange(0.0, 5.0 + 1.0, 1.0).astype(np.float32)
    gap = 5  # same idea as your snippet

    # ---- load baseline propagation outputs ----
    data_ga  = PropagationData(path=os.path.join("data", "lp_np64_dt6.npz"))   # GA
    data_ut  = PropagationData(path=os.path.join("data", "ut_np64_dt6.npz"))   # UT
    data_gmm = PropagationData(path=os.path.join("data", "gmm_np64_dt6.npz"))  # GMM

    # ---- load pre-trained PINN models ----
    scale = get_p_normalize()
    p_prior = load_model(PNet(scale=scale).to(device), "data/p_net(prior).pth", device)
    p_pinn  = load_model(PNet(scale=scale).to(device), "data/p_net.pth",        device)
    p_pgmm  = load_model(TimeToGMM1D().to(device),     "data/p_net(pinn-gmm).pth", device)

    # ---- plot (3D: t, x, p) ----
    set_publication_plot_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for t in t_plot:
        # MC "true"
        if np.isclose(t, 0.0):
            pdf_mc = p_init(x).astype(np.float32)
        else:
            pdf_mc = np.load(f"data/psim_t{t:.1f}.npy").astype(np.float32).reshape(-1,)

        # GA / UT
        _, _, mu_ga, cov_ga = data_ga.get(float(t))
        pdf_ga = multivariate_normal(mean=mu_ga, cov=cov_ga).pdf(x).astype(np.float32)

        _, _, mu_ut, cov_ut = data_ut.get(float(t))
        pdf_ut = multivariate_normal(mean=mu_ut, cov=cov_ut).pdf(x).astype(np.float32)

        # GMM
        _, w, mu, cov = data_gmm.get(float(t))
        pdf_gmm = make_gmm_pdf(w, mu, cov)(x).astype(np.float32)

        # PINNs
        pdf_prior = eval_pnet_1d(p_prior, x, float(t), device).astype(np.float32)
        pdf_pinn  = eval_pnet_1d(p_pinn,  x, float(t), device).astype(np.float32)
        pdf_pgmm  = eval_pnet_1d(p_pgmm,  x, float(t), device).astype(np.float32)

        tt_full = np.full_like(x, t, dtype=np.float32)
        tt_sub  = np.full_like(x[::gap], t, dtype=np.float32)

        # MC (exact style you asked)
        ax.plot(tt_sub, x[::gap], pdf_mc[::gap],
                color=MC_COLOR, linestyle="-", marker="o",
                linewidth=2.0, markersize=6.0)

        # Baselines (colors/linestyles/markers you asked)
        ax.plot(tt_full, x, pdf_ga,  linewidth=1.0, markersize=1.0, markevery=gap, **GA)
        ax.plot(tt_full, x, pdf_ut,  linewidth=1.0, markersize=1.0, markevery=gap, **UT)
        ax.plot(tt_full, x, pdf_gmm, linewidth=1.0, markersize=1.0, markevery=gap, **GMM)

        # PINN family
        ax.plot(tt_full, x, pdf_prior, linewidth=2.0, markersize=3.0, markevery=gap, **PRIOR)
        ax.plot(tt_full, x, pdf_pinn,  linewidth=2.0, markersize=3.0, markevery=gap, **PINN)
        ax.plot(tt_full, x, pdf_pgmm,  linewidth=2.0, **PGMM)  # marker None

    # Legend (match styles)
    handles = [
        Line2D([0],[0], color=MC_COLOR, linestyle="-", marker="o", label=r"$p$ (MC)"),
        Line2D([0],[0], label="GA",  **GA),
        Line2D([0],[0], label="UT",  **UT),
        Line2D([0],[0], label="GMM", **GMM),
        Line2D([0],[0], label="PINN-MLP (prior)",    **PRIOR),
        Line2D([0],[0], label="PINN-MLP", **PINN),
        Line2D([0],[0], label="PINN-GMM", **PGMM),
    ]
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    # ax.set_zlabel("p(x,t)")
    ax.text2D(0.99, 0.8, "PDF", transform=ax.transAxes,
          ha="center", va="bottom")
    ax.legend(handles=handles, loc="upper left", ncol=2)

    out_path = "figs/pdfs.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
