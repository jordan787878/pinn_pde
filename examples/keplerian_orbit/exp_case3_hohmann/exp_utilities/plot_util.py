import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
import torch
import seaborn as sns
from scipy.stats import multivariate_normal
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from matplotlib.patches import Patch, Rectangle
from matplotlib import cm, colors

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.classic_gmm import make_gmm_pdf, GMMWhitenedModel
from utilities._General.util import (set_publication_plot_style, colors_4set, 
                                     tidy_corner_axes, apply_default_locators, custom_save_plot,
                                     linestyles_4set, markers_4set)


x_axis_labels=[r"$r'$", r"$\phi'$", r"$v'_r$", r"$v'_{\phi}$"]


def _rnsphere_to_sphere(constants, input_array, t_prime):
    r = input_array[:, 0] * constants.R
    phi = input_array[:, 1] * constants.PHI + constants.W*constants.T*t_prime
    return np.column_stack((r, phi))


def _sphere_to_cartesian(input_array):
    # Unpack input columns for vectorized operations
    r = input_array[:, 0]  # Radial distance
    phi = input_array[:, 1]  # Azimuthal angle (phi)
    # Convert position from spherical to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y))


def map_marginal_pdf_to_cartesian(constants, pdf_vals_n_rphi, r_phi_grid_pts):
    r = r_phi_grid_pts[:, 0]
    pdf_vals = pdf_vals_n_rphi / (r * constants.R * constants.PHI)
    return pdf_vals


def _eval_marginal_xy_from_gmm(constants, ws, mus, covs, t, n_rphi_grid_pts, num,
                               idx_x1=0, idx_x2=1):
    """
    Given GMM params (ws, mus, covs) at time t, return grids (X, Y, Z) for p(x, y).
    ws:  (K,)
    mus: (K, D)
    covs:(K, D, D)
    """
    idx = np.array([idx_x1, idx_x2])

    # (K, 2) and (K, 2, 2) for the (x1, x2) marginal
    mus_2d = mus[:, idx]
    covs_2d = covs[:, idx[:, None], idx[None, :]]

    pdf_func = make_gmm_pdf(ws, mus_2d, covs_2d)   # (N,2) -> (N,)
    pdf_vals_n_rphi = pdf_func(n_rphi_grid_pts)    # (num^2,)

    # rotating spherical -> spherical -> Cartesian
    r_phi_grid_pts = _rnsphere_to_sphere(constants, n_rphi_grid_pts, t)
    xy_grid_pts = _sphere_to_cartesian(r_phi_grid_pts)

    x_grid = xy_grid_pts[:, 0].reshape(num, num)
    y_grid = xy_grid_pts[:, 1].reshape(num, num)

    pdf_vals_xy = map_marginal_pdf_to_cartesian(constants, pdf_vals_n_rphi, r_phi_grid_pts)
    z_grid = pdf_vals_xy.reshape(num, num)

    return x_grid, y_grid, z_grid


def plot_marginal_pdf_cart(constants, data_mc, data_mc_no_thrust, p_net_gmm=None, num=256):
    # grid in normalized (n_r, n_phi)
    n_r_vals  = np.linspace(constants.N_X1_RANGE[0], constants.N_X1_RANGE[1],
                            num=num, endpoint=True)
    n_phi_vals = np.linspace(constants.N_X2_RANGE[0], constants.N_X2_RANGE[1],
                             num=num, endpoint=True)
    n_r_grid, n_phi_grid = np.meshgrid(n_r_vals, n_phi_vals, indexing="ij")
    n_rphi_grid_pts = np.vstack([n_r_grid.ravel(), n_phi_grid.ravel()]).T  # (num^2, 2)

    set_publication_plot_style()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    for t in constants.T_PRIME_SPAN:

        # ---- Approximate PDF from PINN-GMM ----
        if(p_net_gmm is not None):
            ws_torch, mus_torch, covs_torch = p_net_gmm.weights_means_covs_at(t)
            ws_approx = ws_torch.detach().cpu().numpy()
            mus_approx = mus_torch.detach().cpu().numpy()
            covs_approx = covs_torch.detach().cpu().numpy()
            print("[debug] p_net_gmm weights: ", ws_approx)

            X_a, Y_a, Z_a = _eval_marginal_xy_from_gmm(
                constants, ws_approx, mus_approx, covs_approx, t, n_rphi_grid_pts, num
            )

            # clear, bright colormap (local to each surface)
            norm = colors.Normalize(vmin=Z_a.min(), vmax=Z_a.max())

            surf = ax.plot_surface(
                X_a,
                Y_a,
                Z_a,
                rstride=10,
                cstride=10,
                cmap=cm.viridis_r,
                norm=norm,
                linewidth=0,
                antialiased=True,
                shade=False,   # <- prevents extra darkening
                zorder=1,
            )

        # ---- Reference PDF (black wireframe) ----
        gmm = GMMWhitenedModel.load(data_mc + f"gmm_whitened_t{t:.2f}.npz")
        gmm_params = gmm.print_x_params()
        ws_ref = gmm_params["weights"]
        mus_ref = gmm_params["means_x"]
        covs_ref = gmm_params["covs_x"]
        X_r, Y_r, Z_r = _eval_marginal_xy_from_gmm(
            constants, ws_ref, mus_ref, covs_ref, t, n_rphi_grid_pts, num
        )
        # compute a small vertical offset based on the scale of Z
        Z_r_plot = Z_r
        ax.plot_wireframe(
            X_r,
            Y_r,
            Z_r_plot,          # <-- use lifted version
            rstride=20,
            cstride=20,
            color="k",
            linewidth=1.0,
        )

        # ---- Reference PDF (black wireframe) ----
        gmm = GMMWhitenedModel.load(data_mc_no_thrust + f"gmm_whitened_t{t:.2f}.npz")
        gmm_params = gmm.print_x_params()
        ws_ref = gmm_params["weights"]
        mus_ref = gmm_params["means_x"]
        covs_ref = gmm_params["covs_x"]
        X_r, Y_r, Z_r = _eval_marginal_xy_from_gmm(
            constants, ws_ref, mus_ref, covs_ref, t, n_rphi_grid_pts, num
        )
        # compute a small vertical offset based on the scale of Z
        Z_r_plot = Z_r
        ax.plot_wireframe(
            X_r,
            Y_r,
            Z_r_plot,          # <-- use lifted version
            rstride=20,
            cstride=20,
            color="blue",
            linewidth=0.5,
        )

        # ---- Label time above the reference PDF ----
        # choose the point of maximum reference PDF as anchor
        imax = np.argmax(Z_r_plot)
        ix, iy = np.unravel_index(imax, Z_r_plot.shape)
        x_lbl = X_r[ix, iy]
        y_lbl = Y_r[ix, iy]
        z_lbl = Z_r_plot[ix, iy]

        ax.text(
            x_lbl,
            y_lbl,
            z_lbl,
            rf"$t={t:.2f}T$",
            fontsize=16,
            ha="center",
            va="bottom",
        )

    ax.view_init(20, 66)
    ax.set_xlabel(r"$x$ [m]", labelpad=15)
    ax.set_ylabel(r"$y$ [m]", labelpad=15)
    ax.set_zlabel(r"$p(x, y)$")

    approx_proxy = Line2D([], [], color=cm.viridis_r(0.1), lw=6)
    ref_proxy = Line2D([], [], color="k", lw=1.5)
    ref_proxy_no_thrust = Line2D([], [], color="blue", lw=0.5)
    ax.legend(
        [approx_proxy, ref_proxy, ref_proxy_no_thrust],
        [r"PINN $\hat p$", r"Ref. $p$", r"Ref. $p$ (no thrust)"],
        loc="upper left",             # corner inside the axes
        bbox_to_anchor=(0.7, 0.85),  # (x, y) in axes fraction coords
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )
    custom_save_plot(True, "figs/pdf_cart.pdf")
