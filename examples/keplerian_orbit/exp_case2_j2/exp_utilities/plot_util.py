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

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.classic_gmm import make_gmm_pdf, GMMWhitenedModel
from utilities._General.util import (set_publication_plot_style, colors_4set, 
                                     tidy_corner_axes, apply_default_locators, custom_save_plot,
                                     linestyles_4set, markers_4set)


x_axis_labels=[r"$r'$", r"$\phi'$", r"$v'_r$", r"$v'_{\phi}$"]


def plot_full_corner(constants, t, X_samples,
    bins=256,
    labels=None,
    figsize_per_dim=1.5,
    max_points=500_000,
    mode="heatmap",            # "scatter" or "heatmap"
    cmap="bwr",
    alpha=0.4,
    point_size=2,
    ranges="fixed",               # list of (lo,hi) per dim; if None -> data-driven
    pad_frac=0.02,
    log_counts=True,           # log color scale for heatmap
    PNet_XL_PATH=None,
    p_net_gmm_N1=None,
    p_net_gmm=None,
    data_lp=None,
    data_ut=None,
    data_gmm=None,
    save_plot=False,
):
    set_publication_plot_style(font_size=14)

    X = np.asarray(X_samples)
    N, D = X.shape
    if labels is None:
        labels = [f"x{i+1}" for i in range(D)]

    # --- fixed ranges ---
    if ranges == "fixed":
        ranges = [
            (constants.N_X1_RANGE[0],constants.N_X1_RANGE[1]),
            (constants.N_X2_RANGE[0],constants.N_X2_RANGE[1]),
            (constants.N_X3_RANGE[0],constants.N_X3_RANGE[1]),
            (constants.N_X4_RANGE[0],constants.N_X4_RANGE[1]),
        ]
    # ---- auto ranges ----
    elif ranges == "auto":
        # data-driven min/max
        lo = np.min(X, axis=0)
        hi = np.max(X, axis=0)
        pad_frac = 0.03
        D = X.shape[1]
        # allow pad_frac to be a scalar or per-dimension iterable
        if np.isscalar(pad_frac):
            pad_frac_arr = np.full(D, float(pad_frac))
        else:
            pad_frac_arr = np.asarray(pad_frac, dtype=float).reshape(-1)
            if pad_frac_arr.size != D:
                raise ValueError(f"pad_frac has length {pad_frac_arr.size}, expected {D}")

        span = hi - lo
        # prevent zero span from collapsing range
        eps = 1e-12
        pad = pad_frac_arr * np.maximum(span, eps)
        ranges = [(float(lo[i] - pad[i]), float(hi[i] + pad[i])) for i in range(D)]
    else:
        raise("range for full corner plot not implemented.")

    fig, axes = plt.subplots(
        D, D,
        figsize=(figsize_per_dim*D, figsize_per_dim*D),
        constrained_layout=True,
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}   # small gaps
    )

    # Optional: trim outer margins further (constrained_layout respects these)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    # def _plot_pinn_1d_marginal(x_coords):
    #     filename = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{x_coords}_t{t:.3f}.npz"
    #     if(os.path.exists(filename)):
    #         data_marginal_pinn = np.load(filename)
    #         pdf_values = data_marginal_pinn['pdf']
    #         X_grid = data_marginal_pinn["X_grid"]
    #         ax.plot(X_grid, pdf_values, color=colors_4set[1])

    # Diagonals: 1D histograms (counts)
    for d in range(D):
        ax = axes[d, d]
        lo, hi = ranges[d]

        ax.hist(
            X[:, d],
            bins=bins,
            range=(lo, hi),
            histtype="stepfilled",
            alpha=1.0,
            color=plt.cm.bwr(0.35),     # blue from "bwr"
            edgecolor=plt.cm.bwr(0.35), # optional
            density=True,
        )
        
        # ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if d == D - 1: ax.set_xlabel(labels[d])
        else: ax.set_xticklabels([])
        if d != 0: ax.set_yticklabels([])
        # hide upper triangle in row d
        for k in range(d + 1, D):
            axes[d, k].axis("off")
        if d == 0:
            ax.set_ylabel(labels[d])
        
        # if(PNet_XL_PATH is not None):
        #     _plot_pinn_1d_marginal(d+1)
        
        if(p_net_gmm is not None):
            x_vals = np.linspace(lo, hi, num=128)
            ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
            ws = ws.detach().cpu().numpy()
            if(d == 0):
                print(ws)
            mus = mus.detach().cpu().numpy()
            covs = covs.detach().cpu().numpy()
            pdf_values = np.copy(x_vals) * 0.0
            for k in range(ws.shape[0]):
                pdf_func = multivariate_normal(mean=mus[k,d], cov=covs[k,d,d])
                p_k = pdf_func.pdf(x_vals).reshape(x_vals.shape)
                pdf_values += ws[k] * p_k
            ax.plot(x_vals, pdf_values, color=colors_4set[3])

        if(data_lp is not None):
            x_vals = np.linspace(lo, hi, num=128)
            _, _, mu_6d, cov_6d = data_lp.get(t)
            pdf_values = np.copy(x_vals) * 0.0
            pdf_func = multivariate_normal(mean=mu_6d[d], cov=cov_6d[d,d])
            pdf_values = pdf_func.pdf(x_vals).reshape(x_vals.shape)
            ax.plot(x_vals, pdf_values, color=colors_4set[0])

        if(data_ut is not None):
            x_vals = np.linspace(lo, hi, num=128)
            _, _, mu_6d, cov_6d = data_ut.get(t)
            pdf_values = np.copy(x_vals) * 0.0
            pdf_func = multivariate_normal(mean=mu_6d[d], cov=cov_6d[d,d])
            pdf_values = pdf_func.pdf(x_vals).reshape(x_vals.shape)
            ax.plot(x_vals, pdf_values, color=colors_4set[1])

    # Off-diagonals: scatter or heatmap (counts)
    for i in range(1, D):
        for j in range(i):
            ax = axes[i, j]
            (xlo, xhi), (ylo, yhi) = ranges[j], ranges[i]

            if mode == "scatter":
                idx = np.arange(N)
                if max_points is not None and N > max_points:
                    idx = np.random.default_rng(0).choice(N, size=max_points, replace=False)
                ax.scatter(
                    X[idx, j], X[idx, i],
                    s=point_size, alpha=alpha, edgecolors="none"
                )
            else:  # heatmap of counts
                H, xedges, yedges = np.histogram2d(
                    X[:, j], X[:, i],
                    bins=bins,
                    range=[(xlo, xhi), (ylo, yhi)],
                )
                H = H.T
                if log_counts:
                    pos = H[H > 0]
                    vmax = float(pos.max()) if pos.size else 1.0
                    vmin = float(pos.min()) if pos.size else 1.0  # >=1 to avoid zeros in LogNorm
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = None
                H_ma = np.ma.masked_invalid(H)           # handle NaNs/Infs
                alpha = np.clip(norm(H_ma).filled(0), 0, 1)  # normalize to [0,1]
                ax.imshow(
                    H, origin="lower",
                    extent=(xlo, xhi, ylo, yhi),
                    aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                    # alpha=alpha
                    alpha=0.5,
                )

            relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
            # def _plot_pinn_contour(x_coords):
            #     filename = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz"
            #     if(os.path.isfile(filename)):
            #         data_marginal_pinn = np.load(filename)
            #         pdf_values = data_marginal_pinn['pdf']
            #         X_grid, Y_grid = data_marginal_pinn["X_grid"], data_marginal_pinn["Y_grid"], 
            #         pdf_max = np.max(pdf_values[pdf_values > 0])
            #         # Define levels as percentages of the maximum value
            #         # For example, levels at 5%, 25%, 50%, 75%, and 95% of the max
            #         # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
            #         levels = relative_levels * pdf_max
            #         # Draw the contour lines with the custom levels
            #         ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors_4set[1]], linewidths=1.)
            #         return levels

            # if(PNet_XL_PATH is not None):
            #     x_coords = (j+1, i+1)
            #     pinn_levels = _plot_pinn_contour(x_coords)

            def _plot_pinn_gmm_contour(ws, mus, covs, x_coords, color):
                # print(ws.shape, mus.shape, covs.shape)
                plot_axes = (x_coords[0]-1, x_coords[1]-1)
                xs = np.linspace(xlo, xhi, num=256, endpoint=True)
                ys = np.linspace(ylo, yhi, num=256, endpoint=True)
                X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
                grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

                pdf_values = np.copy(X_grid) * 0.0
                for k in range(ws.shape[0]):
                    ws_k = ws[k]
                    mus_k = mus[k, :]
                    covs_k = covs[k, :, :]
                    marginal_mu = mus_k[list(plot_axes)]
                    marginal_cov = covs_k[np.ix_(list(plot_axes), list(plot_axes))]
                    pdf_func = make_gmm_pdf(1., marginal_mu, marginal_cov)
                    p_k = pdf_func(grid_pts).reshape(X_grid.shape)
                    # p_k = _p_normal(grid_pts, marginal_mu, marginal_cov).reshape(X_grid.shape)
                    pdf_values = pdf_values + ws_k * p_k
                    # if(ws.shape[0] > 1):
                    #     _pdf_max = np.max(p_k).item()
                    #     _levels = np.array([0.01]) * _pdf_max
                    #     ax.contour(X_grid, Y_grid, p_k, levels=_levels, colors=[color], 
                    #             linewidths=0.3, alpha=0.4)
                # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[color], 
                           linewidths=1.)

            # if(p_net_gmm_N1 is not None):
            #     x_coords = (j+1, i+1)
            #     ws, mus, covs = p_net_gmm_N1.weights_means_covs_at(t)
            #     ws = ws.detach().cpu().numpy()
            #     mus = mus.detach().cpu().numpy()
            #     covs = covs.detach().cpu().numpy()
            #     _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors_4set[3])
                            
            if(p_net_gmm is not None):
                x_coords = (j+1, i+1)
                ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
                ws = ws.detach().cpu().numpy()
                mus = mus.detach().cpu().numpy()
                covs = covs.detach().cpu().numpy()
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors_4set[3])

            if(data_lp is not None):
                x_coords = (j+1, i+1)
                _, _, mu_6d, cov_6d = data_lp.get(t)
                plot_axes = (x_coords[0]-1, x_coords[1]-1)
                marginal_mu = mu_6d[list(plot_axes)]
                marginal_cov = cov_6d[np.ix_(list(plot_axes), list(plot_axes))]
                xs = np.linspace(xlo, xhi, num=256, endpoint=True)
                ys = np.linspace(ylo, yhi, num=256, endpoint=True)
                X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
                grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
                pdf_func = make_gmm_pdf(1., marginal_mu, marginal_cov)
                pdf_values = pdf_func(grid_pts).reshape(X_grid.shape)
                # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max
                # Draw the contour lines with the custom levels
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors_4set[0]], 
                           linewidths=1.)
                del pdf_func, pdf_values
                
            if(data_ut is not None):
                x_coords = (j+1, i+1)
                _, _, mu_6d, cov_6d = data_ut.get(t)
                plot_axes = (x_coords[0]-1, x_coords[1]-1)
                marginal_mu = mu_6d[list(plot_axes)]
                marginal_cov = cov_6d[np.ix_(list(plot_axes), list(plot_axes))]
                xs = np.linspace(xlo, xhi, num=256, endpoint=True)
                ys = np.linspace(ylo, yhi, num=256, endpoint=True)
                X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
                grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
                pdf_func = make_gmm_pdf(1., marginal_mu, marginal_cov)
                pdf_values = pdf_func(grid_pts).reshape(X_grid.shape)
                # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max
                # Draw the contour lines with the custom levels
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors_4set[1]], 
                           linewidths=1.)
                del pdf_func, pdf_values

            if(data_gmm is not None):
                x_coords = (j+1, i+1)
                _, ws, mus, covs = data_gmm.get(t)
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors_4set[2])

            # _set_axis_limits_with_buffer(ax, (xlo, xhi), (ylo, yhi))
            # ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D - 1: ax.set_xlabel(labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(labels[i])
            else: ax.set_yticklabels([])
    
    plt.tick_params(axis='both', which='major', labelsize=8)
    tidy_corner_axes(fig, axes, labels=x_axis_labels)
    save_path = "figs/full_corner_t{:.2f}.pdf".format(t)
    custom_save_plot(save_plot, save_path)


def plot_corner_elem(
    constants, t, X_samples, dims,
    *,
    bins=256,
    labels=None,
    ranges="auto",     # "fixed" (from constants) or "auto"
    pad_frac=0.03,
    cmap="bwr",
    log_counts=True,
    PNet_XL_PATH=None,
    p_net_gmm_N1=None,
    p_net_gmm=None,
    data_lp=None,
    data_ut=None,
    data_gmm=None,
    save_plot=False,
):
    """
    dims = (i,) -> 1D histogram for x_i with model overlays (same colors_4set as full plot)
    dims = (i,j) -> 2D heatmap for (x_i, x_j) with model contour overlays
    Indices in dims are 1-based.
    """
    set_publication_plot_style()
    X = np.asarray(X_samples)
    N, D = X.shape
    idx = tuple(int(k) - 1 for k in dims)
    if labels is None:
        labels = [f"x{k+1}" for k in range(D)]

    hist_color = plt.cm.bwr(0.35)          # keep your diagonal histogram color
    hist_edge  = plt.cm.bwr(0.35)

    # ---------- range helpers ----------
    def _fixed_range(k):
        lo, hi = getattr(constants, f"X{k+1}_RANGE")
        return float(lo), float(hi)

    def _auto_range(arr):
        lo, hi = float(np.min(arr)), float(np.max(arr))
        span = hi - lo
        pad = pad_frac * (span if span > 0 else 1.0)
        return lo - pad, hi + pad

    # ---------- 1D (diagonal) ----------
    if len(idx) == 1:
        k = idx[0]
        lo, hi = (_fixed_range(k) if ranges == "fixed" else _auto_range(X[:, k]))
        fig, ax = plt.subplots()

        ax.hist(
            X[:, k], bins=bins, range=(lo, hi),
            histtype="stepfilled", alpha=1.0, density=True,
            color=hist_color, edgecolor=hist_edge
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_xlabel(x_axis_labels[k]); ax.set_ylabel("density")
        ax.set_xlim(lo, hi)

        # ---- overlays (keep same colors_4set/logic as your full function) ----
        # PINN precomputed 1D
        # def _plot_pinn_1d_marginal(k1b):  # k1b = 1-based
        #     if PNet_XL_PATH is None: return
        #     fn = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{k1b}_t{t:.3f}.npz"
        #     if os.path.exists(fn):
        #         data = np.load(fn)
        #         ax.plot(data["X_grid"], data["pdf"], color=colors_4set[-1])
        # _plot_pinn_1d_marginal(k+1)

        # p_net_gmm (mixture’s 1D marginal)
        x_vals = np.linspace(lo, hi, 256)

        if p_net_gmm is not None:
            ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
            ws, mus, covs = (ws.detach().cpu().numpy(),
                             mus.detach().cpu().numpy(),
                             covs.detach().cpu().numpy())
            pdf = np.zeros_like(x_vals)
            for c in range(ws.shape[0]):
                pdf += ws[c] * multivariate_normal(mean=mus[c, k], cov=covs[c, k, k]).pdf(x_vals)
            ax.plot(x_vals, pdf, color=colors_4set[3])

        # linear prop (Gaussian)
        if data_lp is not None:
            _, _, mu6, P6 = data_lp.get(t)
            ax.plot(x_vals, multivariate_normal(mu6[k], P6[k, k]).pdf(x_vals), color=colors_4set[0])

        # UT (Gaussian)
        if data_ut is not None:
            _, _, mu6, P6 = data_ut.get(t)
            ax.plot(x_vals, multivariate_normal(mu6[k], P6[k, k]).pdf(x_vals), color=colors_4set[1])

        if data_gmm is not None:
            _, ws, mus, covs = data_gmm.get(t)
            pdf = np.zeros_like(x_vals)
            for c in range(ws.shape[0]):
                pdf += ws[c] * multivariate_normal(mean=mus[c, k], cov=covs[c, k, k]).pdf(x_vals)
            ax.plot(x_vals, pdf, color=colors_4set[2])

        return fig, ax

    # ---------- 2D (off-diagonal heatmap) ----------
    i, j = idx
    xlo, xhi = (_fixed_range(i) if ranges == "fixed" else _auto_range(X[:, i]))
    ylo, yhi = (_fixed_range(j) if ranges == "fixed" else _auto_range(X[:, j]))

    H, xe, ye = np.histogram2d(X[:, i], X[:, j], bins=bins,
                               range=[(xlo, xhi), (ylo, yhi)])
    H = H.T

    fig, ax = plt.subplots()
    apply_default_locators(ax)
    set_publication_plot_style()
    if log_counts:
        pos = H[H > 0]
        vmin = float(pos.min()) if pos.size else 1.0
        vmax = float(pos.max()) if pos.size else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    ax.imshow(
        H, origin="lower",
        extent=(xlo, xhi, ylo, yhi),
        aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
        alpha=0.5
    )

    # contour helpers (same levels/colors_4set as your full function)
    relative_levels = np.array([0.01, 0.05, 0.50, 0.95])

    # def _plot_pinn_2d_contour(i1b, j1b):
    #     if PNet_XL_PATH is None: return
    #     fn = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{i1b}_x{j1b}_t{t:.3f}.npz"
    #     if os.path.isfile(fn):
    #         data = np.load(fn)
    #         pdf = data["pdf"]
    #         Xg, Yg = data["X_grid"], data["Y_grid"]
    #         pdf_max = float(np.max(pdf[pdf > 0])) if np.any(pdf > 0) else 1.0
    #         levels = relative_levels * pdf_max
    #         ax.contour(Xg, Yg, pdf, levels=levels, colors=[colors_4set[1]], linewidths=1.0)

    def _plot_gmm_contour(ws, mus, covs, color):
        xs = np.linspace(xlo, xhi, 256)
        ys = np.linspace(ylo, yhi, 256)
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([Xg.ravel(), Yg.ravel()])
        pdf = np.zeros_like(Xg, dtype=float)
        for c in range(ws.shape[0]):
            mu_ij = mus[c, [i, j]]
            cov_ij = covs[c][np.ix_([i, j], [i, j])]
            pdf += ws[c] * multivariate_normal(mu_ij, cov_ij).pdf(pts).reshape(Xg.shape)
        pdf_max = float(np.max(pdf)) if np.any(pdf > 0) else 1.0
        levels = relative_levels * pdf_max
        ax.contour(Xg, Yg, pdf, levels=levels, colors=[color], linewidths=1.5)

    # PINN 2D
    # _plot_pinn_2d_contour(i+1, j+1)

    # # p_net_gmm_N1
    # if p_net_gmm_N1 is not None:
    #     ws, mus, covs = p_net_gmm_N1.weights_means_covs_at(t)
    #     _plot_gmm_contour(ws.detach().cpu().numpy(),
    #                       mus.detach().cpu().numpy(),
    #                       covs.detach().cpu().numpy(),
    #                       colors_4set[0])

    # p_net_gmm
    if p_net_gmm is not None:
        ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
        _plot_gmm_contour(ws.detach().cpu().numpy(),
                          mus.detach().cpu().numpy(),
                          covs.detach().cpu().numpy(),
                          colors_4set[3])

    # linear prop (Gaussian)
    if data_lp is not None:
        _, _, mu6, P6 = data_lp.get(t)
        ws = np.array([1.0])
        mus = np.array([mu6])
        covs = np.array([P6])
        _plot_gmm_contour(ws, mus, covs, colors_4set[0])

    # UT (Gaussian)
    if data_ut is not None:
        _, _, mu6, P6 = data_ut.get(t)
        ws = np.array([1.0])
        mus = np.array([mu6])
        covs = np.array([P6])
        _plot_gmm_contour(ws, mus, covs, colors_4set[1])

    # external GMM provider (same color as UT in your full code)
    if data_gmm is not None:
        _, ws, mus, covs = data_gmm.get(t)
        _plot_gmm_contour(ws, mus, covs, colors_4set[2])

    # manual legends
    cmap = plt.get_cmap('bwr')
    c_low  = cmap(0.08)   # low density
    c_high = cmap(0.92)   # high density
    spacer = Line2D([0], [0], linestyle='None', marker=None, alpha=0.0, label='')
    handles = [
        Line2D([0], [0], color=colors_4set[0], lw=2, label='GA'),
        Line2D([0], [0], color=colors_4set[1], lw=2, label='UT'),
        Line2D([0], [0], color=colors_4set[2], lw=2, label='GMM'),
        Line2D([0], [0], color=colors_4set[3], lw=2, label='PINN-GMM'),
        
        # # Density legend entries (marker-only squares)
        # Line2D([0],[0], linestyle='None', marker='s', markersize=12,
        #     markerfacecolor=c_low, markeredgecolor=c_low,
        #     alpha=0.8, label='Ref. PDF, low  density'),
        # Line2D([0],[0], linestyle='None', marker='s', markersize=12,
        #     markerfacecolor=c_high, markeredgecolor=c_high,
        #     alpha=0.8, label='Ref. PDF, high density'),
        # spacer,
        # spacer
    ]
    ax.legend(handles=handles, ncol=2, loc="best")

    ax.set_xlabel(x_axis_labels[i]); ax.set_ylabel(x_axis_labels[j])
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    save_path = "figs/corner_element_t{:.2f}.pdf".format(t)
    custom_save_plot(save_plot, save_path)


def plot_pdf_metrics(metrics, save_plot=False):
    set_publication_plot_style(pair_line_marker_cycle=True)
    plt.figure()
    print(metrics["t"])
    plt.plot(metrics["t"], metrics["rel_error_lp"])
    plt.plot(metrics["t"], metrics["rel_error_ut"])
    plt.plot(metrics["t"], metrics["rel_error_gmm"])
    # plt.plot(metrics["t"], metrics["rel_error_pinn"], color=colors_4set[1])
    # all_zeros = not np.any(metrics["B1_pinn"])
    # if(all_zeros is False):
    #     plt.fill_between(
    #         metrics["t"],
    #         metrics["rel_error_pinn"],
    #         metrics["B1_pinn"],
    #         color=colors_4set[1],
    #         alpha=0.2,
    #     )
    plt.plot(metrics["t"], metrics["rel_error_pinngmm"])
    all_zeros = not np.any(metrics["B1_pinngmm"])
    if(all_zeros is False):
        plt.fill_between(
            metrics["t"],
            metrics["rel_error_pinngmm"]*0.0,
            metrics["B1_pinngmm"],
            alpha=0.3,
            color=colors_4set[3],
            # hatch='//',                 # tilt/density: '/', '//', '///', etc.
            # linewidth=0.0               # hide polygon outline
        )
    # manual legend
    spacer = Line2D([0], [0], linestyle='None', marker=None, alpha=0.0, label='')
    handles = [
        Line2D([0], [0], color=colors_4set[0], lw=2, 
               linestyle=linestyles_4set[0], marker=markers_4set[0], 
               label='GA'),
        Line2D([0], [0], color=colors_4set[1], lw=2, 
               linestyle=linestyles_4set[1], marker=markers_4set[1], 
               label='UT'),
        Line2D([0], [0], color=colors_4set[2], lw=2, 
               linestyle=linestyles_4set[2], marker=markers_4set[2], 
               label='GMM'),

        # Line2D([0], [0], color=colors_4set[1], lw=2, 
        #        linestyle=linestyles_6set[3], marker=markers_6set[3], 
        #        label='PINN-MLP'),
        # Patch(facecolor=colors_4set[1], edgecolor='none',
        #       alpha=0.20, label='Error Bound'),

        Line2D([0], [0], color=colors_4set[3], lw=2, 
               linestyle=linestyles_4set[3], marker=markers_4set[3], 
               label='PINN-GMM'),
        Patch(facecolor=colors_4set[3], edgecolor=colors_4set[3],
                        alpha=0.3,
                        label='Error Bound')
    ]
    plt.legend(handles=handles, ncol=2)
    plt.xlabel("t")
    plt.ylabel("Worst Normalized Error %")
    ymin = np.min(np.array([metrics["rel_error_lp"].min().item(),
                            metrics["rel_error_ut"].min().item(),
                            metrics["rel_error_gmm"].min().item(),
                            ]))
    ymax = np.max(np.array([metrics["rel_error_lp"].max().item(),
                            metrics["rel_error_ut"].max().item(),
                            metrics["rel_error_gmm"].max().item(),
                            ]))
    # print(ymin, ymax)
    plt.ylim([ymin, ymax])
    save_path = "figs/metric-WNE.pdf"
    custom_save_plot(save_plot, save_path)

    plt.figure()
    plt.plot(metrics["t"], metrics["tv_lp"],   label="GA")
    plt.plot(metrics["t"], metrics["tv_ut"], label="UT")
    plt.plot(metrics["t"], metrics["tv_gmm"], label="GMM")
    # plt.plot(metrics["t"], metrics["tv_pinn"], color=colors_4set[1], label="PINN-MLP")
    plt.plot(metrics["t"], metrics["tv_pinngmm"], label="PINN-GMM")
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("Total Variation %")
    ymin = np.min(np.array([metrics["tv_lp"].min().item(),
                            metrics["tv_ut"].min().item(),
                            metrics["tv_gmm"].min().item(),
                            ]))
    ymax = np.max(np.array([metrics["tv_lp"].max().item(),
                            metrics["tv_ut"].max().item(),
                            metrics["tv_gmm"].max().item(),
                            ]))
    # print(ymin, ymax)
    plt.ylim([ymin, ymax])
    save_path = "figs/metric-TV.pdf"
    custom_save_plot(save_plot, save_path)

    # metric 3: negative log liklihood (relative KL)
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_lp"], label="GA")
    plt.plot(metrics["t"], metrics["g_kl_ut"], label="UT")
    plt.plot(metrics["t"], metrics["g_kl_gmm"], label="GMM")
    # plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors_4set[1], label="PINN-MLP")
    plt.plot(metrics["t"], metrics["g_kl_pinngmm"], label="PINN-GMM")
    plt.legend(ncol=2)
    plt.xlabel("t")
    plt.ylabel("Relative Divergence")
    ymin = np.min(np.array([metrics["g_kl_lp"].min().item(),
                            metrics["g_kl_ut"].min().item(),
                            metrics["g_kl_gmm"].min().item(),
                            ]))
    ymax = np.max(np.array([metrics["g_kl_lp"].max().item(),
                            metrics["g_kl_ut"].max().item(),
                            metrics["g_kl_gmm"].max().item(),
                            ]))
    # print(ymin, ymax)
    plt.ylim([ymin, ymax])
    save_path = "figs/metric-RD.pdf"
    custom_save_plot(save_plot, save_path)

    # # Top-512 Error vs PINN Error (Not Good....)
    # i_plot = [1, 3, 5]   # change if you want other times
    # set_publication_plot_style(save_tight_pad=0.3)
    # fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    # plot_idx = 0
    # for i in i_plot:
    #     ax = axes[plot_idx]
    #     top_delta = metrics["topK_delta_p_pinngmm"][i]
    #     top_e1    = metrics["topK_e1_pinngmm"][i]
    #     x = np.arange(len(top_delta))

    #     # --- plot on this subplot ---
    #     # true error (subsampled, with markers)
    #     ax.plot(x, top_delta,
    #             color="#0066FF", linestyle="-", marker="o", markersize=3,
    #             label=r"$|e|$")

    #     # PINN error approximation (full line)
    #     ax.plot(x, top_e1,
    #             color=colors_4set[3], linestyle=linestyles_4set[3],
    #             label=r"$|\hat{e}|$")
        
    #     ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)

    #     t = metrics["t"][i]
    #     ax.set_title(rf"$t = {t:.2f}T$", pad=4)
    #     ax.set_xlabel("State index")

    #     if plot_idx == 0:
    #         ax.set_ylabel("Error (Top-512)")
    #         ax.legend(loc="best")
    #     plot_idx += 1

    plt.show()


def plot_event_triptych_simple(constants, X_event, t, p_net_gmm=None, 
        data_lp=None, data_ut=None, data_gmm=None):
    """
    X_event: (D,2) array for x = [a, P1, P2, Q1, Q2, lambda], D>=6.
    Shows 1x3 panels for (a,lambda), (P1,P2), (Q1,Q2).
    """   
    D = 4 
    def _plot_pinn_gmm_contour(x_ranges, ws, mus, covs, plot_axes, color):
        xlo = x_ranges[plot_axes[0], 0]
        xhi = x_ranges[plot_axes[0], 1]
        ylo = x_ranges[plot_axes[1], 0]
        yhi = x_ranges[plot_axes[1], 1]
        relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
        xs = np.linspace(xlo, xhi, num=256, endpoint=True)
        ys = np.linspace(ylo, yhi, num=256, endpoint=True)
        X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
        grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        pdf_values = np.copy(X_grid) * 0.0
        for k in range(ws.shape[0]):
            ws_k = ws[k]
            mus_k = mus[k, :]
            covs_k = covs[k, :, :]
            marginal_mu = mus_k[list(plot_axes)]
            marginal_cov = covs_k[np.ix_(list(plot_axes), list(plot_axes))]
            pdf_func = make_gmm_pdf(1., marginal_mu, marginal_cov)
            p_k = pdf_func(grid_pts).reshape(X_grid.shape)
            pdf_values = pdf_values + ws_k * p_k
        pdf_max = np.max(pdf_values).item()
        levels = relative_levels * pdf_max
        ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[color], 
                    linewidths=1.)
        
    def _plot_gmm_contour(x_ranges, ws, mus, covs, plot_axes, color):
        xlo = x_ranges[plot_axes[0], 0]
        xhi = x_ranges[plot_axes[0], 1]
        ylo = x_ranges[plot_axes[1], 0]
        yhi = x_ranges[plot_axes[1], 1]
        relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
        xs = np.linspace(xlo, xhi, num=256, endpoint=True)
        ys = np.linspace(ylo, yhi, num=256, endpoint=True)
        X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
        grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        pdf_values = np.copy(X_grid) * 0.0
        for k in range(ws.shape[0]):
            ws_k = ws[k]
            mus_k = mus[k, :]
            covs_k = covs[k, :, :]
            marginal_mu = mus_k[list(plot_axes)]
            marginal_cov = covs_k[np.ix_(list(plot_axes), list(plot_axes))]
            pdf_func = make_gmm_pdf(1., marginal_mu, marginal_cov)
            p_k = pdf_func(grid_pts).reshape(X_grid.shape)
            pdf_values = pdf_values + ws_k * p_k
        pdf_max = np.max(pdf_values).item()
        levels = relative_levels * pdf_max
        ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[color], 
                   linewidths=1.)

    x_ranges = constants._NX_RANGE_NP
          
    pairs  = [(0, 1), (2, 3)]
    labels = ['r', 'ph', 'r_dot', 'phi_dot']

    set_publication_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

    for ax, (ix, iy) in zip(axs, pairs):
        x0, x1 = X_event[ix]
        y0, y1 = X_event[iy]
        print(x0, x1, y0, y1)
        # simple 5% padding (fallback 1.0 if zero-width)
        px = 0.05 * (x1 - x0 if x1 > x0 else 1.0)
        py = 0.05 * (y1 - y0 if y1 > y0 else 1.0)

        # ax.set_xlim(x0 - px, x1 + px)
        # ax.set_ylim(y0 - py, y1 + py)
        ax.set_xlabel(labels[ix])
        ax.set_ylabel(labels[iy])

        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor='tab:green',
                               linewidth=2.0, linestyle='--'))
        
        if(p_net_gmm is not None):
            ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
            ws = ws.detach().cpu().numpy()
            mus = mus.detach().cpu().numpy()
            covs = covs.detach().cpu().numpy()
            _plot_pinn_gmm_contour(x_ranges, ws, mus, covs, [ix, iy], color=colors_4set[3])

        if(data_lp is not None):
            _, ws, mus, covs = data_lp.get(t)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _plot_gmm_contour(x_ranges, ws, mus, covs, [ix, iy], color=colors_4set[0])

        if(data_ut is not None):
            _, ws, mus, covs = data_ut.get(t)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _plot_gmm_contour(x_ranges, ws, mus, covs, [ix, iy], color=colors_4set[1])

        if(data_gmm is not None):
            _, ws, mus, covs = data_gmm.get(t)
            mus = mus.reshape((-1, D))
            covs = covs.reshape((-1, D, D))
            _plot_gmm_contour(x_ranges, ws, mus, covs, [ix, iy], color=colors_4set[2])


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


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D


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


def plot_marginal_pdf_cart(constants, data_mc, p_net_gmm, num=256):
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
        ws_torch, mus_torch, covs_torch = p_net_gmm.weights_means_covs_at(t)
        ws_approx = ws_torch.detach().cpu().numpy()
        mus_approx = mus_torch.detach().cpu().numpy()
        covs_approx = covs_torch.detach().cpu().numpy()

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
        delta_z = 0.002 * (Z_a.max() - Z_a.min() + 1e-12)  # avoid zero range
        Z_r_plot = Z_r + delta_z

        ax.plot_wireframe(
            X_r,
            Y_r,
            Z_r_plot,          # <-- use lifted version
            rstride=20,
            cstride=20,
            color="k",
            linewidth=1.0,
        )

        # ---- Label time above the reference PDF ----
        # choose the point of maximum reference PDF as anchor
        imax = np.argmax(Z_r_plot)
        ix, iy = np.unravel_index(imax, Z_r_plot.shape)
        x_lbl = X_r[ix, iy]
        y_lbl = Y_r[ix, iy]
        z_lbl = Z_r_plot[ix, iy] + 0.01 * (Z_a.max() - Z_a.min() + 1e-12)

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
    ax.legend(
        [approx_proxy, ref_proxy],
        [r"PINN $\hat p$", r"Ref. $p$"],
        loc="upper left",             # corner inside the axes
        bbox_to_anchor=(0.7, 0.85),  # (x, y) in axes fraction coords
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )
    custom_save_plot(True, "figs/pdf_cart.pdf")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


def animate_pinn_surface_with_static_ref_wireframes(
    constants,
    data_mc,
    p_net_gmm,
    num: int = 256,
    # wireframes (Ref. p): shown ONCE for these times
    t_ref_list=None,
    # surface (hat p): animated on a denser time grid
    t_surf_list=None,
    n_surf_frames: int = 80,
    interval_ms: int = 100,
    save_path: str | None = None,
    dpi: int = 150,
    elev: float = 20,
    azim: float = 66,
    rstride_surf: int = 10,
    cstride_surf: int = 10,
    rstride_wire: int = 24,
    cstride_wire: int = 24,
    # visual separation for multiple wireframes
    wire_lift_frac: float = 0.002,
    wire_stack_lift_frac: float = 0.0004,   # extra lift per ref time index (optional)
    wire_alpha: float = 0.9,
    # fixed axes
    z_pad_frac: float = 0.08,
    xy_pad_frac: float = 0.02,
    time_fmt: str = r"$t={:.2f}T$",
):
    """
    - Draw ALL MC reference PDFs (Ref. p) as static wireframes at t_ref_list (default: constants.T_PRIME_SPAN).
    - Animate ONLY the PINN surface (hat p) over a denser time grid t_surf_list
      (default: linspace over the same time span with n_surf_frames frames).
    - Fix x/y/z limits across all frames.

    Requirements (same as your plot function):
      - _eval_marginal_xy_from_gmm(constants, ws, mus, covs, t, grid_pts, num) -> (X, Y, Z)
      - GMMWhitenedModel.load(npz_path) and gmm.print_x_params()
      - set_publication_plot_style()
    """
    # ----------------------------
    # Time grids
    # ----------------------------
    if t_ref_list is None:
        t_ref_list = list(constants.T_PRIME_SPAN)
    else:
        t_ref_list = list(t_ref_list)

    if t_surf_list is None:
        t0 = float(np.min(t_ref_list))
        t1 = float(np.max(t_ref_list))
        t_surf_list = np.linspace(t0, t1, n_surf_frames, endpoint=True)
    else:
        t_surf_list = np.asarray(t_surf_list, dtype=float)
        n_surf_frames = len(t_surf_list)

    # ----------------------------
    # Grid in normalized (n_r, n_phi)
    # ----------------------------
    n_r_vals = np.linspace(constants.N_X1_RANGE[0], constants.N_X1_RANGE[1], num=num, endpoint=True)
    n_phi_vals = np.linspace(constants.N_X2_RANGE[0], constants.N_X2_RANGE[1], num=num, endpoint=True)
    n_r_grid, n_phi_grid = np.meshgrid(n_r_vals, n_phi_vals, indexing="ij")
    n_rphi_grid_pts = np.vstack([n_r_grid.ravel(), n_phi_grid.ravel()]).T  # (num^2, 2)

    # ----------------------------
    # Precompute static reference wireframes
    # ----------------------------
    ref_surfaces = []
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    z_min, z_max = np.inf, -np.inf
    global_z_range_hat = 0.0

    # Also get a coarse bound on hat-p z-range from the ref times (enough for fixed zlim in practice)
    for t in t_ref_list:
        # --- ref (MC-fit GMM) ---
        gmm = GMMWhitenedModel.load(data_mc + f"gmm_whitened_t{t:.2f}.npz")
        gmm_params = gmm.print_x_params()
        ws_ref = gmm_params["weights"]
        mus_ref = gmm_params["means_x"]
        covs_ref = gmm_params["covs_x"]

        X_r, Y_r, Z_r = _eval_marginal_xy_from_gmm(
            constants, ws_ref, mus_ref, covs_ref, t, n_rphi_grid_pts, num
        )
        ref_surfaces.append((t, X_r, Y_r, Z_r))

        x_min = min(x_min, float(np.nanmin(X_r)))
        x_max = max(x_max, float(np.nanmax(X_r)))
        y_min = min(y_min, float(np.nanmin(Y_r)))
        y_max = max(y_max, float(np.nanmax(Y_r)))
        z_min = min(z_min, float(np.nanmin(Z_r)))
        z_max = max(z_max, float(np.nanmax(Z_r)))

        # --- hat (PINN) at same t (for z-limit calibration) ---
        ws_torch, mus_torch, covs_torch = p_net_gmm.weights_means_covs_at(t)
        ws_hat = ws_torch.detach().cpu().numpy()
        mus_hat = mus_torch.detach().cpu().numpy()
        covs_hat = covs_torch.detach().cpu().numpy()
        _, _, Z_hat = _eval_marginal_xy_from_gmm(
            constants, ws_hat, mus_hat, covs_hat, t, n_rphi_grid_pts, num
        )
        z_min = min(z_min, float(np.nanmin(Z_hat)))
        z_max = max(z_max, float(np.nanmax(Z_hat)))
        global_z_range_hat = max(global_z_range_hat, float(np.nanmax(Z_hat) - np.nanmin(Z_hat)))

    # Fixed limits with padding
    x_pad = xy_pad_frac * (x_max - x_min + 1e-12)
    y_pad = xy_pad_frac * (y_max - y_min + 1e-12)
    z_pad = z_pad_frac * (z_max - z_min + 1e-12)

    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    # A consistent lift to avoid z-fighting
    wire_lift = wire_lift_frac * (global_z_range_hat + 1e-12)

    # Give enough headroom for stacked wireframes + time label
    stack_extra = wire_stack_lift_frac * (global_z_range_hat + 1e-12) * max(0, len(t_ref_list) - 1)
    z_lim = (max(0.0, z_min - z_pad), z_max + wire_lift + stack_extra + 2.0 * z_pad)

    # ----------------------------
    # Figure setup
    # ----------------------------
    set_publication_plot_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev, azim)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

    ax.set_xlabel(r"$x$ [m]", labelpad=15)
    ax.set_ylabel(r"$y$ [m]", labelpad=15)
    ax.set_zlabel(r"$p(x, y)$")

    # Legend proxies
    approx_proxy = Line2D([], [], color=cm.viridis_r(0.15), lw=6)
    ref_proxy = Line2D([], [], color="k", lw=1.5)

    ax.legend(
        [approx_proxy, ref_proxy],
        [r"PINN $\hat p$", r"Ref. $p$ (MC)"],
        loc="upper left",
        bbox_to_anchor=(0.7, 0.85),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )

    # ----------------------------
    # Draw ALL static wireframes (once)
    # ----------------------------
    wire_handles = []
    for k, (t, X_r, Y_r, Z_r) in enumerate(ref_surfaces):
        # optional small stacking lift per time index to reduce overlap
        Z_plot = Z_r + wire_lift + k * (wire_stack_lift_frac * (global_z_range_hat + 1e-12))

        h = ax.plot_wireframe(
            X_r, Y_r, Z_plot,
            rstride=rstride_wire,
            cstride=cstride_wire,
            color="k",
            linewidth=1.0,
            alpha=wire_alpha,
            zorder=2,
        )
        wire_handles.append(h)

    # ----------------------------
    # Animated surface (hat p)
    # ----------------------------
    surf_handle = None
    time_text = None

    def _draw_surface(t):
        nonlocal surf_handle, time_text

        # remove old surface/text (keep wireframes)
        if surf_handle is not None:
            surf_handle.remove()
            surf_handle = None
        if time_text is not None:
            time_text.remove()
            time_text = None

        ws_torch, mus_torch, covs_torch = p_net_gmm.weights_means_covs_at(t)
        ws_hat = ws_torch.detach().cpu().numpy()
        mus_hat = mus_torch.detach().cpu().numpy()
        covs_hat = covs_torch.detach().cpu().numpy()

        X_a, Y_a, Z_a = _eval_marginal_xy_from_gmm(
            constants, ws_hat, mus_hat, covs_hat, t, n_rphi_grid_pts, num
        )

        norm = colors.Normalize(vmin=float(np.nanmin(Z_a)), vmax=float(np.nanmax(Z_a)))
        surf_handle = ax.plot_surface(
            X_a, Y_a, Z_a,
            rstride=rstride_surf,
            cstride=cstride_surf,
            cmap=cm.viridis_r,
            norm=norm,
            linewidth=0,
            antialiased=True,
            shade=False,
            zorder=1,
        )

        # place label at the peak of hat p
        imax = int(np.nanargmax(Z_a))
        ix, iy = np.unravel_index(imax, Z_a.shape)
        x_lbl = float(X_a[ix, iy])
        y_lbl = float(Y_a[ix, iy])
        z_lbl = float(Z_a[ix, iy] + 0.5 * z_pad)

        time_text = ax.text(
            x_lbl, y_lbl, z_lbl,
            time_fmt.format(float(t)),
            fontsize=16,
            ha="center",
            va="bottom",
        )

        # re-assert fixed limits/view (some mpl backends reset them)
        ax.view_init(elev, azim)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)

        return []

    def _update(i):
        return _draw_surface(t_surf_list[i])

    # initial surface
    _draw_surface(t_surf_list[0])

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_surf_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        if save_path.lower().endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", dpi=dpi)
        elif save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", dpi=dpi)
        else:
            raise ValueError("save_path must end with .mp4 or .gif (or set save_path=None).")

    return fig, anim
