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

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.util import set_publication_plot_style


colors = sns.color_palette([
    "#000000",  
    "#8C00FF",  
    "#00FF1E",  # orange
    "#FF008C",  # purple
    "#00FBFF",  # green
    "#FF8400",  # brown
    "#999999",  # gray
])


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
):
    set_publication_plot_style(font_size=14)

    X = np.asarray(X_samples)
    N, D = X.shape
    if labels is None:
        labels = [f"x{i+1}" for i in range(D)]

    # --- fixed ranges ---
    if ranges == "fixed":
        ranges = [
            (constants.X1_RANGE[0],constants.X1_RANGE[1]),
            (constants.X2_RANGE[0],constants.X2_RANGE[1]),
            (constants.X3_RANGE[0],constants.X3_RANGE[1]),
            (constants.X4_RANGE[0],constants.X4_RANGE[1]),
            (constants.X5_RANGE[0],constants.X5_RANGE[1]),
            (constants.X6_RANGE[0],constants.X6_RANGE[1]),
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

    def _plot_pinn_1d_marginal(x_coords):
        filename = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{x_coords}_t{t:.3f}.npz"
        if(os.path.exists(filename)):
            data_marginal_pinn = np.load(filename)
            pdf_values = data_marginal_pinn['pdf']
            X_grid = data_marginal_pinn["X_grid"]
            ax.plot(X_grid, pdf_values, color=colors[-1])

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
        
        if(PNet_XL_PATH is not None):
            _plot_pinn_1d_marginal(d+1)
        
        if(p_net_gmm is not None):
            x_vals = np.linspace(lo, hi, num=128)
            ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
            ws = ws.detach().cpu().numpy()
            # if(d == 0):
            #     print(ws)
            mus = mus.detach().cpu().numpy()
            covs = covs.detach().cpu().numpy()
            pdf_values = np.copy(x_vals) * 0.0
            for k in range(ws.shape[0]):
                pdf_func = multivariate_normal(mean=mus[k,d], cov=covs[k,d,d])
                p_k = pdf_func.pdf(x_vals).reshape(x_vals.shape)
                pdf_values += ws[k] * p_k
            ax.plot(x_vals, pdf_values, color=colors[0])

        if(data_lp is not None):
            x_vals = np.linspace(lo, hi, num=128)
            _, _, mu_6d, cov_6d = data_lp.get(t)
            pdf_values = np.copy(x_vals) * 0.0
            pdf_func = multivariate_normal(mean=mu_6d[d], cov=cov_6d[d,d])
            pdf_values = pdf_func.pdf(x_vals).reshape(x_vals.shape)
            ax.plot(x_vals, pdf_values, color=colors[3])

        if(data_ut is not None):
            x_vals = np.linspace(lo, hi, num=128)
            _, _, mu_6d, cov_6d = data_ut.get(t)
            pdf_values = np.copy(x_vals) * 0.0
            pdf_func = multivariate_normal(mean=mu_6d[d], cov=cov_6d[d,d])
            pdf_values = pdf_func.pdf(x_vals).reshape(x_vals.shape)
            ax.plot(x_vals, pdf_values, color=colors[4])

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
            def _plot_pinn_contour(x_coords):
                filename = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz"
                if(os.path.isfile(filename)):
                    data_marginal_pinn = np.load(filename)
                    pdf_values = data_marginal_pinn['pdf']
                    X_grid, Y_grid = data_marginal_pinn["X_grid"], data_marginal_pinn["Y_grid"], 
                    pdf_max = np.max(pdf_values[pdf_values > 0])
                    # Define levels as percentages of the maximum value
                    # For example, levels at 5%, 25%, 50%, 75%, and 95% of the max
                    # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
                    levels = relative_levels * pdf_max
                    # Draw the contour lines with the custom levels
                    ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors[1]], linewidths=1.)
                    return levels

            if(PNet_XL_PATH is not None):
                x_coords = (j+1, i+1)
                pinn_levels = _plot_pinn_contour(x_coords)

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
                    pdf_func = multivariate_normal(mean=marginal_mu, cov=marginal_cov)
                    p_k = pdf_func.pdf(grid_pts).reshape(X_grid.shape)
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

            if(p_net_gmm_N1 is not None):
                x_coords = (j+1, i+1)
                ws, mus, covs = p_net_gmm_N1.weights_means_covs_at(t)
                ws = ws.detach().cpu().numpy()
                mus = mus.detach().cpu().numpy()
                covs = covs.detach().cpu().numpy()
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors[0])
                            
            if(p_net_gmm is not None):
                x_coords = (j+1, i+1)
                ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
                ws = ws.detach().cpu().numpy()
                mus = mus.detach().cpu().numpy()
                covs = covs.detach().cpu().numpy()
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors[0])

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
                pdf_values = _p_normal(grid_pts, marginal_mu, marginal_cov).reshape(X_grid.shape)
                # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max
                # Draw the contour lines with the custom levels
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors[3]], 
                           linewidths=1.)
                
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
                pdf_values = _p_normal(grid_pts, marginal_mu, marginal_cov).reshape(X_grid.shape)
                # relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max
                # Draw the contour lines with the custom levels
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors[4]], 
                           linewidths=1.)

            if(data_gmm is not None):
                x_coords = (j+1, i+1)
                _, ws, mus, covs = data_gmm.get(t)
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors[4])

            # _set_axis_limits_with_buffer(ax, (xlo, xhi), (ylo, yhi))
            # ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D - 1: ax.set_xlabel(labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(labels[i])
            else: ax.set_yticklabels([])
    
    plt.tick_params(axis='both', which='major', labelsize=8)


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
):
    """
    dims = (i,) -> 1D histogram for x_i with model overlays (same colors as full plot)
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
        fig, ax = plt.subplots(figsize=(4, 3))

        ax.hist(
            X[:, k], bins=bins, range=(lo, hi),
            histtype="stepfilled", alpha=1.0, density=True,
            color=hist_color, edgecolor=hist_edge
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_xlabel(labels[k]); ax.set_ylabel("density")
        ax.set_xlim(lo, hi)

        # ---- overlays (keep same colors/logic as your full function) ----
        # PINN precomputed 1D
        def _plot_pinn_1d_marginal(k1b):  # k1b = 1-based
            if PNet_XL_PATH is None: return
            fn = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{k1b}_t{t:.3f}.npz"
            if os.path.exists(fn):
                data = np.load(fn)
                ax.plot(data["X_grid"], data["pdf"], color=colors[-1])

        _plot_pinn_1d_marginal(k+1)

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
            ax.plot(x_vals, pdf, color=colors[0])

        # linear prop (Gaussian)
        if data_lp is not None:
            _, _, mu6, P6 = data_lp.get(t)
            ax.plot(x_vals, multivariate_normal(mu6[k], P6[k, k]).pdf(x_vals), color=colors[3])

        # UT (Gaussian)
        if data_ut is not None:
            _, _, mu6, P6 = data_ut.get(t)
            ax.plot(x_vals, multivariate_normal(mu6[k], P6[k, k]).pdf(x_vals), color=colors[4])

        if data_gmm is not None:
            _, ws, mus, covs = data_gmm.get(t)
            pdf = np.zeros_like(x_vals)
            for c in range(ws.shape[0]):
                pdf += ws[c] * multivariate_normal(mean=mus[c, k], cov=covs[c, k, k]).pdf(x_vals)
            ax.plot(x_vals, pdf, color=colors[5])

        return fig, ax

    # ---------- 2D (off-diagonal heatmap) ----------
    i, j = idx
    xlo, xhi = (_fixed_range(i) if ranges == "fixed" else _auto_range(X[:, i]))
    ylo, yhi = (_fixed_range(j) if ranges == "fixed" else _auto_range(X[:, j]))

    H, xe, ye = np.histogram2d(X[:, i], X[:, j], bins=bins,
                               range=[(xlo, xhi), (ylo, yhi)])
    H = H.T

    fig, ax = plt.subplots(figsize=(4, 4))
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

    # contour helpers (same levels/colors as your full function)
    relative_levels = np.array([0.01, 0.05, 0.50, 0.95])

    def _plot_pinn_2d_contour(i1b, j1b):
        if PNet_XL_PATH is None: return
        fn = f"{PNet_XL_PATH}/pre_compute/marginal_pdfpinn_x{i1b}_x{j1b}_t{t:.3f}.npz"
        if os.path.isfile(fn):
            data = np.load(fn)
            pdf = data["pdf"]
            Xg, Yg = data["X_grid"], data["Y_grid"]
            pdf_max = float(np.max(pdf[pdf > 0])) if np.any(pdf > 0) else 1.0
            levels = relative_levels * pdf_max
            ax.contour(Xg, Yg, pdf, levels=levels, colors=[colors[1]], linewidths=1.0)

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
        ax.contour(Xg, Yg, pdf, levels=levels, colors=[color], linewidths=1.0)

    # PINN 2D
    _plot_pinn_2d_contour(i+1, j+1)

    # p_net_gmm_N1
    if p_net_gmm_N1 is not None:
        ws, mus, covs = p_net_gmm_N1.weights_means_covs_at(t)
        _plot_gmm_contour(ws.detach().cpu().numpy(),
                          mus.detach().cpu().numpy(),
                          covs.detach().cpu().numpy(),
                          colors[0])

    # p_net_gmm
    if p_net_gmm is not None:
        ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
        _plot_gmm_contour(ws.detach().cpu().numpy(),
                          mus.detach().cpu().numpy(),
                          covs.detach().cpu().numpy(),
                          colors[0])

    # linear prop (Gaussian)
    if data_lp is not None:
        _, _, mu6, P6 = data_lp.get(t)
        ws = np.array([1.0])
        mus = np.array([mu6])
        covs = np.array([P6])
        _plot_gmm_contour(ws, mus, covs, colors[3])

    # UT (Gaussian)
    if data_ut is not None:
        _, _, mu6, P6 = data_ut.get(t)
        ws = np.array([1.0])
        mus = np.array([mu6])
        covs = np.array([P6])
        _plot_gmm_contour(ws, mus, covs, colors[4])

    # external GMM provider (same color as UT in your full code)
    if data_gmm is not None:
        _, ws, mus, covs = data_gmm.get(t)
        _plot_gmm_contour(ws, mus, covs, colors[5])

    ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    return fig, ax


def plot_pdf_metrics(metrics):    
    set_publication_plot_style()

    # metric 1
    plt.figure()
    print(metrics["t"])
    plt.plot(metrics["t"], metrics["rel_error_lp"], color=colors[3],   label="LP")
    plt.plot(metrics["t"], metrics["rel_error_ut"], color=colors[4],   label="UT")
    plt.plot(metrics["t"], metrics["rel_error_gmm"], color=colors[5],   label="GMM")
    plt.plot(metrics["t"], metrics["rel_error_pinn"], color=colors[1], label="PINN-MLP")
    # all_zeros = not np.any(metrics["B1_pinn"])
    # if(all_zeros is False):
    #     plt.fill_between(
    #         metrics["t"],
    #         metrics["rel_error_pinn"],
    #         metrics["B1_pinn"],
    #         color=colors[1],
    #         alpha=0.2,
    #         label="PINN Error Bound"
    #     )
    plt.plot(metrics["t"], metrics["rel_error_pinngmm"], color=colors[0], label="PINN-GMM")
    # all_zeros = not np.any(metrics["B1_pinngmm"])
    # if(all_zeros is False):
    #     plt.fill_between(
    #         metrics["t"],
    #         metrics["rel_error_pinngmm"],
    #         metrics["B1_pinngmm"],
    #         color=colors[0],
    #         alpha=0.2,
    #         label="PINN-GMM Error Bound"
    #     )
    plt.legend(loc="upper left", ncol=2)
    plt.xlabel("t")
    plt.ylabel("norm. worst error %")
    plt.grid(True)

    # metric 2
    plt.figure()
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[3],   label="LP")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[4],   label="UT")
    plt.plot(metrics["t"], metrics["tv_gmm"], color=colors[5],   label="GMM")
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[1], label="PINN-MLP")
    plt.plot(metrics["t"], metrics["tv_pinngmm"], color=colors[0], label="PINN-GMM")
    plt.legend(loc="upper left", ncol=2)
    plt.xlabel("t")
    plt.ylabel("total variation %")
    plt.grid(True)

    # metric 3: negative log liklihood (general KL)
    plt.figure()
    print(metrics["t"])
    print(metrics["g_kl_pinn"])
    plt.plot(metrics["t"], metrics["g_kl_lp"], color=colors[3], label="GA")
    plt.plot(metrics["t"], metrics["g_kl_ut"], color=colors[4], label="UT")
    plt.plot(metrics["t"], metrics["g_kl_gmm"], color=colors[5], label="GMM")
    plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors[1], label="PINN-MLP")
    plt.plot(metrics["t"], metrics["g_kl_pinngmm"], color=colors[0], label="PINN-GMM")
    # plt.plot(metrics["t"], metrics["g_kl_pinngmm(uniform)"], color=colors[0], linestyle=":", label="PINN-GMM (uniform)")
    # plt.plot(metrics["t"], metrics["g_kl_pinngmm(no-imp)"], color=colors[0], linestyle="--", label="PINN-GMM (uniform + p0)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("General KL")
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, 1.5*np.array(metrics["g_kl_pinngmm"]).max())

    plt.show()


# --- Helper functions ---


def _p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    scales = np.sqrt(np.diag(cov))  # std per dimension
    cov_scaled = cov / np.outer(scales, scales)
    x_scaled = (x - mean) / scales
    rv = multivariate_normal(mean=np.zeros(len(mean)), cov=cov_scaled)
    pdf_eval = rv.pdf(x_scaled) / np.prod(scales)  # back-transform
    return pdf_eval.reshape(-1,)


def _set_axis_limits_with_buffer(ax, xlim, ylim, buffer_frac=0.05):
    """
    Set axis limits with a relative buffer.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to modify.
    xlim : tuple
        (x_lo, x_hi)
    ylim : tuple
        (y_lo, y_hi)
    buffer_frac : float
        Fraction of the data range to use as buffer on both sides.
    """
    xlo, xhi = xlim
    ylo, yhi = ylim

    # compute buffers
    xbuf = buffer_frac * (xhi - xlo)
    ybuf = buffer_frac * (yhi - ylo)

    ax.set_xlim(xlo - xbuf, xhi + xbuf)
    ax.set_ylim(ylo - ybuf, yhi + ybuf)
