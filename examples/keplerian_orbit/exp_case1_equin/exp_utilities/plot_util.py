import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional
import torch
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.patches import Patch, Rectangle

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.util import (set_publication_plot_style, colors_4set, 
                                     tidy_corner_axes, apply_default_locators, custom_save_plot,
                                     linestyles_4set, markers_4set)
from utilities._General.classic_gmm import make_gmm_pdf


def _integrate_out_others(pdf: np.ndarray, axes_coords: Sequence[np.ndarray], keep_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate a 6-D PDF over all axes except keep_axis using trapezoids (non-uniform grids OK)."""
    f = np.asarray(pdf, dtype=float)
    for ax in sorted([i for i in range(f.ndim) if i != keep_axis], reverse=True):
        f = np.trapz(f, x=np.asarray(axes_coords[ax], dtype=float), axis=ax)
    return np.asarray(axes_coords[keep_axis], dtype=float), f


def p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    scales = np.sqrt(np.diag(cov))  # std per dimension
    cov_scaled = cov / np.outer(scales, scales)
    x_scaled = (x - mean) / scales
    rv = multivariate_normal(mean=np.zeros(len(mean)), cov=cov_scaled)
    pdf_eval = rv.pdf(x_scaled) / np.prod(scales)  # back-transform
    return pdf_eval.reshape(-1,)


# ---------- plotting ----------
def plot_time_curves_3d(x_components, constants, data_sol, data_pinn, data_lp=None, data_ut=None,
                        p_init=None, leg_txt=None, title="Marginal p(x1|t): curves (no interpolation)"):
    """
    Plot one 3D curve per time: (x1, t_fixed, p(x1|t_fixed)).
    No faces between times -> no interpolation across time.
    """
    set_publication_plot_style()

    # pick a seaborn color palette
    colors = sns.color_palette("husl", 3)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_keep = data_sol["x"]; times = data_sol["times"]; M = data_sol["M"]
    for j, t in enumerate(times):
        ax.plot(np.full_like(x_keep, t), x_keep, M[:, j],
                color="black", linestyle="-")
        # if abs(t) < 1e-5 and p_init is not None:
        #     p_init_mc_marginal = M[:, j]
        #     pdf_func = multivariate_normal(
        #         mean=constants.MEAN_I[x_components-1],
        #         cov=constants.COV_I[x_components-1, x_components-1])
        #     p_init_analy_marginal = pdf_func.pdf(x_keep).astype(x_keep.dtype)
        #     rel_acc = (np.max(np.abs(p_init_mc_marginal - p_init_analy_marginal))
        #               / np.max(np.abs(p_init_analy_marginal)))
        #     print(f"[check] rel. accuracy of MC   (marginal to x{x_components:1d}) "
        #           f"--> the deviation between p(t0) MC and p(t0): {100.0*rel_acc:.2f} %")
            # ax.plot(np.full_like(x_keep, t), x_keep, p_init_analy_marginal, lw=2.0, color="green", linestyle=":")

    x_keep = data_pinn["x"]; times = data_pinn["times"]; M = data_pinn["M"]
    for j, t in enumerate(times):
        ax.plot(np.full_like(x_keep, t), x_keep, M[:, j],
                color=colors[0], linestyle="--")
        # if abs(t) < 1e-5 and p_init is not None:
        #     p_init_mc_marginal = M[:, j]
        #     pdf_func = multivariate_normal(
        #         mean=constants.MEAN_I[x_components-1],
        #         cov=constants.COV_I[x_components-1, x_components-1])
        #     p_init_analy_marginal = pdf_func.pdf(x_keep).astype(x_keep.dtype)
        #     rel_acc = (np.max(np.abs(p_init_mc_marginal - p_init_analy_marginal))
        #                 / np.max(np.abs(p_init_analy_marginal)))
        #     print(f"[check] rel. accuracy of PINN   (marginal to x{x_components:1d}) "
        #             f"--> the deviation between p(t0) PINN and p(t0): {100.0*rel_acc:.2f} %")
        #     ax.plot(np.full_like(x_keep, t), x_keep, p_init_analy_marginal, lw=2.0, color="black", linestyle=":")

    if data_lp is not None:
        # x_keep = data1["x"]; times = data1["times"]
        # x_keep = densify_between(x_keep, n_between=2)
        for j, t in enumerate(times):
            _, _mu_lp, _cov_lp = data_lp.get(t)
            _pdf_func = multivariate_normal(
                mean=_mu_lp[x_components-1],
                cov=_cov_lp[x_components-1, x_components-1])
            pdf_lp = _pdf_func.pdf(x_keep).astype(x_keep.dtype)
            ax.plot(np.full_like(x_keep, t), x_keep, pdf_lp,
                    color=colors[1], linestyle="--")
            
    if data_ut is not None:
        # x_keep = data1["x"]; times = data1["times"]
        # x_keep = densify_between(x_keep, n_between=2)
        for j, t in enumerate(times):
            _, _mu_us, _cov_us = data_ut.get(t)
            _pdf_func = multivariate_normal(
                mean=_mu_us[x_components-1],
                cov=_cov_us[x_components-1, x_components-1])
            pdf_us = _pdf_func.pdf(x_keep).astype(x_keep.dtype)
            ax.plot(np.full_like(x_keep, t), x_keep, pdf_us,
                    color=colors[2], lw=3, linestyle=":")

    # labels — add padding to avoid collisions with tick labels
    ax.set_xlabel(r"$t/T$", labelpad=8)
    ax.set_ylabel(r"$x$" + str(x_components), labelpad=10)
    ax.set_zlabel(r"$p(x$" + str(x_components) + r"$,t)$", labelpad=8)

    # give tick labels a bit of breathing room
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='z', pad=2)

    ax.view_init(elev=18, azim=-33)

    # legend
    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", label=r"$p$ MC"),
        Line2D([0], [0], color=colors[0], linestyle="--", label=r"$\hat{p}$ PINN"),
        Line2D([0], [0], color=colors[1], linestyle="--",  label=r"$p$ Linear Prop."),
        Line2D([0], [0], color=colors[2], linestyle=":", lw=3, label=r"$p$ Unscent Trans.")
    ]
    ax.legend(handles=legend_elements, loc="best", frameon=True)
    plt.show()

    # No tight_layout with big pad — constrained_layout already did the work.
    # For export: trim outer whitespace aggressively.
    # savepath = f"figs/Case_6D_Equin_marginal_PDF{int(x_components)}.pdf"
    # plt.savefig(savepath, format="pdf", dpi=300,
    #             bbox_inches='tight', pad_inches=0.5)
    # plt.close()


def _indices_in_dense(time_dense, time_sparse, tol=None):
    time_dense  = np.asarray(time_dense)
    time_sparse = np.asarray(time_sparse)

    if tol is None:  # exact match (fast)
        mask = np.isin(time_dense, time_sparse)
        return np.flatnonzero(mask)

    # tolerant match (no huge memory use)
    order = np.argsort(time_dense)
    td    = time_dense[order]
    mask_sorted = np.zeros(td.shape, dtype=bool)

    for t in np.asarray(time_sparse).ravel():
        left  = np.searchsorted(td, t - tol, side='left')
        right = np.searchsorted(td, t + tol, side='right')
        if right > left:
            mask_sorted[left:right] = True

    return order[np.flatnonzero(mask_sorted)]


def plot_e1_pinn_validation(data1, data2):
    """
    """
    set_publication_plot_style()

    # pick a seaborn color palette
    colors = sns.color_palette("tab10")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    # --- first dataset ---
    y = data1["times"]; z = data1["values"]
    x = np.arange(1, data1["N_trials"]+1, step=1)
    for j, t in enumerate(y):
        color = colors[j % len(colors)]
        ax.plot(x, t, z[j, :], lw=0.3, alpha=0.5,
                color="blue", linestyle="-")
    ax.plot(x[-1], y, z[:, -1], color="blue", linestyle="-", label=r"$\max e_1$")

    # --- second dataset ---
    y = data2["times"]; z = 2.0*data2["values"]
    x = np.arange(1, data2["N_trials"]+1, step=1)
    for j, t in enumerate(y):
        color = colors[j % len(colors)]
        ax.plot(x, t, z[j, :], lw=0.3, alpha=0.5,
                color="green", linestyle="-")
    ax.plot(x[-1], y, z[:, -1], color="green", linestyle="-", label=r"$B_1(\hat{e}_1)$")

    # axis labels and title
    # print("\n")
    ax.set_xlabel("N samples (1E+6)", labelpad=8)
    ax.set_ylabel(r"$t/T$", labelpad=8)
    ax.set_zlabel("Error (Scaled by PDF)", labelpad=8)
    ax.view_init(elev=23, azim=-150)
    fig.tight_layout()
    ax.legend(loc="best")

    # # --- legend with linestyle meaning ---
    # legend_elements = [
    #     Line2D([0], [0], color="black", linestyle="-", lw=1.7, label=r"$\hat{p}$"),
    #     Line2D([0], [0], color="black", linestyle="--", lw=1.7, label=r"$p_{MC}$")
    # ]
    # ax.legend(handles=legend_elements, loc="best")
    savepath = "figs/Case_6D_Equin_validate_e1hat.pdf"
    plt.savefig(savepath, format="pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.5)
    plt.close()
    # plt.show()


def _iter_blocks(sizes, block_sizes):
    """
    Yield 6D index ranges for blocks covering a cartesian grid:
    sizes = [N1,...,N6], block_sizes = [B1,...,B6]
    Yields tuples of slices (s1,e1),...,(s6,e6) with e exclusive.
    """
    ranges = [ [(s, min(s+b, n)) for s in range(0, n, b)]
               for n, b in zip(sizes, block_sizes) ]
    for a1,b1 in ranges[0]:
        for a2,b2 in ranges[1]:
            for a3,b3 in ranges[2]:
                for a4,b4 in ranges[3]:
                    for a5,b5 in ranges[4]:
                        for a6,b6 in ranges[5]:
                            yield (a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6)

def _make_block_points(x_axes, idxs, device, dtype):
    """
    Build a small block’s grid points (K,6) tensor from per-axis arrays and index ranges.
    """
    a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = idxs
    Xs = [x_axes[0][a1:b1], x_axes[1][a2:b2], x_axes[2][a3:b3],
          x_axes[3][a4:b4], x_axes[4][a5:b5], x_axes[5][a6:b6]]
    g = np.meshgrid(*Xs, indexing="ij")  # small block only
    pts = np.stack([gi.ravel() for gi in g], axis=1)  # (K,6) in numpy
    return torch.as_tensor(pts, dtype=dtype, device=device), [b1-a1,b2-a2,b3-a3,b4-a4,b5-a5,b6-a6]

def _open_memmap_npy(path, shape, dtype=np.float32):
    """
    Create/overwrite a .npy file as a memmap so we can write slices incrementally.
    """
    from numpy.lib.format import open_memmap
    # mode='w+' creates the file and allows writing
    return open_memmap(path, mode='w+', dtype=dtype, shape=shape)


# ---- main computation ----
def precompute_pdf_init_streaming(constants, p_init, out_dir, dtype=np.float32,
                             grid_dir="data/grids", block_target_points=1_000_000,
                             inner_batch_points=500_000, device="cpu"):
    """
    Streams the 6D grid in blocks; for each time t in constants.T_PRIME_SPAN:
      - writes pdf_t_<t>.npy incrementally using a memmap
    No full 30^6 grid or pdf is ever kept in memory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # load per-axis grids (numpy 1D arrays)
    x_axes = [
        np.load(os.path.join(grid_dir, f"x{i}s.npy")).astype(dtype, copy=False)
        for i in range(1, 7)
    ]
    sizes = [len(a) for a in x_axes]  # expect [30,30,30,30,30,30]

    # choose roughly cubic block sizes so product ~ block_target_points (cap by axis length)
    root = max(1, round(block_target_points ** (1/6)))
    block_sizes = [min(n, root) for n in sizes]
    # nudge down if still too big
    def prod(v):
        r = 1
        for x in v: r *= x
        return r
    while prod(block_sizes) > block_target_points:
        i = int(np.argmax(block_sizes))
        block_sizes[i] = max(1, block_sizes[i]-1)

    print(f"[info] grid sizes: {sizes}, block sizes: {block_sizes} (~{prod(block_sizes):,} pts/block)")

    t = 0.0
    out_path = os.path.join(out_dir, f"pdf_analy_t{t:.3f}.npy")
    print(f"[time {t:.3f}] writing -> {out_path}")

    # create memmap file for this time
    pdf_mm = _open_memmap_npy(out_path, shape=tuple(sizes), dtype=dtype)

    with torch.no_grad():
        # constant time tensor will be built per inner batch
        for idxs in _iter_blocks(sizes, block_sizes):
            # build block points on device
            pts_block, blk_shape = _make_block_points(x_axes, idxs, device, torch.float32)
            K = pts_block.shape[0]

            # process this block in inner batches to fit GPU/CPU memory
            vals_list = []
            for start in range(0, K, inner_batch_points):
                end = min(start + inner_batch_points, K)
                x_batch = pts_block[start:end, :6]
                x_batch_numpy = x_batch.detach().cpu().numpy()
                # # time batch: (B,1)
                # t_batch = torch.full((x_batch.shape[0], 1), fill_value=t,
                #                     dtype=torch.float32, device=device)

                y = p_init(constants, x_batch_numpy)           # (B,1) or (B,)
                vals_list.append(y)

            vals = np.concatenate(vals_list, axis=0)   # (K,)
            vals = vals.reshape(blk_shape)             # reshape to local block shape

            # write into proper slice of the memmap
            a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = idxs
            pdf_mm[a1:b1, a2:b2, a3:b3, a4:b4, a5:b5, a6:b6] = vals
    # ensure data is flushed
    del pdf_mm
    print("done.")


def precompute_pdf_streaming(constants, p_net, out_dir, dtype=np.float32,
                             grid_dir="data/grids", block_target_points=1_000_000,
                             inner_batch_points=500_000, device="cpu"):
    """
    Streams the 6D grid in blocks; for each time t in constants.T_PRIME_SPAN:
      - writes pdf_t_<t>.npy incrementally using a memmap
    No full 30^6 grid or pdf is ever kept in memory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # load per-axis grids (numpy 1D arrays)
    x_axes = [
        np.load(os.path.join(grid_dir, f"x{i}s.npy")).astype(dtype, copy=False)
        for i in range(1, 7)
    ]
    sizes = [len(a) for a in x_axes]  # expect [30,30,30,30,30,30]

    # choose roughly cubic block sizes so product ~ block_target_points (cap by axis length)
    root = max(1, round(block_target_points ** (1/6)))
    block_sizes = [min(n, root) for n in sizes]
    # nudge down if still too big
    def prod(v):
        r = 1
        for x in v: r *= x
        return r
    while prod(block_sizes) > block_target_points:
        i = int(np.argmax(block_sizes))
        block_sizes[i] = max(1, block_sizes[i]-1)

    print(f"[info] grid sizes: {sizes}, block sizes: {block_sizes} (~{prod(block_sizes):,} pts/block)")

    # set model to eval/device
    p_net.eval().to(device)

    # iterate all requested times
    for t in constants.T_PRIME_SPAN:
        out_path = os.path.join(out_dir, f"pdf_t{t:.3f}.npy")
        print(f"[time {t:.3f}] writing -> {out_path}")

        # create memmap file for this time
        pdf_mm = _open_memmap_npy(out_path, shape=tuple(sizes), dtype=dtype)

        with torch.no_grad():
            # constant time tensor will be built per inner batch
            for idxs in _iter_blocks(sizes, block_sizes):
                # build block points on device
                pts_block, blk_shape = _make_block_points(x_axes, idxs, device, torch.float32)
                K = pts_block.shape[0]

                # process this block in inner batches to fit GPU/CPU memory
                vals_list = []
                for start in range(0, K, inner_batch_points):
                    end = min(start + inner_batch_points, K)
                    x_batch = pts_block[start:end, :6]
                    # time batch: (B,1)
                    t_batch = torch.full((x_batch.shape[0], 1), fill_value=t,
                                        dtype=torch.float32, device=device)

                    y = p_net(x_batch, t_batch)           # (B,1) or (B,)
                    vals_list.append(y.detach().flatten().cpu().numpy())

                vals = np.concatenate(vals_list, axis=0)   # (K,)
                vals = vals.reshape(blk_shape)             # reshape to local block shape

                # write into proper slice of the memmap
                a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = idxs
                pdf_mm[a1:b1, a2:b2, a3:b3, a4:b4, a5:b5, a6:b6] = vals

        # ensure data is flushed
        del pdf_mm
        print(f"[time {t:.3f}] done.")


def precompute_error(constants, p_net, out_dir, mc_dir, dtype=np.float32):
    for t in constants.T_PRIME_SPAN:
        mc_path = mc_dir+f"pdf_t{t:.3f}.npy"
        pdf_mc = np.load(mc_path)
        pinn_path = os.path.join(out_dir, f"pdf_t{t:.3f}.npy")
        pdf_pinn = np.load(pinn_path)
        e1 = pdf_mc - pdf_pinn
        del pdf_mc
        del pdf_pinn
        e1_path = os.path.join(out_dir, f"e1_t{t:.3f}.npy")
        np.save(e1_path, e1)
        del e1
        print(f"[time {t:.3f}] writing -> {e1_path}")


def precompute_e1hat_streaming(constants, net, out_dir, dtype=np.float32,
                             grid_dir="data/grids", block_target_points=1_000_000,
                             inner_batch_points=500_000, device="cpu"):
    """
    Streams the 6D grid in blocks; for each time t in constants.T_PRIME_SPAN:
      - writes pdf_t_<t>.npy incrementally using a memmap
    No full 30^6 grid or pdf is ever kept in memory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # load per-axis grids (numpy 1D arrays)
    x_axes = [
        np.load(os.path.join(grid_dir, f"x{i}s.npy")).astype(dtype, copy=False)
        for i in range(1, 7)
    ]
    sizes = [len(a) for a in x_axes]  # expect [30,30,30,30,30,30]

    # choose roughly cubic block sizes so product ~ block_target_points (cap by axis length)
    root = max(1, round(block_target_points ** (1/6)))
    block_sizes = [min(n, root) for n in sizes]
    # nudge down if still too big
    def prod(v):
        r = 1
        for x in v: r *= x
        return r
    while prod(block_sizes) > block_target_points:
        i = int(np.argmax(block_sizes))
        block_sizes[i] = max(1, block_sizes[i]-1)

    print(f"[info] grid sizes: {sizes}, block sizes: {block_sizes} (~{prod(block_sizes):,} pts/block)")

    # set model to eval/device
    net.eval().to(device)

    # iterate all requested times
    for t in constants.T_PRIME_SPAN:
        if(abs(t) < 1e-5):
            out_path = os.path.join(out_dir, f"e1hat_t{t:.3f}.npy")
            print(f"[time {t:.3f}] writing -> {out_path}")

            # create memmap file for this time
            pdf_mm = _open_memmap_npy(out_path, shape=tuple(sizes), dtype=dtype)

            with torch.no_grad():
                # constant time tensor will be built per inner batch
                for idxs in _iter_blocks(sizes, block_sizes):
                    # build block points on device
                    pts_block, blk_shape = _make_block_points(x_axes, idxs, device, torch.float32)
                    K = pts_block.shape[0]

                    # process this block in inner batches to fit GPU/CPU memory
                    vals_list = []
                    for start in range(0, K, inner_batch_points):
                        end = min(start + inner_batch_points, K)
                        x_batch = pts_block[start:end, :6]
                        # time batch: (B,1)
                        t_batch = torch.full((x_batch.shape[0], 1), fill_value=t,
                                            dtype=torch.float32, device=device)

                        y = net(x_batch, t_batch)           # (B,1) or (B,)
                        vals_list.append(y.detach().flatten().cpu().numpy())

                    vals = np.concatenate(vals_list, axis=0)   # (K,)
                    vals = vals.reshape(blk_shape)             # reshape to local block shape

                    # write into proper slice of the memmap
                    a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = idxs
                    pdf_mm[a1:b1, a2:b2, a3:b3, a4:b4, a5:b5, a6:b6] = vals

            # ensure data is flushed
            del pdf_mm
            print(f"[time {t:.3f}] done.")


def densify_between(v, n_between=1):
    """
    Insert `n_between` evenly spaced points between each consecutive pair in v.
    v: 1D numpy array
    """
    v = np.asarray(v, dtype=float)
    pieces = [np.linspace(v[i], v[i+1], n_between + 2)[:-1]  # drop right endpoint
              for i in range(len(v)-1)]
    return np.concatenate(pieces + [v[-1:]])


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
    # i_plot = [2, 4, 6]   # change if you want other times
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


x_axis_labels=[r"$a$", r"$P_1$", r"$P_2$", 
               r"$Q_1$", r"$Q_2$", r"$\lambda$"]


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
    rar_samples=None,
    data_lp=None,
    data_ut=None,
    data_gmm=None,
    save_plot=False,
):
    """
    dims = (i,) -> 1D histogram for x_i with model overlays (same colors as full plot)
    dims = (i,j) -> 2D heatmap for (x_i, x_j) with model contour overlays
    Indices in dims are 1-based.
    """

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
        apply_default_locators(ax)
        set_publication_plot_style()
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
                ax.plot(data["X_grid"], data["pdf"], color=colors_4set[1], lw=2)

        _plot_pinn_1d_marginal(k+1)

        # p_net_gmm (mixture’s 1D marginal)
        x_vals = np.linspace(lo, hi, 256)

        if p_net_gmm is not None:
            ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
            ws, mus, covs = (ws.detach().cpu().numpy(),
                             mus.detach().cpu().numpy(),
                             covs.detach().cpu().numpy())
            xvals_scaled = (x_vals - constants.MEAN_I[k])/ constants.COV_I[k,k]**0.5
            pdf_scaling = 1 / (constants.COV_I[k,k])**0.5
            pdf = np.zeros_like(x_vals)
            for c in range(ws.shape[0]):
                pdf += ws[c] * multivariate_normal(mean=mus[c, k], cov=covs[c, k, k]).pdf(xvals_scaled) * pdf_scaling
            ax.plot(x_vals, pdf, color=colors_4set[0])

        # linear prop (Gaussian)
        if data_lp is not None:
            _, _, mu6, P6 = data_lp.get(t)
            ax.plot(x_vals, multivariate_normal(mu6[k], P6[k, k]).pdf(x_vals), color=colors_4set[3])

        # UT (Gaussian)
        if data_ut is not None:
            _, _, mu6, P6 = data_ut.get(t)
            ax.plot(x_vals, multivariate_normal(mu6[k], P6[k, k]).pdf(x_vals), color=colors_4set[4])

        if data_gmm is not None:
            _, ws, mus, covs = data_gmm.get(t)
            pdf = np.zeros_like(x_vals)
            for c in range(ws.shape[0]):
                pdf += ws[c] * multivariate_normal(mean=mus[c, k], cov=covs[c, k, k]).pdf(x_vals)
            ax.plot(x_vals, pdf, color=colors_4set[5])

        return fig, ax

    # ---------- 2D (off-diagonal heatmap) ----------
    i, j = idx
    xlo, xhi = (_fixed_range(i) if ranges == "fixed" else _auto_range(X[:, i]))
    ylo, yhi = (_fixed_range(j) if ranges == "fixed" else _auto_range(X[:, j]))

    H, xe, ye = np.histogram2d(X[:, i], X[:, j], bins=bins,
                               range=[(xlo, xhi), (ylo, yhi)])
    H = H.T

    set_publication_plot_style()
    fig, ax = plt.subplots()
    apply_default_locators(ax)
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
            ax.contour(Xg, Yg, pdf, levels=levels, colors=[colors_4set[1]], linewidths=2.0)

    def _plot_gmm_contour(ws, mus, covs, color, scaled_x=False, rar_samples=None):
        xs = np.linspace(xlo, xhi, 256)
        ys = np.linspace(ylo, yhi, 256)
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        if(scaled_x):
            xs_scaled = (xs - constants.MEAN_I[i])/ constants.COV_I[i,i]**0.5
            ys_scaled = (ys - constants.MEAN_I[j])/ constants.COV_I[j,j]**0.5
            X_grid_scaled, Y_grid_scaled = np.meshgrid(xs_scaled, ys_scaled, indexing="ij")
            pts = np.vstack([X_grid_scaled.ravel(), Y_grid_scaled.ravel()]).T
            pdf_scaling = 1 / (constants.COV_I[i,i] * constants.COV_I[j,j])**0.5
        else:
            pts = np.column_stack([Xg.ravel(), Yg.ravel()])
            pdf_scaling = 1.

        if mus.ndim == 1:   mus  = mus[None, :]        # (1, D)
        if covs.ndim == 2:  covs = covs[None, :, :]    # (1, D, D)  (tied covariance)

        K, D = mus.shape
        dims = (int(i), int(j))
        # means: take the two columns across all components -> (K, 2)
        mu_ij = mus[:, dims]
        # covariances: take the 2x2 submatrix across all components -> (K, 2, 2)
        idx = np.ix_(np.arange(covs.shape[0]), dims, dims)
        cov_ij = covs[idx]
        _pdf_func = make_gmm_pdf(ws, mu_ij, cov_ij)
        pdf = pdf_scaling * _pdf_func(pts).reshape(Xg.shape)
        pdf_max = float(np.max(pdf)) if np.any(pdf > 0) else 1.0
        levels = relative_levels * pdf_max
        ax.contour(Xg, Yg, pdf, levels=levels, colors=[color], linewidths=2.0)

        def subset_by_time(
            t_res_rar: np.ndarray,
            x_res_rar: np.ndarray,
            t: float,
            eps: float = 5e-2,
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            """
            Return (t_subset, x_subset, idx) for rows where |t_res_rar - t| <= eps.
            If no rows match, return (None, None, None).
            """
            if t_res_rar.shape[0] != x_res_rar.shape[0]:
                raise ValueError("t_res_rar and x_res_rar must have the same number of rows.")

            tt = np.asarray(t_res_rar).reshape(-1)  # (N,)
            mask = np.isfinite(tt) & np.isclose(tt, t, rtol=0.0, atol=eps)
            idx = np.flatnonzero(mask)

            if idx.size == 0:
                return None, None, None

            return t_res_rar[idx], x_res_rar[idx], idx
        
        def fifo_fade_scatter(ax, x, y, idx=None, s=10,
                      alpha_min=0.05, alpha_max=1.0,
                      base="green", label=""):
            """
            Oldest points = faint, newest = opaque.
            base: 'black' or 'green'
            """
            x = np.asarray(x).ravel()
            y = np.asarray(y).ravel()
            if x.size == 0:
                return

            # FIFO order (small idx = older)
            order = np.argsort(idx) if idx is not None else np.arange(x.size)
            x, y = x[order], y[order]

            n = x.size
            alphas = np.linspace(alpha_min, alpha_max, n)

            # choose base color
            if base not in ("black", "green"):
                raise ValueError("base must be 'black' or 'green'")
            base_code = "k" if base == "black" else "g"
            rgba = np.array(plt.matplotlib.colors.to_rgba(base_code))

            colors = np.repeat(rgba[None, :], n, axis=0)
            colors[:, 3] = alphas  # fade by alpha

            ax.scatter(x, y, s=s, c=colors, marker="x", linewidth=1, label=label)

        if(rar_samples is not None):
            if(abs(t) < 1e-4):
                x_bc_rar = rar_samples["X_BC_RAR"]
                print("visual x_bc_rar data shape: ", x_bc_rar.shape)
                _rar_x = x_bc_rar[:,i]
                _rar_y = x_bc_rar[:,j]
                rar_x = _rar_x * constants.COV_I[i,i]**0.5 + constants.MEAN_I[i] 
                rar_y = _rar_y * constants.COV_I[j,j]**0.5 + constants.MEAN_I[j] 
                fifo_fade_scatter(ax, rar_x, rar_y, base="black",
                                  label="Sample Pool for Initial Loss")
                
            t_res_rar = rar_samples["T_RES_RAR"]
            x_res_rar = rar_samples["X_RES_RAR"]
            t_res_rar_sel, x_res_rar_sel, _ = subset_by_time(t_res_rar, x_res_rar, t)
            if(t_res_rar_sel is not None):
                print("visual x_res_rar data shape: ", x_res_rar.shape)
                _rar_x = x_res_rar_sel[:,i]
                _rar_y = x_res_rar_sel[:,j]
                rar_x = _rar_x * constants.COV_I[i,i]**0.5 + constants.MEAN_I[i] 
                rar_y = _rar_y * constants.COV_I[j,j]**0.5 + constants.MEAN_I[j] 
                fifo_fade_scatter(ax, rar_x, rar_y,
                                  label="Sample Pool for Differential Loss")

    # PINN 2D
    _plot_pinn_2d_contour(i+1, j+1)

    # p_net_gmm_N1
    if p_net_gmm_N1 is not None:
        ws, mus, covs = p_net_gmm_N1.weights_means_covs_at(t)
        _plot_gmm_contour(ws.detach().cpu().numpy(),
                          mus.detach().cpu().numpy(),
                          covs.detach().cpu().numpy(),
                          colors_4set[0])

    # linear prop (Gaussian)
    if data_lp is not None:
        _, w, mu6, P6 = data_lp.get(t)
        mus = np.array([mu6])
        covs = np.array([P6])
        _plot_gmm_contour(w, mus, covs, colors_4set[0])

    # UT (Gaussian)
    if data_ut is not None:
        _, w, mu6, P6 = data_ut.get(t)
        mus = np.array([mu6])
        covs = np.array([P6])
        _plot_gmm_contour(w, mus, covs, colors_4set[1])

    # external GMM provider (same color as UT in your full code)
    if data_gmm is not None:
        _, ws, mus, covs = data_gmm.get(t)
        _plot_gmm_contour(ws, mus, covs, colors_4set[2])

    # p_net_gmm (train in scaled x)
    if p_net_gmm is not None:
        ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
        _plot_gmm_contour(ws.detach().cpu().numpy(),
                          mus.detach().cpu().numpy(),
                          covs.detach().cpu().numpy(),
                          colors_4set[3], scaled_x=True, rar_samples=rar_samples)

    ax.set_xlabel(x_axis_labels[i]); ax.set_ylabel(x_axis_labels[j])
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)

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
    #    # Density legend entries (marker-only squares)
    #     Line2D([0],[0], linestyle='None', marker='s', markersize=12,
    #         markerfacecolor=c_low, markeredgecolor=c_low,
    #         alpha=0.8, label='Ref. PDF, low  density'),
    #     Line2D([0],[0], linestyle='None', marker='s', markersize=12,
    #         markerfacecolor=c_high, markeredgecolor=c_high,
    #         alpha=0.8, label='Ref. PDF, high density'),
    #     spacer,
    #     spacer
    ]
    if(rar_samples is not None):
        if(abs(t) < 1e-5):
            handles.append(
                Line2D([0], [0],
                    linestyle='None', marker='x',
                    markersize=7, markeredgewidth=1.8,
                    markeredgecolor='black',  # or use color='green'
                    label='Samples Pool (Initial Loss)'))
        handles.append(
            Line2D([0], [0],
            linestyle='None', marker='x',
            markersize=7, markeredgewidth=1.8,
            markeredgecolor='green',  # or use color='green'
            label='Samples Pool (Differential Loss)')
        )
    ax.legend(handles=handles, ncol=1, loc="upper right")
    save_path = "figs/corner_element_t{:.2f}.pdf".format(t)
    custom_save_plot(save_plot, save_path)
    

def plot_full_corner(constants, t,
    X_samples,
    bins=256,
    labels=None,
    figsize_per_dim=1.5,
    max_points=500_000,
    mode="heatmap",            # "scatter" or "heatmap"
    cmap="bwr",
    alpha=0.4,
    point_size=2,
    ranges=None,               # list of (lo,hi) per dim; if None -> data-driven
    log_counts=True,           # log color scale for heatmap
    OUTPUT_PATH=None,
    p_net_gmm=None,
    save_plot=False,
):
    set_publication_plot_style(font_size=14)

    X = np.asarray(X_samples)
    N, D = X.shape
    if labels is None:
        labels = [f"x{i+1}" for i in range(D)]

    # ---- auto ranges ----
    # if ranges is None:
    #     if quantile_range is None:
    #         lo = np.min(X, axis=0)
    #         hi = np.max(X, axis=0)
    #     else:
    #         qlo, qhi = quantile_range
    #         lo = np.quantile(X, qlo, axis=0)
    #         hi = np.quantile(X, qhi, axis=0)
    #     pad = pad_frac * (hi - lo + 1e-12)
    #     ranges = [(float(lo[i] - pad[i]), float(hi[i] + pad[i])) for i in range(D)]
    # else:
    #     if len(ranges) != D:
    #         raise ValueError(f"`ranges` must have length {D}")
    #     ranges = [(float(a), float(b)) for (a, b) in ranges]
    # --- fixed ranges ---
    ranges = [
        (constants.X1_RANGE[0],constants.X1_RANGE[1]),
        (constants.X2_RANGE[0],constants.X2_RANGE[1]),
        (constants.X3_RANGE[0],constants.X3_RANGE[1]),
        (constants.X4_RANGE[0],constants.X4_RANGE[1]),
        (constants.X5_RANGE[0],constants.X5_RANGE[1]),
        (constants.X6_RANGE[0],constants.X6_RANGE[1]),
    ]

    fig, axes = plt.subplots(
        D, D,
        figsize=(figsize_per_dim*D, figsize_per_dim*D),
        constrained_layout=True,
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}   # small gaps
    )

    # Optional: trim outer margins further (constrained_layout respects these)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    def _plot_pinn_1d_marginal(x_coords):
        data_marginal_pinn = np.load(
        f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords}_t{t:.3f}.npz")
        if(data_marginal_pinn is not None):
            pdf_values = data_marginal_pinn['pdf']
            X_grid = data_marginal_pinn["X_grid"]
            ax.plot(X_grid, pdf_values, color=colors_4set[1])
        # pdf_max = np.max(pdf_values[pdf_values > 0])
        # Define levels as percentages of the maximum value
        # For example, levels at 5%, 25%, 50%, 75%, and 95% of the max
        # relative_levels = np.array([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95])
        # levels = relative_levels * pdf_max
        # Draw the contour lines with the custom levels
        # ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[sns_colors[0]], linewidths=0.5)

    # Diagonals: 1D histograms (counts)
    for d in range(D):
        ax = axes[d, d]
        lo, hi = ranges[d]
        hist_color = plt.cm.bwr(0.35)          # keep your diagonal histogram color
        hist_edge  = plt.cm.bwr(0.35)
        ax.hist(
            X[:, d], bins=bins, range=(lo, hi),
            histtype="stepfilled", alpha=1.0, density=True,
            color=hist_color, edgecolor=hist_edge
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if d == D - 1: ax.set_xlabel(x_axis_labels[d])
        else: ax.set_xticklabels([])
        if d != 0: ax.set_yticklabels([])
        if d == 0: ax.set_ylabel(x_axis_labels[d])
        # hide upper triangle in row d
        for k in range(d + 1, D):
            axes[d, k].axis("off")
        # if(OUTPUT_PATH is not None):
        #     _plot_pinn_1d_marginal(d+1)

        def _plot_pinn_gmm_1d(ws, mus, covs, plot_axes, color):
            # print(ws.shape, mus.shape, covs.shape)
            # print(plot_axes)
            xs = np.linspace(lo, hi, num=256, endpoint=True)
            xs_scaled = (xs - constants.MEAN_I[plot_axes])/ constants.COV_I[plot_axes,plot_axes]**0.5
            # grid_pts = np.vstack([X_grid_scaled.ravel(), Y_grid_scaled.ravel()]).T
            pdf_scaling = 1 / (constants.COV_I[plot_axes,plot_axes])**0.5

            pdf_values = np.copy(xs) * 0.0
            for k in range(ws.shape[0]):
                ws_k = ws[k]
                mus_k = mus[k, :]
                covs_k = covs[k, :, :]
                marginal_mu = mus_k[plot_axes]
                marginal_cov = covs_k[plot_axes, plot_axes]
                # print(marginal_mu)
                pdf_func = multivariate_normal(mean=marginal_mu, cov=marginal_cov)
                p_k = pdf_func.pdf(xs_scaled).reshape(xs_scaled.shape) * pdf_scaling
                pdf_values = pdf_values + ws_k * p_k
            ax.plot(xs, pdf_values, color=color)

        if(p_net_gmm is not None):
            plot_axes = d
            ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
            ws = ws.detach().cpu().numpy()
            # print("GMM weights: ", ws, np.sum(ws))
            mus = mus.detach().cpu().numpy()
            covs = covs.detach().cpu().numpy()
            _plot_pinn_gmm_1d(ws, mus, covs, plot_axes, colors_4set[3])

    # Off-diagonals: scatter or heatmap (counts)
    for i in range(1, D):
        for j in range(i):
            ax = axes[i, j]
            (xlo, xhi), (ylo, yhi) = ranges[j], ranges[i]
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
                filename = f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz"
                if(os.path.isfile(filename)):
                    data_marginal_pinn = np.load(filename)
                    pdf_values = data_marginal_pinn['pdf']
                    X_grid, Y_grid = data_marginal_pinn["X_grid"], data_marginal_pinn["Y_grid"], 
                    pdf_max = np.max(pdf_values[pdf_values > 0])
                    # Define levels as percentages of the maximum value
                    # For example, levels at 5%, 25%, 50%, 75%, and 95% of the max
                    levels = relative_levels * pdf_max
                    # Draw the contour lines with the custom levels
                    ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors_4set[1]], linewidths=1.0)
                # else:
                #     print(x_coords)

            # if(OUTPUT_PATH is not None):
            #     x_coords = (j+1, i+1)
            #     _plot_pinn_contour(x_coords)

            def _plot_pinn_gmm_contour(ws, mus, covs, x_coords, color):
                # print(ws.shape, mus.shape, covs.shape)
                plot_axes = (x_coords[0]-1, x_coords[1]-1)
                # print(plot_axes)
                xs = np.linspace(xlo, xhi, num=256, endpoint=True)
                ys = np.linspace(ylo, yhi, num=256, endpoint=True)
                X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
                xs_scaled = (xs - constants.MEAN_I[plot_axes[0]])/ constants.COV_I[plot_axes[0],plot_axes[0]]**0.5
                ys_scaled = (ys - constants.MEAN_I[plot_axes[1]])/ constants.COV_I[plot_axes[1],plot_axes[1]]**0.5
                X_grid_scaled, Y_grid_scaled = np.meshgrid(xs_scaled, ys_scaled, indexing="ij")
                grid_pts = np.vstack([X_grid_scaled.ravel(), Y_grid_scaled.ravel()]).T
                # print(xs_scaled.min(), xs_scaled.max())
                # print(ys_scaled.min(), ys_scaled.max())
                pdf_scaling = 1 / (constants.COV_I[plot_axes[0],plot_axes[0]] * constants.COV_I[plot_axes[1],plot_axes[1]])**0.5

                pdf_values = np.copy(X_grid) * 0.0
                for k in range(ws.shape[0]):
                    ws_k = ws[k]
                    mus_k = mus[k, :]
                    covs_k = covs[k, :, :]
                    marginal_mu = mus_k[list(plot_axes)]
                    marginal_cov = covs_k[np.ix_(list(plot_axes), list(plot_axes))]
                    # print(marginal_mu)
                    pdf_func = multivariate_normal(mean=marginal_mu, cov=marginal_cov)
                    p_k = pdf_func.pdf(grid_pts).reshape(X_grid.shape) * pdf_scaling
                    pdf_values = pdf_values + ws_k * p_k
                    # if(ws.shape[0] > 1):
                    #     _pdf_max = np.max(p_k).item()
                    #     _levels = np.array([0.01]) * _pdf_max
                    #     ax.contour(X_grid, Y_grid, p_k, levels=_levels, colors=[color], 
                    #             linewidths=0.5, alpha=0.4)
                # print(pdf_values.shape, X_grid.shape, Y_grid.shape)
                # print(pdf_values)
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max # ; print(pdf_max, levels)
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[color], linewidths=1.)

            if(p_net_gmm is not None):
                x_coords = (j+1, i+1)
                ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
                ws = ws.detach().cpu().numpy()
                # print("GMM weights: ", ws, np.sum(ws))
                mus = mus.detach().cpu().numpy()
                covs = covs.detach().cpu().numpy()
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors_4set[3])

            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D - 1: ax.set_xlabel(x_axis_labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(x_axis_labels[i])
            else: ax.set_yticklabels([])
    
    plt.tick_params(axis='both', which='major', labelsize=8)
    tidy_corner_axes(fig, axes, labels=x_axis_labels)
    save_path = "figs/full_corner_t{:.2f}.pdf".format(t)
    custom_save_plot(save_plot, save_path)


def plot_event_triptych_simple(constants, X_event, t, p_net_gmm=None, data_lp=None):
    """
    X_event: (D,2) array for x = [a, P1, P2, Q1, Q2, lambda], D>=6.
    Shows 1x3 panels for (a,lambda), (P1,P2), (Q1,Q2).
    """    

    def _plot_pinn_gmm_contour(x_ranges, ws, mus, covs, plot_axes, color):
        xlo = x_ranges[plot_axes[0], 0]
        xhi = x_ranges[plot_axes[0], 1]
        ylo = x_ranges[plot_axes[1], 0]
        yhi = x_ranges[plot_axes[1], 1]
        relative_levels = np.array([0.01, 0.05, 0.50, 0.95])
        xs = np.linspace(xlo, xhi, num=256, endpoint=True)
        ys = np.linspace(ylo, yhi, num=256, endpoint=True)
        X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
        xs_scaled = (xs - constants.MEAN_I[plot_axes[0]])/ constants.COV_I[plot_axes[0],plot_axes[0]]**0.5
        ys_scaled = (ys - constants.MEAN_I[plot_axes[1]])/ constants.COV_I[plot_axes[1],plot_axes[1]]**0.5
        X_grid_scaled, Y_grid_scaled = np.meshgrid(xs_scaled, ys_scaled, indexing="ij")
        grid_pts = np.vstack([X_grid_scaled.ravel(), Y_grid_scaled.ravel()]).T
        pdf_scaling = 1 / (constants.COV_I[plot_axes[0],plot_axes[0]] * constants.COV_I[plot_axes[1],plot_axes[1]])**0.5
        pdf_values = np.copy(X_grid) * 0.0
        for k in range(ws.shape[0]):
            ws_k = ws[k]
            mus_k = mus[k, :]
            covs_k = covs[k, :, :]
            marginal_mu = mus_k[list(plot_axes)]
            marginal_cov = covs_k[np.ix_(list(plot_axes), list(plot_axes))]
            pdf_func = make_gmm_pdf(1., marginal_mu, marginal_cov)
            p_k = pdf_func(grid_pts).reshape(X_grid.shape) * pdf_scaling
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

    x_ranges = constants.unscale_box(constants._NX_RANGE_NP)
          
    pairs  = [(0, 5), (1, 2), (3, 4)]
    labels = ['a', 'P1', 'P2', 'Q1', 'Q2', 'lambda']

    set_publication_plot_style()
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)

    for ax, (ix, iy) in zip(axs, pairs):
        x0, x1 = X_event[ix]
        y0, y1 = X_event[iy]
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
            mus = mus.reshape((1, -1))
            covs = covs.reshape((1, 6, 6))
            _plot_gmm_contour(x_ranges, ws, mus, covs, [ix, iy], color=colors_4set[0])
