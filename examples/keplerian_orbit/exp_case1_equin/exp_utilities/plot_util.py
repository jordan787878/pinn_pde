import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
import torch
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter



def set_publication_plot_style(font_family='Times New Roman', font_size=18):
    """
    Update Matplotlib settings to use publication-ready fonts.

    Parameters:
        font_family (str): Font family to be used for all texts.
        font_size (int): Base font size for labels, titles, legends, and ticks.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.titlesize'] = font_size
    plt.rcParams['lines.linewidth'] = 2


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


def plot_pdf_metrics(metrics):
    set_publication_plot_style()

    colors = sns.color_palette("husl", 4)

    plt.figure()
    print(metrics["t"])
    plt.plot(metrics["t"], metrics["rel_error_lp"], color=colors[1],   label="LP")
    plt.plot(metrics["t"], metrics["rel_error_ut"], color=colors[2],   label="UT")
    plt.plot(metrics["t"], metrics["rel_error_pinn"], color=colors[-1], label="PINN-MLP")
    all_zeros = not np.any(metrics["B1_pinn"])
    if(all_zeros is False):
        plt.fill_between(
            metrics["t"],
            metrics["rel_error_pinn"],
            metrics["B1_pinn"],
            color=colors[-1],
            alpha=0.2,
            label="PINN Error Bound"
        )
    plt.plot(metrics["t"], metrics["rel_error_pinngmm"], color=colors[0], label="PINN-GMM")
    all_zeros = not np.any(metrics["B1_pinngmm"])
    if(all_zeros is False):
        plt.fill_between(
            metrics["t"],
            metrics["rel_error_pinngmm"],
            metrics["B1_pinngmm"],
            color=colors[0],
            alpha=0.2,
            label="PINN-GMM Error Bound"
        )
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("norm. worst error %")
    plt.grid(True)

    plt.figure()
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[1],   label="LP")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[2],   label="UT")
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[-1], label="PINN-MLP")
    plt.plot(metrics["t"], metrics["tv_pinngmm"], color=colors[0], label="PINN-GMM")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("total variation %")
    plt.grid(True)

    # metric 3: negative log liklihood (relative KL)
    plt.figure()
    plt.plot(metrics["t"], metrics["g_kl_lp"], color=colors[1],   label="LP")
    plt.plot(metrics["t"], metrics["g_kl_ut"], color=colors[2],   label="UT")
    # plt.plot(metrics["t"], metrics["rel_kl_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.plot(metrics["t"], metrics["g_kl_pinn"], color=colors[-1], label="PINN-MLP")
    plt.plot(metrics["t"], metrics["g_kl_pinngmm"], color=colors[0], label="PINN-GMM")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("General KL")
    plt.grid(True)

    plt.show()



def plot_single_corner(t, constants, 
    x_coord1=1, x_coord2=1,
    X_samples=None,
    data_marginal_pinn=None,
    gaussian_lp=None,
    gaussian_ut=None,
    p_net_gmm=None,
    bins=300,
    cmap="cividis", #"Greys", "magma", "cividis", "rocket_r"
    ranges=None,               # list of (lo,hi) per dim; if None -> data-driven
    quantile_range=(0.001, 0.999),  # set to None to use full min/max
    log_counts=True,           # log color scale for heatmap
    ):

    xrange = getattr(constants, f"X{x_coord1}_RANGE")
    yrange = getattr(constants, f"X{x_coord2}_RANGE")

    set_publication_plot_style()

    labels=["x"+str(x_coord1), "x"+str(x_coord2)]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", 4)

    # heatmap plot from samples drawn from the true distribution
    xlo = X_samples[:, x_coord1-1].min()# x_first_unscaled.min()
    xhi = X_samples[:, x_coord1-1].max()#x_first_unscaled.max()
    ylo = X_samples[:, x_coord2-1].min()#x_second_unscaled.min()
    yhi = X_samples[:, x_coord2-1].max()#x_second_unscaled.max()
    H, xedges, yedges = np.histogram2d(
        X_samples[:, x_coord1-1], X_samples[:, x_coord2-1],
        bins=bins,
        range=[(xlo, xhi), (ylo, yhi)],
    )
    H = H.T
    if log_counts:
        pos = H[H > 0]
        vmax = float(pos.max()) if pos.size else 1.0
        vmin = float(pos.min()) if pos.size else 1.0  # >=1 to avoid zeros in LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
        # norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None
    H_ma = np.ma.masked_invalid(H)           # handle NaNs/Infs
    alpha = np.clip(norm(H_ma).filled(0), 0, 1)  # normalize to [0,1]
    ax.imshow(
        H, origin="lower",
        extent=(xlo, xhi, ylo, yhi),
        aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
        alpha=alpha
        # alpha=0.5,
    )

    relative_levels = np.array([1e-3, 0.01, 0.05, 0.50, 0.95])
    # contour plot from data_marginal_pinn
    if(data_marginal_pinn is not None):
        X_grid = data_marginal_pinn['X_grid']
        Y_grid = data_marginal_pinn['Y_grid']
        pdf_values = data_marginal_pinn['pdf']
        pdf_max = np.max(pdf_values[pdf_values > 0])
        levels = relative_levels * pdf_max
        # print(levels)
        # Draw the contour lines with the custom levels
        ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors[-1]], 
                   linewidths=2)
    
    # contour of LP
    if(gaussian_lp is not None):
        mu_6d, cov_6d = gaussian_lp
        plot_axes = (x_coord1-1, x_coord2-1)
        marginal_mu = mu_6d[list(plot_axes)]
        marginal_cov = cov_6d[np.ix_(list(plot_axes), list(plot_axes))]

        xs = np.linspace(xrange[0], xrange[1], num=256, endpoint=True)
        ys = np.linspace(yrange[0], yrange[1], num=256, endpoint=True)
        X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
        grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

        pdf_values = p_normal(grid_pts, marginal_mu, marginal_cov).reshape(X_grid.shape)
        pdf_max = np.max(pdf_values[pdf_values > 0])
        # Define levels as percentages of the maximum value
        levels = relative_levels * pdf_max; print(levels)
        # Draw the contour lines with the custom levels
        ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors[1]], 
                linewidths=2.)

    # contour of UT
    if(gaussian_ut is not None):
        mu_6d, cov_6d = gaussian_ut
        plot_axes = (x_coord1-1, x_coord2-1)
        marginal_mu = mu_6d[list(plot_axes)]
        marginal_cov = cov_6d[np.ix_(list(plot_axes), list(plot_axes))]

        xs = np.linspace(xrange[0], xrange[1], num=256, endpoint=True)
        ys = np.linspace(yrange[0], yrange[1], num=256, endpoint=True)
        X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
        grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

        pdf_values = p_normal(grid_pts, marginal_mu, marginal_cov).reshape(X_grid.shape)
        pdf_max = np.max(pdf_values[pdf_values > 0])
        # Define levels as percentages of the maximum value
        levels = relative_levels * pdf_max; print(levels)
        # Draw the contour lines with the custom levels
        ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[colors[2]], 
                linewidths=2.)
        
    def _plot_pinn_gmm_contour(ws, mus, covs, x_coords, color):
        # print(ws.shape, mus.shape, covs.shape)
        plot_axes = (x_coords[0]-1, x_coords[1]-1)
        # print(plot_axes)
        xs = np.linspace(xrange[0], xrange[1], num=256, endpoint=True)
        ys = np.linspace(yrange[0], yrange[1], num=256, endpoint=True)
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
        ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[color], linewidths=2.)

    if(p_net_gmm is not None):
        x_coords = (x_coord1, x_coord2)
        ws, mus, covs = p_net_gmm.weights_means_covs_at(t)
        ws = ws.detach().cpu().numpy()
        print("GMM weights: ", ws, np.sum(ws))
        mus = mus.detach().cpu().numpy()
        covs = covs.detach().cpu().numpy()
        _plot_pinn_gmm_contour(ws, mus, covs, x_coords, colors[0])
        
    # Create the legend handles
    handles = []
    if gaussian_lp is not None:
        handles.append(plt.Line2D([], [], color=colors[1], label="LP", linewidth=2))
    if gaussian_ut is not None:
        handles.append(plt.Line2D([], [], color=colors[2], label="UT", linewidth=2))
    if data_marginal_pinn is not None:
        handles.append(plt.Line2D([], [], color=colors[-1], label="PINN-MLP", linewidth=2))
    if p_net_gmm is not None:
        handles.append(plt.Line2D([], [], color=colors[0], label="PINN-GMM", linewidth=2))
    ax.legend(handles=handles, loc='best')

    b=0.05
    ax.set_xlim(*np.ptp((x:=X_samples[:,x_coord1-1]))*np.array([-b,1+b])+x.min())
    ax.set_ylim(*np.ptp((y:=X_samples[:,x_coord2-1]))*np.array([-b,1+b])+y.min())
    # ax.set_xlim(xrange[0], xrange[1])
    # ax.set_ylim(yrange[0], yrange[1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f'2D Marginal PDF for {labels[0]} and {labels[1]}')
    return ax


def plot_full_corner(constants, t,
    X_samples,
    bins=200,
    labels=None,
    figsize_per_dim=1.5,
    max_points=500_000,
    mode="heatmap",            # "scatter" or "heatmap"
    cmap="cividis",
    alpha=0.4,
    point_size=2,
    ranges=None,               # list of (lo,hi) per dim; if None -> data-driven
    log_counts=True,           # log color scale for heatmap
    OUTPUT_PATH=None,
    p_net_gmm=None,
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

    sns_colors = sns.color_palette("husl", 3)
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
            ax.plot(X_grid, pdf_values, color=sns_colors[0])
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
        ax.hist(
            X[:, d],
            bins=bins,
            range=(lo, hi),
            histtype="stepfilled",
            alpha=0.85,
            color="steelblue",
            density=True,
        )
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if d == D - 1: ax.set_xlabel(labels[d])
        else: ax.set_xticklabels([])
        if d != 0: ax.set_yticklabels([])
        if d == 0: ax.set_ylabel(labels[d])
        # hide upper triangle in row d
        for k in range(d + 1, D):
            axes[d, k].axis("off")
        if(OUTPUT_PATH is not None):
            _plot_pinn_1d_marginal(d+1)

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
            print("GMM weights: ", ws, np.sum(ws))
            mus = mus.detach().cpu().numpy()
            covs = covs.detach().cpu().numpy()
            _plot_pinn_gmm_1d(ws, mus, covs, plot_axes, sns_colors[0])

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
                    alpha=alpha
                    # alpha=0.5,
                )                
                # ax.imshow(
                #     H, origin="lower",
                #     extent=(xlo, xhi, ylo, yhi),
                #     aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
                # )

            relative_levels = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95])

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
                    ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[sns_colors[0]], linewidths=0.5)
                # else:
                #     print(x_coords)

            if(OUTPUT_PATH is not None):
                x_coords = (j+1, i+1)
                _plot_pinn_contour(x_coords)

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
                print("GMM weights: ", ws, np.sum(ws))
                mus = mus.detach().cpu().numpy()
                covs = covs.detach().cpu().numpy()
                _plot_pinn_gmm_contour(ws, mus, covs, x_coords, sns_colors[0])

            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D - 1: ax.set_xlabel(labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(labels[i])
            else: ax.set_yticklabels([])
    
    plt.tick_params(axis='both', which='major', labelsize=8)
