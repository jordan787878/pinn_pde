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
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator


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


def plot_pdf_metrics(metrics, data_normalize_e1_pinn_max=None):
    colors = sns.color_palette("husl", 3)

    # Get PINN Error Bound over time
    if(data_normalize_e1_pinn_max is not None):
        times = data_normalize_e1_pinn_max["times"]
        # synchronize times
        idx_times = _indices_in_dense(times, metrics["t"], tol=1e-4)
        times = times[idx_times]
        e1_pinn_max = data_normalize_e1_pinn_max["values"][:, -1]
        B1 = 2.*e1_pinn_max[idx_times]
        print(times)
        print(B1)

    plt.figure()
    plt.plot(metrics["t"], metrics["rel_error_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    if(data_normalize_e1_pinn_max is not None):
        plt.fill_between(
            metrics["t"],
            metrics["rel_error_pinn"],
            100.*B1,
            color=colors[0],
            alpha=0.2,
            label="PINN Error Bound"
        )
    plt.plot(metrics["t"], metrics["rel_error_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["rel_error_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("worst rel. error %")

    plt.figure()
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("total variation %")

    # metric 3: negative log liklihood (relative KL)
    plt.figure()
    plt.plot(metrics["t"], metrics["rel_kl_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["rel_kl_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["rel_kl_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    # plt.plot(metrics["t"], metrics["rel_kl_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Negative Log Likelihood")

    plt.show()


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


def corner_plot_single(
    constants,
    X_samples,
    pdf_vals=None,                # optional weights
    bins=200,
    labels=None,
    figsize_per_dim=1.5,
    normalize_weights=False,
    max_points=10_000,           # for scatter mode speed
    mode="heatmap",               # "scatter" or "heatmap"
    cmap="viridis",
    alpha=1.0,                    # scatter transparency
    point_size=3
):
    """
    Corner plot with 1D weighted histograms and either scatter or heatmap for off-diagonals.
    """
    X = np.asarray(X_samples)
    if pdf_vals is None:
        w = np.ones(X.shape[0])
    else:
        w = np.asarray(pdf_vals)
    if normalize_weights and np.sum(w) > 0:
        w = w / np.sum(w)

    N, D = X.shape
    if labels is None:
        labels = [f"x{i+1}" for i in range(D)]

    ranges = [
        (constants.X1_RANGE[0],constants.X1_RANGE[1]),
        (constants.X2_RANGE[0],constants.X2_RANGE[1]),
        (constants.X3_RANGE[0],constants.X3_RANGE[1]),
        (constants.X4_RANGE[0],constants.X4_RANGE[1]),
        (constants.X5_RANGE[0],constants.X5_RANGE[1]),
        (constants.X6_RANGE[0],constants.X6_RANGE[1]),
    ]
    # mins = np.quantile(X, 0.001, axis=0)
    # maxs = np.quantile(X, 0.999, axis=0)
    # pad  = 0.02 * (maxs - mins + 1e-12)
    # ranges = [(float(mins[i]-pad[i]), float(maxs[i]+pad[i])) for i in range(X.shape[1])]


    fig, axes = plt.subplots(D, D, figsize=(figsize_per_dim*D, figsize_per_dim*D))
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    # Diagonals: weighted 1D histograms
    for d in range(D):
        ax = axes[d, d]
        lo, hi = ranges[d]
        ax.hist(
            X[:, d],
            bins=bins,
            range=(lo, hi),
            weights=w,
            histtype="stepfilled",
            alpha=0.8,
            color="steelblue"
        )
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if d == D-1:
            ax.set_xlabel(labels[d])
        else:
            ax.set_xticklabels([])
        if d != 0:
            ax.set_yticklabels([])

        # Hide upper triangle
        for k in range(d+1, D):
            axes[d, k].axis("off")

    # Off-diagonals: scatter or heatmap
    for i in range(1, D):
        for j in range(i):
            ax = axes[i, j]
            (xlo, xhi), (ylo, yhi) = ranges[j], ranges[i]

            if mode == "scatter":
                # Subsample for speed if too many points
                idx = np.arange(N)
                if max_points is not None and N > max_points:
                    idx = np.random.default_rng(0).choice(N, size=max_points, replace=False)
                sc = ax.scatter(
                    X[idx, j], X[idx, i],
                    c=w[idx],
                    cmap=cmap,
                    # alpha=alpha,
                    s=point_size,
                    edgecolors="none"
                )

            elif mode == "heatmap":
                H, xedges, yedges = np.histogram2d(
                    X[:, j], X[:, i],
                    bins=bins,
                    range=[(xlo, xhi), (ylo, yhi)],
                    weights=w
                )
                H = H.T
                ax.imshow(
                    H,
                    origin="lower",
                    extent=(xlo, xhi, ylo, yhi),
                    aspect="auto",
                    cmap=cmap,
                    norm=LogNorm(vmax=H.max() if H.max() > 0 else 1)
                )

            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)

            if i == D-1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])

    fig.tight_layout()


def dim_ranges_minmax(X, ignore_nan=True):
    X = np.asarray(X)
    mins = np.nanmin(X, axis=0) if ignore_nan else np.min(X, axis=0)
    maxs = np.nanmax(X, axis=0) if ignore_nan else np.max(X, axis=0)
    # shape (6, 2): [[x1_min, x1_max], ..., [x6_min, x6_max]]
    return np.stack([mins, maxs], axis=1)


def corner_compare_overlay_lower(
    constants,
    X, wA, wB,
    bins=200,
    labels=None,
    figsize_per_dim=1.5,
    normalize_weights=False,
    cmapA="Blues",
    cmapB="Reds",
    alphaA=0.3,
    alphaB=0.3,
    add_contours=False,           # optional thin outlines to help in overlaps
    contour_colorA="#1f4ab8",
    contour_colorB="#9a1b1b",
):
    """
    Corner plot comparing two weighted distributions using *overlaid* lower-triangle heatmaps.
      - Diagonal: overlaid 1D weighted histograms (A vs B)
      - Lower triangle: for each (i,j), overlay 2D weighted histograms of A and B
      - Upper triangle: hidden

    Args:
        X:  (N,D) shared sample locations
        wA: (N,) weights of distribution A (e.g., reference)
        wB: (N,) weights of distribution B (e.g., PINN)
    """
    range_npy = dim_ranges_minmax(X)  # array of shape (6, 2)
    print(range_npy)
    # ranges = [tuple(r) for r in range_npy]
    ranges = [
        (constants.X1_RANGE[0],constants.X1_RANGE[1]),
        (constants.X2_RANGE[0],constants.X2_RANGE[1]),
        (constants.X3_RANGE[0],constants.X3_RANGE[1]),
        (constants.X4_RANGE[0],constants.X4_RANGE[1]),
        (constants.X5_RANGE[0],constants.X5_RANGE[1]),
        (constants.X6_RANGE[0],constants.X6_RANGE[1]),
    ]

    X = np.asarray(X)
    wA = np.asarray(wA).reshape(-1)
    wB = np.asarray(wB).reshape(-1)
    if X.ndim != 2: raise ValueError("X must be (N, D).")
    if wA.shape[0] != X.shape[0] or wB.shape[0] != X.shape[0]:
        raise ValueError("wA and wB must have length N.")
    if normalize_weights:
        sA, sB = wA.sum(), wB.sum()
        if sA > 0: wA = wA / sA
        if sB > 0: wB = wB / sB

    N, D = X.shape
    if labels is None:
        labels = [f"x{i+1}" for i in range(D)]

    fig, axes = plt.subplots(D, D, figsize=(figsize_per_dim*D, figsize_per_dim*D))
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    # Diagonals: overlay 1D histograms
    for d in range(D):
        ax = axes[d, d]
        lo, hi = ranges[d]
        ax.hist(X[:, d], bins=bins, range=(lo, hi), weights=wA,
                histtype="stepfilled", alpha=0.55, color="#3b82f6", label="A")
        ax.hist(X[:, d], bins=bins, range=(lo, hi), weights=wB,
                histtype="stepfilled", alpha=0.55, color="#ef4444", label="B")
        ax.hist(X[:, d], bins=bins, range=(lo, hi), weights=wA,
                histtype="step", color="#1f4ab8", linewidth=0.9)
        ax.hist(X[:, d], bins=bins, range=(lo, hi), weights=wB,
                histtype="step", color="#9a1b1b", linewidth=0.9)
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if d == D-1: ax.set_xlabel(labels[d])
        else: ax.set_xticklabels([])
        if d != 0: ax.set_yticklabels([])
        for k in range(d+1, D):
            axes[d, k].axis("off")

    # Lower triangle: overlay heatmaps with shared vmax per cell
    for i in range(1, D):
        for j in range(i):
            ax = axes[i, j]
            (xlo, xhi), (ylo, yhi) = ranges[j], ranges[i]

            HA, xedges, yedges = np.histogram2d(
                X[:, j], X[:, i], bins=bins,
                range=[(xlo, xhi), (ylo, yhi)], weights=wA
            )
            HB, _, _ = np.histogram2d(
                X[:, j], X[:, i], bins=bins,
                range=[(xlo, xhi), (ylo, yhi)], weights=wB
            )
            HA, HB = HA.T, HB.T
            vmax = float(max(HA.max(), HB.max(), 1e-12))

            # Plot A then B with transparency, same normalization
            ax.imshow(
                HA, origin="lower",
                extent=(xlo, xhi, ylo, yhi), aspect="auto",
                cmap=cmapA, alpha=alphaA,
                norm=LogNorm(vmin=1e-12, vmax=vmax)
            )
            ax.imshow(
                HB, origin="lower",
                extent=(xlo, xhi, ylo, yhi), aspect="auto",
                cmap=cmapB, alpha=alphaB,
                norm=LogNorm(vmin=1e-12, vmax=vmax)
            )

            # Optional thin outlines to help where colors overlap
            if add_contours and (HA.max() > 0 or HB.max() > 0):
                levels = np.geomspace(max(vmax*1e-3, 1e-12), vmax, 5)
                ax.contour(
                    0.5*(xedges[:-1]+xedges[1:]),
                    0.5*(yedges[:-1]+yedges[1:]),
                    HA, levels=levels, colors=contour_colorA, linewidths=0.5
                )
                ax.contour(
                    0.5*(xedges[:-1]+xedges[1:]),
                    0.5*(yedges[:-1]+yedges[1:]),
                    HB, levels=levels, colors=contour_colorB, linewidths=0.5, linestyles="--"
                )

            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D-1: ax.set_xlabel(labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(labels[i])
            else: ax.set_yticklabels([])

    axes[0,0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()



from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

def corner_plot_from_samples(constants,
    X_samples,
    bins=200,
    labels=None,
    figsize_per_dim=1.5,
    max_points=500_000,
    mode="heatmap",            # "scatter" or "heatmap"
    cmap="viridis",
    alpha=0.4,
    point_size=2,
    ranges=None,               # list of (lo,hi) per dim; if None -> data-driven
    quantile_range=(0.001, 0.999),  # set to None to use full min/max
    pad_frac=0.02,
    log_counts=True,           # log color scale for heatmap
):
    X = np.asarray(X_samples)
    N, D = X.shape
    if labels is None:
        labels = [f"x{i+1}" for i in range(D)]

    # ---- ranges ----
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
    ranges = [
        (constants.X1_RANGE[0],constants.X1_RANGE[1]),
        (constants.X2_RANGE[0],constants.X2_RANGE[1]),
        (constants.X3_RANGE[0],constants.X3_RANGE[1]),
        (constants.X4_RANGE[0],constants.X4_RANGE[1]),
        (constants.X5_RANGE[0],constants.X5_RANGE[1]),
        (constants.X6_RANGE[0],constants.X6_RANGE[1]),
    ]

    fig, axes = plt.subplots(D, D, figsize=(figsize_per_dim*D, figsize_per_dim*D))
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

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
            color="steelblue"
        )
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if d == D - 1: ax.set_xlabel(labels[d])
        else: ax.set_xticklabels([])
        if d != 0: ax.set_yticklabels([])

        # hide upper triangle in row d
        for k in range(d + 1, D):
            axes[d, k].axis("off")

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
                ax.imshow(
                    H, origin="lower",
                    extent=(xlo, xhi, ylo, yhi),
                    aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
                )

            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D - 1: ax.set_xlabel(labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(labels[i])
            else: ax.set_yticklabels([])

    fig.tight_layout()
