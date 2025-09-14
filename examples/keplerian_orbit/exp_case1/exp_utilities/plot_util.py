import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
import torch
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)


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
    ranges="fixed",               # list of (lo,hi) per dim; if None -> data-driven
    pad_frac=0.02,
    log_counts=True,           # log color scale for heatmap
    OUTPUT_PATH=None,
    data_lp=None,
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

    sns_colors = sns.color_palette("husl", 3)
    fig, axes = plt.subplots(D, D, figsize=(figsize_per_dim*D, figsize_per_dim*D))
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    def _plot_pinn_1d_marginal(x_coords):
        filename = f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords}_t{t:.3f}.npz"
        if(os.path.exists(filename)):
            data_marginal_pinn = np.load(filename)
            pdf_values = data_marginal_pinn['pdf']
            X_grid = data_marginal_pinn["X_grid"]
            ax.plot(X_grid, pdf_values, color=sns_colors[0])

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
        if(OUTPUT_PATH is not None):
            _plot_pinn_1d_marginal(d+1)

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

            def _plot_pinn_contour(x_coords):
                filename = f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz"
                if(os.path.isfile(filename)):
                    data_marginal_pinn = np.load(filename)
                    pdf_values = data_marginal_pinn['pdf']
                    X_grid, Y_grid = data_marginal_pinn["X_grid"], data_marginal_pinn["Y_grid"], 
                    pdf_max = np.max(pdf_values[pdf_values > 0])
                    # Define levels as percentages of the maximum value
                    # For example, levels at 5%, 25%, 50%, 75%, and 95% of the max
                    relative_levels = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95])
                    levels = relative_levels * pdf_max
                    # Draw the contour lines with the custom levels
                    ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[sns_colors[0]], linewidths=1.5)
                    return levels

            if(OUTPUT_PATH is not None):
                x_coords = (j+1, i+1)
                pinn_levels = _plot_pinn_contour(x_coords)

            if(data_lp is not None):
                x_coords = (j+1, i+1)
                _, mu_6d, cov_6d = data_lp.get(t)
                plot_axes = (x_coords[0]-1, x_coords[1]-1)
                marginal_mu = mu_6d[list(plot_axes)]
                marginal_cov = cov_6d[np.ix_(list(plot_axes), list(plot_axes))]
                xs = np.linspace(xlo, xhi, num=64, endpoint=True)
                ys = np.linspace(ylo, yhi, num=64, endpoint=True)
                X_grid, Y_grid = np.meshgrid(xs, ys, indexing="ij")
                grid_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
                pdf_values = p_normal(grid_pts, marginal_mu, marginal_cov).reshape(X_grid.shape)
                relative_levels = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95])
                pdf_max = np.max(pdf_values).item()
                levels = relative_levels * pdf_max
                # Draw the contour lines with the custom levels
                ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors=[sns_colors[1]], 
                           linewidths=1.5)

            # set_axis_limits_with_buffer(ax, (xlo, xhi), (ylo, yhi))
            # ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            if i == D - 1: ax.set_xlabel(labels[j])
            else: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(labels[i])
            else: ax.set_yticklabels([])
    plt.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout()


### helper


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


def set_axis_limits_with_buffer(ax, xlim, ylim, buffer_frac=0.05):
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


##### Obsolete Below #####


def _integrate_out_others(pdf: np.ndarray, axes_coords: Sequence[np.ndarray], keep_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate a 6-D PDF over all axes except keep_axis using trapezoids (non-uniform grids OK)."""
    f = np.asarray(pdf, dtype=float)
    for ax in sorted([i for i in range(f.ndim) if i != keep_axis], reverse=True):
        f = np.trapz(f, x=np.asarray(axes_coords[ax], dtype=float), axis=ax)
    return np.asarray(axes_coords[keep_axis], dtype=float), f


def compute_marginals_over_time(
    times: np.ndarray,
    data_dir: str = "data/1e+5",
    grid_dir: str = "data/grids",
    axes_files: Sequence[str] = ("x1s.npy","x2s.npy","x3s.npy","x4s.npy","x5s.npy","x6s.npy"),
    keep_axis: int = 0,
    filename_fmt: str = "pdf_t{:.3f}.npy",
):
    """Return x1 grid, kept times (filtered to existing files), and M[:, j]=p(x1|t_j)."""
    times = np.asarray(times, dtype=float).ravel()

    # load grids
    axes = [np.load(os.path.join(grid_dir, fn)) for fn in axes_files]
    x_keep = np.asarray(axes[keep_axis], dtype=float)

    # filter to times with existing files
    file_exists = np.array([os.path.isfile(os.path.join(data_dir, filename_fmt.format(t))) for t in times], dtype=bool)
    times_kept = times[file_exists]
    if times_kept.size == 0:
        raise FileNotFoundError(f"No matching files in '{data_dir}' using format '{filename_fmt}'")
    print(times_kept)

    # preallocate
    M = np.empty((x_keep.size, times_kept.size), dtype=float)

    # process each snapshot
    for j, t in enumerate(times_kept):
        fpath = os.path.join(data_dir, filename_fmt.format(t))
        f = np.load(fpath)
        if f.ndim != 6:
            raise ValueError(f"{fpath} is not 6-D (got {f.ndim})")
        # shape checks
        for i in range(6):
            if f.shape[i] != axes[i].size:
                raise ValueError(f"Shape mismatch at axis {i}: pdf {f.shape[i]} vs axis {axes[i].size} in {fpath}")

        # if clip_negatives:
        #     f = np.clip(f, 0.0, None)
        # # normalize full 6-D snapshot
        # g = f.copy()
        # for ax in reversed(range(6)):
        #     g = np.trapz(g, x=np.asarray(axes[ax], dtype=float), axis=ax)
        # Z = float(g)
        # if not np.isfinite(Z) or Z <= 0:
        #     raise ValueError(f"Non-positive/invalid total mass in {fpath}")
        # f /= Z

        # marginal on keep_axis
        xk, mk = _integrate_out_others(f, axes, keep_axis)
        # mk = np.clip(mk, 0.0, None)
        # # ensure each marginal integrates to ~1 over xk
        area = np.trapz(mk, x=xk)
        print("[check] pdf total: ", t, area)
        if area > 0 and np.isfinite(area):
            mk /= area

        M[:, j] = mk

    return x_keep, times_kept, M

# ---------- plotting ----------
def plot_time_curves_3d(x_components, constants, data1, data2=None, p_init=None, title="Marginal p(x1|t): curves (no interpolation)"):
    """
    Plot one 3D curve per time: (x1, t_fixed, p(x1|t_fixed)).
    No faces between times -> no interpolation across time.
    """
    set_publication_plot_style()

    # pick a seaborn color palette
    colors = sns.color_palette("tab10")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- first dataset ---
    x_keep = data1["x"]; times = data1["times"]; M = data1["M"]
    for j, t in enumerate(times):
        color = colors[j % len(colors)]
        ax.plot(np.full_like(x_keep, t), x_keep, M[:, j],
                color="blue", linestyle="--")
        if abs(t) < 1e-5 and p_init is not None:
                p_init_mc_marginal = M[:, j]
                pdf_func = multivariate_normal(
                    mean=constants.N_MEAN_I[x_components-1],
                    cov=constants.N_COV_I[x_components-1, x_components-1])
                p_init_analy_marginal = pdf_func.pdf(x_keep).astype(x_keep.dtype)
                rel_acc = (np.max(np.abs(p_init_mc_marginal - p_init_analy_marginal))
                          / np.max(np.abs(p_init_analy_marginal)))
                print(f"[check] rel. accuracy of PINN   (marginal to x{x_components:1d}) "
                      f"--> the deviation between p(t0) PINN and p(t0): {100.0*rel_acc:.2f} %")

    # --- second dataset ---
    if data2 is not None:
        x_keep = data2["x"]; times = data2["times"]; M = data2["M"]
        for j, t in enumerate(times):
            color = colors[j % len(colors)]
            ax.plot(np.full_like(x_keep, t), x_keep, M[:, j],
                    color="black", linestyle="-", marker="o", markersize=3)
            if abs(t) < 1e-5 and p_init is not None:
                p_init_mc_marginal = M[:, j]
                pdf_func = multivariate_normal(
                    mean=constants.N_MEAN_I[x_components-1],
                    cov=constants.N_COV_I[x_components-1, x_components-1])
                p_init_analy_marginal = pdf_func.pdf(x_keep).astype(x_keep.dtype)
                rel_acc = (np.max(np.abs(p_init_mc_marginal - p_init_analy_marginal))
                          / np.max(np.abs(p_init_analy_marginal)))
                print(f"[check] rel. accuracy of MC   (marginal to x{x_components:1d}) "
                      f"--> the deviation between p(t0) MC and p(t0): {100.0*rel_acc:.2f} %")
                ax.plot(np.full_like(x_keep, t), x_keep, p_init_analy_marginal, lw=2.0, color="green", linestyle=":")

    # labels — add padding to avoid collisions with tick labels
    ax.set_xlabel(r"$t/T$", labelpad=8)
    ax.set_ylabel(r"$x$" + str(x_components), labelpad=10)
    ax.set_zlabel(r"$p(x$" + str(x_components) + r"$,t)$", labelpad=8)

    # give tick labels a bit of breathing room
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='z', pad=2)

    ax.view_init(elev=25, azim=-60)

    # legend
    legend_elements = [
        Line2D([0], [0], color="blue", linestyle="--", label=r"$\hat{p}$"),
        Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=3, label=r"$p_{MC}$"),
        Line2D([0], [0], color="green", linestyle=":",  lw=2.0, label=r"$p(t_0)$ (analy.)")
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True)

    # No tight_layout with big pad — constrained_layout already did the work.
    # For export: trim outer whitespace aggressively.
    savepath = f"figs/Case_6D_marginal_PDF{int(x_components)}.pdf"
    plt.savefig(savepath, format="pdf", dpi=300,
                bbox_inches='tight', pad_inches=0.5)
    plt.close()


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
