import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple
import torch
import seaborn as sns
from scipy.stats import norm, multivariate_normal
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)


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

    # pick a seaborn color palette
    colors = sns.color_palette("tab10")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    # --- first dataset ---
    x_keep = data1["x"]; times = data1["times"]; M = data1["M"]
    for j, t in enumerate(times):
        color = colors[j % len(colors)]
        ax.plot(np.full_like(x_keep, t), x_keep, M[:, j], lw=1.7,
                color=color, linestyle="-")
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
                ax.plot(np.full_like(x_keep, t), x_keep, p_init_analy_marginal, lw=2.0, color="black", linestyle=":")

    # --- second dataset ---
    if data2 is not None:
        x_keep = data2["x"]; times = data2["times"]; M = data2["M"]
        for j, t in enumerate(times):
            color = colors[j % len(colors)]
            ax.plot(np.full_like(x_keep, t), x_keep, M[:, j], lw=1.7,
                    color=color, linestyle="--")
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
                ax.plot(np.full_like(x_keep, t), x_keep, p_init_analy_marginal, lw=2.0, color="black", linestyle=":")

    # axis labels and title
    print("\n")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$x$"+str(x_components))
    ax.set_zlabel(r"$p(x$"+str(x_components)+r"$,t)$")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()

    # --- legend with linestyle meaning ---
    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", lw=1.7, label=r"$\hat{p}$"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.7, label=r"$p_{MC}$")
    ]
    ax.legend(handles=legend_elements, loc="best")

    plt.show()


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
