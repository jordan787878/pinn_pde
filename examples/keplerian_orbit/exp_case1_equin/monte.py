import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from typing import Sequence, Tuple, List
from scipy.stats import norm, multivariate_normal
from exp_utilities.constants import Case1_6D_Constants, Case1_6D_Constants_Equin
from exp_utilities.plot_util import plot_time_curves_3d
import sys
sys.path.insert(0, '../utilities/')
from _General.astrodynamics import *

GRID_FOLDER = "data/grids/"
SAMPLES_FOLDER = "data/samples/"

constants = Case1_6D_Constants_Equin()
np.random.seed(0)


def p_sol_monte(t=0.0, linespace_num=31, stat_sample=10000000):
    global constants
    X = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=stat_sample).astype(np.float32)
    
    for i in tqdm(range(stat_sample), desc="Propagating samples"):
        x1, x2, x3, x4, x5, x6 = X[i, :]
        # update state
        x6 = x6 + np.sqrt(constants.MU_EARTH/x1**3)* constants.T * t
        X[i,:] = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float32)
    
    # Define bins for each dimension
    bins_x1 = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x2 = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x3 = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x4 = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x5 = np.linspace(constants.X5_RANGE[0], constants.X5_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x6 = np.linspace(constants.X6_RANGE[0], constants.X6_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)

    # Digitize to find bin indices for each dimension
    bin_indices_x1 = np.digitize(X[:, 0], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[:, 1], bins_x2) - 1
    bin_indices_x3 = np.digitize(X[:, 2], bins_x3) - 1
    bin_indices_x4 = np.digitize(X[:, 3], bins_x4) - 1  
    bin_indices_x5 = np.digitize(X[:, 4], bins_x5) - 1
    bin_indices_x6 = np.digitize(X[:, 5], bins_x6) - 1  

    # Initialize frequency
    # NOTE: for 6d PDF, N_d > 51 exceeds memory
    N_d = linespace_num-1
    frequency = np.zeros((N_d, N_d, N_d, N_d, N_d, N_d)).astype(np.float32)

    # Count occurrences in each bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        idx_x1 = bin_indices_x1[i]
        idx_x2 = bin_indices_x2[i]
        idx_x3 = bin_indices_x3[i]
        idx_x4 = bin_indices_x4[i]
        idx_x5 = bin_indices_x5[i]
        idx_x6 = bin_indices_x6[i]

        # Check if the indices are valid
        if (0 <= idx_x1 < frequency.shape[0] and
            0 <= idx_x2 < frequency.shape[1] and
            0 <= idx_x3 < frequency.shape[2] and
            0 <= idx_x4 < frequency.shape[3] and
            0 <= idx_x5 < frequency.shape[4] and
            0 <= idx_x6 < frequency.shape[5]
            ):
            frequency[idx_x1, idx_x2, idx_x3, idx_x4, idx_x5, idx_x6] += 1
            # print("bin into: ", idx_x1, idx_x2, idx_x3, idx_x4)

    # Normalize the frequency to get the probability density
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    dx3 = bins_x3[1] - bins_x3[0]
    dx4 = bins_x4[1] - bins_x4[0]
    dx5 = bins_x5[1] - bins_x5[0]
    dx6 = bins_x6[1] - bins_x6[0]
    frequency /= (dx1 * dx2 * dx3 * dx4 * dx5 * dx6 * stat_sample) 
    # # NOTE: I do not normalize the p with "r"*dr*dphi*dr_dot*dphi_dot. Instead, a simple version is used.

    # Check the sum of the probability density function
    print("[check] sum pdf(monte) = 1.0", np.sum(frequency) * dx1 * dx2 * dx3 * dx4 * dx5 * dx6)

    # Calculate the midpoints for bins (optional, depending on your needs)
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
    midpoints_x4 = (bins_x4[:-1] + bins_x4[1:]) / 2
    midpoints_x5 = (bins_x5[:-1] + bins_x5[1:]) / 2
    midpoints_x6 = (bins_x6[:-1] + bins_x6[1:]) / 2
    return midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4, midpoints_x5, midpoints_x6, frequency


def p_init(constants, x):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    # print(constants.COV_I)
    # pdf_func = multivariate_normal(mean=constants.MEAN_I, cov=constants.COV_I)
    # pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    # return pdf_eval
    # (the following is more stable due to decoupling)
    pdf_eval = np.float32(1.0)
    for i in range(6):
        pdf_func_i = multivariate_normal(mean=constants.MEAN_I[i], cov=constants.COV_I[i,i])
        x_i = x[:,i].astype(np.float32)
        pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,1).astype(x_i.dtype)
    return pdf_eval


def p_init_scaled(constants, x):
    """
    x is from the scaled cooridnate: x_i = (x_i_notscaled - mu_i)/std
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    pdf_func = multivariate_normal(mean=constants.N_MEAN_I, cov=constants.N_COV_I)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    return pdf_eval


def p_sol(constants, x, t_prime):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    # shift x
    x1 = x[:, 0]
    x6 = x[:, 5]
    x6_shift = x6 - (t_prime * constants.T) * np.sqrt(constants.MU_EARTH/ x1**3)
    pdf_eval = np.float32(1.0)
    for i in range(6):
        pdf_func_i = multivariate_normal(mean=constants.MEAN_I[i], cov=constants.COV_I[i,i])
        if(i == 5):
            x_i = x6_shift
        else:
            x_i = x[:,i].astype(np.float32)
        pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,1).astype(x_i.dtype)
    return pdf_eval


def p_sol_scaled(constants, x, t_prime):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    # shift x
    x1 = x[:, 0]*constants.COV_I[0,0]**0.5 + constants.MEAN_I[0]
    x6 = x[:, 5]
    x6_shift = x6 - (t_prime) * np.sqrt(constants.MU_EARTH/ x1**3) * constants.T / (constants.COV_I[5,5]**0.5)
    pdf_eval = np.float32(1.0)
    for i in range(6):
        pdf_func_i = multivariate_normal(mean=constants.N_MEAN_I[i], cov=constants.N_COV_I[i,i])
        if(i == 5):
            x_i = x6_shift
        else:
            x_i = x[:,i].astype(np.float32)
        pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,1).astype(x_i.dtype)
    return pdf_eval


# def p_sol_vectorize(constants, x, t_prime, return_log=False, dtype_out=np.float32):
#     """
#     Diagonal-Gaussian evaluator in the log domain.
#     - Does NOT modify x in place.
#     - Assumes: diag(constants.COV_I) > 0 and x[:,0] > 0.
#     """
#     _TWO_PI = np.float32(2.0 * np.pi)
#     x = np.asarray(x, dtype=np.float32, order="C")
#     N, D = x.shape
#     assert D == 6

#     mean = np.asarray(constants.MEAN_I, dtype=np.float32).reshape(1, D)
#     var  = np.asarray(np.diag(constants.COV_I), dtype=np.float32).reshape(1, D)

#     # Optional defensive checks (no clipping applied)
#     if not np.all(var > 0):
#         raise ValueError("COV_I diagonal must be strictly positive.")
#     if not np.all(x[:, 0] > 0):
#         raise ValueError("x1 must be > 0 for sqrt(MU_EARTH / x1**3).")

#     inv_var = 1.0 / var

#     # Shift for x6: g(x1) = sqrt(MU_EARTH / x1^3) * (T * t')
#     g = (t_prime * constants.T) * np.sqrt(constants.MU_EARTH / (x[:, 0] ** 3), dtype=np.float32)

#     # Differences; special-case dim 5 (x6)
#     diff = x - mean
#     diff[:, 5] = x[:, 5] - g - mean[0, 5]

#     # log N(x|mu, diag(var)) = -0.5 * [ sum_i( (diff_i^2 / var_i) + log(2π var_i) ) ]
#     quad = np.sum(diff * diff * inv_var, axis=1, dtype=np.float32)
#     sum_log_2pi_var = np.sum(np.log(_TWO_PI * var, dtype=np.float32))
#     logp = -0.5 * (quad + sum_log_2pi_var)

#     if return_log:
#         return logp.astype(dtype_out)
#     return np.exp(logp, dtype=np.float32).astype(dtype_out)


def wrap_to_pi(theta):
    """
    Wrap angles to [-pi, pi).
    """
    return (theta + np.pi) % (2*np.pi) - np.pi


def fit_p_init_Gaussian(stat_sample=1000000):
    constants = Case1_6D_Constants()
    X = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)
    
    # convert rns to cartesian
    X_cart = sphere_to_cartesian(rnsphere_to_sphere(X, constants.TI, constants))
    for i in tqdm(range(stat_sample), desc="Propagating samples"):
        _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(X_cart[i,:], constants.MU_EARTH)
        _m = true_to_mean_anomaly(_e, _nu)

        x1 = _a
        x2 = _e*np.sin(_w+_RAAN)
        x3 = _e*np.cos(_w+_RAAN)
        x4 = np.sin(_i)*np.sin(_RAAN)/(1.+np.cos(_i))
        x5 = np.sin(_i)*np.cos(_RAAN)/(1.+np.cos(_i))
        x6 = _m + _w + _RAAN
        x6 = wrap_to_pi(x6)
        X[i, :] = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float32)
    
    
    # now I want to fit a single Gaussian for each column of X, how can I do so? and return the mean and covariance of each fitting.
    mu_vec  = X.mean(axis=0)                         # shape (6,)
    var_vec = X.var(axis=0, ddof=0)                  # shape (6,)
    cov_diag = np.diag(var_vec)                      # 6x6 diagonal covariance
    min_vec = X.min(axis=0)              # shape (6,)
    max_vec = X.max(axis=0)              # shape (6,)

    for i in range(X.shape[1]):
        x_col = X[:, i]
        pdf_func = multivariate_normal(mean=mu_vec[i], cov=cov_diag[i,i])
        x_points = np.linspace(min_vec[i], max_vec[i], endpoint=True)
        pdf_values = pdf_func.pdf(x_points).reshape(-1,)
        plt.figure()
        plt.hist(x_col, 20, density=True) # this is a count of data --> needs to be normalized to a pdf
        plt.plot(x_points, pdf_values) # this is a fitted pdf
    plt.show()
    return mu_vec, cov_diag, min_vec, max_vec


def generate_data(data_folder, N_samples):
    t_span = constants.T_PRIME_SPAN
    mc_time = []
    for t_prime in t_span:
        start_time = time.time()
        
        x1s, x2s, x3s, x4s, x5s, x6s, pdf = p_sol_monte(t=t_prime, linespace_num=31, stat_sample=N_samples)   
        mc_time.append(time.time() - start_time)
        np.save(data_folder+"pdf_t{:.3f}.npy".format(t_prime), pdf)
        if t_prime == 0.0:
            np.save(GRID_FOLDER+"x1s.npy", x1s)
            np.save(GRID_FOLDER+"x2s.npy", x2s)
            np.save(GRID_FOLDER+"x3s.npy", x3s)
            np.save(GRID_FOLDER+"x4s.npy", x4s)
            np.save(GRID_FOLDER+"x5s.npy", x5s)
            np.save(GRID_FOLDER+"x6s.npy", x6s)

    np.save(data_folder+"mc_time.npy", np.array(mc_time))


def print_mc_time(mc_folder):
    mc_time = np.load(mc_folder+"mc_time.npy") # print mc computation time
    print("[check] MC computation time: ", mc_time)


def test_pfunc():
    global constants
    N_samples = 100
    _x = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], N_samples),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], N_samples),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], N_samples),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], N_samples),
        np.random.uniform(constants.X5_RANGE[0], constants.X5_RANGE[1], N_samples),
        np.random.uniform(constants.X6_RANGE[0], constants.X6_RANGE[1], N_samples),
    ])
    p_from_init = p_init(constants, _x).reshape(-1,)
    p_from_sol = p_sol(constants, _x, 0.0).reshape(-1,)
    print("[check] ", np.max(np.abs(p_from_init-p_from_sol)))
    

# def _iter_blocks(sizes, block_sizes):
#     """
#     Yield 6D index ranges for blocks covering a cartesian grid:
#     sizes = [N1,...,N6], block_sizes = [B1,...,B6]
#     Yields tuples of slices (s1,e1),...,(s6,e6) with e exclusive.
#     """
#     ranges = [ [(s, min(s+b, n)) for s in range(0, n, b)]
#                for n, b in zip(sizes, block_sizes) ]
#     for a1,b1 in ranges[0]:
#         for a2,b2 in ranges[1]:
#             for a3,b3 in ranges[2]:
#                 for a4,b4 in ranges[3]:
#                     for a5,b5 in ranges[4]:
#                         for a6,b6 in ranges[5]:
#                             yield (a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6)

# def _make_block_points(x_axes, idxs, device, dtype):
#     """
#     Build a small block’s grid points (K,6) tensor from per-axis arrays and index ranges.
#     """
#     a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = idxs
#     Xs = [x_axes[0][a1:b1], x_axes[1][a2:b2], x_axes[2][a3:b3],
#           x_axes[3][a4:b4], x_axes[4][a5:b5], x_axes[5][a6:b6]]
#     g = np.meshgrid(*Xs, indexing="ij")  # small block only
#     pts = np.stack([gi.ravel() for gi in g], axis=1)  # (K,6) in numpy
#     return torch.as_tensor(pts, dtype=dtype, device=device), [b1-a1,b2-a2,b3-a3,b4-a4,b5-a5,b6-a6]

# def _open_memmap_npy(path, shape, dtype=np.float32):
#     """
#     Create/overwrite a .npy file as a memmap so we can write slices incrementally.
#     """
#     from numpy.lib.format import open_memmap
#     # mode='w+' creates the file and allows writing
#     return open_memmap(path, mode='w+', dtype=dtype, shape=shape)

# def precompute_pdfsol_streaming(constants, p_sol, out_dir, dtype=np.float32,
#                              grid_dir="data/grids", block_target_points=1_000_000,
#                              inner_batch_points=500_000, device="cpu"):
#     """
#     Streams the 6D grid in blocks; for each time t in constants.T_PRIME_SPAN:
#       - writes pdf_t_<t>.npy incrementally using a memmap
#     No full 30^6 grid or pdf is ever kept in memory.
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     # load per-axis grids (numpy 1D arrays)
#     x_axes = [
#         np.load(os.path.join(grid_dir, f"x{i}s.npy")).astype(dtype, copy=False)
#         for i in range(1, 7)
#     ]
#     sizes = [len(a) for a in x_axes]  # expect [30,30,30,30,30,30]

#     # choose roughly cubic block sizes so product ~ block_target_points (cap by axis length)
#     root = max(1, round(block_target_points ** (1/6)))
#     block_sizes = [min(n, root) for n in sizes]
#     # nudge down if still too big
#     def prod(v):
#         r = 1
#         for x in v: r *= x
#         return r
#     while prod(block_sizes) > block_target_points:
#         i = int(np.argmax(block_sizes))
#         block_sizes[i] = max(1, block_sizes[i]-1)

#     print(f"[info] grid sizes: {sizes}, block sizes: {block_sizes} (~{prod(block_sizes):,} pts/block)")

#     # iterate all requested times
#     for t in constants.T_PRIME_SPAN:
#         out_path = os.path.join(out_dir, f"pdfsol_t{t:.3f}.npy")
#         print(f"[time {t:.3f}] writing -> {out_path}")

#         # create memmap file for this time
#         pdf_mm = _open_memmap_npy(out_path, shape=tuple(sizes), dtype=dtype)

#         with torch.no_grad():
#             # constant time tensor will be built per inner batch
#             for idxs in _iter_blocks(sizes, block_sizes):
#                 # build block points on device
#                 pts_block, blk_shape = _make_block_points(x_axes, idxs, device, torch.float32)
#                 K = pts_block.shape[0]

#                 # process this block in inner batches to fit GPU/CPU memory
#                 vals_list = []
#                 for start in range(0, K, inner_batch_points):
#                     end = min(start + inner_batch_points, K)
#                     x_batch = pts_block[start:end, :6]
#                     x_batch_numpy = x_batch.detach().cpu().numpy()
#                     y = p_sol(constants, x_batch_numpy, t)           # (B,1) or (B,)
#                     vals_list.append(y)

#                 vals = np.concatenate(vals_list, axis=0)   # (K,)
#                 vals = vals.reshape(blk_shape)             # reshape to local block shape

#                 # write into proper slice of the memmap
#                 a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6 = idxs
#                 pdf_mm[a1:b1, a2:b2, a3:b3, a4:b4, a5:b5, a6:b6] = vals

#         # ensure data is flushed
#         del pdf_mm
#         print(f"[time {t:.3f}] done.")


def _integrate_out_others(pdf: np.ndarray, axes_coords: Sequence[np.ndarray], keep_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate a 6-D PDF over all axes except keep_axis using trapezoids (non-uniform grids OK)."""
    f = np.asarray(pdf, dtype=float)
    for ax in sorted([i for i in range(f.ndim) if i != keep_axis], reverse=True):
        f = np.trapz(f, x=np.asarray(axes_coords[ax], dtype=float), axis=ax)
    return np.asarray(axes_coords[keep_axis], dtype=float), f


def _integrate_all_axes(f: np.ndarray, axes_coords: Sequence[np.ndarray]) -> float:
    """∫ f(x) dx over all axes via repeated trapezoids."""
    g = np.asarray(f, dtype=float)
    for ax in reversed(range(g.ndim)):
        g = np.trapz(g, x=np.asarray(axes_coords[ax], dtype=float), axis=ax)
    return float(g)


def mean_from_pdf_uniform_box(X, pdf_vals, eps=1e-300):
    """
    X: (N,6) samples drawn uniformly from a bounded box
    pdf_vals: (N,) p(X) evaluated at those samples (needn't be normalized)
    returns: (6,) estimated mean vector under p
    """
    w = np.clip(np.asarray(pdf_vals, dtype=np.float64), 0.0, None)
    denom = w.sum() + eps
    mu = (w[:, None] * np.asarray(X, dtype=np.float64)).sum(axis=0) / denom
    return mu


def compute_marginals_over_time(
    times: np.ndarray,
    data_dir: str = "data/1e+6",
    grid_dir: str = "data/grids",
    axes_files: Sequence[str] = ("x1s.npy","x2s.npy","x3s.npy","x4s.npy","x5s.npy","x6s.npy"),
    keep_axis: int = 0,
    filename_fmt: str = "pdf_t{:.3f}.npy",
):
    """Return x_keep, kept times, M[:, j]=p(x_keep|t_j), plus mean comparisons (joint vs marginal)."""
    times = np.asarray(times, dtype=float).ravel()

    # load grids
    axes = [np.load(os.path.join(grid_dir, fn)) for fn in axes_files]
    x_keep = np.asarray(axes[keep_axis], dtype=float)

    # filter to times with existing files
    file_exists = np.array([os.path.isfile(os.path.join(data_dir, filename_fmt.format(t))) for t in times], dtype=bool)
    times_kept = times[file_exists]
    if times_kept.size == 0:
        raise FileNotFoundError(f"No matching files in '{data_dir}' using format '{filename_fmt}'")

    # preallocate
    M = np.empty((x_keep.size, times_kept.size), dtype=float)
    mean_joint = np.empty(times_kept.size, dtype=float)
    mean_marg  = np.empty(times_kept.size, dtype=float)
    mass6D     = np.empty(times_kept.size, dtype=float)  # ∫ p(x) dx over 6D
    mass1D     = np.empty(times_kept.size, dtype=float)  # ∫ p(x_keep) dx_keep

    # broadcast shape helper for x_keep on the keep_axis
    shape = [1]*6
    shape[keep_axis] = x_keep.size
    xk_grid = x_keep.reshape(shape)

    x_grids = constants.get_xinputs_on_grids("data/grids/").detach().numpy()

    # process each snapshot
    for j, t in enumerate(times_kept):
        fpath = os.path.join(data_dir, filename_fmt.format(t))
        print(fpath)
        # f_precompute = np.asarray(np.load(fpath), dtype=float)
        f = p_sol(constants, x_grids, t).reshape((30,30,30,30,30,30))
        print("p_sol done.")

        if f.ndim != 6:
            raise ValueError(f"{fpath} is not 6-D (got {f.ndim})")
        for i in range(6):
            if f.shape[i] != axes[i].size:
                raise ValueError(f"Shape mismatch at axis {i}: pdf {f.shape[i]} vs axis {axes[i].size} in {fpath}")

        # ---- (A) mean from the 1D marginal ----
        xk, mk = _integrate_out_others(f, axes, keep_axis)
        area_1d = np.trapz(mk, x=xk)
        mu_marg = (np.trapz(xk * mk, x=xk) / area_1d) if np.isfinite(area_1d) and area_1d > 0 else np.nan

        # ---- (B) mean directly from the 6D joint ----
        denom_6d = _integrate_all_axes(f, axes)                    # ∫ p(x) dx
        numer_6d = _integrate_all_axes(f * xk_grid, axes)          # ∫ x_k p(x) dx
        mu_joint = (numer_6d / denom_6d) if np.isfinite(denom_6d) and denom_6d > 0 else np.nan

        # store + print comparison
        M[:, j] = mk
        mean_joint[j] = mu_joint
        mean_marg[j]  = mu_marg
        mass6D[j]     = denom_6d
        mass1D[j]     = area_1d

        print(
            f"[t={t:.6g}] mass6D={denom_6d:.6e}  mass1D={area_1d:.6e}  "
            f"mean_joint={mu_joint:.9g}  mean_marg={mu_marg:.9g}  "
            f"Δ={mu_joint - mu_marg:+.3e}"
        )

    return x_keep, times_kept, M


def compute_marginal_pdf_along_x6():
    i = 6
    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")
    x5s = np.load("data/grids/x5s.npy")
    x6s = np.load("data/grids/x6s.npy")
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dx5 = x5s[1] - x5s[0]
    x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid = np.meshgrid(
        x1s, x2s, x3s, x4s, x5s, x6s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel(), x6_grid.ravel()]).T
    # init
    marginal_pdf_over_time = np.empty((x6s.size, constants.T_PRIME_SPAN.size))
    for t_idx, t in enumerate(constants.T_PRIME_SPAN):
        print(t)
        pdf_values = p_sol(constants, grid_points, t).reshape(x1_grid.shape)
        pdf_at_t = np.sum(pdf_values, axis=(0,1,2,3,4)) * (dx1*dx2*dx3*dx4*dx5)
        marginal_pdf_over_time[:, t_idx] = pdf_at_t
    np.savez("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz", x=x6s, times=constants.T_PRIME_SPAN, M=marginal_pdf_over_time)


def main():
    # --- Fit p_init in Equinoctial element set ---
    # mu_vec, cov_diag, min_vec, max_vec = fit_p_init_Gaussian(stat_sample=1000000)
    # np.savez("data/constants.npz", mu_vec=mu_vec, 
    #          cov_diag=cov_diag,
    #          min_vec=min_vec,
    #          max_vec=max_vec)    

    # --- Generate data ---
    global constants; constants.test_printout()
    data_folder = "data/1e+6"
    # generate_data(data_folder+"/", 100000)



    # --- Test MC results ---
    # print_mc_time(data_folder)

    # --- Pre-computation for plotting data ---
    # 1) save p(t0) max
    # p_max = p_sol(constants, constants.MEAN_I.reshape(1,-1), 0.0) # compute maximum PDF at t0 (directly evaluated at the mean of initial Gaussian)
    # print(p_max.item())
    # np.savez("data/pre_compute/p_init_max.npz", value=np.float32(p_max.item()), label="mean at Gaussian")

    # 2) save marginal pdf over time
    # for i in range(1,7): # compute marginalized PDF (to each dimension) over discrete time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir=data_folder,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdf_t{:.3f}.npy",
    #     )
    #     np.savez(data_folder+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    # precompute_pdf_init_streaming(constants, p_init, "data/1e+6")
    
    # --- Plots ---
    # for i in range(1, 7):
    #     _data = np.load(data_folder+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(i, constants, _data, title="Marginal p(x"+str(i)+",t)")

    # precompute_pdfsol_streaming(constants, p_sol, "data/pre_compute")

    # compute_marginal_pdf_along_x6()

    # for i in range(6,7): # compute marginalized PDF (to each dimension) over discrete time
    #     # x, t_kept, M = compute_marginals_over_time(
    #     #     times=constants.T_PRIME_SPAN,   # time array you mentioned
    #     #     data_dir="data/pre_compute",
    #     #     grid_dir="data/grids",
    #     #     keep_axis=(i-1),                # first dimension
    #     #     filename_fmt="pdfsol_t{:.3f}.npy",
    #     # )
    #     x, t, pdf_at_i_axis = compute_marginal_pdf(i)
    #     np.savez("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz", x=x, times=t, M=pdf_at_i_axis)
   
    # test_pfunc()

    # --- Plots ---
    # This plot validates the "analytical" joint PDF by p_sol() --> can used to validate error.
    # for i in range(6, 7):
    #     data_MC = np.load(data_folder+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     data_sol = np.load("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(i, constants, data_MC, data_sol,
    #                         leg_txt = ["p MC", "p sol."], title="Marginal p(x"+str(i)+",t)")
    

if __name__ == "__main__":
    main()