import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Assuming the following imports and setup are the same as your original code
from monte import p_init_scaled
from exp_utilities.constants import Case1_6D_Constants_Equin
from _General.neuralnetworks import PNet_Scaled, PNet_XL, load_trained_model

# --- Initial setup (as in your original script) ---
constants = Case1_6D_Constants_Equin()
device = "cpu"


def marginal_pinn(
    p_net, t_val, constants,
    plot_axes,
    num_linespace=256,
    num_samples_mc=10000,
    batch_size=16,
    device="cpu",
    save_path=None,
    renormalize_over_window=True
):
    xrange = getattr(constants, f"N_X{plot_axes[0]+1}_RANGE")
    yrange = getattr(constants, f"N_X{plot_axes[1]+1}_RANGE")
    x1_vals_scaled = np.linspace(xrange[0], xrange[1], num=num_linespace, endpoint=True)
    x6_vals_scaled = np.linspace(yrange[0], yrange[1], num=num_linespace, endpoint=True)

    # axes
    all_axes = [0,1,2,3,4,5]
    integration_axes = [i for i in all_axes if i not in plot_axes]  # [1,2,3,4]

    # fixed integration ranges (DO NOT change with x6 plot window)
    range_names = [f'N_X{i+1}_RANGE' for i in range(6)]
    integration_ranges = [getattr(constants, range_names[i]) for i in integration_axes]
    domain_volume = np.prod([r[1]-r[0] for r in integration_ranges])

    # grid
    X1, X6 = np.meshgrid(x1_vals_scaled, x6_vals_scaled, indexing="ij")
    grid_pts = np.stack([X1.ravel(), X6.ravel()], axis=1)  # shape [G, 2]

    # one MC set reused for all grid points
    integration_samples = np.random.uniform(
        low=[r[0] for r in integration_ranges],
        high=[r[1] for r in integration_ranges],
        size=(num_samples_mc, len(integration_axes))
    )

    # batched evaluation
    pdf_flat = np.zeros(grid_pts.shape[0], dtype=np.float32)
    # for s in range(0, grid_pts.shape[0], batch_size):
    for s in tqdm(range(0, grid_pts.shape[0], batch_size), desc="Processing batch"):
        e = min(s + batch_size, grid_pts.shape[0])
        batch = grid_pts[s:e]  # [B,2]

        # tile to [B*num_samples_mc, :]
        plot_tiled = np.repeat(batch, num_samples_mc, axis=0)
        integ_tiled = np.tile(integration_samples, (batch.shape[0], 1))

        # assemble full 6D inputs
        full = np.empty((plot_tiled.shape[0], 6), dtype=np.float32)
        full[:, plot_axes[0]] = plot_tiled[:, 0]  # x1
        full[:, plot_axes[1]] = plot_tiled[:, 1]  # x6
        for col_i, ax in enumerate(integration_axes):
            full[:, ax] = integ_tiled[:, col_i]

        t_tensor = torch.full((full.shape[0], 1), t_val, dtype=torch.float32, device=device)
        x_tensor = torch.from_numpy(full).to(device)

        with torch.no_grad():
            vals = p_net(x_tensor, t_tensor).view(batch.shape[0], num_samples_mc)
            # mean over MC, then scale by domain volume
            pdf_flat[s:e] = (vals.mean(dim=1) * domain_volume).cpu().numpy()

    pdf = pdf_flat.reshape(X1.shape)

    # Unscaled values
    x1_vals = x1_vals_scaled*constants.COV_I[plot_axes[0], plot_axes[0]]**0.5 + constants.MEAN_I[plot_axes[0]]
    x6_vals = x6_vals_scaled*constants.COV_I[plot_axes[1], plot_axes[1]]**0.5 + constants.MEAN_I[plot_axes[1]]
    X1_grid, X6_grid = np.meshgrid(x1_vals, x6_vals, indexing="ij")

    if renormalize_over_window:
        # ONLY do this if you want a probability *over this exact window*
        # (e.g., for computing a mean within fixed bounds).
        dx1 = (x1_vals[-1] - x1_vals[0]) / max(1, len(x1_vals)-1)
        dx6 = (x6_vals[-1] - x6_vals[0]) / max(1, len(x6_vals)-1)
        Z = pdf.sum()*dx1*dx6
        if Z > 0:
            pdf = pdf / Z

    if(save_path):
        np.savez(save_path, X_grid=X1_grid, Y_grid=X6_grid, pdf=pdf)
        print(f"Marginal distribution saved to {save_path}.npz")

    return X1_grid, X6_grid, pdf


def marginal_pinn_1d(
    p_net, t_val, constants,
    plot_axis,                 # int in [0..5]
    num_linespace=256,
    num_samples_mc=500000,
    batch_size=64,
    device="cpu",
    save_path=None,
    renormalize_over_window=True,
    mc_chunk=100000            # process MC samples in chunks to limit memory
):
    """
    Monte-Carlo marginalization to a 1D PDF for x_{plot_axis} at time t_val.
    Returns (X_grid, pdf), both in *unscaled* physical units.
    """
    # -----------------------------
    # 1) Define evaluation grid (scaled space)
    # -----------------------------
    x_range = getattr(constants, f"N_X{plot_axis+1}_RANGE")
    x_vals_scaled = np.linspace(x_range[0], x_range[1], num=num_linespace, endpoint=True)

    # axes bookkeeping
    all_axes = [0,1,2,3,4,5]
    integration_axes = [i for i in all_axes if i != plot_axis]

    # fixed integration ranges (scaled)
    range_names = [f'N_X{i+1}_RANGE' for i in range(6)]
    integration_ranges = [getattr(constants, range_names[i]) for i in integration_axes]
    domain_volume = np.prod([r[1]-r[0] for r in integration_ranges])

    # -----------------------------
    # 2) One MC set reused for all grid points (scaled)
    # -----------------------------
    integration_samples = np.random.uniform(
        low=[r[0] for r in integration_ranges],
        high=[r[1] for r in integration_ranges],
        size=(num_samples_mc, len(integration_axes))
    ).astype(np.float32)

    # -----------------------------
    # 3) Batched evaluation over x-grid; stream MC in chunks
    # -----------------------------
    pdf_vals = np.zeros(len(x_vals_scaled), dtype=np.float32)

    for s in tqdm(range(0, len(x_vals_scaled), batch_size), desc="1D marginal batches"):
        e = min(s + batch_size, len(x_vals_scaled))
        x_batch = x_vals_scaled[s:e]  # [B]

        # We'll accumulate mean over MC with chunking to avoid huge tensors
        acc = torch.zeros((e - s,), dtype=torch.float32, device=device)
        n_total = 0

        for mc_s in range(0, num_samples_mc, mc_chunk):
            mc_e = min(mc_s + mc_chunk, num_samples_mc)
            mc_block = integration_samples[mc_s:mc_e, :]  # [M_chunk, k]

            # tile: for each x in batch, pair with all mc_block rows
            B = (e - s)
            M = (mc_e - mc_s)
            # assemble full 6D inputs
            full = np.empty((B * M, 6), dtype=np.float32)
            # set the plotted axis
            full[:, plot_axis] = np.repeat(x_batch, M)

            # fill integration axes
            for col_i, ax in enumerate(integration_axes):
                full[:, ax] = np.tile(mc_block[:, col_i], B)

            # tensors
            x_tensor = torch.from_numpy(full).to(device)
            t_tensor = torch.full((x_tensor.shape[0], 1), t_val, dtype=torch.float32, device=device)

            with torch.no_grad():
                vals = p_net(x_tensor, t_tensor).view(B, M).mean(dim=1)  # mean over MC chunk
                acc += vals
                n_total += 1

        # average across chunks and scale by domain volume
        pdf_vals[s:e] = (acc / max(1, n_total) * domain_volume).cpu().numpy()

    # -----------------------------
    # 4) Unscale x-axis to physical units
    # -----------------------------
    std_i = float(constants.COV_I[plot_axis, plot_axis]**0.5)
    mu_i  = float(constants.MEAN_I[plot_axis])
    x_vals = x_vals_scaled * std_i + mu_i

    # -----------------------------
    # 5) Optional renormalization over the plotted window (physical units)
    # -----------------------------
    if renormalize_over_window:
        dx = (x_vals[-1] - x_vals[0]) / max(1, len(x_vals)-1)
        Z = float(np.sum(pdf_vals) * dx)
        if Z > 0:
            pdf_vals = pdf_vals / Z

    # -----------------------------
    # 6) Save (optional) and return
    # -----------------------------
    if save_path:
        np.savez(save_path, X_grid=x_vals, pdf=pdf_vals)
        print(f"1D marginal saved to {save_path}.npz")

    return x_vals, pdf_vals



# --- Example Usage ---
if __name__ == '__main__':
    OUTPUT_PATH = "output/v0_scaled_T0.3"

    # p_net = PNet_Scaled(constants, input_feature=7)
    p_net = PNet_XL(constants, input_feature=7)
    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale_torch = torch.tensor(p_max, dtype=torch.float32)
    p_net.scale = scale_torch
    # load p_net
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    # marginalize 2D: select x_coords
    x_coords = (4, 5)
    plot_axes = tuple(c - 1 for c in x_coords)
    for t in [constants.T_PRIME_SPAN[-1]]:
        save_path = f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz"
        X_grid, Y_grid, pdf_values = marginal_pinn(
            p_net, t, constants, #x1_vals_scaled, x6_vals_scaled,
            plot_axes,
            save_path=save_path
        )

    # Marginalize 1D: select x_coord
    # x_coord = 6
    # plot_axs = x_coord-1
    # for t in [constants.T_PRIME_SPAN[-1]]:
    #     save_path = f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coord}_t{t:.3f}.npz"
    #     marginal_pinn_1d(p_net, t, constants, plot_axs, save_path=save_path)
