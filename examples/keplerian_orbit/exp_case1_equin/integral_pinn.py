import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Assuming the following imports and setup are the same as your original code
from monte import p_init_scaled
from exp_utilities.constants import Case1_6D_Constants_Equin
from _General.neuralnetworks import PNet_Scaled, load_trained_model

# --- Initial setup (as in your original script) ---
constants = Case1_6D_Constants_Equin()
device = "cpu"


def marginal_pinn(
    p_net, t_val, constants,
    plot_axes,
    num_linespace=200,
    num_samples_mc=10000,
    batch_size=128,
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


# --- Example Usage ---
if __name__ == '__main__':
    OUTPUT_PATH = "output/v0_scaled_T0.3"

    p_net = PNet_Scaled(constants, input_feature=7)
    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale_torch = torch.tensor(p_max, dtype=torch.float32)
    p_net.scale = scale_torch
    # load p_net
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    # Select coordinates to marginalize
    x_coords = (1,6)

    plot_axes = tuple(c - 1 for c in x_coords)
    for t in constants.T_PRIME_SPAN:
        save_path = f"{OUTPUT_PATH}/pre_compute/marginal_pdfpinn_x{x_coords[0]}_x{x_coords[1]}_t{t:.3f}.npz"

        X_grid, Y_grid, pdf_values = marginal_pinn(
            p_net, t, constants, #x1_vals_scaled, x6_vals_scaled,
            plot_axes,
            num_linespace=256,
            num_samples_mc=10000,
            batch_size=16,
            save_path=save_path
        )

        # Quick visualization
        # data = np.load(save_path)
        # X_grid = data["X_grid"]; Y_grid = data["Y_grid"]; pdf_values = data["pdf"]
        # pdf_max = np.max(pdf_values[pdf_values > 0])
        # # Define levels as percentages of the maximum value
        # relative_levels = np.array([0.01, 0.25, 0.50, 0.75, 0.99])
        # levels = relative_levels * pdf_max
        # fig, ax = plt.subplots(figsize=(10, 8))
        # # Draw the contour lines with the custom levels
        # ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors='black', linewidths=0.5)
        # ax.set_xlim(X_grid.min(), X_grid.max())
        # ax.set_ylim(Y_grid.min(), Y_grid.max())
        # # ax.set_title(f'2D Marginal PDF for {labels[0]} and {labels[1]}')
        # ax.grid(True)
        # plt.show()
