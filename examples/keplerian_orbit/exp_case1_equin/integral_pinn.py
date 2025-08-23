import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Assuming the following imports and setup are the same as your original code
from monte import p_init_scaled
from exp_utilities.constants import Case1_6D_Constants_Equin
from _General.neuralnetworks import PNet_Scaled, load_trained_model

# --- Initial setup (as in your original script) ---
constants = Case1_6D_Constants_Equin()

p_net = PNet_Scaled(constants, input_feature=7)
_x_at_mean = constants.N_MEAN_I.copy()
p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
scale = p_max
scale_torch = torch.tensor(scale, dtype=torch.float32)
p_net.scale = scale_torch
OUTPUT_PATH = "output/v0_scaled"
p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
p_net.to(device)


def monte_carlo_marginal_pdf_batch(p_net, t_val, plot_var_points, integration_limits, num_samples_mc=10000, batch_size=256):
    """
    Computes the 2D marginal PDF using Monte Carlo integration and batch processing.
    
    Args:
        p_net (nn.Module): The 6D PDF neural network.
        t_val (float): The time value for the PDF evaluation.
        plot_var_points (np.array): A 2D array of grid points for the plot variables.
        integration_limits (list): Limits for variables x3-x6.
        num_samples_mc (int): Number of random samples for the Monte Carlo integration.
        batch_size (int): Number of (x1, x2) grid points to process in each batch.

    Returns:
        np.array: A 2D array of marginal PDF values.
    """
    total_grid_points = plot_var_points.shape[0]
    num_integration_dims = len(integration_limits)
    
    marginal_values = np.zeros(total_grid_points)

    # Calculate the volume of the integration domain
    domain_volume = np.prod([limit[1] - limit[0] for limit in integration_limits])
    
    # Generate all Monte Carlo samples once
    integration_samples_mc = np.random.uniform(
        low=[limit[0] for limit in integration_limits],
        high=[limit[1] for limit in integration_limits],
        size=(num_samples_mc, num_integration_dims)
    )

    for i in range(0, total_grid_points, batch_size):
        print(f"Processing batch {i // batch_size + 1} of {total_grid_points // batch_size + 1}")
        end = min(i + batch_size, total_grid_points)
        batch_size_actual = end - i
        
        plot_batch = plot_var_points[i:end]

        # Prepare the full input tensor
        plot_tiled = np.repeat(plot_batch, num_samples_mc, axis=0)
        integration_tiled = np.tile(integration_samples_mc, (batch_size_actual, 1))
        
        combined_batch = np.hstack([plot_tiled, integration_tiled])

        # Prepare the time tensor
        t_tensor = torch.full((combined_batch.shape[0], 1), t_val, dtype=torch.float32).to(device)
        
        inputs_tensor = torch.from_numpy(combined_batch).float().to(device)

        with torch.no_grad():
            outputs = p_net(inputs_tensor, t_tensor)

        outputs_reshaped = outputs.view(batch_size_actual, num_samples_mc).mean(dim=1).cpu().numpy()
        marginal_values[i:end] = outputs_reshaped * domain_volume
        
    return marginal_values


def visualize_and_save_marginal_mc(p_net, t_val, constants, plot_axes, 
                                   num_plot_points=50, num_samples_mc=10000, 
                                   batch_size=128, save_path=None):
    # ... (rest of the visualization and saving logic, with the new marginalization call) ...
    # This is similar to your existing function, but calls monte_carlo_marginal_pdf_batch instead.
    # The inputs_array construction needs to be modified for the random samples.
    
    # ... (steps 1-2 remain the same) ...
    all_axes = set(range(6))
    plot_axes_set = set(plot_axes)
    integration_axes_set = sorted(list(all_axes - plot_axes_set))
    
    if len(plot_axes) != 2:
        raise ValueError("plot_axes must be a tuple of two integers.")
    
    axis_names = [f'x{i+1}' for i in range(6)]
    plot_x_name, plot_y_name = axis_names[plot_axes[0]], axis_names[plot_axes[1]]

    range_names = [f'N_X{i+1}_RANGE' for i in range(6)]
    plot_x_range = np.linspace(getattr(constants, range_names[plot_axes[0]])[0], getattr(constants, range_names[plot_axes[0]])[-1], num_plot_points)
    plot_y_range = np.linspace(getattr(constants, range_names[plot_axes[1]])[0], getattr(constants, range_names[plot_axes[1]])[-1], num_plot_points)
    
    X_plot_grid, Y_plot_grid = np.meshgrid(plot_x_range, plot_y_range, indexing="ij")
    plot_var_points = np.stack([X_plot_grid.flatten(), Y_plot_grid.flatten()], axis=1)

    # 3. Use Monte Carlo for integration
    integration_ranges = [getattr(constants, range_names[i]) for i in integration_axes_set]
    
    # Calculate the volume of the integration domain
    domain_volume = np.prod([r[1] - r[0] for r in integration_ranges])

    # 4. Generate all Monte Carlo samples at once to save memory and time in loops
    # This part should not be tiled upfront as it's a huge memory sink
    integration_samples_mc = np.random.uniform(
        low=[limit[0] for limit in integration_ranges],
        high=[limit[1] for limit in integration_ranges],
        size=(num_samples_mc, len(integration_ranges))
    )
    
    marginal_values_flat = np.zeros(X_plot_grid.size)

    # 5. Perform the batch evaluation and integration with Monte Carlo
    total_grid_points = X_plot_grid.size
    
    for i in range(0, total_grid_points, batch_size):
        print(f"Processing batch {i // batch_size + 1} of {total_grid_points // batch_size + 1}")
        end = min(i + batch_size, total_grid_points)
        batch_size_actual = end - i
        
        plot_batch = plot_var_points[i:end]

        plot_tiled = np.repeat(plot_batch, num_samples_mc, axis=0)
        integration_tiled = np.tile(integration_samples_mc, (batch_size_actual, 1))

        # Reorder variables to match p_net input
        combined_batch = np.empty((plot_tiled.shape[0], 6))
        combined_batch[:, plot_axes] = plot_tiled
        combined_batch[:, integration_axes_set] = integration_tiled
        
        t_tensor = torch.full((combined_batch.shape[0], 1), t_val, dtype=torch.float32).to(device)
        inputs_tensor = torch.from_numpy(combined_batch).float().to(device)
        
        with torch.no_grad():
            outputs = p_net(inputs_tensor, t_tensor)

        outputs_reshaped = outputs.view(batch_size_actual, -1).mean(dim=1).cpu().numpy()
        marginal_values_flat[i:end] = outputs_reshaped * domain_volume

    marginal_pdf_values = marginal_values_flat.reshape(X_plot_grid.shape)
    
    # # ... (rest of the visualization and saving, remains the same) ...
    # fig, ax = plt.subplots(figsize=(10, 8))
    # contourf = ax.contourf(X_plot_grid, Y_plot_grid, marginal_pdf_values, cmap=cm.viridis, levels=10)
    # ax.contour(X_plot_grid, Y_plot_grid, marginal_pdf_values, colors='black', linewidths=0.5)
    # ax.set_xlabel(plot_x_name)
    # ax.set_ylabel(plot_y_name)
    # ax.set_title(f'2D Marginal PDF p({plot_x_name}, {plot_y_name} | t={t_val}) - Monte Carlo Contour Plot')
    # fig.colorbar(contourf, ax=ax, label='Marginal PDF Value')
    # plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, x_first=plot_x_range, x_second=plot_y_range, pdf=marginal_pdf_values)
        print(f"Marginal distribution saved to {save_path}.npz")

    return X_plot_grid, Y_plot_grid, marginal_pdf_values


# --- Example Usage ---
if __name__ == '__main__':
    constants.test_printout()

    t = 0.1
    OUTPUT_PATH = "output/v0_scaled"

    # Use the Monte Carlo approach
    X_plot_grid, Y_plot_grid, marginal_pdf_values = visualize_and_save_marginal_mc(
        p_net=p_net, 
        t_val=t, 
        constants=constants,
        plot_axes=(0, 5), 
        num_plot_points=100, 
        num_samples_mc=10000, # Increased samples for better accuracy
        batch_size=50, 
        save_path=OUTPUT_PATH+"/pre_compute/marginal_pdfpinn_x1_x6_t{:.3f}.npz".format(t)
    )

    # Load the data
    data = np.load(OUTPUT_PATH+"/pre_compute/marginal_pdfpinn_x1_x6_t{:.3f}.npz".format(t))
    labels=["x1", "x6"]
    x_first = data['x_first']
    x_second = data['x_second']
    X_grid, Y_grid = np.meshgrid(x_first, x_second, indexing="ij")
    pdf_values = data['pdf']
    pdf_max = np.max(pdf_values[pdf_values > 0])
    # Define levels as percentages of the maximum value
    # For example, levels at 5%, 25%, 50%, 75%, and 95% of the max
    relative_levels = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
    levels = relative_levels * pdf_max
    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw the contour lines with the custom levels
    ax.contour(X_grid, Y_grid, pdf_values, levels=levels, colors='black', linewidths=0.5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f'2D Marginal PDF for {labels[0]} and {labels[1]} (filtered)')
    plt.show()
