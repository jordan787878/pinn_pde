import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, HMC

# Assuming the following imports and setup are the same as your original code
from monte import p_init_scaled
from exp_utilities.constants import Case1_6D_Constants_Equin
from _General.neuralnetworks import PNet_Scaled, load_trained_model

# --- Initial setup (as in your original script) ---
constants = Case1_6D_Constants_Equin()
device = "cpu"


def generate_samples_hmc(p_net, t_val, num_samples, warmup_steps, initial_params):
    """
    Generates samples from the unnormalized p_net distribution using HMC in Pyro.
    """
    # Potential function is the negative log-density
    def custom_potential_fn(params):
        x = params['x']  # shape: [batch_size, 6]
        t_tensor = torch.full((x.shape[0], 1), t_val, dtype=torch.float32, device=x.device)
        
        # Calculate the log-density, with a small epsilon for stability
        log_prob = torch.log(p_net(x, t_tensor).squeeze() + 1e-10)
        
        # The potential is the negative of the log-probability
        return -log_prob.sum()

    # Create the HMC kernel
    hmc_kernel = HMC(potential_fn=custom_potential_fn)

    # Run the MCMC sampler
    mcmc = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=initial_params)
    mcmc.run()

    # Get the samples
    samples = mcmc.get_samples(group_by_chain=False)['x'].reshape(-1, 6)
    return samples


# --- Example Usage ---
if __name__ == '__main__':
    OUTPUT_PATH = "output/v0_scaled_T0.3"

    p_net = PNet_Scaled(constants, input_feature=7)
    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale = p_max
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    # load p_net
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    num_samples_hmc = 1000  # Number of samples to collect
    warmup_steps_hmc = 500  # Burn-in steps for HMC

    t = constants.T_PRIME_SPAN[1]
    print(f"Generating HMC samples for t = {t:.3f}...")
    # Define initial parameters for the sampler
    # Start near the mean or a reasonable location.
    # The shape should be [num_chains, 6].
    initial_params_tensor = torch.tensor([constants.N_MEAN_I], dtype=torch.float32, device=device)
    # Run the sampler
    hmc_samples = generate_samples_hmc(
        p_net,
        t,
        num_samples=num_samples_hmc,
        warmup_steps=warmup_steps_hmc,
        initial_params={'x': initial_params_tensor}
    )

    # hmc_samples will be a torch tensor of shape [num_samples, 6]
    print("Shape of HMC samples:", hmc_samples.shape)
    print("First 5 HMC samples:\n", hmc_samples[:5])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the 2D marginal contour from your MC code
    # ax.contour(X_grid, Y_grid, pdf_values, levels=5, cmap='viridis', linewidths=0.5)

    # Plot a 2D histogram of the HMC samples
    # ax.hist2d(
    #     hmc_samples_np[:, 0], hmc_samples_np[:, 5],
    #     bins=50,
    #     cmap='Reds',
    #     density=True,
    #     alpha=0.5
    # )
    ax.scatter(hmc_samples[:, 0], hmc_samples[:, 5])

    plt.show()
