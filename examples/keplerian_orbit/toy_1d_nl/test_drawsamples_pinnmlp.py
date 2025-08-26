import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from main import PNet, get_p_normalize, load_train_model


# 1. Define the Target Log-Probability Function
def get_log_prob_pnet(p_net_model, x, t):
    """
    Computes the log-probability for a given sample x at a specific time t.
    The p_net model outputs the unnormalized density, so we take the logarithm.
    """
    # Ensure x and t are tensors of the correct shape
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32).reshape(-1,1)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([[t]], dtype=torch.float32).reshape(-1,1)

    # Get the unnormalized density from the network and take the log
    unnormalized_prob = p_net_model(x.view(-1,1), t.view(-1,1))
    log_prob = torch.log(unnormalized_prob)
    
    return log_prob.squeeze()

# 2. Metropolis-Hastings MCMC Algorithm
def metropolis_hastings_pnet(
    p_net_model, num_samples, input_dim, proposal_std, initial_state, time_t, burn_in=1000
):
    """
    Samples from the PDF defined by the p_net at a specific time using 
    the Metropolis-Hastings algorithm.
    
    Args:
        p_net_model (nn.Module): The MLP that represents the unnormalized density.
        num_samples (int): The total number of samples to generate.
        input_dim (int): The dimensionality of the state space.
        proposal_std (float): Standard deviation of the isotropic Gaussian proposal.
        initial_state (np.ndarray): The starting point for the MCMC chain.
        time_t (float): The time at which to sample from the network.
        burn_in (int): Number of initial samples to discard.
        
    Returns:
        np.ndarray: An array of accepted samples from the target distribution.
    """
    # Create the time tensor once
    time_tensor = torch.tensor([[time_t]], dtype=torch.float32)
    
    # Initialize the chain and storage
    chain = [initial_state]
    current_state = torch.tensor(initial_state, dtype=torch.float32)
    
    # Get the initial log-probability
    current_log_prob = get_log_prob_pnet(p_net_model, current_state, time_tensor)
    
    # Run the MCMC loop
    for i in range(num_samples - 1):
        # Propose a new state from a symmetric proposal distribution
        proposal_noise = torch.randn(input_dim) * proposal_std
        proposed_state = current_state + proposal_noise
        
        # Get the log-probability of the proposed state
        proposed_log_prob = get_log_prob_pnet(p_net_model, proposed_state, time_tensor)
        
        # Calculate the log-acceptance ratio
        log_acceptance_ratio = proposed_log_prob - current_log_prob
        
        # Generate a random number to decide acceptance
        rand_uniform = torch.rand(1).log()
        
        # Accept or reject the proposal
        if log_acceptance_ratio > rand_uniform:
            current_state = proposed_state
            current_log_prob = proposed_log_prob
        
        # Store the current state in the chain
        chain.append(current_state.numpy())

    return np.array(chain[burn_in:])


# Example Usage
if __name__ == "__main__":
    p_net = PNet()
    p_net.scale = get_p_normalize()
    p_net = load_train_model(p_net, PATH="data/p_net.pth")

        # --- Step 2: Run the MCMC sampler for a given time `t`
    time_t = 5.0  # The specific time to sample at
    input_dim = 1 # We are sampling for the variable `x`
    num_samples = 50000
    burn_in = 5000
    proposal_std = 1.0
    initial_state = np.array([0.0])
    
    print(f"Running MCMC for time t={time_t}...")
    samples = metropolis_hastings_pnet(
        p_net, num_samples, input_dim, proposal_std, initial_state, time_t, burn_in
    )
    print("MCMC finished.")

    # --- Step 3: Visualize the results
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=80, density=True, alpha=0.6, color='b', label=f'MCMC Samples at t={time_t}')
    
    # Plot the target PDF defined by the PNet for comparison
    x_range = np.linspace(-5, 5, 200).reshape(-1, 1)
    time_tensor_range = torch.full((x_range.shape[0], 1), time_t)
    
    with torch.no_grad():
        unnormalized_densities = p_net(torch.tensor(x_range, dtype=torch.float32), time_tensor_range).numpy().squeeze()
    
    # Normalize the densities for plotting
    area = np.trapz(unnormalized_densities, x_range.squeeze())
    true_density = unnormalized_densities / area

    x_grid = np.load("data/xsim.npy").astype(np.float32)
    pdf_mc = np.load("data/psim_t{:.1f}.npy".format(time_t)).astype(np.float32).reshape(-1,)
    
    plt.plot(x_range, true_density, 'r-', linewidth=2, label=f'Target PDF (from p_net at t={time_t})')
    plt.plot(x_grid, pdf_mc, "black", label="p MC")
    plt.title(f'MCMC Sampling from p_net(*, {time_t})')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()