import numpy as np
from scipy.special import erf


def check_pdf_integral(problem, trained_model, method=""):
    x_grid = problem['x'] # (N_grid, d)
    N_grid = x_grid.shape[0]
    p0 = problem['p0'] # with respect to x_grid (N_grid, )
    domain_bounds = problem['domain_bounds']
    domain_volume = problem['domain_V']
    Pr_p0 = np.sum(p0) * domain_volume / N_grid

    if(method == "analy"):
        gmm_params = trained_model.get_gmm_paramters()
        Pr_opt = integral_torchgmm(gmm_params, domain_bounds)
    else:
        p_opt = trained_model(x_grid).detach().cpu().numpy()
        Pr_opt = np.sum(p_opt) * domain_volume / N_grid
    print("[check] integral pdf domain p0: {:.4f}, p_opt: {:.4f}".format(
        Pr_p0, Pr_opt
    ))


def check_pdf_interval(problem, trained_model):
    p0_grid = problem['p0']
    x_grid = problem['x']
    B = problem['B']
    N_grid = x_grid.shape[0]
    p_opt = trained_model(x_grid).detach().cpu().numpy()
    sat = (p_opt >= (p0_grid - B)) & (p_opt <= (p0_grid + B))
    percentage_satisfied = (sat.sum().item() / N_grid) * 100
    print(f"[check] interval satisfied for {percentage_satisfied:.4f}% over the domain grid.")
    # # Get the indices of the violations.
    # violation_indices = (~sat).nonzero(as_tuple=False).squeeze(1)  # shape: [num_violations]
    # if violation_indices.numel() > 0:
    #     num_violations = violation_indices.shape[0]
    #     print(f"Number of constraint violations: {num_violations}")
    #     # Set the number of top violations you want to print.
    #     top_N = 5  # you can change this to any number you prefer
    #     violation_p_values = p_target_full[violation_indices]
    #     sorted_order = torch.argsort(violation_p_values, descending=True)
    #     top_violation_indices = violation_indices[sorted_order][:top_N]
    #     print(f"Example violations (top {top_N} largest p_model values):")
    #     for idx in top_violation_indices:
    #         i = idx.item()
    #         # Assuming that state_full is an array (or tensor) holding the state coordinates for sample i.
    #         # If state_full is a tensor, you might want to convert it to a list or numpy array.
    #         # state_coords = state_full[i]
    #         print(f"  Sample {i}: p_target = {p_target_full[i].item():.6f}, p_model = {p_model_full[i].item():.6f}")
    # else:
    #     print("No constraint violations found.")


def integral_torchgmm(gmm_params, region_bounds):
    w_np, m_np, cov_np = gmm_params
    d = m_np.shape[1]
    N = m_np.shape[0]
    Pr = 0.0
    for i in range(N):
        Pr_i = 1.0
        w = w_np[i]
        mu = m_np[i,:]
        sigma = np.sqrt(cov_np[i, :])
        for j in range(d):
            lb = region_bounds[j, 0]
            up = region_bounds[j, 1]
            Pr_i *= 0.5*( erf( (up-mu[j])/(np.sqrt(2)*sigma[j]) ) - erf( (lb-mu[j])/(np.sqrt(2)*sigma[j]) ) )
        Pr += w*Pr_i
    return Pr