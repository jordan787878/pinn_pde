import numpy as np
import torch


def solve_numer_integral(problem, N_res=50):
    """
    compute upper bound of P* by numerical integral
    """
    constants = problem['constants']
    B = problem['B']
    t = problem['time']
    p_net = problem['p_net']
    target_bounds = problem['target_bounds']
    x1s = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], N_res)
    x2s = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], N_res)
    x3s = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], N_res)
    x4s = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], N_res)
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dV = dx1 * dx2 * dx3 * dx4
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    p0 = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape).flatten()
    # Create a mask for grid points in the target region.
    mask = ((x1_grid >= target_bounds[0, 0]) & (x1_grid <= target_bounds[0, 1]) &
            (x2_grid >= target_bounds[1, 0]) & (x2_grid <= target_bounds[1, 1]) &
            (x3_grid >= target_bounds[2, 0]) & (x3_grid <= target_bounds[2, 1]) &
            (x4_grid >= target_bounds[3, 0]) & (x4_grid <= target_bounds[3, 1]))
    mask_flat = mask.flatten()
    if(sum(mask_flat) <= 0.0):
        return 0.0, None
    # --- Solving ---
    inside = mask_flat
    outside = ~inside
    p0_inside = p0[inside]
    p0_outside = p0[outside]
    # direct integral inside target
    Pr = 0.0
    for p in p0_inside:
        Pr += (p+B)*dV
    # 1 - integral outside target
    Pr_neg = 0.0
    for p in p0_outside:
        if((p-B) >= 0):
            Pr_neg += (p-B)*dV
    Pr_neg = 1 - Pr_neg
    return min(Pr, Pr_neg), None