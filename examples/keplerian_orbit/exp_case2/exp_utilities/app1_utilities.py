import numpy as np
import exp_utilities.plot_utilites as exp_plot
# import cvxpy as cp
import torch
import os
import sys
sys.path.insert(0, '../utilities/')
import FunctionalOpt.src as funcOpt


def set_target(constants):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    # specify a fixedtarget region in spherical coordinate
    ## tar1
    target_r = np.array([21.3, 21.8])*constants.R
    target_ph = np.array([-3.5, 1.5])*constants.PHI + constants.W*constants.T*(0.1)

    ## tar2
    # target_r = np.array([21.3, 21.8])*constants.R
    # target_ph = np.array([-3.5, 1.5])*constants.PHI + constants.W*constants.T*(0.19)

    return target_r, target_ph


def get_prob_MC(constants, t_span, target_r, targer_ph):
    data_list = []
    for t in t_span:
        pr_array = compute_prob_event_monte(constants, target_r, targer_ph, t)
        data = np.insert(pr_array, 0, t)
        data_list.append(data)
    data_array = np.vstack(data_list)
    print("Prob result:")
    print(data_array)
    np.save('data/app1/tar1/prob/mcs.npy', data_array)


def get_prob_PINN(constants, t_span, target_r, targer_ph, networks, options):
    data_list = []
    for t in t_span:
        model = None
        pr, model = compute_prob_event(constants, target_r, targer_ph, t, networks, N_discret=50)
        data_list.append([t, pr])
        # --- save model ----
        if(model is not None and options['save_result']):
            torch.save(model.state_dict(), 
                options["path_pdf_models"]+options["label"]+"_t{:.3f}.pth".format(t))

    data_array = np.array(data_list)
    print("Prob result:")
    print(data_array)
    # --- save results ---
    if(options['save_result']):
        np.save(options["path_prob"]+options["label"]+".npy", data_array)


def compute_prob_event_monte(constants, target_r, targer_ph, t):
    # convert target_r target_phi to normalized coordinate
    target_region = np.array([
        [target_r[0], target_r[1]]/constants.R,   # r bounds
        [(targer_ph[0]-constants.W*constants.T*t)/constants.PHI, 
         (targer_ph[1]-constants.W*constants.T*t)/constants.PHI],   # phi bounds
        [constants._X3_RANGE[0], constants._X3_RANGE[1]],  
        [constants._X4_RANGE[0], constants._X4_RANGE[1]]
    ])

    # [1e+5, 1e+6]
    pr_list = []
    mc_folders = ["data/1e+5/", "data/1e+6/"]
    N_monte = len(mc_folders)

    # data_folder
    x1s = np.load("data/grids/x1s.npy")
    x2s = np.load("data/grids/x2s.npy")
    x3s = np.load("data/grids/x3s.npy")
    x4s = np.load("data/grids/x4s.npy")
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dV = dx1 * dx2 * dx3 * dx4

    # Create a mask for grid points in the target region.
    mask = ((x1_grid >= target_region[0, 0]) & (x1_grid <= target_region[0, 1]) &
            (x2_grid >= target_region[1, 0]) & (x2_grid <= target_region[1, 1]) &
            (x3_grid >= target_region[2, 0]) & (x3_grid <= target_region[2, 1]) &
            (x4_grid >= target_region[3, 0]) & (x4_grid <= target_region[3, 1]))
    mask_flat = mask.flatten()
    # print(sum(mask_flat)) # if this is zero, then no further computation, directly set zero.

    # (case: 1)
    if(sum(mask_flat) <= 0.0):
        for j in range(N_monte):
            pr_list.append(0.0)
        return np.array(pr_list)
    
    # (case: 2)
    for j in range(N_monte):
        file_name = mc_folders[j]+"pdf_t{:.3f}.npy".format(t)
        if not os.path.exists(file_name):
            pr_list.append(np.NaN)
        else:
            pdf_mc_grid = np.load(mc_folders[j]+"pdf_t{:.3f}.npy".format(t))
            pr = np.sum(pdf_mc_grid[mask]) * dV
            pr_list.append(pr)
    return pr_list


def compute_prob_event(constants, target_r, targer_ph, t, networks, N_discret = 50):
    p_net, e1_net_seq1, e1_net_seq2 = networks

    # convert target_r target_phi to normalized coordinate
    target_region = np.array([
        [target_r[0], target_r[1]]/constants.R,   # r bounds
        [(targer_ph[0]-constants.W*constants.T*t)/constants.PHI, 
         (targer_ph[1]-constants.W*constants.T*t)/constants.PHI],   # phi bounds
        [constants._X3_RANGE[0], constants._X3_RANGE[1]],  
        [constants._X4_RANGE[0], constants._X4_RANGE[1]]
    ])
    x1s = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], N_discret)
    x2s = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], N_discret)
    x3s = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], N_discret)
    x4s = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], N_discret)
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dV = dx1 * dx2 * dx3 * dx4

    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    p0 = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

    # sequence selection
    if(t < 0.5*constants.TF/constants.T):
        e1_nn  = e1_net_seq1(grid_points_tensor, t_tensor).detach().numpy().ravel()
    else:
        e1_nn  = e1_net_seq2(grid_points_tensor, t_tensor).detach().numpy().ravel()
    B = 2.0 * np.max(np.abs(e1_nn))

    # Create a mask for grid points in the target region.
    mask = ((x1_grid >= target_region[0, 0]) & (x1_grid <= target_region[0, 1]) &
            (x2_grid >= target_region[1, 0]) & (x2_grid <= target_region[1, 1]) &
            (x3_grid >= target_region[2, 0]) & (x3_grid <= target_region[2, 1]) &
            (x4_grid >= target_region[3, 0]) & (x4_grid <= target_region[3, 1]))
    mask_flat = mask.flatten()
    print("\n[check] t, sum(mask), B: ", t, sum(mask_flat), B)
    if(sum(mask_flat) <= 0.0):
        return 0.0, None

    # --- Formulate well-posed problem ---
    domain_bounds = np.array([constants.X1_RANGE, constants.X2_RANGE, constants.X3_RANGE, constants.X4_RANGE])
    target_bounds = funcOpt.helpers.get_valid_target_bounds(domain_bounds, target_region)
    p0_flat = p0.flatten()
    problem = {
        'x': grid_points_tensor,
        'mask_flat': mask_flat,
        'p0': p0_flat,
        'p_net': p_net,
        'B': B,
        'domain_bounds': domain_bounds,
        'target_bounds': target_bounds,
        'time': t,
        'constants': constants,
        'Pr_p0_est': np.sum(p0_flat[mask_flat])*dV
    }

    # --- Solving ---
    pr, model = funcOpt.solver.solve_funcopt(problem)
    return pr, model