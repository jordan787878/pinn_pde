import numpy as np
import exp_utilities.plot_utilites as exp_plot
import torch
import os
import sys
sys.path.insert(0, '../utilities/')
import FunctionalOpt.src as funcOpt
from _General.util import get_valid_target_bounds, compute_volume
from _General.generalsolvers import solve_numer_integral, solve_linearprogram


def get_prob_MC(constants, t_span, target_r, targer_ph):
    data_list = []
    for t in t_span:
        pr_array = compute_prob_event_monte(constants, target_r, targer_ph, t)
        data = np.insert(pr_array, 0, t)
        data_list.append(data)
    data_array = np.vstack(data_list)
    print("Prob result (MC):")
    print(data_array)
    return data_array


def get_prob_PINN(constants, t_span, target_r, targer_ph, networks, options):
    data_list = []
    for t in t_span:
        model = None
        pr, model = compute_prob_event(constants, target_r, targer_ph, t, networks, options, N_discret=50)
        data_list.append([t, pr])
        # --- save model ----
        if(model is not None and options['save_result']):
            torch.save(model.state_dict(), 
                options["path_pdf_models"]+options["label"]+"_t{:.3f}.pth".format(t))

    data_array = np.array(data_list)
    print("Prob result of "+options["label"])
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


def compute_prob_event(constants, target_r, targer_ph, t, networks, options, N_discret = 50):
    # convert target_r target_phi to normalized coordinate
    target_region = np.array([
        [target_r[0], target_r[1]]/constants.R,   # r bounds
        [(targer_ph[0]-constants.W*constants.T*t)/constants.PHI, 
         (targer_ph[1]-constants.W*constants.T*t)/constants.PHI],   # phi bounds
        [constants._X3_RANGE[0], constants._X3_RANGE[1]],  
        [constants._X4_RANGE[0], constants._X4_RANGE[1]]
    ])
    domain_bounds = np.array([constants.X1_RANGE, constants.X2_RANGE, constants.X3_RANGE, constants.X4_RANGE])
    target_bounds = get_valid_target_bounds(domain_bounds, target_region)
    # --- if the intersection of target and domain is empty ---
    if(target_bounds is None):
        return 0.0, None

    # --- Formulate well-posed problem ---
    problem = {
        'networks': networks,
        'domain_bounds': domain_bounds,
        'target_bounds': target_bounds,
        'domain_V' : compute_volume(domain_bounds),
        'target_V' : compute_volume(target_bounds),
        'time': t,
        'constants': constants,
        'path_pdf_models' : options["path_pdf_models"],
        'label': options['label'],
        'use_trained_gmm': options["use_trained_gmm"],
        'show_loss_landscape': options["show_loss_landscape"],
        'trained_gmm_path': options["path_pdf_models"]+options["label"],
        'num_iterations': options["num_iterations"],
        "label_parent_gmm": options["label_parent_gmm"],
    }

    # --- Solving ---
    if options["solver"] == "fo":
        problem = form_problem_exp_case2(problem, need_grid=True)
        pr, model = funcOpt.solver.solve_funcopt(problem)
    elif options["solver"] == "ni":
        problem = form_problem_exp_case2(problem)
        pr, model = solve_numer_integral(problem)
    elif options["solver"] == "lp":
        problem = form_problem_exp_case2(problem)
        pr, model = solve_linearprogram(problem)
    else:
        raise("solver in options is not implemented")
    return pr, model


def form_problem_exp_case2(problem, need_grid=False, N_res=50):
    constants = problem['constants']
    t = problem['time']
    p_net, e1_net_seq1, e1_net_seq2 = problem['networks']
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
    p0_flat = p0.flatten()
    # Create a mask for grid points in the target region.
    mask = ((x1_grid >= target_bounds[0, 0]) & (x1_grid <= target_bounds[0, 1]) &
            (x2_grid >= target_bounds[1, 0]) & (x2_grid <= target_bounds[1, 1]) &
            (x3_grid >= target_bounds[2, 0]) & (x3_grid <= target_bounds[2, 1]) &
            (x4_grid >= target_bounds[3, 0]) & (x4_grid <= target_bounds[3, 1]))
    mask_flat = mask.flatten()
    problem['Pr_p0_est'] = p0_flat[mask_flat].sum()*dV
    problem['dV'] = dV
    problem['p0'] = p0_flat
    problem['mask'] = mask_flat
    if(need_grid):
        problem['x'] = grid_points_tensor

    # sequence selection
    if(t < 0.5*constants.TF/constants.T):
        e1_nn  = e1_net_seq1(grid_points_tensor, t_tensor).detach().numpy().ravel()
    else:
        e1_nn  = e1_net_seq2(grid_points_tensor, t_tensor).detach().numpy().ravel()
    problem['B'] = 2.0 * np.max(np.abs(e1_nn))
    problem['B_marginal'] = problem['B'] * problem["target_V"]
    print("[debug] integral of B over target domain: {:.4f}".format(problem['B_marginal']))

    return problem
