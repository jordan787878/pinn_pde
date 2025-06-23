import numpy as np
import exp_utilities.plot_utilites as exp_plot

import cvxpy as cp

import torch
import torch.optim as optim

import os
import sys
sys.path.insert(0, '../utilities/')
import FunctionalOpt.src as funcOpt


def set_target(constants, p_net, data_folder, show_plot=False):
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

    if(show_plot):
        exp_plot.plot_target(constants, p_net, target_r, target_ph, data_folder)

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
    # np.save('data/app1/pr_mcs.npy', data_array)


def get_prob_PINN(constants, t_span, target_r, targer_ph, networks):
    data_list = []
    for t in t_span:
        model = None
        pr, model = compute_prob_event(constants, target_r, targer_ph, t, networks, N_discret=50)
        data_list.append([t, pr])
        # --- save model ----
        # if(model is not None):
        #     torch.save(model.state_dict(), "data/app1/tar1/gmm_64_t{:.3f}.pth".format(t))

    data_array = np.array(data_list)
    print("Prob result:")
    print(data_array)
    # --- save results ---
    # np.save('data/app1/prob_diaggmmx64.npy', data_array)


def compute_prob_event_monte(constants, target_r, targer_ph, t, N_monte=4, data_folder="data/"):
    # convert target_r target_phi to normalized coordinate
    target_region = np.array([
        [target_r[0], target_r[1]]/constants.R,   # r bounds
        [(targer_ph[0]-constants.W*constants.T*t)/constants.PHI, 
         (targer_ph[1]-constants.W*constants.T*t)/constants.PHI],   # phi bounds
        [constants._X3_RANGE[0], constants._X3_RANGE[1]],  
        [constants._X4_RANGE[0], constants._X4_RANGE[1]]
    ])

    # [1e+8, 1e+7, 1e+6, 1e+5]
    pr_list = []
    mc_folders = ["data/1e+8/", "data/1e+7/", "data/1e+6/", "data/1e+5/"]

    # data_folder
    x1s = np.load(data_folder+"x1s.npy")
    x2s = np.load(data_folder+"x2s.npy")
    x3s = np.load(data_folder+"x3s.npy")
    x4s = np.load(data_folder+"x4s.npy")
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


def compute_prob_event(constants, target_r, targer_ph, t, networks, N_discret = 50, data_folder="data/"):
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

    n1, n2, n3, n4 = len(x1s), len(x2s), len(x3s), len(x4s)
    N = n1 * n2 * n3 * n4

    # Create a mask for grid points in the target region.
    mask = ((x1_grid >= target_region[0, 0]) & (x1_grid <= target_region[0, 1]) &
            (x2_grid >= target_region[1, 0]) & (x2_grid <= target_region[1, 1]) &
            (x3_grid >= target_region[2, 0]) & (x3_grid <= target_region[2, 1]) &
            (x4_grid >= target_region[3, 0]) & (x4_grid <= target_region[3, 1]))
    mask_flat = mask.flatten()
    print("\n[check] t, sum(mask), B: ", t, sum(mask_flat), B)

    # Formulate the problem
    if(sum(mask_flat) <= 0.0):
        return 0.0, None
    else: 
        # # (1.0) only phat
        # p0_flat = p0.flatten()
        # pr = np.sum(p0_flat[mask_flat])*dV
        # return pr
    
        # (1.1) naive: integral (phat + B1) dx
        # p0_flat = p0.flatten()
        # pr = np.sum(p0_flat[mask_flat])*dV + sum(mask_flat)*dV*B
        # return pr

        # (2.0) linear program
        # p0_flat = p0.flatten()
        # # Decision variable: p_vec is now a vector of length N.
        # p_vec = cp.Variable(N)
        # # Constraints.
        # constraints = [
        #     cp.sum(p_vec) * dV == 1,  # total mass constraint.
        #     p_vec >= 0,                              # non-negativity.
        #     p_vec >= p0_flat - B,                      # lower bound.
        #     p_vec <= p0_flat + B                       # upper bound.
        # ]
        # # Objective: maximize probability mass in target region minus smoothness penalty.
        # objective = cp.Maximize(
        #     cp.sum(cp.multiply(p_vec, mask_flat)) * dV #- lambda_reg * smoothness_penalty
        # )
        # # Set up and solve the problem.
        # prob = cp.Problem(objective, constraints)
        # result = prob.solve()
        # pr = np.sum(p_vec.value[mask_flat]) * dV
        # print("NN+Error Upper Bound value (target region):", pr)
        # print("Total mass (should be 1):", np.sum(p_vec.value) * dV)
        # return pr

        # (3.0) FO
        p0_flat = p0.flatten()
        pr, model = compute_Pr_Guass(constants, t, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, dV, B)
        return pr, model
    

def compute_Pr_Guass(constants, t, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, dV, B):
    device = torch.device("cpu")  # or torch.device("cuda") if available.

    # --- Formulate problem ---
    domain_bounds = np.array([constants.X1_RANGE, constants.X2_RANGE, constants.X3_RANGE, constants.X4_RANGE])
    target_region = np.array([target_region[0,:], 
                              target_region[1,:], 
                              constants.X3_RANGE, 
                              constants.X4_RANGE])
    target_bounds = funcOpt.helpers.get_valid_target_bounds(domain_bounds, target_region)
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
    }

    # --- Init model ---
    # trained_model = RBFDensity(num_basis=256).to(device) # (default degree of basis)
    trained_model = funcOpt.models.TorchGMM(constants).to(device)

    # --- Train model ---
    trained_model  = train_model(problem, trained_model, num_iterations=20000)

    # --- Load model ---
    # trained_model.load_state_dict(torch.load("data/app1/tar1/pdf_models/gmm_64_t{:.3f}.pth".format(t)))
    
    # --- Check constraints ---
    trained_model.eval()
    funcOpt.helpers.check_pdf_integral(problem, trained_model, method="analy")
    funcOpt.helpers.check_pdf_interval(problem, trained_model)
    
    # --- Evaluate Probability
    Pr_p0 = np.sum(p0_flat[mask_flat])*dV
    Pr_opt = funcOpt.helpers.integral_torchgmm(trained_model.get_gmm_paramters(), problem['target_bounds'])
    
    print("Integrated phat over target region: {:.4f}".format(Pr_p0))
    print("Prob. pGMM over target region: {:.4f}".format(Pr_opt))
    # visual_fo_rbf(t, p_net, trained_model, target_region)

    return Pr_opt, trained_model
    

def train_model(problem, model, num_iterations=1000, batch_size=256, device=torch.device("cpu")):
    # --- Set a fixed seed for reproducibility ---
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Extract from problem
    constants = problem['constants']
    t = problem['time']
    p_net = problem['p_net'] 
    p0_flat = problem['p0']
    mask_flat = problem['mask_flat']
    grid_points_tensor = problem['x']
    B = problem['B']
    domain_bounds = problem['domain_bounds']
    target_bounds = problem['target_bounds']

    # --- Set optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    domain_V = funcOpt.helpers.compute_volume(domain_bounds)
    target_V = funcOpt.helpers.compute_volume(target_bounds)
    
    # Convert the target numpy array to a torch tensor once.
    p0_tensor = torch.from_numpy(p0_flat).to(device)

    _p_model_add = model(grid_points_tensor).view(-1,)
    _deviation = torch.abs(_p_model_add - p0_tensor)
    _, indices = torch.topk(_deviation, batch_size, largest=True)
    x_dom = grid_points_tensor[indices].view(-1, 4)
    p_target_dom = p0_tensor[indices].view(-1)

    _p_reg_add = p0_tensor[mask_flat].view(-1,)
    _x_red_add = grid_points_tensor[mask_flat].view(-1,4)
    _, indices = torch.topk(_p_reg_add, batch_size, largest=True)
    x_reg = _x_red_add[indices].view(-1,4)
    p_target_reg = _p_reg_add[indices].view(-1,)

    x_dom = torch.cat((x_dom, x_reg), dim=0)
    p_target_dom = torch.cat((p_target_dom, p_target_reg), dim=0)

    best_loss = np.inf
    for it in range(num_iterations):
        optimizer.zero_grad()

        # random samples
        _x_dom = constants.sample_points(batch_size, domain_bounds)
        _t_dom = torch.full((batch_size, 1), t)
        _p_target_dom = p_net(_x_dom, _t_dom).view(-1,)
        _x_reg = constants.sample_points(batch_size, target_bounds)
        _t_reg = torch.full((batch_size, 1), t)
        _p_target_reg = p_net(_x_reg, _t_reg).view(-1,)
        x_dom_train = torch.cat((x_dom, _x_dom, _x_reg), dim=0)
        p_target_dom_train = torch.cat((p_target_dom, _p_target_dom, _p_target_reg), dim=0)
        x_reg_train = torch.cat((x_reg, _x_reg), dim=0)

        # [hard constraints loss]
        p_model_dom  = model(x_dom_train).view(-1,)
        hc = (p_model_dom >= (p_target_dom_train - B)) & (p_model_dom <= (p_target_dom_train + B))
        violation_mask = ~hc  # True for datapoints that do NOT satisfy the constraint
        vio_percent = 100.0*(violation_mask.sum()/p_target_dom_train.shape[0])
        if(sum(violation_mask) > 0):
            loss_full = torch.mean(0.5*(p_model_dom[violation_mask] - p_target_dom_train[violation_mask]) ** 2)*domain_V
        else:
            loss_full = torch.zeros(1)

        # [max prob. over target region]
        p_model_reg = model(x_reg_train).view(-1,)
        region_mass_estimate = torch.mean(p_model_reg)*target_V
        loss_region = -region_mass_estimate

        # --- Combine loss and optimize ---
        total_loss = loss_full + loss_region*1e-1
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        # --- Print progress and update the best model ---
        if it % int(num_iterations/10) == 0:
            print(f"violation percent: {vio_percent:.4f}%")
            print(f"Iteration {it:4d}, loss: {total_loss.item()}, loss hc: {loss_full.item()}, loss reg: {loss_region.item()}")
        if(total_loss.item() < best_loss):
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_loss = total_loss.item()
            print(f"[Saved] Iteration {it:4d}, loss: {total_loss.item()}, {loss_full.item()}, {loss_region.item()}")
            print(f"   violation percent: {vio_percent:.4f}%")
            
    # After training, load the best model state.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Returning best model with total loss: {best_loss}")

    return model


def simple_interpolate(arr):
    """
    Given a 1D numpy array, return a new array that inserts the average
    of each pair of adjacent elements between them.
    
    For example:
    If arr = [0.0, 0.4, 0.8]
    then the result will be [0.0, 0.2, 0.4, 0.6, 0.8]
    """
    # Number of original elements
    n = len(arr)
    # New array length will be (2*n - 1)
    new_arr = np.empty(2 * n - 1, dtype=arr.dtype)
    
    # Place the original values in the even indices of the new array
    new_arr[0::2] = arr
    
    # Calculate averages and place in the odd indices
    new_arr[1::2] = (arr[:-1] + arr[1:]) * 0.5
    
    return new_arr
