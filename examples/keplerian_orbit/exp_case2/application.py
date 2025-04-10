import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata
import argparse
from constants import Case2_4D_Constants
from monte import p_init, get_p_init_max
from pnet_models import PNet
from e1net_models import E1Net
import sys
import os
# Get the parent directory of the current directory (exp1)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from post.post_exp_cas2 import check_train_results, load_trained_model, get_max_e1_init
import cvxpy as cp
import scipy.sparse as sp
from matplotlib.patches import Patch
from scipy.linalg import kron


"""
select the NN models in output/
"""
PNET_PATH = "output/v2/p_net.pth"
E1NET_PATH_SEQ1 = "output/v2/e1_net_seq1.pth" # sequence 1
E1NET_PATH_SEQ2 = "output/v2/e1_net_seq2.pth"
DATA_FOLDER = "data/"
TRAIN_FLAG = False
constants = Case2_4D_Constants()
# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def visual_phat_trainings():
    """
    visualize phat training results using intermediate saved model
    """
    global constants

    # load p_net
    p_net = PNet(scale=get_p_init_max())
    p_net = load_trained_model(p_net, path="output/v2/p_net_5.pth", method="new"); p_net.eval()

    # Create a figure with a black background
    fig = plt.figure(figsize=(8, 6), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Adjust the axis panes and tick labels to be visible on a black background
    ax.xaxis.pane.set_facecolor('black')
    ax.yaxis.pane.set_facecolor('black')
    ax.zaxis.pane.set_facecolor('black')
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # t_span = simple_interpolate(constants.T_PRIME_SPAN)
    t_span = constants.T_PRIME_SPAN
    for t_prime in t_span:
    # for t_prime in np.linspace(constants.T_PRIME_SPAN[0], constants.T_PRIME_SPAN[-1], num=6):
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx1 = x1s[1] - x1s[0] # dr'
        dx2 = x2s[1] - x2s[0] # dphi'
        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
        sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
        print("[check] sum p_nn (N-sphere): ", sum_p_nn)

        # load pdf_monte on the domain
        t_label = t_prime
        if(t_prime in constants.T_PRIME_SPAN):
            pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))
            pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

        # convert pdf(r, phi)
        # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
        pdf_nn_rphi_data = np.empty((0,5))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        if(t_label in constants.T_PRIME_SPAN):
            pdf_mo_rphi_data = np.empty((0,4))
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                if(t_label in constants.T_PRIME_SPAN):
                    pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
        if(t_label in constants.T_PRIME_SPAN):
            z_mo = pdf_mo_rphi_data[:, 3]/(r) 

        # _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
        # __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
        # ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
        # print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

        # visualize p(x,y) using interpolation
        _grid_resolution = 50
        num_strides = 5
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), 
                                    np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        if(t_label in constants.T_PRIME_SPAN):
            grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

        if(t_prime == t_span[0]):
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8, label="NN")
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-", label="Monte")
        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8)
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-")
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Optional: Add a color bar
    # Labels and title
    ax.legend()
    ax.view_init(40, -133) # viewing angle
    ax.set_xlabel('X, m', color='white')
    ax.set_ylabel('Y, m', color='white')
    ax.set_zlabel('PDF Value', color='white')
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))
    plt.show()


def visual_app1(p_net, constants, show_plot=False):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    # specify a fixedtarget region in spherical coordinate
    target_r = np.array([21.4, 21.7])*constants.R
    target_ph = np.array([-12.0, 12.0])*constants.PHI + constants.W*constants.T*(0.08)

    if(show_plot):
        # Create a figure with a black background
        fig = plt.figure(figsize=(9, 7), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        z_max = -np.inf

        # Adjust the axis panes and tick labels to be visible on a black background
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.line.set_color('white')
        ax.yaxis.line.set_color('white')
        ax.zaxis.line.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        # t_span = simple_interpolate(constants.T_PRIME_SPAN)
        t_span = constants.T_PRIME_SPAN
        for t_prime in t_span:
        # for t_prime in np.linspace(constants.T_PRIME_SPAN[0], constants.T_PRIME_SPAN[-1], num=6):
            print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
            x1s = np.load(DATA_FOLDER+"x1s.npy")
            x2s = np.load(DATA_FOLDER+"x2s.npy")
            x3s = np.load(DATA_FOLDER+"x3s.npy")
            x4s = np.load(DATA_FOLDER+"x4s.npy")

            # obtain pdf_nn on the domain
            x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
            grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
            grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
            t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
            pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

            dx1 = x1s[1] - x1s[0] # dr'
            dx2 = x2s[1] - x2s[0] # dphi'
            dx3 = x3s[1] - x3s[0]
            dx4 = x4s[1] - x4s[0]

            # marginalize to spherical position (r, phi)
            pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
            sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
            print("[check] sum p_nn (N-sphere): ", sum_p_nn)

            # load pdf_monte on the domain
            t_label = t_prime
            if(t_prime in constants.T_PRIME_SPAN):
                pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))
                pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

            # convert pdf(r, phi)
            # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
            pdf_nn_rphi_data = np.empty((0,5))
            x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
            if(t_label in constants.T_PRIME_SPAN):
                pdf_mo_rphi_data = np.empty((0,4))
            for i in range(len(x1_grid)):
                for j in range(len(x2_grid)):
                    # extract [r', phi', p(r',phi')]
                    Nr = x1_grid[i,j]
                    Nphi = x2_grid[i,j]
                    r = Nr*constants.R
                    phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                    t = constants.T*t_prime
                    pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                    if(t_label in constants.T_PRIME_SPAN):
                        pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

            # convert pdf(r,phi) to pdf(x,y)
            x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
            y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
            r = pdf_nn_rphi_data[:, 0]
            z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
            if(t_label in constants.T_PRIME_SPAN):
                z_mo = pdf_mo_rphi_data[:, 3]/(r) 
                max_z_t_prime = np.max(z_mo)
                z_max = max(z_max, max_z_t_prime)

            # _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
            # __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
            # ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
            # print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

            # visualize p(x,y) using interpolation
            _grid_resolution = 50
            num_strides = 5
            grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), 
                                        np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
            # Interpolate scattered data onto the grid
            grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
            if(t_label in constants.T_PRIME_SPAN):
                grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

            if(t_prime == t_span[0]):
                surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8, label="NN")
                if(t_label in constants.T_PRIME_SPAN):
                    surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                            edgecolor='white', linewidth=1.0, linestyle="-", label="Monte")
            else:
                surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8)
                if(t_label in constants.T_PRIME_SPAN):
                    surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                            edgecolor='white', linewidth=1.0, linestyle="-")
            # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Optional: Add a color bar
        
        # Plot the target volume.
        print("z_max: ", z_max)
        target_plot = plot_target_volume(ax, target_r, target_ph, z_max)

        # Labels and title
        ax.legend()
        ax.view_init(40, -133) # viewing angle
        ax.set_xlabel('X, m', color='white')
        ax.set_ylabel('Y, m', color='white')
        ax.set_zlabel('PDF Value', color='white')
        ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))
        plt.show()

    return target_r, target_ph


def compute_prob_event_monte(target_r, targer_ph, t, N_monte=4, data_folder="data/"):
    # convert target_r target_phi to normalized coordinate
    target_region = np.array([
        [target_r[0], target_r[1]]/constants.R,   # r bounds
        [(targer_ph[0]-constants.W*constants.T*t)/constants.PHI, 
         (targer_ph[1]-constants.W*constants.T*t)/constants.PHI],   # phi bounds
        [constants._X3_RANGE[0], constants._X3_RANGE[1]],  
        [constants._X4_RANGE[0], constants._X4_RANGE[1]]
    ])

    # 1e+8, 1e+7, 1e+6, 1e+5
    pr_list = []
    mc_folders = ["data/", "data/1e+7/", "data/1e+6/", "data/1e+5/"]

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
    if(sum(mask_flat) <= 0.0):
        for j in range(N_monte):
            pr_list.append(0.0)
        return np.array(pr_list)
    else:
        if(t in constants.T_PRIME_SPAN):
            for j in range(N_monte):
                pdf_mc_grid = np.load(mc_folders[j]+"pdf_t{:.3f}.npy".format(t))
                pr = np.sum(pdf_mc_grid[mask]) * dV
                pr_list.append(pr)
            return pr_list
        else:
            for j in range(N_monte):
                pr_list.append(np.NaN)
            return pr_list


def compute_prob_event(target_r, targer_ph, t, p_net, e1_net_seq1, e1_net_seq2, N_discret = 50, data_folder="data/"):
    global constants
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
    # print(sum(mask_flat)) # if this is zero, then no further computation, directly set zero.
    if(sum(mask_flat) <= 0.0):
        return 0.0
    else: 
        # linear program
        # # Flatten p0 and mask to vector form.
        p0_flat = p0.flatten()
        # Decision variable: p_vec is now a vector of length N.
        p_vec = cp.Variable(N)
        # Constraints.
        constraints = [
            cp.sum(p_vec) * dV == 1,  # total mass constraint.
            p_vec >= 0,                              # non-negativity.
            p_vec >= p0_flat - B,                      # lower bound.
            p_vec <= p0_flat + B                       # upper bound.
        ]

        # Objective: maximize probability mass in target region minus smoothness penalty.
        objective = cp.Maximize(
            cp.sum(cp.multiply(p_vec, mask_flat)) * dV #- lambda_reg * smoothness_penalty
        )
        # Set up and solve the problem.
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        pr = np.sum(p_vec.value[mask_flat]) * dV
        # print("NN+Error Upper Bound value (target region):", pr2)
        print("Total mass (should be 1):", np.sum(p_vec.value) * dV)
        return pr


def app1(p_net, e1_net_seq1, e1_net_seq2):
    global constants
    
    # target region in (r, phi)
    target_r, target_phi = visual_app1(p_net, constants, show_plot=False)

    # Define a t_span to evaluate Pr(Event)
    t_span = simple_interpolate(constants.T_PRIME_SPAN)
    t_span = simple_interpolate(t_span)

    # Pr(Event) when pdf are obtained by MC
    # data_list = []
    # for t in t_span:
    #     pr_array = compute_prob_event_monte(target_r, target_phi, t)
    #     data_list.append(pr_array)
    #     print(t, pr_array)
    # # Convert the list to a NumPy array.
    # data_array = np.vstack(data_list)
    # # Now data_array is an array of shape (N, 3), where N = len(t_span).
    # print("Data array:")
    # print(data_array)
    # np.save('data/app1/pr_mcs.npy', data_array)

    # # Pr(Event) when pdf are obtained by MC vs. PINN + B1
    data_list = []
    for t in t_span:
        pr = compute_prob_event(target_r, target_phi, t, p_net, e1_net_seq1, e1_net_seq2, N_discret=40)
        data_list.append([t, pr])
        print(t, pr)
    # Convert the list to a NumPy array.
    data_array = np.array(data_list)
    # Now data_array is an array of shape (N, 3), where N = len(t_span).
    print("Data array:")
    print(data_array)
    np.save('data/app1/pr_nn_Nd40.npy', data_array)


def plot_app1(N_mc = 1):
    pr_mcs = np.load('data/app1/pr_mcs.npy')
    print(pr_mcs)
    pr_nn  = np.load('data/app1/pr_nn.npy')
    t_span = pr_nn[:,0]
    plt.figure()
    plt.plot(t_span, pr_nn[:,1], "o-", color="black", label="PINN with Error Bound")
    for j in range(N_mc):
        pr = pr_mcs[:,j]
        mask = ~np.isnan(pr)
        plt.plot(t_span[mask], pr[mask], "x:", label="M.C.")
    plt.ylabel(r"$Pr(x \in TAR)$")
    plt.xlabel("t (of orbit period)")
    plt.legend()
    plt.show()


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


def plot_target_volume(ax, target_r, target_ph, z_max, grid_resolution=30, 
                       color='red', alpha=0.2, edge_color='white', edge_width=2):
    """
    Plot an extruded 3D volume for a target region defined in spherical coordinates
    and display its edges.

    The volume is created by extruding the target region, defined by:
      - target_r: an array-like of two elements (min, max) for the radial coordinate,
      - target_ph: an array-like of two elements (min, max) for the angular coordinate (ϕ, in radians).

    The extrusion is performed from z = 0 up to z = z_max.

    Parameters:
      ax             : A matplotlib 3D axis object.
      target_r       : Array-like with two elements specifying the min and max radial limits.
      target_ph      : Array-like with two elements specifying the min and max angular limits (in radians).
      z_max          : The maximum z-value (height) for the extruded volume.
      grid_resolution: Number of grid points to use along each dimension (default: 50).
      color          : Face color for the volume surfaces (default: 'red').
      alpha          : Transparency of the volume surfaces (default: 1.0).
      edge_color     : Color to use for the edge lines (default: black).
      edge_width     : Line width for the edge lines (default: 2).

    Returns:
      surfaces: A dictionary containing:
          - 'top': top surface,
          - 'bottom': bottom surface,
          - 'side1', 'side2', 'side3', 'side4': side surfaces,
          - 'edges': a dictionary of edge line objects ('top_edge', 'bottom_edge', 'vertical_edges').
    """
    # Create grid arrays for r and ϕ.
    r_vals = np.linspace(np.min(target_r), np.max(target_r), grid_resolution)
    phi_vals = np.linspace(np.min(target_ph), np.max(target_ph), grid_resolution)
    
    # Create mesh grid for the top and bottom surfaces.
    grid_tar_r, grid_tar_ph = np.meshgrid(r_vals, phi_vals, indexing='ij')
    x_tar = grid_tar_r * np.cos(grid_tar_ph)
    y_tar = grid_tar_r * np.sin(grid_tar_ph)
    
    # Plot the top surface (z = z_max) and bottom surface (z = 0)
    surf_top = ax.plot_surface(x_tar, y_tar, np.full_like(x_tar, z_max),
                               facecolor=color, edgecolor='none', alpha=alpha)
    surf_bottom = ax.plot_surface(x_tar, y_tar, np.zeros_like(x_tar),
                                  facecolor=color, edgecolor='none', alpha=alpha)
    
    # Define vertical grid along z for the sides.
    z_side = np.linspace(0, z_max, grid_resolution)
    
    # Side 1: r = target_r.min() (vary ϕ)
    x_side1 = np.min(target_r) * np.cos(phi_vals)
    y_side1 = np.min(target_r) * np.sin(phi_vals)
    X_side1, Z_side1 = np.meshgrid(x_side1, z_side, indexing='ij')
    Y_side1, _       = np.meshgrid(y_side1, z_side, indexing='ij')
    surf_side1 = ax.plot_surface(X_side1, Y_side1, Z_side1,
                                facecolor=color, edgecolor='none', alpha=alpha, label="target region")
    
    # Side 2: r = target_r.max() (vary ϕ)
    x_side2 = np.max(target_r) * np.cos(phi_vals)
    y_side2 = np.max(target_r) * np.sin(phi_vals)
    X_side2, Z_side2 = np.meshgrid(x_side2, z_side, indexing='ij')
    Y_side2, _       = np.meshgrid(y_side2, z_side, indexing='ij')
    surf_side2 = ax.plot_surface(X_side2, Y_side2, Z_side2,
                                facecolor=color, edgecolor='none', alpha=alpha)
    
    # Side 3: ϕ = target_ph.min() (vary r)
    x_side3 = r_vals * np.cos(np.min(target_ph))
    y_side3 = r_vals * np.sin(np.min(target_ph))
    X_side3, Z_side3 = np.meshgrid(x_side3, z_side, indexing='ij')
    Y_side3, _       = np.meshgrid(y_side3, z_side, indexing='ij')
    surf_side3 = ax.plot_surface(X_side3, Y_side3, Z_side3,
                                facecolor=color, edgecolor='none', alpha=alpha)
    
    # Side 4: ϕ = target_ph.max() (vary r)
    x_side4 = r_vals * np.cos(np.max(target_ph))
    y_side4 = r_vals * np.sin(np.max(target_ph))
    X_side4, Z_side4 = np.meshgrid(x_side4, z_side, indexing='ij')
    Y_side4, _       = np.meshgrid(y_side4, z_side, indexing='ij')
    surf_side4 = ax.plot_surface(X_side4, Y_side4, Z_side4,
                                facecolor=color, edgecolor='none', alpha=alpha)
    
    # --- Add edge lines ---
    # Compute corner coordinates in the top and bottom planes.
    r_min = np.min(target_r)
    r_max = np.max(target_r)
    phi_min = np.min(target_ph)
    phi_max = np.max(target_ph)
    
    # Four corners (completing the loop by repeating the first point)
    top_corners = np.array([
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), z_max],
        [r_min * np.cos(phi_max), r_min * np.sin(phi_max), z_max],
        [r_max * np.cos(phi_max), r_max * np.sin(phi_max), z_max],
        [r_max * np.cos(phi_min), r_max * np.sin(phi_min), z_max],
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), z_max]
    ])
    bottom_corners = np.array([
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), 0],
        [r_min * np.cos(phi_max), r_min * np.sin(phi_max), 0],
        [r_max * np.cos(phi_max), r_max * np.sin(phi_max), 0],
        [r_max * np.cos(phi_min), r_max * np.sin(phi_min), 0],
        [r_min * np.cos(phi_min), r_min * np.sin(phi_min), 0]
    ])
    
    # Draw top and bottom edges.
    # edge_top, = ax.plot(top_corners[:, 0], top_corners[:, 1], top_corners[:, 2],
    #                     color=edge_color, lw=edge_width)
    # edge_bottom, = ax.plot(bottom_corners[:, 0], bottom_corners[:, 1], bottom_corners[:, 2],
    #                        color=edge_color, lw=edge_width)
    
    # Draw vertical edges connecting top and bottom corners.
    vertical_edges = []
    for i in range(4):
        ve, = ax.plot([top_corners[i, 0], bottom_corners[i, 0]],
                      [top_corners[i, 1], bottom_corners[i, 1]],
                      [top_corners[i, 2], bottom_corners[i, 2]],
                      color=edge_color, lw=edge_width)
        vertical_edges.append(ve)
    
    edges = {
        # 'top_edge': edge_top,
        # 'bottom_edge': edge_bottom,
        'vertical_edges': vertical_edges
    }
    
    surfaces = {
        'top': surf_top,
        'bottom': surf_bottom,
        'side1': surf_side1,
        'side2': surf_side2,
        'side3': surf_side3,
        'side4': surf_side4,
        'edges': edges
    }
    return surfaces


def create_second_difference_matrix(n):
    """
    Create the second-difference matrix D for a 1D grid of length n.
    (D p)[i] = p[i+1] - 2*p[i] + p[i-1], for i = 1, ..., n-2.
    D is of shape (n-2, n).
    """
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i]   = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    return D


def main():
    global constants

    ### Visualization ###
    # visual_phat_trainings()
    # plot_app1()
    # return

    p_net = PNet(scale=get_p_init_max())
    p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()
    e1_net_seq1 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
    e1_net_seq1 = load_trained_model(e1_net_seq1, path=E1NET_PATH_SEQ1, method="new"); e1_net_seq1.eval()
    e1_net_seq2 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
    e1_net_seq2 = load_trained_model(e1_net_seq2, path=E1NET_PATH_SEQ2, method="new"); e1_net_seq2.eval()

    ### Application ###
    app1(p_net, e1_net_seq1, e1_net_seq2)


if __name__ == "__main__":
    main()