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
from post.post_exp_cas2 import check_train_results, load_trained_model, get_max_e1_init, plot_train_loss
import cvxpy as cp
import scipy.sparse as sp
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import seaborn as sns
import torch.optim as optim
import torch.distributions as D

"""
select the NN models in output/
"""
PNET_PATH = "output/v2/p_net.pth"
E1NET_PATH_SEQ1 = "output/v2/e1_net_seq1.pth" # seq 1
E1NET_PATH_SEQ2 = "output/v2/e1_net_seq2.pth" # seq 2
DATA_FOLDER = "data/"
TRAIN_FLAG = False
constants = Case2_4D_Constants()
# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def set_publication_plot_style(font_family='Times New Roman', font_size=18):
    """
    Update Matplotlib settings to use publication-ready fonts.

    Parameters:
        font_family (str): Font family to be used for all texts.
        font_size (int): Base font size for labels, titles, legends, and ticks.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.titlesize'] = font_size
    plt.rcParams['lines.linewidth'] = 2


# Paper
def visual_phat_trainings(idx=0):
    """
    visualize phat training results using intermediate saved model
    """
    set_publication_plot_style()

    global constants

    # load p_net
    p_net = PNet(scale=get_p_init_max())
    if(idx >= 1):
        p_net = load_trained_model(p_net, path="output/v2/p_net_"+str(idx)+".pth", method="new"); p_net.eval()
    else:
        p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()

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
                                        edgecolor='white', linewidth=1.0, linestyle="-", label="MC")
                x_text = (grid_x.min() + grid_x.max()) / 2
                y_text = (grid_y.min() + grid_y.max()) / 2
                ax.text(x_text, y_text, 1.1*np.max(z_nn), f"t={t_label:.2f}", color="white", fontsize=14)

        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, cmap='viridis', alpha=0.8)
            if(t_label in constants.T_PRIME_SPAN):
                surf2 = ax.plot_wireframe(grid_x, grid_y, grid_z_mo, color="none", rstride=num_strides, cstride=num_strides, 
                                        edgecolor='white', linewidth=1.0, linestyle="-")
                x_text = (grid_x.min() + grid_x.max()) / 2
                y_text = (grid_y.min() + grid_y.max()) / 2
                ax.text(x_text, y_text, 1.1*np.max(z_nn), f"t={t_label:.2f}", color="white", fontsize=14)
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Optional: Add a color bar
    # Labels and title
    ax.legend()
    ax.view_init(40, -133) # viewing angle
    ax.set_xlabel('X, m', color='white', labelpad=10)
    ax.set_ylabel('Y, m', color='white', labelpad=10)
    ax.set_zlabel('PDF', color='white', labelpad=10)
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))

    # Save the figure as a PDF, with minimal extra margins and no clipping of labels
    plt.tight_layout(pad=0.3)
    # fig.savefig("figs/v2phat_"+str(idx)+".pdf", format='pdf')
    plt.show()


# Paper
def visual_e1hat_training(idx=0):
    """
    """
    set_publication_plot_style()

    global constants

    # load p_net
    p_net = PNet(scale=get_p_init_max())
    p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()
    if(idx >= 1):
        e1_net_seq1 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
        e1_net_seq1 = load_trained_model(e1_net_seq1, path="output/v2/e1_net_seq1_"+str(idx)+".pth", method="new"); e1_net_seq1.eval()
        e1_net_seq2 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
        e1_net_seq2 = load_trained_model(e1_net_seq2, path="output/v2/e1_net_seq2_"+str(idx)+".pth", method="new"); e1_net_seq2.eval()
    else:
        e1_net_seq1 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
        # e1_net_seq1 = load_trained_model(e1_net_seq1, path=E1NET_PATH_SEQ1, method="new"); e1_net_seq1.eval()
        e1_net_seq2 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
        # e1_net_seq2 = load_trained_model(e1_net_seq2, path=E1NET_PATH_SEQ2, method="new"); e1_net_seq2.eval()

    x1s = np.load(DATA_FOLDER+"x1s.npy")
    x2s = np.load(DATA_FOLDER+"x2s.npy")
    x3s = np.load(DATA_FOLDER+"x3s.npy")
    x4s = np.load(DATA_FOLDER+"x4s.npy")

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    ymin = [-0.025, -0.070]
    ymax = [0.019,  0.042]

    t_span = [constants.T_PRIME_SPAN[2], constants.T_PRIME_SPAN[-1]]
    for i in range(len(t_span)):
        t_prime = t_span[i]
        pdf_true = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        if(t_prime >= 0.5*constants.TF/constants.T):
            e1_net = e1_net_seq1
        else:
            e1_net = e1_net_seq2
        e1 = pdf_true - pdf_nn
        e1_nn  = e1_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
        
        e1_vec = e1.reshape(-1)
        e1_nn_vec = e1_nn.reshape(-1)
        a1 = np.max(np.abs(e1_vec - e1_nn_vec)) / np.max(np.abs(e1_nn_vec))
        print("a1 (t=", np.round(t_prime,2),"): ", np.round(a1,3))
        # print(E_x1, E_x2, E_x3, E_x4)
        max_e1 = np.max(np.abs(e1_vec))
        max_e1_nn = np.max(np.abs(e1_nn_vec))
        # print(max_e1, max_e1_nn)
        B1 = 2.0 * max_e1_nn

        gap = 1
        # Generate an array for the x-axis based on indices
        x = np.arange(len(e1_vec))[::gap]
        # Select every n-th element from the data array for the y-axis
        y1 = e1_vec[::gap]
        # y2 = e1_nn_vec[::gap]
        axs[i].plot(x, y1, "black", linewidth=0.5, rasterized=True, label="MC")
        # axs[i].plot(x, y2, "blue",  linewidth=0.5, rasterized=True, label="NN")
        # axs[i].fill_between(x, y1=0.0*x+B1, y2=0.0*0-B1, 
        #                     color="green", edgecolor="none", alpha=0.1, label=r"$B_1$")
        axs[i].set_ylabel("Error")
        axs[i].grid(True)
        axs[i].text(0.02, 0.98, f"t={t_prime:.2f}T", transform=axs[i].transAxes,
                    ha='left', va='top', color='black', fontsize=18,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        axs[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        # axs[i].set_ylim([ymin[i], ymax[i]])
        # ymin, ymax = axs[i].get_ylim()
        # print(f"Subplot {i}: ymin = {ymin}, ymax = {ymax}")

    axs[1].set_xlabel("4D state idx")
    axs[0].legend(loc="lower left", ncol=3)
    # fig.subplots_adjust(left=0.1)
    plt.tight_layout(pad=0.2)
    # fig.savefig("figs/v2e1hat_"+str(idx)+".pdf", format='pdf')
    plt.show()
    # plt.fill_between(idx_plot, y1=0.0*idx_plot+B1, y2=0.0*idx_plot-B1, 
    #                      color="green", alpha=0.2, label=r"$B$")


# Paper
def set_and_visual_target_app1(p_net, constants, show_plot=False):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    # specify a fixedtarget region in spherical coordinate
    target_r = np.array([21.3, 21.8])*constants.R
    target_ph = np.array([-3.5, 1.5])*constants.PHI + constants.W*constants.T*(0.1)

    if(show_plot):
        # Create a figure with a black background
        fig = plt.figure(figsize=(8, 6), facecolor='black')
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
        # Save the figure as a PDF, with minimal extra margins and no clipping of labels
        plt.tight_layout(pad=0.3)
        fig.savefig("figs/target_X.pdf", format='pdf')
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
    print("\n[check] t, sum(mask), B: ", t, sum(mask_flat), B)
    if(sum(mask_flat) <= 0.0):
        return 0.0
    else: 
        # # (1.0) only phat
        p0_flat = p0.flatten()
        pr = np.sum(p0_flat[mask_flat])*dV
        return pr
    
        # (1.1) naive: integral (phat + B1) dx
        # p0_flat = p0.flatten()
        # pr = np.sum(p0_flat[mask_flat])*dV + sum(mask_flat)*dV*B
        # return pr

        # (2.0) linear program
        # Flatten p0 and mask to vector form.
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
        # print("Total mass (should be 1):", np.sum(p_vec.value) * dV)
        return pr

        # (3.0) Learning
        # p0_flat = p0.flatten()
        # pr = compute_Pr_Guass(t, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, dV, B)
        # return pr


def app1(p_net, e1_net_seq1, e1_net_seq2):
    global constants
    
    # target region in (r, phi)
    target_r, target_phi = set_and_visual_target_app1(p_net, constants, show_plot=False)

    # Define a t_span to evaluate Pr(Event)
    # t_span = [constants.T_PRIME_SPAN[2]]
    t_span = simple_interpolate(constants.T_PRIME_SPAN)
    t_span = simple_interpolate(t_span)
    
    # # Pr(Event) when pdf are obtained by MC
    # data_list = []
    # for t in t_span:
    #     pr_array = compute_prob_event_monte(target_r, target_phi, t)
    #     data = np.insert(pr_array, 0, t)
    #     data_list.append(data)
    # # Convert the list to a NumPy array.
    # data_array = np.vstack(data_list)
    # # Now data_array is an array of shape (N, 3), where N = len(t_span).
    # print("Data array:")
    # print(data_array)
    # np.save('data/app1/pr_mcs.npy', data_array)

    # Pr(Event) when pdf are obtained by MC vs. PINN + B1, with finer t_span (possibly continuous)
    refine = 2
    for i in range(refine):
        t_span = simple_interpolate(t_span)
    data_list = []
    for t in t_span:
        pr = compute_prob_event(target_r, target_phi, t, p_net, e1_net_seq1, e1_net_seq2, N_discret=50)
        data_list.append([t, pr])
    # Convert the list to a NumPy array.
    data_array = np.array(data_list)
    # Now data_array is an array of shape (N, 3), where N = len(t_span).
    print("Data array:")
    print(data_array)
    np.save('data/app1/pr_nn_Nd50_phat+B.npy', data_array)


# Paper
def plot_app1(N_mc=1):
    set_publication_plot_style()

    pr_mcs = np.load('data/app1/pr_mcs.npy')
    
    pr_nn_data_labels = ["data/app1/pr_nn_Nd50_onlyphat.npy", "data/app1/pr_nn_Nd50_phat+B.npy", "data/app1/pr_nn_Nd50_LP.npy", "data/app1/pr_nn_Nd50_gmm7.npy"]
    plot_labels = [r"$\hat{p}$",r"$\int_{X^{'}} \hat{p}+B_1 dx$", r"LP($\hat{p},B_1$)", r"GMM($\hat{p},B_1$)"]
    
    pr_nn_data = []
    for j in range(len(pr_nn_data_labels)):
        pr_nn_data.append(np.load(pr_nn_data_labels[j]))

    fig = plt.figure(figsize=(8,6))
    colors = sns.color_palette("husl", len(pr_nn_data_labels))

    for j in range(N_mc):
        t_span = pr_mcs[:,0]
        pr = pr_mcs[:,j+1] 
        mask = ~np.isnan(pr)
        plt.plot(t_span[mask], pr[mask], color="black", linestyle="--", marker="d", label="M.C.")

    for j in range(len(pr_nn_data_labels)):
        pr_nn_data_i = pr_nn_data[j]
        t_span = pr_nn_data_i[:,0]
        plt.plot(t_span, pr_nn_data_i[:,1], color=colors[j], label=plot_labels[j]) 
        # plt.fill_between(t_span, y1=0.0*t_span, y2=pr_nn_data_i[:,1],
        #                  color=colors[j], edgecolor="none", alpha=0.2)

    plt.grid(True)
    plt.ylabel(r"$Pr(x \in X^{'})$")
    plt.xlabel("t")
    plt.legend(loc="upper left", ncol=2)
    plt.ylim([0.0, 1.5])
    # Get current axes, and then obtain and reformat the xticks.
    plt.tight_layout(pad=0.2)
    # Define the tick positions.
    ticks = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    # Create corresponding labels with "T" appended.
    tick_labels = [f"{tick:.2f}T" for tick in ticks]

    # Get the current axis.
    ax = plt.gca()
    # Set the tick positions.
    ax.set_xticks(ticks)
    # Set the tick labels.
    ax.set_xticklabels(tick_labels)
    fig.savefig("figs/v2pr.pdf", format='pdf')
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
                       color='red', alpha=0.3, edge_color='gray', edge_width=2.5):
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
    edge_top, = ax.plot(top_corners[:, 0], top_corners[:, 1], top_corners[:, 2],
                        color=edge_color, lw=edge_width)
    edge_bottom, = ax.plot(bottom_corners[:, 0], bottom_corners[:, 1], bottom_corners[:, 2],
                           color=edge_color, lw=edge_width)
    
    # Draw vertical edges connecting top and bottom corners.
    vertical_edges = []
    for i in range(4):
        ve, = ax.plot([top_corners[i, 0], bottom_corners[i, 0]],
                      [top_corners[i, 1], bottom_corners[i, 1]],
                      [top_corners[i, 2], bottom_corners[i, 2]],
                      color=edge_color, lw=edge_width)
        vertical_edges.append(ve)
    
    surfaces = {
        'top': surf_top,
        'bottom': surf_bottom,
        'side1': surf_side1,
        'side2': surf_side2,
        'side3': surf_side3,
        'side4': surf_side4,
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


### Functional Optimization
class GaussianMixtureModel(nn.Module):
    def __init__(self, num_components=4, input_dim_t=1, output_dim_x=4, hidden_dim=64, verbose=False):
        """
        Constructs a neural network that outputs the parameters of a Gaussian Mixture Model.
        For each t, the network outputs a normalized pdf p(x,t) over x in ℝ⁴ as a weighted sum
        of K Gaussians.
        
        Args:
            input_dim_t (int): Dimension of the time input t. (e.g., 1 if t is scalar)
            output_dim_x (int): Dimension of x (here 4).
            hidden_dim (int): Number of hidden neurons.
            num_components (int): Number of mixture components (here 2).
            verbose (bool): If True, prints debug information.
        """
        super(GaussianMixtureModel, self).__init__()
        self.input_dim_t = input_dim_t
        self.output_dim_x = output_dim_x  # e.g., 4
        self.num_components = num_components
        self.verbose = verbose

        # Two hidden layers.
        self.fc1 = nn.Linear(input_dim_t, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4*hidden_dim)
        # For each mixture component, we need:
        #  - output_dim_x for the mean (4)
        #  - output_dim_x*(output_dim_x+1)//2 for the covariance lower-triangular parameters (10)
        #  - 1 for the weight (logits)
        comp_param_dim = output_dim_x + (output_dim_x * (output_dim_x + 1)) // 2 + 1  # 4 + 10 + 1 = 15
        final_out_dim = num_components * comp_param_dim  # For K=2: 30
        self.fc_out = nn.Linear(4*hidden_dim, final_out_dim)
    
    def forward(self, x_input, t):
        """
        Args:
            x_input (Tensor): Points at which to evaluate the PDF, shape [batch, output_dim_x].
            t (Tensor): Time input of shape [batch, input_dim_t].
        
        Returns:
            pdf (Tensor): Evaluated PDF at x_input, shape [batch, 1].
        """
        # Pass t through the hidden layers.
        h1 = F.relu(self.fc1(t))
        h2 = F.relu(self.fc2(h1))
        out = self.fc_out(h2)  # shape: [batch, final_out_dim]
        
        batch_size = out.shape[0]
        K = self.num_components
        d = self.output_dim_x  # For x in ℝ⁴.
        comp_param_dim = d + (d*(d+1))//2 + 1  # =15
        
        # Reshape output into [batch, K, comp_param_dim].
        out = out.view(batch_size, K, comp_param_dim)
        
        # Extract mean vectors for each component. Shape: [batch, K, d]
        means = out[:, :, :d]
        # (Optional: add constant offsets to each dimension.)
        for i in range(d):
            means[:, :, i] = means[:, :, i] + constants.N_MEAN_I[i]
        
        # Extract covariance parameters for each component. Shape: [batch, K, d*(d+1)//2]
        cov_params = out[:, :, d : d + (d*(d+1))//2]
        
        # Extract weight logits for each component. Shape: [batch, K]
        weight_logits = out[:, :, -1]
        # Normalize to obtain valid mixture weights.
        weights = F.softmax(weight_logits, dim=1)  # sums to 1 across components
        
        # Build lower-triangular matrices L for each mixture component.
        L = torch.zeros(batch_size, K, d, d, device=out.device)
        idx = 0
        for i in range(d):
            for j in range(i+1):
                if i == j:
                    # Diagonals: exponentiate to enforce positivity.
                    L[:, :, i, j] = torch.exp(cov_params[:, :, idx])
                else:
                    L[:, :, i, j] = cov_params[:, :, idx]
                idx += 1
        
        # Compute covariance matrices for each component: cov = L Lᵀ.
        # Reshape L for batch-matrix multiplication.
        L_reshaped = L.view(batch_size * K, d, d)
        cov = torch.bmm(L_reshaped, L_reshaped.transpose(1, 2))
        cov = cov.view(batch_size, K, d, d)
        
        # Create a multivariate normal distribution for each component.
        components = D.MultivariateNormal(loc=means, covariance_matrix=cov)
        # Create a categorical distribution for the mixture weights.
        cat = D.Categorical(probs=weights)
        # Create the mixture distribution.
        mixture = D.MixtureSameFamily(cat, components)
        
        # Evaluate the log probability (and hence density) at x_input.
        log_prob = mixture.log_prob(x_input)  # shape: [batch]
        pdf = torch.exp(log_prob).unsqueeze(1)  # shape: [batch, 1]
        
        if self.verbose:
            print("[check] mixture weights:", weights[0])
            # print("[check] mixture means (first example):", means[0])
            # print("[check] mixture covariances (first example):", cov[0])
        return pdf
    

class GaussianModel(nn.Module):
    global constants
    def __init__(self, input_dim_t=1, output_dim_x=4, hidden_dim=64, verbose=False):
        """
        Constructs a neural network p that for each t outputs a normalized pdf p(x,t)
        over x in R^4 by outputting the parameters of a multivariate Gaussian.
        
        Args:
            input_dim_t (int): Dimension of time input t.
            output_dim_x (int): Dimension of x (here 4).
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(GaussianModel, self).__init__()
        self.input_dim_t = input_dim_t
        self.output_dim_x = output_dim_x  # here x ∈ ℝ⁴
        self.verbose = verbose

        # Two hidden layers
        self.fc1 = nn.Linear(input_dim_t, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # The final output layer produces:
        #   - output_dim_x numbers for the mean vector, and
        #   - output_dim_x*(output_dim_x+1)//2 numbers for the lower-triangular part of L.
        final_out_dim = output_dim_x + (output_dim_x * (output_dim_x + 1)) // 2
        self.fc_out = nn.Linear(hidden_dim, final_out_dim)
    def forward(self, x_input, t):
        """
        Args:
            t (Tensor): Time input of shape [batch, input_dim_t].
        
        Returns:
            dist (torch.distributions.MultivariateNormal): A Gaussian distribution 
                with parameters depending on t.
        """
        # Pass through the network
        h = F.gelu(self.fc1(t))
        h = F.gelu(self.fc2(h))
        out = self.fc_out(h)  # shape: [batch, final_out_dim]
        
        batch_size = out.shape[0]
        n = self.output_dim_x
        
        # Extract the mean vector (first n outputs)
        mean = out[:, :n]  # shape: [batch, 4]
        for i in range(4):
            mean[:,i] = mean[:,i] + constants.N_MEAN_I[i]
        
        # The remaining outputs parameterize the lower-triangular matrix L
        cov_params = out[:, n:]  # shape: [batch, n(n+1)/2]
        L = torch.zeros(batch_size, n, n, device=out.device)
        
        idx = 0
        for i in range(n):
            for j in range(i+1):
                if i == j:
                    # Diagonal entries: exponentiate to ensure they are positive.
                    L[:, i, j] = torch.exp(cov_params[:, idx])
                else:
                    L[:, i, j] = cov_params[:, idx]
                idx += 1
        
        # Construct covariance matrix: Sigma = L Lᵀ (always positive definite)
        cov = torch.bmm(L, L.transpose(1, 2))

        if(self.verbose):
            print("[check] pnn_gmm mean: ", mean[0,:])
            print("[check] pnn_gmm cov:  ", cov[0,:,:])

        # Create a multivariate normal distribution with these parameters.
        # By construction, this density is normalized over R^4.
        dist = D.MultivariateNormal(mean, covariance_matrix=cov)

        # Evaluate the density at x_input.
        log_prob = dist.log_prob(x_input)       # shape: [batch]
        pdf = torch.exp(log_prob).unsqueeze(1)  # shape: [batch, 1]
        return pdf


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(t, target_region, p_net, p0_flat, grid_points_tensor, B, 
                num_iterations=2000, batch_size=300, device=torch.device("cpu"), lambda_region=1000.0):
    # Create the model and optimizer.
    model = GaussianMixtureModel(num_components=7).to(device)
    model.apply(init_weights_He)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Add a learning rate scheduler.
    # For example, using StepLR to decay the learning rate by a factor of 0.9 every 10,000 iterations.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    domain_bounds = np.array([constants.X1_RANGE, constants.X2_RANGE, constants.X3_RANGE, constants.X4_RANGE])
    target_bounds = np.array([target_region[0,:], target_region[1,:], 
                              constants.X3_RANGE, 
                              constants.X4_RANGE])
    
    # Convert the target numpy array to a torch tensor once.
    p0_tensor = torch.from_numpy(p0_flat).to(device)
    # # Select a fixed mini-batch from the full dataset based on sorted p0 values.
    # indices = torch.argsort(p0_tensor, descending=True)[:batch_size]
    # x_fix = grid_points_tensor[indices].view(-1, 4)
    # t_fix = torch.full((x_fix.shape[0], 1), t).view(-1, 1)
    # p_target_fix = p0_tensor[indices].view(-1,)
    # # Create the region data once.
    # x_region = grid_points_tensor[torch.from_numpy(mask_flat)].view(-1, 4)
    # t_region = torch.full((x_region.shape[0], 1), t).view(-1, 1)
    # N_region = x_region.shape[0]
    
    best_loss = np.inf
    best_violation = np.inf
    best_model_state = None
    
    for it in range(num_iterations):
        optimizer.zero_grad()

        x_batch = constants.sample_points(batch_size, domain_bounds)
        t_batch = torch.full((x_batch.shape[0], 1), t).view(-1, 1)
        p_target_batch = p_net(x_batch, t_batch).view(-1,)

        random_indices = torch.randperm(grid_points_tensor.shape[0])[:batch_size]
        x_fix = grid_points_tensor[random_indices].view(-1, 4)
        t_fix = torch.full((x_fix.shape[0], 1), t).view(-1, 1)
        p_target_fix = p0_tensor[random_indices].view(-1,)
        p_target_hc = torch.cat((p_target_batch, p_target_fix), dim=0)
        # p_target_hc = p_target_batch

        p_model_batch = model(x_batch, t_batch).view(-1,)
        p_model_fix   = model(x_fix, t_fix).view(-1,)
        p_model_hc    = torch.cat((p_model_batch, p_model_fix), dim=0)
        # p_model_hc = p_model_batch
        hc = (p_model_hc >= (p_target_hc - B)) & (p_model_hc <= (p_target_hc + B))
        if not hc.all():
            loss_full = 1e+10*torch.mean((p_target_hc - p_model_hc) ** 2)
        else:
            loss_full = 0.0*torch.mean((p_target_hc - p_model_hc) ** 2)

        # --- Region Loss (Mass Maximization) ---
        x_tar = constants.sample_points(batch_size, target_bounds)
        t_tar = torch.full((x_tar.shape[0], 1), t).view(-1, 1)
        p_model_tar = model(x_tar, t_tar).view(-1,)
        region_mass_estimate = torch.mean(p_model_tar)
        loss_region = -(region_mass_estimate)

        total_loss = loss_full + lambda_region*loss_region
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Every 100 iterations, print progress and update the best model.
        # if it % int(num_iterations/10) == 0:
        #     print(f"Iteration {it:4d}, mini-batch loss: {total_loss.item()}, loss full: {loss_full.item()}, loss region: {lambda_region*loss_region.item()}")
        
        # (0. naive) current_loss = total_loss.item()
        # (1. validation loss)
        N_scale = 10
        x_batch = constants.sample_points(N_scale*batch_size, domain_bounds)
        t_batch = torch.full((x_batch.shape[0], 1), t).view(-1, 1)
        p_target_batch = p_net(x_batch, t_batch).view(-1,)
        random_indices = torch.randperm(grid_points_tensor.shape[0])[:10*batch_size]
        x_fix = grid_points_tensor[random_indices].view(-1, 4)
        t_fix = torch.full((x_fix.shape[0], 1), t).view(-1, 1)
        p_target_fix = p0_tensor[random_indices].view(-1,)
        p_target_hc = torch.cat((p_target_batch, p_target_fix), dim=0)
        p_model_batch = model(x_batch, t_batch).view(-1,)
        p_model_fix   = model(x_fix, t_fix).view(-1,)
        p_model_hc    = torch.cat((p_model_batch, p_model_fix), dim=0)
        hc = (p_model_hc >= (p_target_hc - B)) & (p_model_hc <= (p_target_hc + B))
        violations = float(100.0 * (~hc).sum().item() / p_target_hc.shape[0])
        if(violations < best_violation or violations < (100.0-99.9999) ):
            x_tar = constants.sample_points(N_scale*batch_size, target_bounds)
            t_tar = torch.full((x_tar.shape[0], 1), t).view(-1, 1)
            p_model_tar = model(x_tar, t_tar).view(-1,)
            region_mass_estimate = torch.mean(p_model_tar)
            current_loss = -(region_mass_estimate)
            if(current_loss < best_loss):
                best_violation = violations
                best_loss = current_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"[Saved] Iteration {it:4d}, vio%: {violations}, loss: {current_loss}")

    # After training, load the best model state.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Returning best model with total loss: {best_loss}")
    
    return model


# Testing routine.
def compute_Pr_Guass(t, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, dV, B):
    global constants
    device = torch.device("cpu")  # or torch.device("cuda") if available.

    trained_model = train_model(t, target_region, p_net, p0_flat, grid_points_tensor, B)
    
    # # Check constraints over the full domain.
    x_full = grid_points_tensor.view(-1,4)
    p_target_full = torch.from_numpy(p0_flat)
    N_full = len(p_target_full)
    t_full = torch.full((N_full, 1), t).view(-1,1)
    p_model_full = trained_model(x_full, t_full).view(-1,)
    B_check = B  # Bound to check (can be different than training bound if desired)
    print("B: ", B_check)
    constraint_mask = (p_model_full >= (p_target_full - B_check)) & (p_model_full <= (p_target_full + B_check))
    percentage_satisfied = (constraint_mask.sum().item() / N_full) * 100
    print(f"Constraint satisfied for {percentage_satisfied:.2f}% of samples over the full domain.")
    violation_indices = (~constraint_mask).nonzero(as_tuple=False)
    if violation_indices.numel() > 0:
        print(f"Number of constraint violations: {violation_indices.shape[0]}")
        print("Example violations (first 5):")
        for idx in violation_indices[:5]:
            i = idx.item()
            print(f"  Sample {i}: p_target = {p_target_full[i].item():.6f}, p_model = {p_model_full[i].item():.6f}")
    else:
        print("All samples satisfy the constraint.")
    
    # Evaluate integrated masses over the target region.
    trained_model.verbose = True
    x_eval = grid_points_tensor[torch.from_numpy(mask_flat)].view(-1,4)
    t_eval= torch.full((x_eval.shape[0], 1), t).view(-1,1)
    learned_pdf_values = trained_model(x_eval, t_eval).view(-1,).detach().cpu().numpy()
    integrated_true_pdf = np.sum(p0_flat[mask_flat])*dV
    integrated_learned_pdf = np.sum(learned_pdf_values)*dV
    
    print("\nIntegrated phat PDF over target region: {:.4f}".format(integrated_true_pdf))
    print("Integrated Learned PDF over target region: {:.4f}".format(integrated_learned_pdf))

    return integrated_learned_pdf


def main():
    global constants

    ### Visualization ###
    # visual_phat_trainings()
    # visual_e1hat_training()
    # plot_train_loss(E1NET_PATH_SEQ2)
    plot_app1()
    return

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