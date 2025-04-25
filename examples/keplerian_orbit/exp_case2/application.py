import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata
from utilities.constants import Case2_4D_Constants
from monte import p_init, get_p_init_max
from pnet_models import PNet, PNet_FO
from e1net_models import E1Net
import sys
import os
# Get the parent directory of the current directory (exp1)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utilities.post_exp_cas2 import check_train_results, load_trained_model, get_max_e1_init, plot_train_loss
import cvxpy as cp
import scipy.sparse as sp
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import seaborn as sns
import torch.optim as optim
import torch.distributions as D
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily


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
    ## tar1
    target_r = np.array([21.3, 21.8])*constants.R
    target_ph = np.array([-3.5, 1.5])*constants.PHI + constants.W*constants.T*(0.1)

    ## tar2
    # target_r = np.array([21.3, 21.8])*constants.R
    # target_ph = np.array([-3.5, 1.5])*constants.PHI + constants.W*constants.T*(0.19)

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
        # fig.savefig("figs/target_X.pdf", format='pdf')
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

    # [1e+8, 1e+7, 1e+6, 1e+5]
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
        pr, model = compute_Pr_Guass(t, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, dV, B)
        return pr


def app1(p_net, e1_net_seq1, e1_net_seq2):
    global constants
    
    # target region in (r, phi)
    target_r, target_phi = set_and_visual_target_app1(p_net, constants, show_plot=False)

    # Define a t_span to evaluate Pr(Event)
    # t_span = simple_interpolate(constants.T_PRIME_SPAN)
    # t_span = simple_interpolate(t_span)
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
    dt = 0.005
    t_span = np.arange(0.00, 0.20+dt, dt)
    # t_span = [0.10]
    data_list = []
    for t in t_span:
        pr = compute_prob_event(target_r, target_phi, t, p_net, e1_net_seq1, e1_net_seq2, N_discret=50)
        data_list.append([t, pr])
    # Convert the list to a NumPy array.
    data_array = np.array(data_list)
    # Now data_array is an array of shape (N, 3), where N = len(t_span).
    print("Data array:")
    print(data_array)
    np.save('data/app1/pr_nn_Nd50_gmmx64(new).npy', data_array)


# Paper
def plot_app1(N_mc=1):
    set_publication_plot_style()

    parent_folder = "data/app1/tar1/"
    pr_mcs = np.load(parent_folder+"pr_mcs.npy")
    
    pr_nn_data_labels = [parent_folder+"pr_nn_Nd50_onlyphat.npy", 
                         parent_folder+"pr_nn_Nd50_phat+B.npy", 
                         parent_folder+"pr_nn_Nd50_LP.npy", 
                        #  parent_folder+"pr_nn_Nd50_FOx128(new).npy",
                         parent_folder+"pr_nn_Nd50_gmmx64(iter-10k).npy", # weight of region_loss 1e-2
                         parent_folder+"pr_nn_Nd50_gmmx64(iter-20k).npy", # weight of region_loss 1e-2
                         parent_folder+"pr_nn_Nd50_gmmx64(new).npy", # weight of region_loss 1e-1 with half random samples
                        #  parent_folder+"pr_nn_Nd50_FOx256.npy"
                         ]
    plot_labels = [r"$\hat{p}$",
                   r"$\int_{X^{'}} \hat{p}+B_1 dx$", 
                   r"LP($\hat{p},B_1$)", 
                #    r"FO($\hat{p},B_1$) RBFx256(new)",
                   r"FO($\hat{p},B_1$) GMMx64(10k det.)",
                   r"FO($\hat{p},B_1$) GMMx64(20k det.)",
                   r"FO($\hat{p},B_1$) GMMx64(20k quasi)",
                #    r"FO($\hat{p},B_1;\theta=256$)"
                   ]
    plot_fills  = [False, False, True, True, True, True]
    plot_style =  ["--", "-", "-", "-", "-", "-"]

    pr_nn_data = []
    for j in range(len(pr_nn_data_labels)):
        pr_nn_data.append(np.load(pr_nn_data_labels[j]))

    fig = plt.figure(figsize=(8,6))
    colors = sns.color_palette("husl", len(pr_nn_data_labels))

    for j in range(N_mc):
        t_span = pr_mcs[:,0]
        pr = pr_mcs[:,j+2] # [should change back to] pr = pr_mcs[:,j+1] 
        mask = ~np.isnan(pr)
        plt.plot(t_span[mask], pr[mask], color="black", linestyle="", marker="o", markersize=4, label="MC")
        print(t_span[mask], pr[mask])

    for j in range(len(pr_nn_data_labels)):
        pr_nn_data_i = pr_nn_data[j]
        t_span = pr_nn_data_i[:,0]
        plt.plot(t_span, pr_nn_data_i[:,1], color=colors[j], linestyle=plot_style[j], label=plot_labels[j])
        if(plot_fills[j]): 
            plt.fill_between(t_span, y1=0.0*t_span, y2=pr_nn_data_i[:,1],
                             color=colors[j], edgecolor="none", alpha=0.1)

    plt.grid(True)
    plt.ylabel(r"$Pr(x \in X^{'})$")
    plt.xlabel("t")
    plt.legend(loc="upper left", ncol=2)
    plt.ylim([-0.05, 1.5])
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


# Paper
def plot_app1_onlymc(N_mc=4):
    set_publication_plot_style()

    pr_mcs = np.load('data/app1/pr_mcs.npy')
    plot_labels = ["1e+8", "1e+7", "1e+6", "1e+5"]
    # pr_nn_data_labels = ["data/app1/pr_nn_Nd50_onlyphat.npy", "data/app1/pr_nn_Nd50_phat+B.npy", "data/app1/pr_nn_Nd50_LP.npy", "data/app1/pr_nn_Nd50_gmm7.npy"]
    # plot_labels = [r"$\hat{p}$",r"$\int_{X^{'}} \hat{p}+B_1 dx$", r"LP($\hat{p},B_1$)", r"F.O.($\hat{p},B_1$)"]

    fig = plt.figure(figsize=(8,6))
    colors = sns.color_palette("husl", pr_mcs.shape[1])

    for j in range(N_mc):
        t_span = pr_mcs[:,0]
        pr_base = pr_mcs[:,1] 
        pr      = pr_mcs[:,j+1]

        # Create a mask to filter out NaN values.
        mask_nan = ~np.isnan(pr)
        mask_nan_based = ~np.isnan(pr_base)
        # Create a mask that selects only the time points in t_plot_list.
        mask_time = np.isin(t_span, constants.T_PRIME_SPAN)
        # Combine the two masks.
        mask = mask_nan & mask_nan_based & mask_time

        plt.plot(t_span[mask], pr[mask] - pr_base[mask], color=colors[j], linestyle="--", marker="d", label="M.C. samples="+plot_labels[j])

    plt.grid(True)
    plt.ylabel(r"$\Delta Pr(x \in X^{'})$")
    plt.xlabel("t")
    plt.legend(loc="lower right")
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
    fig.savefig("figs/prmcs.pdf", format='pdf')
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


def generate_sobol_points_4d(N_points, bounds):
    """
    Generate N_points in a 4D space with given bounds using a Sobol sequence.
    
    Args:
        N_points (int): Number of points to generate.
        bounds (np.ndarray): Array of shape (4,2) where each row is [lower, upper].
        
    Returns:
        np.ndarray: An array of shape (N_points, 4) of Sobol-generated sample points.
    """
    # Create a Sobol engine for 4 dimensions. (scramble=True gives more randomness)
    sobol_engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True)
    x_tensor = sobol_engine.draw(N_points)  # shape: [N_points, 4], in [0,1]
    samples = x_tensor.numpy()  # convert to NumPy array
    # Scale each dimension to its respective bounds.
    scaled_samples = np.zeros_like(samples)
    for i in range(4):
        a, b = bounds[i, 0], bounds[i, 1]
        scaled_samples[:, i] = a + (b - a) * samples[:, i]
    return scaled_samples


class TorchGMM(nn.Module):
    def __init__(self, num_components=64, n_features=4):
        """
        Constructs a differentiable Gaussian Mixture Model using PyTorch distributions.

        Args:
            num_components (int): Number of mixture components.
            n_features (int): Dimensionality of each Gaussian component (e.g., 4 for a 4D density).
        """
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        
        # Learnable parameters:
        # Unnormalized mixture logit weights (will be converted to probabilities via softmax)
        self.logits = nn.Parameter(torch.zeros(num_components))
        # Means of shape [num_components, n_features]
        self.means = nn.Parameter(torch.randn(num_components, n_features)) + torch.from_numpy(constants.N_MEAN_I)
        # Raw lower-triangular matrix for each component. We have one per component.
        # This parameter is of shape [num_components, n_features, n_features].
        self.raw_scale_tril = nn.Parameter(torch.randn(num_components, n_features, n_features))
    
    def get_distribution(self):
        """
        Constructs the MixtureSameFamily distribution representing the GMM.
        The covariance for each component is built from a lower-triangular matrix
        whose diagonal is enforced positive.
        """
        # Obtain lower triangular matrices. We use torch.tril to zero out the upper part.
        # Then, we ensure the diagonal is positive via softplus.
        scale_tril = torch.tril(self.raw_scale_tril)
        # Create a mask for diagonal elements: shape (n_features, n_features)
        diag_mask = torch.eye(self.n_features, device=scale_tril.device).bool()
        
        # Apply softplus to the diagonals of each component.
        # Loop over components; if you prefer a vectorized approach you could reshape and index.
        scale_tril_fixed = scale_tril.clone()
        for k in range(self.num_components):
            scale_tril_fixed[k][diag_mask] = F.softplus(scale_tril[k][diag_mask])
        
        # Create the categorical distribution from the logits.
        cat = Categorical(logits=self.logits)
        # Create the component multivariate normals.
        comp = MultivariateNormal(loc=self.means, scale_tril=scale_tril_fixed)
        # The mixture model is a MixtureSameFamily distribution.
        gmm = MixtureSameFamily(cat, comp)
        return gmm

    def forward(self, x):
        """
        Evaluate the density at input x.
        
        Args:
            x (Tensor): Input tensor with shape [batch_size, n_features]
            
        Returns:
            pdf (Tensor): Evaluated density for each input, with shape [batch_size]
        """
        gmm = self.get_distribution()
        # Return the probability density, computed as exp(log_prob)
        return gmm.log_prob(x).exp()


class RBFDensity(nn.Module):
    def __init__(self, input_dim=4, num_basis=20):
        """
        Approximates a function p(x) as a weighted sum of normalized Gaussian RBFs,
        designed so that p(x) is a valid pdf:
        
           p(x) = sum_i w_i * phi_i(x)
        
        with phi_i(x) defined as a normalized Gaussian over R^4.
        
        The centers of each RBF are adjusted by adding constant offsets defined in
        constants.N_MEAN_I.
        
        Args:
            input_dim (int): Dimensionality of input (should be 4).
            num_basis (int): Number of RBF basis functions.
        """
        super(RBFDensity, self).__init__()
        self.input_dim = input_dim   # This should be 4.
        self.num_basis = num_basis
        
        # Learnable centers: shape [num_basis, input_dim]
        self.centers = nn.Parameter(torch.randn(num_basis, input_dim))
        
        # Learnable log-bandwidths (one per basis). Use softplus later to ensure positivity.
        self.covs = nn.Parameter(torch.ones(num_basis))
        
        # Learnable logits for weights; using softmax will enforce nonnegativity and sum-to-one.
        self.A = nn.Parameter(torch.ones(num_basis)/num_basis)

    def forward(self, x):
        """
        Evaluate the density p(x) for a batch of input points x (shape [batch, input_dim]).
        
        Returns:
            p (Tensor): The pdf evaluated at x, shape [batch].
        """
        batch = x.shape[0]
        K = self.num_basis
        
        # Get sigma with softplus for numerical stability.
        covs = torch.square(self.covs) + 1e-10
        
        # Compute normalized weights via softmax.
        A = self.A + 1e-10
        sum_A = torch.sum(A**2)
        weights = A**2 / sum_A  # shape: [K]
        
        # Adjust centers by adding the constant offset.
        offset = torch.tensor(constants.N_MEAN_I, device=self.centers.device, dtype=self.centers.dtype)
        effective_centers = self.centers + offset.unsqueeze(0)  # shape: [K, input_dim]

        pdf = torch.zeros(batch)
        for i in range(K):
            mean_i = effective_centers[i, :]
            cov_i  = covs[i]
            m = torch.distributions.MultivariateNormal(
                loc=mean_i,
                covariance_matrix=cov_i * torch.eye(4)
            )
            pdf_values = m.log_prob(x).exp()
            pdf = pdf + weights[i] * pdf_values
        return pdf
    

def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def get_valid_target_bounds(domain_bounds, target_bounds):
    """
    Compute the intersection of target_bounds and domain_bounds.
    
    Parameters:
      domain_bounds: (4, 2) np.array containing the min and max for each dimension
      target_bounds: (4, 2) np.array containing the min and max for each dimension
      
    Returns:
      valid_target_bounds: (4, 2) np.array representing the intersection bounds
      
    Raises:
      ValueError: if in any dimension the bounds do not intersect
    """
    # Compute the valid lower and upper bounds per dimension
    valid_lower = np.maximum(domain_bounds[:, 0], target_bounds[:, 0])
    valid_upper = np.minimum(domain_bounds[:, 1], target_bounds[:, 1])
    
    # Check for a valid intersection in each dimension
    if np.any(valid_lower > valid_upper):
        raise ValueError("The target region does not intersect with the domain bounds in at least one dimension.")
    
    # Stack the lower and upper bounds to form a valid bounds array.
    valid_target_bounds = np.stack([valid_lower, valid_upper], axis=1)
    return valid_target_bounds


def filter_points_in_valid_domain(x, valid_bounds):
    """
    Filters out points in x (N x 4) that lie within the valid target domain bounds.

    Parameters:
      x (torch.Tensor): A tensor of shape (N, 4) containing the points.
      valid_bounds (np.array or torch.Tensor): An array or tensor of shape (4, 2)
          where each row represents [lower_bound, upper_bound] for that dimension.

    Returns:
      torch.Tensor: A tensor containing only the points that satisfy all dimension bounds.
    """
    # Ensure valid_bounds is a torch.Tensor (if it's not already)
    if not isinstance(valid_bounds, torch.Tensor):
        valid_bounds = torch.tensor(valid_bounds, dtype=x.dtype, device=x.device)
    
    # Start with a mask of all True values (for N points)
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    
    # Loop over each dimension and update the mask
    for dim in range(x.shape[1]):
        lower_bound = valid_bounds[dim, 0]
        upper_bound = valid_bounds[dim, 1]
        mask &= (x[:, dim] >= lower_bound) & (x[:, dim] <= upper_bound)
    
    # Use the mask to filter the points
    return x[mask]


def violation_percentage_loss(p_model, p_target, B, k=100.0):
    """
    Computes a differentiable loss that approximates the percentage of violations.
    
    A violation is when |p_model - p_target| > B.
    The loss is given by the mean of a steep sigmoid that approximates the indicator function.
    
    Parameters:
      p_model (torch.Tensor): predictions (any shape, but must match p_target)
      p_target (torch.Tensor): target values (same shape as p_model)
      B (float): the acceptable bound threshold
      k (float): steepness parameter for the sigmoid (larger values make the sigmoid sharper)
    
    Returns:
      torch.Tensor: a scalar loss that approximates the fraction of violations.
    """
    # Compute the absolute difference between model and target
    diff = torch.abs(p_model - p_target)
    
    # Use sigmoid to approximate the indicator
    # When diff == B, the output is 0.5; for diff < B, it tends toward 0; for diff > B, toward 1.
    violation_approx = torch.sigmoid(k * (diff - B))
    
    # The mean approximates the percentage (fraction) of points that violate the bound.
    return torch.mean(violation_approx)


def compute_volume(bounds):
    """
    Compute the volume of an axis-aligned hyper-rectangle in 4D space.
    
    Parameters:
      bounds (np.array): A (4, 2) array where each row is [min, max] for a dimension.
      
    Returns:
      float: The volume of the hyper-rectangle.
    """
    # Compute the length of the interval in each dimension.
    side_lengths = bounds[:, 1] - bounds[:, 0]
    
    # Volume is the product of all side lengths.
    volume = np.prod(side_lengths)
    return volume


def train_model(t, model, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, B, 
                num_iterations=1000, batch_size=256, device=torch.device("cpu"), lambda_region=1.0):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # Add a learning rate scheduler.
    # For example, using StepLR to decay the learning rate by a factor of 0.9 every 10,000 iterations.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    domain_bounds = np.array([constants.X1_RANGE, constants.X2_RANGE, constants.X3_RANGE, constants.X4_RANGE])
    target_region = np.array([target_region[0,:], target_region[1,:], 
                              constants.X3_RANGE, 
                              constants.X4_RANGE])
    target_bounds = get_valid_target_bounds(domain_bounds, target_region)
    domain_V = compute_volume(domain_bounds)
    target_V = compute_volume(target_bounds)
    
    # Convert the target numpy array to a torch tensor once.
    p0_tensor = torch.from_numpy(p0_flat).to(device)

    _p_model_add = model(grid_points_tensor).view(-1,)
    _deviation = torch.abs(_p_model_add - p0_tensor)
    _, indices = torch.topk(_deviation, batch_size, largest=True)
    # Now use the selected indices to pick the corresponding training points from the violating set
    x_dom = grid_points_tensor[indices].view(-1, 4)
    p_target_dom = p0_tensor[indices].view(-1)

    _p_reg_add = p0_tensor[mask_flat].view(-1,)
    _x_red_add = grid_points_tensor[mask_flat].view(-1,4)
    _, indices = torch.topk(_p_reg_add, batch_size, largest=True)
    x_reg = _x_red_add[indices].view(-1,4)
    p_target_reg = _p_reg_add[indices].view(-1,)
    # print(p_target_reg.shape, x_reg.shape)

    x_dom = torch.cat((x_dom, x_reg), dim=0)
    p_target_dom = torch.cat((p_target_dom, p_target_reg), dim=0)

    best_loss = np.inf
    max_Pr = 0.0
    
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

        # combined loss
        total_loss = loss_full + loss_region*1e-1

        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        # Every 100 iterations, print progress and update the best model.
        if it % int(num_iterations/10) == 0:
            print(f"violation percent: {vio_percent:.4f}%")
            print(f"Iteration {it:4d}, loss: {total_loss.item()}, loss hc: {loss_full.item()}, loss reg: {loss_region.item()}")
            # print(f"Iteration {it:4d}, loss: {total_loss}, loss hc: {loss_full}, loss tar: {loss_region}, {loss_region_hc}")

        if(total_loss.item() < best_loss):
            # _x_reg = constants.sample_points(10*batch_size, target_bounds)
            # p_model_reg = model(_x_reg).view(-1,)
            # region_mass_estimate = torch.mean(p_model_reg)*target_V
            # if(region_mass_estimate.item() > max_Pr and vio_percent < 30.0):
            if(True):
                # max_Pr = region_mass_estimate.item()
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_loss = total_loss.item()
                print(f"[Saved] Iteration {it:4d}, loss: {total_loss.item()}, {loss_full.item()}, {loss_region.item()}")
                print(f"   violation percent: {vio_percent:.4f}%")
                # print(f"[Saved] Iteration {it:4d}, Region mass {max_Pr}, loss: {total_loss}, {loss_full}, {loss_region}, {loss_region_hc}")
                # print(f"   violation percent: {vio_percent:.4f}%, max Pr: {max_Pr:.3f}, compute Pr: {region_mass_estimate.item():.3f}")
            
    # After training, load the best model state.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Returning best model with total loss: {best_loss}")

    return model, max_Pr, target_bounds


def compute_Pr_Guass(t, target_region, p_net, p0_flat, mask_flat, grid_points_tensor, dV, B):
    global constants
    device = torch.device("cpu")  # or torch.device("cuda") if available.
    domain_bounds = np.array([constants.X1_RANGE, constants.X2_RANGE, constants.X3_RANGE, constants.X4_RANGE])
    target_bounds = get_valid_target_bounds(domain_bounds, target_region)

    # Create model 
    # trained_model = RBFDensity(num_basis=256).to(device) # (default degree of basis)
    trained_model = TorchGMM().to(device)

    ### [train model] ###
    trained_model, _ , _  = train_model(t, trained_model, target_region, 
                                p_net, p0_flat, mask_flat, grid_points_tensor, B, num_iterations=20000)
    torch.save(trained_model.state_dict(), "data/app1/tar1/gmm_64_t{:.3f}.pth".format(t))

    ### [load model] ###
    # trained_model.load_state_dict(torch.load("data/app1/tar1/fo_rbf_128_t{:.3f}.pth".format(t)))

    trained_model.eval()
    
    # # Check constraints over the full domain.
    x_full = grid_points_tensor.view(-1,4)
    p_target_full = torch.from_numpy(p0_flat)
    N_full = len(p_target_full)
    p_model_full = trained_model(x_full).view(-1,)
    print(f"[Check] sum of p (NN, FO): {torch.sum(p_target_full)*dV:.4f}, {torch.sum(p_model_full)*dV:.4f}")

    B_check = B  # Bound to check (can be different than training bound if desired)
    print("B: ", B_check)
    constraint_mask = (p_model_full >= (p_target_full - B_check)) & (p_model_full <= (p_target_full + B_check))
    percentage_satisfied = (constraint_mask.sum().item() / N_full) * 100
    print(f"Constraint satisfied for {percentage_satisfied:.4f}% of samples over the full domain.")
    # Get the indices of the violations.
    violation_indices = (~constraint_mask).nonzero(as_tuple=False).squeeze(1)  # shape: [num_violations]
    if violation_indices.numel() > 0:
        num_violations = violation_indices.shape[0]
        print(f"Number of constraint violations: {num_violations}")
    
        # Set the number of top violations you want to print.
        top_N = 5  # you can change this to any number you prefer
        violation_p_values = p_target_full[violation_indices]
        sorted_order = torch.argsort(violation_p_values, descending=True)
        top_violation_indices = violation_indices[sorted_order][:top_N]
        
        print(f"Example violations (top {top_N} largest p_model values):")
        for idx in top_violation_indices:
            i = idx.item()
            # Assuming that state_full is an array (or tensor) holding the state coordinates for sample i.
            # If state_full is a tensor, you might want to convert it to a list or numpy array.
            # state_coords = state_full[i]
            print(f"  Sample {i}: p_target = {p_target_full[i].item():.6f}, p_model = {p_model_full[i].item():.6f}")
    else:
        print("No constraint violations found.")
    
    # Evaluate integrated masses over the target region.
    trained_model.verbose = True
    integrated_true_pdf = np.sum(p0_flat[mask_flat])*dV

    # Evaluate p_FO(X_tar) at the same grid
    x_eval = grid_points_tensor[torch.from_numpy(mask_flat)].view(-1,4)
    p_FO_tar = trained_model(x_eval).view(-1,).detach().cpu().numpy()
    learned_pdf_values = np.sum(p_FO_tar)*dV

    # [NOTE] for the learned pdf, we could compute it by directly sampling in the target region.
    # batch_size = 1024*4*4*4*4*4
    # x_reg = constants.sample_points(batch_size, target_bounds)
    # p_model_reg = trained_model(x_reg).view(-1,)
    # region_mass_estimate = torch.mean(p_model_reg)*compute_volume(target_bounds)
    
    print("\nIntegrated phat PDF over target region: {:.4f}".format(integrated_true_pdf))
    print("Integrated Learned PDF over target region: {:.4f}".format(learned_pdf_values))
    # print("Integrated Learned PDF over target region: (random uniform samples) {:.4f}".format(region_mass_estimate))

    # visual_fo_rbf(t, p_net, trained_model, target_region)

    return learned_pdf_values, trained_model


def visual_fo_rbf(t, p_net, model, target_region, N_discret=50):
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

    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    p0 = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

    # plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p0_2D =  np.sum(p0, axis=(2,3)) * dx3 * dx4
    p_rbf = model(grid_points_tensor).detach().numpy().reshape(x1_grid.shape)
    p_rbf_2D = np.sum(p_rbf, axis=(2,3)) * dx3 * dx4
    X, Y =  np.meshgrid(x1s, x2s, indexing="ij")
    ax.plot_surface(X, Y, p0_2D, color="none", rstride=2, cstride=2, 
                    edgecolor='black', linewidth=0.5, label=r"$\hat{p}$")
    ax.plot_surface(X, Y, p_rbf_2D, color="none", rstride=2, cstride=2, 
                    edgecolor='blue', linestyle="--", linewidth=0.5, label=r"$p_{FO}$")
    
    # Create the patch for the target region.
    # Assume target_region is a 2x2 array:
    #   target_region[0] = [r_min, r_max]
    #   target_region[1] = [phi_min, phi_max]
    r_bounds = target_region[0, :]    # [r_min, r_max]
    phi_bounds = target_region[1, :]  # [phi_min, phi_max]
    
    # Define the corners of the rectangular patch at z = 0.
    patch_vertices = [
        [r_bounds[0], phi_bounds[0], 0],
        [r_bounds[1], phi_bounds[0], 0],
        [r_bounds[1], phi_bounds[1], 0],
        [r_bounds[0], phi_bounds[1], 0],
    ]
    # Create a Poly3DCollection and add it to the 3D axis.
    patch = Poly3DCollection([patch_vertices], facecolor='red', alpha=0.5, edgecolor='k', label=r"$X^'_{tar}$")
    ax.add_collection3d(patch)
    
    ax.set_xlabel(r"$r'$")
    ax.set_ylabel(r"$\phi'$")
    ax.set_zlabel("PDF")
    plt.legend()
    plt.show()


def main():
    global constants

    ### Visualization ###
    # visual_phat_trainings()
    # visual_e1hat_training()
    # plot_train_loss(E1NET_PATH_SEQ2)
    # plot_app1_onlymc()
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