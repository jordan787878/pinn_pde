import numpy as np
import torch
from monte import get_p_init_max
from exp_utilities.constants import Case2_4D_Constants
import exp_utilities.plot_utilites as exp_plot
from baseline_methods import SAVE_PATH_LINEAR_PROPAGATE, get_linear_propagation_data
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, load_trained_model
from scipy.stats import multivariate_normal
constants = Case2_4D_Constants()


def p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x 4), N is the sample size
    """
    pdf_func = multivariate_normal(mean=mean, cov=cov)
    pdf_eval = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
    return pdf_eval


def p_total_variation(p1, p2):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    print(np.sum(np.abs(p1-p2)))


def visual_different_methods(constants, p_net, data_mc=None, data_lp=None, save_plots=False, save_plot_path=None):
    """
    visualize phat training results using final or intermediate saved model
    """
    exp_plot.set_publication_plot_style()

    # Create a figure with a black background
    fig = exp_plot.plt.figure(figsize=(8, 6), facecolor='black')
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
        x1s = np.load("data/grids/x1s.npy")
        x2s = np.load("data/grids/x2s.npy")
        x3s = np.load("data/grids/x3s.npy")
        x4s = np.load("data/grids/x4s.npy")

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
            pdf_monte = np.load(data_mc+"pdf_t{:.3f}.npy".format(t_prime))
            pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

            _, _mu_lp, _cov_lp = get_linear_propagation_data(data_lp, t_prime)
            pdf_lp = p_normal(grid_points, _mu_lp, _cov_lp).reshape(x1_grid.shape)
            pdf_lp_Nrphi =  np.sum(pdf_lp, axis=(2,3)) * dx3 * dx4

        p_total_variation(pdf_nn, pdf_monte)
        p_total_variation(pdf_lp, pdf_monte)

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
        grid_z_nn = exp_plot.griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        if(t_label in constants.T_PRIME_SPAN):
            grid_z_mo = exp_plot.griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

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
    exp_plot.plt.tight_layout(pad=0.3)
    if(save_plots):
        if(save_plot_path is not None):
            print("Save plot to: ", save_plot_path)
            fig.savefig(save_plot_path, format='pdf')
    else:
        exp_plot.plt.show()


def compare_methods():
    p_net = PNet(constants, scale=get_p_init_max(constants))
    p_net = load_trained_model(p_net, path="output/v0/p_net.pth"); p_net.eval()

    data_linear_propagate = np.load(SAVE_PATH_LINEAR_PROPAGATE)

    visual_different_methods(constants, p_net, 
                                      data_mc="data/1e+6/", 
                                      data_lp=data_linear_propagate,
                                      save_plots=False, save_plot_path=None)


def main():
    compare_methods()


if __name__ == "__main__":
    main()