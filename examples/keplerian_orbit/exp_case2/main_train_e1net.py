"""
train a p_nn that is parameterized by GMM

"""
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
from monte import p_init, get_p_init_max, p_init_perturb
from main_train_pnetreg import diff_opt_p
from main_train_pnet_gmm import PNet_GMM
from pnet_models import PNet
from e1net_models import E1Net

# standard training
E1NET_PATH = "output/e1_net.pth"
DATA_FOLDER = "data/"
device = "cpu"
TRAIN_FLAG = False
constants = Case2_4D_Constants()

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        # init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=50000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = 0.02914694

    N0_samples = 2000
    Nr_samples = 2000
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    x, t = constants.sample_res_points(Nr_samples)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc).to(device)
        e_i = p_i - phat_i
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt_p(x, t, p_net)
        res_e = diff_opt_p(x, t, e1_net)
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)

        # tv
        # epsilon = 1e-1
        # z_bc = x_bc + torch.randn_like(x_bc) * epsilon
        # p_i = p_init(z_bc.detach().numpy())
        # p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        # phat_i = p_net(z_bc, t_bc).to(device)
        # e_i = p_i - phat_i
        # ehat_i = e1_net(z_bc, t_bc).to(device)
        # ic_z = e_i - ehat_i
        # tv_loss_ic = mse_cost_function(ic_z/normalize, ic/normalize)

        loss = mse_u + 1e-1*mse_res
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,   ",res:", mse_res.data,
                #   ",tv:",tv_loss_ic.data
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'train_time': train_time,
                    }, E1NET_PATH)
            min_loss = loss.data
            FLAG = True

        # RAR
        if(epoch % 100 == 0 and FLAG):
            x_bc_rar, t_bc_rar = constants.sample_init_points(S)
            x_rar, t_rar = constants.sample_res_points(S)
            # add initial points
            p_i = p_init(x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            e_i = p_i - phat_i
            ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # add residual points
            res_p = diff_opt_p(x_rar, t_rar, p_net)/normalize
            res_e = diff_opt_p(x_rar, t_rar, e1_net)/normalize
            res = res_e + res_p
            max_error= torch.max(torch.abs(res))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def load_trained_model(net, path, method="old"):
    print("[load model from: "+ path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if(method == "new"):
        loss_history = np.array(checkpoint['loss_history'])
        print("best epoch: ", epoch, ", min loss:", np.min(loss_history), ", train time:", checkpoint['train_time'])
    else:
        print("best epoch: ", epoch, ", min loss:", checkpoint['loss'], ", train time:", checkpoint['train_time'])
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    # plt.figure()
    # plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("pnet loss")
    # plt.tight_layout()
    # plt.savefig("figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    # plt.close()
    return net


def check_train_results(e1_net, p_net, t=0.0):
    """
    """
    global constants
    # load p(monte)
    x1s = np.load(DATA_FOLDER+"x1s.npy")
    x2s = np.load(DATA_FOLDER+"x2s.npy")
    x3s = np.load(DATA_FOLDER+"x3s.npy")
    x4s = np.load(DATA_FOLDER+"x4s.npy")
    pdf_true = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t))
    # print("[check] monte joint pdf shape, type: ", pdf_monte.shape, pdf_monte.dtype)
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    if(t == 0.0):
        pdf_true = p_init(grid_points).reshape(x1_grid.shape) # obtain analytical p(true)
        # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)

    # Grid spacings (assumed uniform)
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    dV = dx1 * dx2 * dx3 * dx4
    E_x1 = np.sum(x1_grid * pdf_true) * dV
    E_x2 = np.sum(x2_grid * pdf_true) * dV
    E_x3 = np.sum(x3_grid * pdf_true) * dV
    E_x4 = np.sum(x4_grid * pdf_true) * dV

    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t).to(device)
    # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    # if(t == 0): # [test]
    #     pdf_nn = p_init_perturb(grid_points).reshape(x1_grid.shape)
    # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    e1_nn  = e1_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    e1 = pdf_true - pdf_nn
    
    e1_vec = e1.reshape(-1)
    e1_nn_vec = e1_nn.reshape(-1)
    a1 = np.max(np.abs(e1_vec - e1_nn_vec)) / np.max(np.abs(e1_nn_vec))
    print("a1 (t=", np.round(t,2),"): ", np.round(a1,3))
    # print(E_x1, E_x2, E_x3, E_x4)
    max_e1 = np.max(np.abs(e1_vec))
    max_e1_nn = np.max(np.abs(e1_nn_vec))
    print(max_e1, max_e1_nn)
    B1 = 2.0 * max_e1_nn 

    idx_plot = np.arange(1, 1+len(e1_vec))
    plt.figure
    plt.plot(idx_plot, e1_vec, "black")
    plt.plot(idx_plot, e1_nn_vec, "b--", linewidth=0.2)
    plt.fill_between(idx_plot, y1=0.0*idx_plot+B1, y2=0.0*idx_plot-B1, 
                         color="green", alpha=0.3, label=r"$B$")
    plt.show()

    return max_e1


def check_pdfnn_marginalize(p_net, t=0.0):
    """
    marginalize the joint pdf to a single coordinate, and compare it to the true analytical pdf at init time
    the true analytical pdf is obtained by 2 ways (but equivalent)
    1. using the univariate normal
    2. first compute the joint pdf (on the meshgrid) using multivariate normal, then marginalize
    """
    global constants
    print("[result] pdf_nn vs monte (or analytical at t=TI)")

    # load p(monte)
    x1s = np.load(DATA_FOLDER+"x1s.npy")
    x2s = np.load(DATA_FOLDER+"x2s.npy")
    x3s = np.load(DATA_FOLDER+"x3s.npy")
    x4s = np.load(DATA_FOLDER+"x4s.npy")
    pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t))
    print("[check] monte joint pdf shape, type: ", pdf_monte.shape, pdf_monte.dtype)

    # prepare grid points 
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    
    # if t == 0.0, obtain analytical p(true)
    if(t == 0.0):
        pdf_true = p_init(grid_points).reshape(x1_grid.shape)
        print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)

    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t).to(device)
    print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]

    # create figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for j in range(4):
        if(j == 0):
            ax = axs[0,0]
            x_axis = x1s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(1,2,3)) * dx2 * dx3 * dx4
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(1,2,3)) * dx2 * dx3 * dx4
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX1)
            ax.set_ylabel(r"$r$")
            if(t == 0.0):
                # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                marginalize_pdf_true = np.sum(pdf_true, axis=(1,2,3)) * dx2 * dx3 * dx4
                ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        elif(j == 1):
            ax = axs[0,1]
            x_axis = x2s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,2,3)) * dx1 * dx3 * dx4
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,2,3)) * dx1 * dx3 * dx4
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX2)
            ax.set_ylabel(r"$\phi$")
            if(t == 0.0):
                    # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                    marginalize_pdf_true = np.sum(pdf_true, axis=(0,2,3)) * dx1 * dx3 * dx4
                    ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        elif(j == 2):
            ax = axs[1,0]
            x_axis = x3s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,1,3)) * dx1 * dx2 * dx4
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,1,3)) * dx1 * dx2 * dx4
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX3)
            ax.set_ylabel(r"$v_r$")
            if(t == 0.0):
                    # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                    marginalize_pdf_true = np.sum(pdf_true, axis=(0,1,3)) * dx1 * dx2 * dx4
                    ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        else:
            ax = axs[1,1]
            x_axis = x4s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,1,2)) * dx1 * dx2 * dx3
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,1,2)) * dx1 * dx2 * dx3
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX4)
            ax.set_ylabel(r"$v_\phi$")
            if(t == 0.0):
                    # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                    marginalize_pdf_true = np.sum(pdf_true, axis=(0,1,2)) * dx1 * dx2 * dx3
                    ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        ax.legend()
    plt.show()


def test_nn_cartesian_pdf_xy(p_net, model_name):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    the contour plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    global constants
    # Create the contour plot
    plt.figure(figsize=(8, 6))
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime).to(device)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4

        # convert pdf
        pdf_nn_rphi_data = np.empty((0,4))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1])  
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r**2)  
        # Define the grid where you want to plot the contours
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),  # Adjust 100 to get finer resolution
                                     np.linspace(y.min(), y.max(), 100))
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        cp = plt.contourf(grid_x, grid_y, grid_z_nn, levels=15, cmap="viridis", alpha=1.0)  # Filled contour plot
        # plt.colorbar(cp)  # Add a colorbar to indicate the values
    # Labels and title
    plt.axis('equal')
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.title(model_name+'(x,y) over T='+str(constants.TF)+' sec.')
    plt.show()


def check_pdfnn_cartesian_wrt_monte(p_net, model_name):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    global constants
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")
        # load pdf_monte on the domain
        pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime).to(device)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx1 = x1s[1] - x1s[0] # dr'
        dx2 = x2s[1] - x2s[0] # dphi'
        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]
        dV = dx1*dx2*dx3*dx4
        # mean_x1 = np.sum(x1_grid * pdf_monte) * dV
        # mean_x2 = np.sum(x2_grid * pdf_monte) * dV
        # mean_x3 = np.sum(x3_grid * pdf_monte) * dV
        # mean_x4 = np.sum(x4_grid * pdf_monte) * dV
        # print("[check] pdf_monte mean, sum p: ", mean_x1, mean_x2, mean_x3, mean_x4, np.sum(pdf_monte)*dV)

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
        pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

        sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
        print("[check] sum p_nn (N-sphere): ", sum_p_nn)

        # convert pdf(r, phi)
        # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
        pdf_nn_rphi_data = np.empty((0,5))
        pdf_mo_rphi_data = np.empty((0,4))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
        z_mo = pdf_mo_rphi_data[:, 3]/(r) 

        _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
        __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
        ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
        print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

        # visualize p(x,y) using interpolation
        _grid_resolution = 70
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

        if(t_prime == 0.0):
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--", label=model_name)
            surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, color="none", rstride=3, cstride=3, edgecolor='blue', linewidth=0.5, label="Monte")
        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--")
            surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, color="none", rstride=3, cstride=3, edgecolor='blue', linewidth=0.5)
        # Optional: Add a color bar
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # Labels and title
    ax.legend()
    ax.set_xlabel('X, m')
    ax.set_ylabel('Y, m')
    ax.set_zlabel('PDF Value')
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))
    ax.view_init(elev=30, azim=-120)
    # Show the plot
    plt.show()


def main():
    global constants

    # p_net = PNet(scale=get_p_init_max()).to(device)
    # p_net = load_trained_model(p_net, path="output/p_net.pth"); p_net.eval()
    p_net = PNet(scale=get_p_init_max()).to(device)
    p_net = load_trained_model(p_net, path="output/p_net_reg.pth", method="new"); p_net.eval()
    # p_net = PNet_GMM().to(device)
    # p_net = load_trained_model(p_net, path="output/p_net_gmm.pth", method="new"); p_net.eval()

    e1_net = E1Net().to(device)
    e1_net.apply(init_weights_He)

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_model(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=300000); print("e1_net train complete")
    e1_net = load_trained_model(e1_net, path=E1NET_PATH, method="new"); e1_net.eval()

    ### Post-process ###
    for t_prime in constants.T_PRIME_SPAN:
        check_train_results(e1_net, p_net, t_prime)
    # for t_prime in constants.T_PRIME_SPAN:
    #    check_pdfnn_marginalize(p_net, t=t_prime)
    # test_nn_cartesian_pdf_xy(p_net, model_name="p_net")
    # check_pdfnn_cartesian_wrt_monte(p_net, model_name="p_net")
    # test_nn_cartesian_pdf_xy(p_net_gmm, model_name="p_net_gmm")
    # check_pdfnn_cartesian_wrt_monte(p_net_gmm, model_name="p_net_gmm")
    
    ### Finish printout ###
    if(TRAIN_FLAG == False):
        print("[complete] 4d perfect keplerian orbit Case2 (e1_net)")
    else:
        print("[complete] 4d perfect keplerian orbit Case2 with pre-trained models (e1_net)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Set the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()