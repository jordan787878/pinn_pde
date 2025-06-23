"""
The baseline of keplerian orbit using rotating spherical coordiante and dynamics.
It is similar to the Case 2 of the paper: Uncertainty propagation in orbital mechanics via tensor decompostion, except
here we do not have process noise, and J2 perturbation.
This case is a circular orbit on plannar motion. Hence, we reduce the dynamics to 4D.

Issues:
    1) NOTE: the true initial pdf is treated as R^4! Hence, it is the simple sum that adds to 1, not the r*dr*d(angle)...
       What is the pdf(r, theta) such that (r,theta) are normally distributed, AND int pdf(r,theta) "r"*dr*dtheta = 1?
    2) NOTE: the rotating-normalized angle: phi' (is it still angle?)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal, norm
from scipy.interpolate import griddata
import argparse
from constants import Case2_4D_Constants
from monte import p_init, get_p_init_max


DATA_FOLDER = "data/"
device = "cpu"
TRAIN_FLAG = False
constants = Case2_4D_Constants()


def dyn_f1(x):
    return x[:,2]

def dyn_f2(x):
    return x[:,3]

def dyn_f3(x):
    global constants
    return constants.T**2 *x[:,0] *(constants.W +constants.PHI *x[:,3]/constants.T)**2 -constants.T**2 *constants.MU_EARTH/(constants.R**3 * x[:,0]**2)

def dyn_f4(x):
    return -2*constants.T*x[:,2]*(constants.W + constants.PHI * x[:,3]/constants.T)/(x[:,0]*constants.PHI)


def diff_opt_p(x, t, p_net, verbose=False):
    output = p_net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x1 = output_x[:,0].view(-1,1)
    output_x2 = output_x[:,1].view(-1,1)
    output_x3 = output_x[:,2].view(-1,1)
    output_x4 = output_x[:,3].view(-1,1)
    f1 = dyn_f1(x).view(-1,1)
    f2 = dyn_f2(x).view(-1,1)
    f3 = dyn_f3(x).view(-1,1)
    f4 = dyn_f4(x).view(-1,1)
    f4_x = torch.autograd.grad(f4, x, grad_outputs=torch.ones_like(f4), create_graph=True)[0]
    f4_x4 = f4_x[:,3].view(-1,1)
    residual = output_t + output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + f4_x4*output
    # print(residual.dtype)
    return residual


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class PNet(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(PNet, self).__init__()
        self.hidden_layer1 = (nn.Linear(5,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        _x1 = (x[:,0].view(-1, 1) - 0.5*(constants.X1_RANGE[1]+constants.X1_RANGE[0]))/(0.5*(constants.X1_RANGE[1]-constants.X1_RANGE[0]))
        _x2 = (x[:,1].view(-1, 1) - 0.5*(constants.X2_RANGE[1]+constants.X2_RANGE[0]))/(0.5*(constants.X2_RANGE[1]-constants.X2_RANGE[0]))
        _x3 = (x[:,2].view(-1, 1) - 0.5*(constants.X3_RANGE[1]+constants.X3_RANGE[0]))/(0.5*(constants.X3_RANGE[1]-constants.X3_RANGE[0]))
        _x4 = (x[:,3].view(-1, 1) - 0.5*(constants.X4_RANGE[1]+constants.X4_RANGE[0]))/(0.5*(constants.X4_RANGE[1]-constants.X4_RANGE[0]))
        _t  = t/(constants.TF/constants.T)
        inputs = torch.cat([_x1, _x2, _x3, _x4, _t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        return output
                

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constans
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    # samples of initial condition
    _x_bc_normal = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=2000).astype(np.float32)
    _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
    _x_bc = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], 2000),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], 2000),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], 2000),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], 2000),
    ])
    _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)
    x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * constants.TI).to(device)
    print("x_bc shape type: ", x_bc.shape, x_bc.dtype)
    print("t_bc shape type: ", t_bc.shape, t_bc.dtype)

    # samples of residual
    _x_normal = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=2000).astype(np.float32)
    _x_normal = torch.tensor(_x_normal, dtype=torch.float32, requires_grad=True)
    _x = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], 2000),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], 2000),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], 2000),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], 2000),
    ])
    _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
    x = torch.cat((_x_normal, _x), dim=0)
    t = np.random.uniform(constants.TI, constants.TF/constants.T, len(x)),
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
    print("x shape type: ", x.shape, x.dtype)
    print("t shape type: ", t.shape, t.dtype)

    # RAR
    S = 30000
    FLAG = False
    
    PATH = "output/p_net.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(phat_i/normalize, p_i/normalize)

        # Loss based on PDE
        res_p = diff_opt_p(x, t, p_net)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_p/normalize, all_zeros)

        # Frequnecy Loss
        # res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_input = torch.cat([res_x, res_t], axis=1)
        # norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
        # mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + (constants.TF/constants.T)*mse_res
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                #    ",res_freq:", mse_norm_res_input.data
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # RAR
        # if (epoch%100 == 0 and FLAG):
        #     _x1_RAR = (torch.rand(S, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
        #     _x2_RAR = (torch.rand(S, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
        #     x_RAR = torch.cat((_x1_RAR, _x2_RAR), dim=1)
        #     t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
        #     t0_RAR = 0.0*t_RAR + ti
        #     p_bc_RAR = p_init_torch(x_RAR)
        #     phat_bc_RAR = p_net(x_RAR, t0_RAR)
        #     max_ic_error = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR))/normalize
        #     # print("RAR max IC: ", max_ic_error.data)
        #     if(max_ic_error > 5e-3):
        #         max_abs_ic, max_index = torch.topk(torch.abs(phat_bc_RAR.squeeze() - p_bc_RAR.squeeze()), 10)
        #         x_max = x_RAR[max_index,:].clone().detach()
        #         t_max = t0_RAR[max_index].clone().detach()
        #         x_bc = torch.cat((x_bc, x_max), dim=0)
        #         t_bc = torch.cat((t_bc, t_max), dim=0)
        #         print("... RAR IC, add: ", x_max[0,:].data, t_max[0].item(), max_abs_ic[0].item())
        #     res_RAR = res_func(x_RAR, t_RAR, p_net)/normalize
        #     max_res_RAR = torch.max(torch.abs(res_RAR))
        #     if(max_res_RAR > 0.0):
        #         max_abs_res, max_index = torch.topk(torch.abs(res_RAR.squeeze()), 10)
        #         x_max = x_RAR[max_index,:].clone()
        #         t_max = t_RAR[max_index].clone()
        #         x = torch.cat((x, x_max), dim=0)
        #         t = torch.cat((t, t_max), dim=0)
        #         print("... RAR Res, add: ", x_max[0,:].data, t_max[0].item(), max_abs_res[0].item())
        #     FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

    np.save("output/p_net_train_loss.npy", np.array(loss_history))


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    print("[load pnet model from: "+ PATH)
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("pnet loss")
    plt.tight_layout()
    plt.savefig("figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    return p_net


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

    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]

    # prepare grid points 
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    
    # if t == 0.0, obtain analytical p(true)
    if(t == 0.0):
        pdf_true = p_init(grid_points).reshape(x1_grid.shape) # NOTE: the true initial pdf is treated as R^4! Hence, it is the simple sum that adds to 1 ...
        print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
        ### checking sum of p_init ### 
        # _pdf_true_rphi = np.sum(pdf_true, axis=(2,3)) * dx3 * dx4
        # # convert pdf
        # pdf_true_rphi_sum = 0.0
        # # pdf_true_rphi_data = np.empty((0,4))
        # _x1_grid, _x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        # for i in range(len(_x1_grid)):
        #     for j in range(len(_x2_grid)):
        #         _r = _x1_grid[i,j]*constants.R
        #         # phi = _x2_grid[i,j]*constants.PHI + constants.W*constants.T*0.0
        #         # t = constants.T*0.0
        #         # pdf_true_rphi_data = np.vstack((pdf_true_rphi_data, np.array([r, phi, t, pdf_true[i,j]/(constants.R*constants.PHI)])))
        #         pdf_true_rphi_sum = pdf_true_rphi_sum + (_pdf_true_rphi[i,j]/(constants.R*constants.PHI))*(dx1*constants.R)*(dx2*constants.PHI)*_r
        # print("[check] pdf_true(r, phi) sum: ", pdf_true_rphi_sum)


    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t).to(device)
    print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)

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
    fig.suptitle("t= {:.1f} sec".format(t*constants.T))
    plt.show()


def test_nn_cartesian_pdf_xy(p_net):
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
    plt.title('pdf_nn(x,y) over T='+str(constants.TF)+' sec.')
    plt.show()


def check_pdfnn_cartesian_wrt_monte(p_net):
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
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--", label="NN")
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
    # Show the plot
    plt.show()


def main():
    global constants
    p_net = PNet().to(device)
    p_net.apply(init_weights_He)
    p_net.scale = get_p_init_max()

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pth", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()

    ### Post-process ###
    for t_prime in constants.T_PRIME_SPAN:
        check_pdfnn_marginalize(p_net, t=t_prime)

    # test_nn_cartesian_pdf_xy(p_net)

    check_pdfnn_cartesian_wrt_monte(p_net)
    
    ### Finish printout ###
    if(TRAIN_FLAG == False):
        print("[complete] 4d perfect keplerian orbit baseline")
    else:
        print("[complete] 4d perfect keplerian orbit baseline with pre-trained models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()