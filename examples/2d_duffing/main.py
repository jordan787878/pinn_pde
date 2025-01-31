import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from constants import MyConstantsDuffing
import argparse
from torch.utils.data import Dataset, DataLoader

MONTE_FLAG = False
TRAIN_FLAG = False
constants = MyConstantsDuffing()
# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def diff_opt(x, t, net, verbose=False):
    global constants
    output = net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    residual = output_t

    x1 = x[:,0].view(-1,1)
    x2 = x[:,1].view(-1,1)
    output_x1 = output_x[:, 0].view(-1, 1)
    output_x2 = output_x[:, 1].view(-1, 1)

    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(output_x.size(1)):
        grad2 = torch.autograd.grad(output_x[:, i], x, grad_outputs=torch.ones_like(output_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    output_xx = torch.stack(hessian, dim=-1)
    output_x2x2 = output_xx[:, 1, 1].view(-1, 1)

    residual = residual + x2*output_x1 + constants.A2*output + \
               (constants.A1*x1 + constants.A2*x2 + constants.A3*x1**3)*output_x2 - \
               (0.025)*output_x2x2
    if(verbose):
        print(residual.dtype, residual.shape)
    return residual


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)

# P net
class Net(nn.Module):
    global constants
    def __init__(self, scale=1.0):
        super(Net, self).__init__()
        self.scale = scale
        num_hidden_layers=5
        neurons=60
        # List to hold layers
        layers = []
        # Input layer
        layers.append(nn.Linear(constants.DIM+1, neurons))
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
        # Output layer
        layers.append(nn.Linear(neurons, 1))
        # Register all layers
        self.layers = nn.ModuleList(layers)
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out   = F.gelu(self.layers[0](inputs))
        for layer in self.layers[1:-1]: 
            out = F.gelu(layer(out))
        out = F.softplus(self.layers[-1](out))
        return out
    
# E1 net
class E1Net(nn.Module):
    global constants
    def __init__(self, scale=1.0):
        super(E1Net, self).__init__()
        self.scale = scale
        num_hidden_layers=5
        neurons=100
        # List to hold layers
        layers = []
        # Input layer
        layers.append(nn.Linear(constants.DIM+1, neurons))
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
        # Output layer
        layers.append(nn.Linear(neurons, 1))
        # Register all layers
        self.layers = nn.ModuleList(layers)
    def forward(self, x, t):
        inputs = torch.cat([x, t/constants.TF], dim=1)
        out   = F.gelu(self.layers[0](inputs))
        for layer in self.layers[1:-1]: 
            out = F.gelu(layer(out))
        out = (self.layers[-1](out)) * self.scale
        return out
    

def get_ini_samples(num_samples=400):
    global constants
    _x_bc_normal = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=num_samples).astype(np.float32)
    _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
    _x_bc = np.column_stack([
        np.random.uniform(constants.X_RANGE[0,0], constants.X_RANGE[0,1], num_samples),
        np.random.uniform(constants.X_RANGE[1,0], constants.X_RANGE[1,1], num_samples),
    ])
    _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)
    x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * constants.TI)
    # print("x_bc shape type: ", x_bc.shape, x_bc.dtype)
    # print("t_bc shape type: ", t_bc.shape, t_bc.dtype)
    return x_bc, t_bc


def get_res_samples(num_samples=400):
    global constants
    _x_normal = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=num_samples).astype(np.float32)
    _x_normal = torch.tensor(_x_normal, dtype=torch.float32, requires_grad=True)
    _x = np.column_stack([
        np.random.uniform(constants.X_RANGE[0,0], constants.X_RANGE[0,1], num_samples),
        np.random.uniform(constants.X_RANGE[1,0], constants.X_RANGE[1,1], num_samples),
    ])
    _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
    x = torch.cat((_x_normal, _x), dim=0)
    t = np.random.uniform(constants.TI, constants.TF, len(x)),
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
    # print("x shape type: ", x.shape, x.dtype)
    # print("t shape type: ", t.shape, t.dtype)
    return x, t
    

class MyDataset(Dataset):
    def __init__(self, x_bc, t_bc, x, t):
        self.x_bc = x_bc
        self.t_bc = t_bc
        self.x = x
        self.t = t
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x_bc[idx], self.t_bc[idx], self.x[idx], self.t[idx]
  

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale
    N_samples = 500
    batch_size = 300

    # samples of initial condition
    x_bc, t_bc = get_ini_samples(num_samples=N_samples)
    # samples of residual
    x, t = get_res_samples(num_samples=N_samples)
    # Create dataset
    dataset = MyDataset(x_bc, t_bc, x, t)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # RAR
    S = 10000
    FLAG = 0
    w_reg = 1e-1
    
    start_time = time.time()

    for epoch in range(iterations):
        # # Dataset loss & RAR
        if(epoch > 0):
            # zero the gradeint
            optimizer.zero_grad()
            p_i = constants.p_init(x_bc.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc, t_bc)
            res_p = diff_opt(x, t, p_net)/normalize
            mse_u = mse_cost_function(phat_i/normalize, p_i/normalize)
            all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False)
            mse_res = mse_cost_function(res_p, all_zeros)

            res_x = torch.autograd.grad(res_p, x, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
            res_t = torch.autograd.grad(res_p, t, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
            res_input = torch.cat([res_x, res_t], axis=1)
            norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
            mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

            loss_dataset = mse_u + constants.TF*(mse_res + w_reg*mse_norm_res_input)
            loss_history.append(loss_dataset.item())

            if(loss_dataset.item() < 5e-4):
                train_time = time.time() - start_time
                # Print the iteration number (and optionally the loss)
                print(f'Epoch [{epoch+1}/{iterations}], Dataset Loss: {loss_dataset.item():.4f}, ic: {mse_u.item():.4f}, res: {mse_res.item():.4f}')
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': p_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dataset.data,
                        'train_time': train_time,
                        'scale': p_net.scale
                         }, constants._PATH_PNET)
                np.save(constants._PATH_PNET_LOSS, np.array(loss_history))
                return

            if(loss_dataset.item() < 0.95*min_loss):
                train_time = time.time() - start_time
                # Print the iteration number (and optionally the loss)
                print(f'Epoch [{epoch+1}/{iterations}], Dataset Loss: {loss_dataset.item():.4f}, ic: {mse_u.item():.4f}, res: {mse_res.item():.4f}')
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': p_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dataset.data,
                        'train_time': train_time,
                        'scale': p_net.scale
                         }, constants._PATH_PNET)
                min_loss = loss_dataset.item()
                FLAG += 1

            if(FLAG > 2):
                # random sample points
                x_bc_rar, t_bc_rar = get_ini_samples(num_samples=S)
                x_rar, t_rar = get_res_samples(num_samples=N_samples)
                # add ic points
                p_i = constants.p_init(x_bc_rar.detach().numpy())
                p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
                phat_i = p_net(x_bc_rar, t_bc_rar)
                max_error = torch.max(torch.abs(p_i - phat_i))/normalize
                if(max_error > 5e-3):
                    max_value, max_index = torch.topk(torch.abs(p_i.squeeze() - phat_i.squeeze()), 20)
                    x_max = x_bc_rar[max_index,:].clone().detach()
                    t_max = t_bc_rar[max_index].clone().detach()
                    x_bc = torch.cat((x_bc, x_max), dim=0)
                    t_bc = torch.cat((t_bc, t_max), dim=0)
                    print("... RAR IC, add: ", x_max[0,:].data, f' {t_max[0].item():.3f}, {max_error.item():.3f}')
                # add residual points
                res_p = diff_opt(x_rar, t_rar, p_net)/normalize
                max_error= torch.max(torch.abs(res_p))
                if(max_error > 5e-3):
                    max_value, max_index = torch.topk(torch.abs(res_p.squeeze()), 20)
                    x_max = x_rar[max_index,:].clone()
                    t_max = t_rar[max_index].clone()
                    x = torch.cat((x, x_max), dim=0)
                    t = torch.cat((t, t_max), dim=0)
                    print("... RAR Res, add: ", x_max[0,:].data, f' {t_max[0].item():.3f}, {max_error.item():.3f}')
                # Recreate dataset
                dataset = MyDataset(x_bc, t_bc, x, t)
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)     
                # reset flag
                FLAG = 0

        # mini-batch training
        for iter, (x_bc_k, t_bc_k, x_k, t_k) in enumerate(data_loader, start=1): 
            # zero the gradeint
            optimizer.zero_grad()

            # Loss based on boundary conditions
            p_i = constants.p_init(x_bc_k.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_k, t_bc_k)
            mse_u = mse_cost_function(phat_i/normalize, p_i/normalize)

            # Loss based on PDE
            res_p = diff_opt(x_k, t_k, p_net)/normalize
            all_zeros = torch.zeros((len(t_k),1), dtype=torch.float32, requires_grad=False)
            mse_res = mse_cost_function(res_p, all_zeros)

            # Gradient Loss
            res_x = torch.autograd.grad(res_p, x_k, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
            res_t = torch.autograd.grad(res_p, t_k, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
            res_input = torch.cat([res_x, res_t], axis=1)
            norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
            mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

            loss = mse_u + constants.TF*(mse_res + w_reg*mse_norm_res_input)
            loss.backward(retain_graph=True) 

        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    np.save(constants._PATH_PNET_LOSS, np.array(loss_history))


def train_e1_net(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    N_samples = 500
    batch_size = 300

    # samples of initial condition
    x_bc, t_bc = get_ini_samples(num_samples=N_samples)
    # samples of residual
    x, t = get_res_samples(num_samples=N_samples)
    # Create dataset
    dataset = MyDataset(x_bc, t_bc, x, t)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # RAR
    S = 10000
    FLAG = 0
    
    start_time = time.time()

    for epoch in range(iterations):
        # # Dataset loss & RAR
        if(epoch > 0):
            p_i = constants.p_init(x_bc.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc, t_bc)
            e_i = p_i - phat_i
            ehat_i = e1_net(x_bc, t_bc)
            res_p = diff_opt(x, t, p_net)
            res_e = diff_opt(x, t, e1_net)
            mse_u = mse_cost_function(ehat_i/normalize, e_i/normalize)
            mse_res = mse_cost_function(res_e/normalize, -res_p/normalize)
            loss_dataset = mse_u + constants.TF*(mse_res)
            loss_history.append(loss_dataset.item())
            
            if(loss_dataset.item() < 0.95*min_loss):
                train_time = time.time() - start_time
                # Print the iteration number (and optionally the loss)
                print(f'Epoch [{epoch+1}/{iterations}], Dataset Loss: {loss_dataset.item():.4f}, ic: {mse_u.item():.4f}, res: {mse_res.item():.4f}')
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dataset.data,
                        'train_time': train_time,
                        'scale': e1_net.scale
                         }, constants._PATH_E1NET)
                min_loss = loss_dataset.item()
                FLAG += 1

            if(FLAG > 2):
                # random sample points
                x_bc_rar, t_bc_rar = get_ini_samples(num_samples=S)
                x_rar, t_rar = get_res_samples(num_samples=N_samples)
                # add ic points
                p_i = constants.p_init(x_bc_rar.detach().numpy())
                p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
                phat_i = p_net(x_bc_rar, t_bc_rar)
                e_i = p_i - phat_i
                ehat_i = e1_net(x_bc_rar, t_bc_rar)
                res_p = diff_opt(x_rar, t_rar, p_net)/normalize
                res_e = diff_opt(x_rar, t_rar, e1_net)/normalize
                max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
                if(max_error > 5e-3):
                    max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 20)
                    x_max = x_bc_rar[max_index,:].clone().detach()
                    t_max = t_bc_rar[max_index].clone().detach()
                    x_bc = torch.cat((x_bc, x_max), dim=0)
                    t_bc = torch.cat((t_bc, t_max), dim=0)
                    print("... RAR IC, add: ", x_max[0,:].data, f' {t_max[0].item():.3f}, {max_error.item():.3f}')
                # add residual points
                max_error= torch.max(torch.abs(res_e + res_p))
                if(max_error > 5e-3):
                    max_value, max_index = torch.topk(torch.abs(res_e.squeeze() + res_p.squeeze()), 20)
                    x_max = x_rar[max_index,:].clone()
                    t_max = t_rar[max_index].clone()
                    x = torch.cat((x, x_max), dim=0)
                    t = torch.cat((t, t_max), dim=0)
                    print("... RAR Res, add: ", x_max[0,:].data, f' {t_max[0].item():.3f}, {max_error.item():.3f}')
                # Recreate dataset
                dataset = MyDataset(x_bc, t_bc, x, t)
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)     
                # reset flag
                FLAG = 0

        # mini-batch training
        for iter, (x_bc_k, t_bc_k, x_k, t_k) in enumerate(data_loader, start=1): 
            # zero the gradeint
            optimizer.zero_grad()

            # Loss based on boundary conditions
            p_i = constants.p_init(x_bc_k.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_k, t_bc_k)
            e_i = p_i - phat_i
            ehat_i = e1_net(x_bc_k, t_bc_k)
            mse_u = mse_cost_function(ehat_i/normalize, e_i/normalize)

            # Loss based on PDE
            res_p = diff_opt(x_k, t_k, p_net)
            res_e = diff_opt(x_k, t_k, e1_net)
            mse_res = mse_cost_function(res_e/normalize, -res_p/normalize)

            loss = mse_u + constants.TF*mse_res
            loss.backward(retain_graph=True) 

        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    np.save(constants._PATH_E1NET_LOSS, np.array(loss_history))


def pos_p_net_train(p_net):
    print("[load pnet model from: "+ constants._PATH_PNET)
    checkpoint = torch.load(constants._PATH_PNET)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    p_net.scale = checkpoint['scale']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    loss_history = np.load(constants._PATH_PNET_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("pnet loss")
    plt.tight_layout()
    plt.savefig(constants._FOLDER+"figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    return p_net


def pos_e1_net_train(e1_net):
    print("[load pnet model from: "+ constants._PATH_E1NET)
    checkpoint = torch.load(constants._PATH_E1NET)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    e1_net.scale = checkpoint['scale']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("e1net best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    # loss_history = np.load(constants._PATH_E1NET_LOSS)
    # min_loss = min(loss_history)
    # plt.figure()
    # plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("pnet loss")
    # plt.tight_layout()
    # plt.savefig(constants._FOLDER+"figs/e1net_loss_history.pdf", format='pdf', dpi=300)
    # plt.close()
    return e1_net


def check_pnn_result(p_net):
    global constants
    grid_points_struct = constants.load_gridpoints_from_monte()
    grid_points = grid_points_struct[-1]
    dx1 = grid_points_struct[0][1] - grid_points_struct[0][0]
    dx2 = grid_points_struct[2][1] - grid_points_struct[2][0]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]
    print("[check] x1_grid x2_grid x3_grid shape: ", x1_grid.shape, x1_grid.dtype)
    for t in constants.T_SPAN:
        print("[check] t=",t)
        # load true pdf(t)
        pdf_true = constants.load_p_sol_monte(t)
        print("[check] pdf_true dtype: ", pdf_true.dtype)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': '3d'})
        ax = axs
        ax.plot_wireframe(x1_grid, x2_grid, pdf_true.reshape(x1_grid.shape), 
                          color="black", linewidth=0.5, alpha=0.7, label=r"p")
        ax.plot_wireframe(x1_grid, x2_grid, pdf_nn.reshape(x1_grid.shape), 
                          color="red", linewidth=1.0, alpha=0.7, label=r"\hat{p}")
        plt.show()
        # if(t == 0.0):
        #     pdf_init = constants.p_init(grid_points)
        
        # fig, axs = plt.subplots(3, 1, figsize=(8, 6))
        # ax = axs[0]
        # ax.plot(grid_points_struct[0], np.sum(pdf_true.reshape(x1_grid.shape), axis=(1,2))*dx2*dx3, "black")
        # ax.plot(grid_points_struct[0], np.sum(pdf_nn.reshape(x1_grid.shape), axis=(1,2))*dx2*dx3, "r--")
        # ax.set_xlim(constants.X_RANGE[0,0], constants.X_RANGE[0,1])
        # if(t == 0.0):
        #     ax.plot(grid_points_struct[0], np.sum(pdf_init.reshape(x1_grid.shape), axis=(1,2))*dx2*dx3, "b:")

        # ax = axs[1]
        # ax.plot(grid_points_struct[2], np.sum(pdf_true.reshape(x1_grid.shape), axis=(0,2))*dx1*dx3, "black")
        # ax.plot(grid_points_struct[2], np.sum(pdf_nn.reshape(x1_grid.shape), axis=(0,2))*dx1*dx3, "r--")
        # ax.set_xlim(constants.X_RANGE[1,0], constants.X_RANGE[1,1])
        # if(t == 0.0):
        #     ax.plot(grid_points_struct[2], np.sum(pdf_init.reshape(x1_grid.shape), axis=(0,2))*dx1*dx3, "b:")

        # ax = axs[2]
        # ax.plot(grid_points_struct[4], np.sum(pdf_true.reshape(x1_grid.shape), axis=(0,1))*dx1*dx2, "black")
        # ax.plot(grid_points_struct[4], np.sum(pdf_nn.reshape(x1_grid.shape), axis=(0,1))*dx1*dx2, "r--")
        # ax.set_xlim(constants.X_RANGE[2,0], constants.X_RANGE[2,1])
        # if(t == 0.0):
        #     ax.plot(grid_points_struct[4], np.sum(pdf_init.reshape(x1_grid.shape), axis=(0,1))*dx1*dx2, "b:")

        # plt.show()

        # # NOTE
        # if(t > 0.0):
        #     return
        # # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
        # # compute p(true)
        # # pdf_true = constants.p_sol(grid_points, t)
        # pdf_true = constants.p_init(grid_points)
        # # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
        # # obtain pdf(nn)
        # grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        # t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        # pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        # # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
        # # print-out
        # error_init = pdf_true - pdf_nn
        # print(" =result= p_nn(t={:.1f}) normalized error: {:.4f}".format(t, np.max(np.abs(error_init))/p_net.scale))
        
        # # visualization (marginalized to 2 cooridnates)
        # fig, axs = plt.subplots(1, 3, figsize=(10, 6), subplot_kw={'projection': '3d'})
        # ax = axs[0]
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(pdf_true.reshape(grid_points_struct[1].shape), axis=(2))*dx, color="black", linewidth=0.5, alpha=0.5)
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(pdf_nn.reshape(grid_points_struct[1].shape), axis=(2))*dx, color="red", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_xlabel('x'); ax.set_ylabel('y')
        # ax.view_init(45, -135)

        # ax = axs[1]
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(pdf_true.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="black", linewidth=0.5, alpha=0.5)
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(pdf_nn.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="red", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_xlabel('y'); ax.set_ylabel('z')
        # ax.view_init(45, -135)

        # ax = axs[2]
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(pdf_true.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="black", linewidth=0.5, alpha=0.5)
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(pdf_nn.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="red", linewidth=0.5, alpha=0.5, linestyle="--")
        # ax.set_xlabel('x'); ax.set_ylabel('z')
        # ax.view_init(45, -135)

        # plt.savefig(constants._FOLDER+'figs/p_vs_pnn.pdf', format='pdf', dpi=300)
        # plt.close()


def check_e1nn_result(e1_net, p_net):
    global constants
    grid_points_struct = constants.load_gridpoints_from_monte()
    grid_points = grid_points_struct[-1]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]
    eS_ratio_data = []
    a1_data = []
    gap_data = []
    print("[check] x1_grid x2_grid x3_grid shape: ", x1_grid.shape, x1_grid.dtype)
    for t in constants.T_SPAN:
        print("[check] t=",t)
        # load true pdf(t)
        pdf_true = constants.load_p_sol_monte(t)
        print("[check] pdf_true dtype: ", pdf_true.dtype)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()

        e1 = pdf_true - pdf_nn.reshape(x1_grid.shape)
        e1_nn = e1_net(grid_points_tensor, t_tensor).detach().numpy()

        a1 = np.max(np.abs(e1.reshape(-1,1) - e1_nn))/ np.max(np.abs(e1_nn))
        a1_data.append(a1)
        eS = 2.0*np.max(np.abs(e1_nn))
        eS_ratio = eS/ np.max(np.abs(pdf_true.reshape(-1)))
        eS_ratio_data.append(eS_ratio)
        gap = (eS - np.max(np.abs(e1)))/ np.max(np.abs(pdf_true))
        gap_data.append(gap)
        print("=result= a1: {:.3f}, eS_ratio: {:.3f}, max(e1): {:.3f}, eS: {:.3f}".format(a1, eS_ratio, np.max(np.abs(e1)), eS))

        fig, axs = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': '3d'})
        ax = axs
        ax.plot_wireframe(x1_grid, x2_grid, e1.reshape(x1_grid.shape), 
                          color="black", linewidth=0.5, alpha=0.7, label=r"e_1")
        ax.plot_wireframe(x1_grid, x2_grid, e1_nn.reshape(x1_grid.shape), 
                          color="red", linewidth=1.0, alpha=0.7, label=r"\hat{e}_1")
        plt.show()
    print("[check] a1_data shape: ", np.array(a1_data).shape)
    print( " =result= max a1      : {:.3f}, var a1: {:.4f}".format(np.max(np.array(a1_data)),
                                                                   np.var(np.array(a1_data))))    
    print( " =result= min gap     : {:.3f}, max gap     : {:.3f}".format(np.min(np.array(gap_data)), np.max(np.array(gap_data))))
    print( " =result= max eS_ratio: {:.3f}, avg eS_ratio: {:.3f}".format(np.max(np.array(eS_ratio_data)),
                                                                         np.mean(np.array(eS_ratio_data))))


def get_e1init_max(p_net):
    global constants
    grid_points_struct = constants.prepare_gridpoints()
    grid_points = grid_points_struct[-1]
    # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    # compute p(true)
    pdf_true = constants.p_init(grid_points)
    # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * constants.TI)
    # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
    # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    # print-out
    error_init = pdf_true - pdf_nn
    error_init_max = np.max(np.abs(error_init))
    print("[check] e1(t0) max: {:.5f}".format(error_init_max))
    return error_init_max


def main():
    global constants
    if(MONTE_FLAG):
        constants.construct_p_sol()

    p_net = Net(scale=constants.get_pinit_max())
    e1_net = E1Net()

    p_net.apply(init_weights)
    e1_net.apply(init_weights)

    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=1000); print("p_net train complete")
    p_net = pos_p_net_train(p_net); p_net.eval()
    check_pnn_result(p_net)

    e1_net.scale = get_e1init_max(p_net)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if(TRAIN_FLAG):
        train_e1_net(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=50000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net); e1_net.eval()
    check_e1nn_result(e1_net, p_net)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Set the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()



