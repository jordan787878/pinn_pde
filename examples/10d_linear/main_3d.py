import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from constants import MyConstants3D
import argparse
from torch.utils.data import Dataset, DataLoader


TRAIN_FLAG = False
constants = MyConstants3D()
# Set a fixed seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)


def dyn_f1(x):
    global constants
    return constants.A_TENSOR[0,0]*x[:,0] + constants.A_TENSOR[0,1]*x[:,1] + constants.A_TENSOR[0,2]*x[:,2]

def dyn_f2(x):
    global constants
    return constants.A_TENSOR[1,0]*x[:,0] + constants.A_TENSOR[1,1]*x[:,1] + constants.A_TENSOR[1,2]*x[:,2]

def dyn_f3(x):
    global constants
    return constants.A_TENSOR[2,0]*x[:,0] + constants.A_TENSOR[2,1]*x[:,1] + constants.A_TENSOR[2,2]*x[:,2]


def diff_opt(x, t, net, verbose=False):
    output = net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x1 = output_x[:,0].view(-1,1)
    output_x2 = output_x[:,1].view(-1,1)
    output_x3 = output_x[:,2].view(-1,1)
    f1 = dyn_f1(x).view(-1,1)
    f2 = dyn_f2(x).view(-1,1)
    f3 = dyn_f3(x).view(-1,1)
    f1_x = torch.autograd.grad(f1, x, grad_outputs=torch.ones_like(f1), create_graph=True)[0]
    f1_x1 = f1_x[:,0].view(-1,1)
    f2_x = torch.autograd.grad(f2, x, grad_outputs=torch.ones_like(f2), create_graph=True)[0]
    f2_x2 = f2_x[:,1].view(-1,1)
    f3_x = torch.autograd.grad(f3, x, grad_outputs=torch.ones_like(f3), create_graph=True)[0]
    f3_x3 = f3_x[:,2].view(-1,1)
    residual = output_t + output_x1*f1 + f1_x1*output \
                        + output_x2*f2 + f2_x2*output \
                        + output_x3*f3 + f3_x3*output
    
    # (if noise is present)
    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(output_x.size(1)):
        grad2 = torch.autograd.grad(output_x[:, i], x, grad_outputs=torch.ones_like(output_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    output_xx = torch.stack(hessian, dim=-1)
    output_x1x1 = output_xx[:, 0, 0].view(-1, 1)
    output_x2x2 = output_xx[:, 1, 1].view(-1, 1)
    output_x3x3 = output_xx[:, 2, 2].view(-1, 1)
    residual = residual - 0.5*(constants.L_TENSOR[0,0]*constants.L_TENSOR[0,0]*output_x1x1 + \
                               constants.L_TENSOR[1,1]*constants.L_TENSOR[1,1]*output_x2x2 + \
                               constants.L_TENSOR[2,2]*constants.L_TENSOR[2,2]*output_x3x3)

    if(verbose):
        print(residual.dtype, residual.shape)
    return residual


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)


# p_net
class Net(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(constants.DIM+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.gelu((self.hidden_layer1(inputs)))
        layer2_out = F.gelu((self.hidden_layer2(layer1_out)))
        layer3_out = F.gelu((self.hidden_layer3(layer2_out)))
        layer4_out = F.gelu((self.hidden_layer4(layer3_out)))
        layer5_out = F.gelu((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        return output
    

# e1 net
class E1Net(nn.Module):
    global constants
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(constants.DIM+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.GELU()
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = self.scale * output
        return output
   

def get_ini_samples(num_samples=400):
    global constants
    _x_bc_normal = np.random.multivariate_normal(constants.MEAN_I, constants.COV_I, size=num_samples).astype(np.float32)
    _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
    _x_bc = np.column_stack([
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
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
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
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
    N_samples = 1000 # 1000 (base)
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
    FLAG = False
    
    start_time = time.time()

    for epoch in range(iterations):
        # # Dataset loss & RAR
        if(epoch > 0):
            p_i = constants.p_init(x_bc.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc, t_bc)
            res_p = diff_opt(x, t, p_net)/normalize
            mse_u = mse_cost_function(phat_i/normalize, p_i/normalize)
            all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False)
            mse_res = mse_cost_function(res_p, all_zeros)
            loss_dataset = mse_u + constants.TF*(mse_res)
            loss_history.append(loss_dataset.item())

            if(loss_dataset.item() < 1e-4):
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
                FLAG = True

            if(FLAG):
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
                FLAG = False

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

            loss = mse_u + constants.TF*mse_res
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
    N_samples = 150 # 150(base)
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
    FLAG = False
    
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
                FLAG = True

            if(FLAG):
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
                FLAG = False

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
    # loss_history = np.load(constants._PATH_PNET_LOSS)
    # min_loss = min(loss_history)
    # plt.figure()
    # plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("pnet loss")
    # plt.tight_layout()
    # plt.savefig(constants._FOLDER+"figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    # plt.close()
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
    grid_points_struct = constants.prepare_gridpoints()
    grid_points = grid_points_struct[-1]
    dx = grid_points_struct[0][1] - grid_points_struct[0][0]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]
    x3_grid = grid_points_struct[5]
    print("[check] x1_grid x2_grid x3_grid shape: ", x1_grid.shape, x2_grid.shape, x3_grid.shape)
    for t in constants.T_SPAN:
        # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
        # compute p(true)
        pdf_true = constants.p_sol(grid_points, t)
        # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
        # obtain pdf(nn)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
        # print-out
        error_init = pdf_true - pdf_nn
        print(" =result= p_nn(t={:.1f}) normalized error: {:.4f}".format(t, np.max(np.abs(error_init))/p_net.scale))
        
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

        # plt.show()
        # plt.savefig(constants._FOLDER+'figs/p_vs_pnn.pdf', format='pdf', dpi=300)
        # plt.close()


def check_e1nn_result(e1_net, p_net):
    global constants
    grid_points_struct = constants.prepare_gridpoints(grid_num=100)
    grid_points = grid_points_struct[-1]
    dx = grid_points_struct[0][1] - grid_points_struct[0][0]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]
    x3_grid = grid_points_struct[5]
    
    gap_data = []
    eSratio_data = []
    a1_data = []
    for t in constants.T_SPAN:
        # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
        # compute p(true)
        pdf_true = constants.p_sol(grid_points, t)
        # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
        # obtain pdf(nn)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
        e1 = pdf_true - pdf_nn
        e1_nn = e1_net(grid_points_tensor, t_tensor).detach().numpy()
        e2 = e1 - e1_nn
        eS = 2.0*np.max(np.abs(e1_nn))
        a1 = np.max(np.abs(e2))/np.max(np.abs(e1_nn))
        a1_data.append(a1)
        gap = (eS - np.max(np.abs(e1)))/ np.max(np.abs(pdf_true))
        gap_data.append(gap)
        eSratio = eS / np.max(np.abs(pdf_true))
        eSratio_data.append(eSratio)

        # # visualization (marginalized to 2 cooridnates)
        # # E1net plot
        # fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw={'projection': '3d'})
        # ax = axs[0]
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
        #                   color="black", linewidth=0.5, alpha=0.7, label=r"e")
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
        #                   color="red", linewidth=0.2, alpha=0.7, linestyle="--", label=r"$\hat{e}_1$")
        # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Error')
        # ax.legend()
        # ax.view_init(20, -135)

        # ax = axs[1]
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="black", linewidth=0.5, alpha=0.7)
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
        # ax.set_xlabel('y'); ax.set_ylabel('z'); ax.set_zlabel('Error')
        # ax.view_init(20, -135)
        # ax = axs[2]
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="black", linewidth=0.5, alpha=0.7)
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
        # ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('Error')
        # ax.view_init(20, -135)

        # # Reduce white space between subplots and around the figure
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
        # # Add a text to the top-left corner of the entire figure
        # fig.text(
        #     0.05, 0.9, "t={:.1f}".format(t), 
        #     fontsize=16, color='black',
        # )
        # plt.show()

        # # Error bound plot
        # fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw={'projection': '3d'})
        # ax = axs[0]
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
        #                   color="black", linewidth=0.5, alpha=0.7, label=r"e")
        # ax.plot_wireframe(x1_grid[:, :, 0], x2_grid[:, :, 0], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(2))*dx, 
        #                   color="red", linewidth=0.2   , alpha=0.7, linestyle="--", label=r"$\hat{e}_1$")
        # ax.plot_surface(x1_grid[:, :, 0], x2_grid[:, :, 0], x1_grid[:, :, 0]*0.0 + 2.0*eS, color="green", alpha=0.2, label=r"$e_S$")
        # ax.plot_surface(x1_grid[:, :, 0], x2_grid[:, :, 0], x1_grid[:, :, 0]*0.0 - 2.0*eS, color="green", alpha=0.2)
        # ax.legend()
        # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Error')
        # ax.view_init(15, -135)

        # ax = axs[1]
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="black", linewidth=0.5, alpha=0.7)
        # ax.plot_wireframe(x2_grid[0, :, :], x3_grid[0, :, :], np.sum(e1_nn.reshape(grid_points_struct[1].shape), axis=(0))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
        # ax.plot_surface(x2_grid[0, :, :], x3_grid[0, :, :], x2_grid[0, :, :]*0.0 + 2.0*eS, color="green", alpha=0.2)
        # ax.plot_surface(x2_grid[0, :, :], x3_grid[0, :, :], x2_grid[0, :, :]*0.0 - 2.0*eS, color="green", alpha=0.2)
        # ax.set_xlabel('y'); ax.set_ylabel('z'); ax.set_zlabel('Error')
        # ax.view_init(15, -135)

        # ax = axs[2]
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="black", linewidth=0.5, alpha=0.7)
        # ax.plot_wireframe(x1_grid[:, 0, :], x3_grid[:, 0, :], np.sum(e1.reshape(grid_points_struct[1].shape), axis=(1))*dx, color="red", linewidth=0.2, alpha=0.7, linestyle="--")
        # ax.plot_surface(x1_grid[:, 0, :], x3_grid[:, 0, :], x1_grid[:, 0, :]*0.0 + 2.0*eS, color="green", alpha=0.2)
        # ax.plot_surface(x1_grid[:, 0, :], x3_grid[:, 0, :], x1_grid[:, 0, :]*0.0 - 2.0*eS, color="green", alpha=0.2)
        # ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('Error')
        # ax.view_init(15, -135)

        # # Reduce white space between subplots and around the figure
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
        # # Add a text to the top-left corner of the entire figure
        # fig.text(
        #     0.05, 0.9, "t={:.1f}".format(t), 
        #     fontsize=16, color='black',
        # )
        # plt.show()

    print( " =result= max a1      : {:.3f}, var a1: {:.4f}".format(np.max(np.array(a1_data)),
                                                                   np.var(np.array(a1_data))))    
    print( " =result= max gap     : {:.3f}, min gap     : {:.3f}".format(np.max(np.array(gap_data)), np.min(np.array(gap_data))))
    print( " =result= max eS_ratio: {:.3f}, avg eS_ratio: {:.3f}".format(np.max(np.array(eSratio_data)),
                                                                         np.mean(np.array(eSratio_data))))


def get_e1init_max(p_net):
    global constants
    grid_points_struct = constants.prepare_gridpoints()
    grid_points = grid_points_struct[-1]
    t = constants.TI
    # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    # compute p(true)
    pdf_true = constants.p_sol(grid_points, t)
    # print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)
    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t)
    # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
    # print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    # print-out
    error_init = pdf_true - pdf_nn
    error_init_max = np.max(np.abs(error_init))
    print("[check] e1(t0) max: {:.5f}".format(error_init_max))
    return error_init_max


def plot_train_loss():
    global constants
    path_1 = constants._PATH_PNET_LOSS
    path_2 = constants._PATH_E1NET_LOSS
    loss_history_1 = np.load(path_1)
    loss_history_2 = np.load(path_2)
    print(loss_history_1)
    print(loss_history_2)
    min_loss_1 = min(loss_history_1)
    min_loss_2 = min(loss_history_2)
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    axs[0].plot(np.arange(len(loss_history_1)), loss_history_1, "black", linewidth=1.0)
    axs[0].set_ylim([min_loss_1, 10*min_loss_1])
    axs[1].plot(np.arange(len(loss_history_2)), loss_history_2, "black", linewidth=1.0)
    axs[1].set_ylim([min_loss_2, 10*min_loss_2])
    axs[0].grid(linewidth=0.5)
    axs[1].grid(linewidth=0.5)
    axs[1].set_xlabel("epochs")
    axs[0].set_ylabel("train loss: "+r"$\hat{p}$")
    axs[1].set_ylabel("train loss: "+r"$\hat{e}_1$")
    plt.tight_layout()
    fig.savefig(constants._FOLDER+'figs/train_loss.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    global constants
    p_net = Net(scale=constants.get_pinit_max())
    e1_net = E1Net()
    p_net.apply(init_weights)
    e1_net.apply(init_weights)

    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=5000); print("p_net train complete")
    p_net = pos_p_net_train(p_net); p_net.eval()
    check_pnn_result(p_net)

    e1_net.scale = get_e1init_max(p_net)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if(TRAIN_FLAG):
        train_e1_net(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=10000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net); e1_net.eval()

    check_e1nn_result(e1_net, p_net)
    plot_train_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Set the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()