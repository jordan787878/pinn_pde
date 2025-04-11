import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from constants import MyConstants7D_Time
import argparse
from torch.utils.data import Dataset, DataLoader


FOLDER_INTERMED = "../meta/data_7dtvou/"
TRAIN_FLAG = False
constants = MyConstants7D_Time() #(testing non singular 5D A, will turn to 7D later)
# Set a fixed seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)


def dyn_f1(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,0,i]*x[:,i]
    return result

def dyn_f2(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,1,i]*x[:,i]
    return result

def dyn_f3(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,2,i]*x[:,i]
    return result

def dyn_f4(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,3,i]*x[:,i]
    return result

def dyn_f5(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,4,i]*x[:,i]
    return result

def dyn_f6(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,5,i]*x[:,i]
    return result

def dyn_f7(x, t):
    global constants
    result = torch.zeros(x.shape[0])
    cos_t = torch.cos(t.view(-1, 1))
    A_t = constants.A_TENSOR + constants.DA_TENSOR * cos_t.view(-1, 1, 1)
    for i in range(constants.DIM):
        result = result + A_t[:,6,i]*x[:,i]
    return result


def diff_opt(x, t, net, verbose=False):
    output = net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    residual = output_t

    f1 = dyn_f1(x, t).view(-1,1)
    f2 = dyn_f2(x, t).view(-1,1)
    f3 = dyn_f3(x, t).view(-1,1)
    f4 = dyn_f4(x, t).view(-1,1)
    f5 = dyn_f5(x, t).view(-1,1)
    f6 = dyn_f6(x, t).view(-1,1)
    f7 = dyn_f7(x, t).view(-1,1)
    fs = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)

    for i in range(constants.DIM):
        fi = fs[:,i].view(-1,1)
        fi_x = torch.autograd.grad(fi, x, grad_outputs=torch.ones_like(fi), create_graph=True)[0]
        fi_xi = fi_x[:,i].view(-1,1)
        residual = residual + output_x[:,i].view(-1,1)*fi + fi_xi*output
    
    # (if noise is present) Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(output_x.size(1)):
        grad2 = torch.autograd.grad(output_x[:, i], x, grad_outputs=torch.ones_like(output_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    output_xx = torch.stack(hessian, dim=-1)
    for i in range(constants.DIM):
        residual = residual - 0.5*((constants.L_TENSOR[i,i]**2)*output_xx[:, i, i].view(-1,1))

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
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
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
        np.random.uniform(constants.X_RANGE[0], constants.X_RANGE[1], num_samples),
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
    N_samples = 150 # 1000 (base)
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

    # Save intermediate enet
    save_count = 1
    save_loss = 10.0
    
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
                         }, constants._PATH_E1NET+".pth")
                min_loss = loss_dataset.item()
                FLAG = True

            if(loss_dataset.item() < 0.75*save_loss):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dataset.data,
                        'train_time': train_time,
                        'scale': e1_net.scale
                         }, constants._PATH_E1NET+"-"+str(save_count)+".pth")
                save_count = save_count + 1
                save_loss = loss_dataset.item()


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
    # e1_net = pos_e1_net_train(e1_net); e1_net.eval()
    # a1_data = compute_stat(p_net, e1_net)
    # np.savez(FOLDER_INTERMED+'a1_data_'+str(save_count)+'.npz', loss=loss.item(), a1_data=a1_data)


def pos_p_net_train(p_net):
    print("[load pnet model from: "+ constants._PATH_PNET)
    checkpoint = torch.load(constants._PATH_PNET)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    p_net.scale = checkpoint['scale']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'], ', scale: ', p_net.scale)
    loss_history = np.load(constants._PATH_PNET_LOSS)
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
    print("[load e1net model from: "+ constants._PATH_E1NET+".pth")
    checkpoint = torch.load(constants._PATH_E1NET+".pth")
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    e1_net.scale = checkpoint['scale']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("e1net best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'], ', scale: ', e1_net.scale)
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
    grid_points = constants.generate_random_samples(num_samples=1000000)
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
        print( " =result= p_nn(t={:.1f}) normalized error: {:.4f}".format(t, np.max(np.abs(error_init))/p_net.scale) )
        # plt.figure()
        # plt.plot(np.arange(1, len(grid_points)+1, 1), pdf_true, "black", linewidth=0.5, label="p")
        # plt.plot(np.arange(1, len(grid_points)+1, 1), pdf_nn, "r--", alpha=0.5, linewidth=0.5, label=r"\hat{p}")
        # plt.show()


def compute_stat(p_net, e1_net):
    global constants
    a1_data = []
    for t in constants.T_SPAN:
        grid_points = constants.generate_random_samples(num_samples=10000000)
        # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
        # compute p(true)
        pdf_true = constants.p_sol(grid_points, t)
        # obtain pdf(nn)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        # print("[check] pdf_true, pdf_nn shape dtype:", pdf_true.shape, pdf_true.dtype, 
        #                                                pdf_nn.shape, pdf_nn.dtype)
        e1 = pdf_true - pdf_nn
        e1_nn = e1_net(grid_points_tensor, t_tensor).detach().numpy()
        e2 = e1 - e1_nn
        a1 = np.max(np.abs(e2)) / np.max(np.abs(e1_nn))
        a1_data.append(a1)
    return np.array(a1_data)


def check_e1nn_result(e1_net, p_net):
    global constants
    gap_data = []
    eSratio_data = []
    e1ratio_data = []
    a1_data = []
    for t in constants.T_SPAN:
        grid_points = constants.generate_random_samples(num_samples=10000000)
        # print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
        # compute p(true)
        start_time = time.time()
        pdf_true = constants.p_sol(grid_points, t)
        print("N.I. time (sec): ", time.time() - start_time)
        # obtain pdf(nn)
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t
        # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        # print("[check] pdf_true, pdf_nn shape dtype:", pdf_true.shape, pdf_true.dtype, 
        #                                                pdf_nn.shape, pdf_nn.dtype)
        e1 = pdf_true - pdf_nn
        e1_nn = e1_net(grid_points_tensor, t_tensor).detach().numpy()
        e2 = e1 - e1_nn
        eS = 2.0*np.max(np.abs(e1_nn))
        a1 = np.max(np.abs(e2)) / np.max(np.abs(e1_nn))
        a1_data.append(a1)
        gap = (eS - np.max(np.abs(e1)))/ np.max(np.abs(pdf_true))
        gap_data.append(gap)
        eSratio = eS / np.max(np.abs(pdf_true))
        eSratio_data.append(eSratio)
        e1ratio_data.append( np.max(np.abs(e1))/np.max(np.abs(pdf_true)) )
        # print("[check] p_true(t) max: {:.6f}".format(ptrue_max))
        
    a1_list = np.array(a1_data)
    gap_list = np.array(gap_data)
    norm_B2_list = eSratio_data
    print("[info] max a1: ", np.round(np.max(a1_list),2), ", avg a1:", np.round(np.mean(a1_list),2), ", std a1:", np.round(np.std(a1_list),3))
    print("[info] min gap: ", np.round(np.min(gap_list),3), ", avg gap:", np.round(np.mean(gap_list),3))
    print("[info] avg B2_norm: ", np.round(np.mean(norm_B2_list),2), ", std B2_norm:", np.round(np.std(norm_B2_list),3))
    
    # visualization
    # fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    # axs[0].plot(constants.T_SPAN, eSratio_data, "green", label=r"$e_S$ (normalized)")
    # axs[0].plot(constants.T_SPAN, e1ratio_data, "black", label=r"$\max|e_1|$ (normalized)")
    # axs[0].set_ylabel("error (normalized)")
    # axs[0].legend(loc="upper left", fontsize=14)
    # axs[1].plot(constants.T_SPAN, a1_data, "black")
    # axs[1].set_xlabel("t")
    # axs[1].set_ylabel(r"$\alpha_1$")
    # plt.tight_layout()
    # fig.savefig(constants._FOLDER+'figs/result.pdf', format='pdf', dpi=300, bbox_inches='tight')
    # plt.close()


def plot_train_loss():
    global constants
    path_1 = constants._PATH_PNET_LOSS
    path_2 = constants._PATH_E1NET_LOSS
    loss_history_1 = np.load(path_1)
    loss_history_2 = np.load(path_2)
    min_loss_1 = min(loss_history_1)
    min_loss_2 = min(loss_history_2)

    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 22,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 22,              # Title font size
    "axes.labelsize": 22,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 20,             # Legend text size
    "legend.title_fontsize": 20        # Legend title size (if you use legend titles)
    })

    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    axs[0].plot(np.arange(len(loss_history_1)), loss_history_1, color="black", linewidth=1.0)
    axs[0].set_ylim([min_loss_1, 10*min_loss_1])
    axs[1].plot(np.arange(len(loss_history_2)), loss_history_2, color="black", linewidth=1.0)
    axs[1].set_ylim([min_loss_2, 10*min_loss_2])
    axs[0].set_xlabel("iterations")
    axs[1].set_xlabel("iterations")
    axs[0].set_ylabel("train loss: "+r"$\hat{p}$")
    axs[1].set_ylabel("train loss: "+r"$\hat{e}_1$")
    axs[0].grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    axs[1].grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    plt.tight_layout()
    fig.savefig(constants._FOLDER+'/figs/7dtvou_trainloss.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def get_e1init_max(p_net):
    global constants
    grid_points = constants.generate_random_samples()
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


def process_intermediate():
    global constants
    N_models = 12
    p_net = Net()
    e1_net = E1Net()
    p_net = pos_p_net_train(p_net); p_net.eval()
    for i in range(1,N_models+1):
        checkpoint = torch.load(constants._PATH_E1NET+"-"+str(i)+".pth")
        e1_net.load_state_dict(checkpoint['model_state_dict'])
        e1_net.scale = checkpoint['scale']
        loss = checkpoint['loss']
        a1_data = compute_stat(p_net, e1_net)
        np.savez(FOLDER_INTERMED+'a1_data_'+str(i)+'.npz', loss=loss.item(), a1_data=a1_data)
    # last model
    checkpoint = torch.load(constants._PATH_E1NET+".pth")
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    e1_net.scale = checkpoint['scale']
    loss = checkpoint['loss']
    a1_data = compute_stat(p_net, e1_net)
    np.savez(FOLDER_INTERMED+'a1_data_'+str(N_models+1)+'.npz', loss=loss.item(), a1_data=a1_data)



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
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=1000); print("p_net train complete") # 20000 (base)
    p_net = pos_p_net_train(p_net); p_net.eval()
    # check_pnn_result(p_net)

    e1_net.scale = get_e1init_max(p_net)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if(TRAIN_FLAG):
        train_e1_net(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=5000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net); e1_net.eval()
    check_e1nn_result(e1_net, p_net)

    plot_train_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Set the TRAIN_FLAG
    TRAIN_FLAG = args.train
    if(TRAIN_FLAG == 2):
        process_intermediate()
    else:
        main()