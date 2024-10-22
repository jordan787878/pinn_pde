import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from tqdm import tqdm
import warnings
import time
from matplotlib.ticker import ScalarFormatter
import argparse


FOLDER = "exp1/main/"
TRAIN_FLAG = False

device = "cpu"; print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

n_d = 1
mu = 0.002   # Drift
sigma = 0.01 # Volatility
S_0 = 100
x_low = 90
x_hig = 110

t0 = 1.0
T_end = 6.0
t1s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

a = mu
b = sigma
d = 0.0

batch_size = 1000
RAR_THRESHOLD = 5e-3


def gbm_transition_density(S_t, t):
    """
    Calculate the transition density p(S_t | S_0) for Geometric Brownian Motion (GBM).

    Parameters:
    - S_t: The value of the process at time t (current value).
    - S_0: The initial value of the process (value at time t=0).
    - mu: The drift coefficient.
    - sigma: The volatility coefficient.
    - t: Time elapsed.

    Returns:
    - The probability density p(S_t | S_0).
    """
    # Calculate the mean and variance of the log-normal distribution
    mean = np.log(S_0) + (mu - 0.5 * sigma**2) * t
    variance = sigma**2 * t
    # Compute the probability density function
    coeff = 1 / (S_t * sigma * np.sqrt(2 * np.pi * t))
    exponent = - (np.log(S_t) - mean)**2 / (2 * variance)
    density = coeff * np.exp(exponent)
    return density


def p_init(x):
    return gbm_transition_density(x, t0)


def p_sol(x,t):
    return gbm_transition_density(x,t)


def get_p0_max():
    x = np.linspace(x_low, x_hig, num=500)
    p0 = p_init(x)
    return np.max(np.abs(p0))


def res_func(x,t, net, verbose=False):
    out = net(x,t)
    out_x = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    out_t = torch.autograd.grad(out, t, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    out_xx = torch.autograd.grad(out_x, x, grad_outputs=torch.ones_like(out_x), create_graph=True)[0]
    Lp = a*out + (a*x+d)*out_x - 0.5*b*b*(2*out + x*x*out_xx)
    residual = out_t + Lp
    if(verbose):
        print(Lp.shape)
        # print(out_xx.shape)
    return residual


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        _x = (x-S_0)/S_0
        inputs = torch.cat([_x,t], axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        return output


def train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_abs_p_ti, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/p_net.pt"
    PATH_LOSS = FOLDER+"output/p_net_train_loss.npy"
    iterations_per_decay = 1000

    # Define the mean and covariance matrix
    _mean = torch.tensor([100.0])
    _covariance_matrix = torch.tensor([[1.0]])
    mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    x_bc = (torch.rand(int(0.5*batch_size), 1)*(x_hig-x_low)+x_low).to(device)
    x_bc_normal = mvn.sample((int(0.5*batch_size),)).to(device)
    x_bc_normal = torch.clamp(x_bc_normal, min=x_low, max=x_hig)
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)

    t = (torch.rand(int(1.0*batch_size), 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    x = (torch.rand(len(t), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    S = 30000
    FLAG = False
    normalize = max_abs_p_ti
    start_time = time.time()

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        u_bc = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # Loss based on PDE
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = res_func(x, t, p_net)/normalize
        mse_res = mse_cost_function(res_out, all_zeros)

        res_g = res_out; x_g = x; t_g = t
        res_x = torch.autograd.grad(res_g, x_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        res_t = torch.autograd.grad(res_g, t_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        mse_norm_res_input = torch.mean(res_x**2 + res_t**2)

        # loss
        loss = mse_u + 5.0*(mse_res + mse_norm_res_input)
        loss_history.append(loss.data)

        # Termination
        if(loss.data < 7e-3):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ", loss:", loss.data, ", ic:",mse_u.data, ", res:",mse_res.data, 
                  ", res freq:", mse_norm_res_input.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time
                    }, PATH)
            np.save(PATH_LOSS, np.array(loss_history))
            return

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ", loss:", loss.data, ", ic:",mse_u.data, ", res:",mse_res.data, 
                  ", res grad:", mse_norm_res_input.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time
                    }, PATH)
            np.save(PATH_LOSS, np.array(loss_history))
            min_loss = loss.data
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

            t0_RAR = 0.0*t_RAR + t0
            p_bc_RAR = p_init(x_RAR.detach().numpy())
            p_bc_RAR = Variable(torch.from_numpy(p_bc_RAR).float(), requires_grad=False).to(device)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            max_ic_error = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR))/normalize
            print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > RAR_THRESHOLD):
                # max_abs_ic, max_index = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR), dim=0)
                max_abs_ic, max_index = torch.topk(torch.abs(phat_bc_RAR.squeeze() - p_bc_RAR.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t0_RAR[max_index].clone()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... IC add [x,t]:", x_max.data, t_max.data, max_abs_ic.data)

            res_RAR = res_func(x_RAR, t_RAR, p_net)/normalize
            mean_res_error = torch.mean(torch.abs(res_RAR))
            print("... RAR mean res: ", mean_res_error.data)
            if(mean_res_error > RAR_THRESHOLD):
                # max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                max_abs_res, max_index = torch.topk(torch.abs(res_RAR.squeeze()), 5)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)
                
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best pnet epoch: ", epoch, ", loss: " , loss.data, ", train time: ", checkpoint['train_time'])
    # see training result
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/pnet_loss_history.png")
    plt.close()
    return p_net


def show_p_net_results(p_net):
    x = np.linspace(x_low, x_hig, num=100).reshape(-1,1)
    plt.figure(figsize=(8,6))
    markers = ['o', 's', 'D', '^', 'v', '<']
    for i in range(len(t1s)):
        t1 = t1s[i]
        p_true = p_sol(x, t1)
        plt.plot(x,p_true,marker=markers[i], markersize=3, label="t="+str(t1))
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.grid(linewidth=0.5)
    plt.savefig(FOLDER+"figs/p_sol.png")

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_ti = Variable(torch.from_numpy(x*0+t0).float(), requires_grad=True).to(device)
    p     = p_init(x)
    p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    e1_true = p - p_hat
    max_abs_e1_ti = max(abs(e1_true))[0]

    fig, axs = plt.subplots(6, 1, figsize=(6, 6))
    # Determine global min and max for y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    p_monte_list = []
    p_hat_list = []
    e1_list = []
    limit_margin = 0.1
    for t1 in t1s:
        p_monte = p_sol(x, t1).reshape(-1,1)
        p_monte_list.append(p_monte)
        current_min = p_monte.min()
        current_max = p_monte.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy()
        p_hat_list.append(p_hat)
        e1 = p_monte - p_hat
        e1_list.append(e1)
    for i, (ax1, p_monte, p_hat) in enumerate(zip(axs, p_monte_list, p_hat_list)):
        if i == 0:
            ax1.plot(x, p_monte, "blue", label=r"$p$")
            ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, p_monte, "blue")
            ax1.plot(x, p_hat, "red", linestyle="--")
        ax1.set_ylim(global_min-limit_margin, global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_result.png")
    plt.close()

    fig, axs = plt.subplots(6, 1, figsize=(6, 6))
    global_min = float('inf')
    global_max = float('-inf')
    res_list = []
    for t1 in t1s:
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        res = res_func(pt_x, pt_t1, p_net).data.cpu().numpy()
        current_min = res.min()
        current_max = res.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        res_list.append(res)
    for i, (ax1, res) in enumerate(zip(axs, res_list)):
        if i == 0:
            ax1.plot(x, res, "red", linestyle="--", label=r"$r_1$")
            ax1.legend()
        else:
            ax1.plot(x, res, "red", linestyle="--")
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_residual.png")
    plt.close()

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    for i, (ax1, e1) in enumerate(zip(axs, e1_list)):
        if i == 0:
            ax1.plot(x, e1, "blue", linestyle="-", label=r"$e_1$")
            ax1.legend()
        else:
            ax1.plot(x, e1, "blue", linestyle="-")
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_error.png")
    plt.close()

    return max_abs_e1_ti
    

class E1Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 64
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Softplus()
    def forward(self, x, t):
        _x = (x-S_0)/S_0
        inputs = torch.cat([_x,t], axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = self.scale * output
        return output
    

def e1_res_func(x, t, e1_net, p_net, verbose=False):
    pres = res_func(x, t, p_net)
    out = e1_net(x,t)
    out_x = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    out_t = torch.autograd.grad(out, t, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    out_xx = torch.autograd.grad(out_x, x, grad_outputs=torch.ones_like(out_x), create_graph=True)[0]
    Lp = a*out + (a*x+d)*out_x - 0.5*b*b*(2*out + x*x*out_xx)
    residual = out_t + Lp + pres
    return residual


def diff_e1(x, t, e1_net):
    out = e1_net(x,t)
    out_x = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    out_t = torch.autograd.grad(out, t, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    out_xx = torch.autograd.grad(out_x, x, grad_outputs=torch.ones_like(out_x), create_graph=True)[0]
    Lp = a*out + (a*x+d)*out_x - 0.5*b*b*(2*out + x*x*out_xx)
    return out_t + Lp


def train_e1_net(e1_net, optimizer, scheduler1, mse_cost_function, p_net, max_abs_e1_x_0, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    PATH_LOSS = FOLDER+"output/e1_net_train_loss.npy"
    iterations_per_decay = 1000

    # Define the mean and covariance matrix
    _mean = torch.tensor([100.0])
    _covariance_matrix = torch.tensor([[1.0]])
    mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    x_bc = (torch.rand(int(0.5*batch_size), 1)*(x_hig-x_low)+x_low).to(device)
    x_bc_normal = mvn.sample((int(0.5*batch_size),)).to(device)
    x_bc_normal = torch.clamp(x_bc_normal, min=x_low, max=x_hig)
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)

    t = (torch.rand(int(1.0*batch_size), 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    x = (torch.rand(len(t), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    S = 100000
    FLAG = False
    normalize = max_abs_e1_x_0
    start_time = time.time()

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p_bc = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = (p_bc - phat_bc).detach()
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # Loss based on residual
        diff_e1_hat = diff_e1(x, t, e1_net)
        diff_e1_target = -res_func(x, t, p_net).detach()
        mse_res = mse_cost_function(diff_e1_hat/normalize, diff_e1_target/normalize)

        res_out = e1_res_func(x, t, e1_net, p_net)/normalize
        res_g = res_out; x_g = x; t_g = t
        res_x = torch.autograd.grad(res_g, x_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        res_t = torch.autograd.grad(res_g, t_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        mse_res_grad = torch.mean(res_x**2 + res_t**2)

        # Combining the loss functions
        loss = mse_u + 5.0*(mse_res + mse_res_grad)
        loss_history.append(loss.data)

        # Termination
        if(loss.data < 8e-3):
            train_time = time.time() - start_time
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, 
                  ",res:", mse_res.data, 
                  ",res grad:", mse_res_grad.data
                )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time
                    }, PATH)
            np.save(PATH_LOSS, np.array(loss_history))
            return

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, 
                  ",res:", mse_res.data, 
                  ",res grad:", mse_res_grad.data
                )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time
                    }, PATH)
            np.save(PATH_LOSS, np.array(loss_history))
            min_loss = loss.data 
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

            t0_RAR = 0.0*t_RAR + t0
            ic_hat_RAR = e1_net(x_RAR, t0_RAR)/normalize
            p_bc_RAR = p_init(x_RAR.detach().numpy())
            p_bc_RAR = Variable(torch.from_numpy(p_bc_RAR).float(), requires_grad=False).to(device)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            ic_RAR = (p_bc_RAR - phat_bc_RAR)/normalize
            max_ic_error = torch.max(torch.abs(ic_RAR - ic_hat_RAR))
            print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > RAR_THRESHOLD):
                # max_abs_ic, max_index = torch.max(torch.abs(ic_RAR - ic_hat_RAR), dim=0)
                ic_diff = ic_RAR - ic_hat_RAR
                max_abs_ic, max_index = torch.topk(torch.abs(ic_diff.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t0_RAR[max_index].clone()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... IC add [x,t]:", x_max.data, t_max.data, max_abs_ic.data)

            res_RAR = e1_res_func(x_RAR, t_RAR, e1_net, p_net)/normalize
            mean_res_error = torch.mean(torch.abs(res_RAR))
            print("... RAR mean res: ", mean_res_error.data)
            if(mean_res_error > RAR_THRESHOLD):
                # max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                max_abs_res, max_index = torch.topk(torch.abs(res_RAR.squeeze()), 5)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, max_abs_res.data)
            
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler1.step()


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best e1net epoch: ", epoch, ", loss: ", loss.data, ", train time: ", checkpoint['train_time'])
    # see training result
    # keys = e1_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(e1_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 5*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/e1net_loss_history.png")
    plt.close()
    return e1_net


def show_e1_net_results(p_net, e1_net):
    plt.rcParams['font.size'] = 14
    t1s = [2.0, 4.0, 6.0]
    x = np.linspace(x_low, x_hig, num=100).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    p0 = p_init(x)

    # Determine global min and max for y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    alpha_1_list = []
    r1_list = []
    r2_list = []
    for t1 in t1s:
        p_monte = p_sol(x,t1)
        p_monte_list.append(p_monte)
        current_min = p_monte.min()
        current_max = p_monte.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy()
        p_hat_list.append(p_hat)
        e1 = p_monte - p_hat
        e1_list.append(e1)
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e1_hat_list.append(e1_hat)
        alpha_1 = max(abs(e1 - e1_hat)) / max(abs(e1_hat))
        alpha_1_list.append(alpha_1)
        r1 = res_func(pt_x, pt_t1, p_net).data.cpu().numpy()
        r1_list.append(r1)
        r2 = e1_res_func(pt_x, pt_t1, e1_net, p_net).data.cpu().numpy()
        r2_list.append(r2)

    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    for i, (t1, ax1, e1, e1_hat, alpha_1) in enumerate(zip(t1s, axs, e1_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))[0]
        if i == 0:
            ax1.plot(x, e1, "black", linewidth=1, label=r"$e_s$")
            ax1.plot(x, e1_hat, "red", linewidth=1, linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.3, label=r"$e_S$")
            ax1.legend(loc="upper right")
        else:
            # if(t1 == 6.0):
            #     ax1.scatter(x[61], e1[61])
            #     print(x[61], e1[61])
            ax1.plot(x, e1, linewidth=1, color="black")
            ax1.plot(x, e1_hat, "red", linewidth=1, linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.3)
        # ax1.set_ylim([-1.5*error_bound, 1.5*error_bound])
        # ax1.set_xlim([92, 108])
        # print out
        # if(t1 == 0.0):
        #     print("t1=",t1, ", a1=", alpha_1_exact)
        print("t1=",t1, ", a1 [Monte]=", alpha_1)
        if(i == 2):
            ax1.set_xlabel('x')
        if(i < 2):
            ax1.set_xticks([])
        # if(i == 1):
        #     ax1.set_yticks(np.array([-0.03, 0.00, 0.03]))
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_xlim([92, 108])
        ax1.set_xticks(np.array([92, 96, 100, 104, 108]))
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.95, "t:"+str(t1)+", "+r"$\alpha_1:$"+str(np.round(alpha_1[0],2)), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=16)
    plt.tight_layout(pad=0.2, h_pad=0.1)
    fig.savefig(FOLDER+'figs/e1hat_result.pdf', format='pdf', dpi=300)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    for i, (t1, ax1, p, p_hat, e1_hat, alpha_1) in enumerate(zip(t1s, axs, p_monte_list, p_hat_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))[0]
        if i == 0:
            ax1.plot(x, p, "black", linewidth=1.0, label=r"$p_s$")
            ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
            # ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.3, label=r"$e_S$")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, p, "black")
            ax1.plot(x, p_hat, "red", linewidth=1.0, linestyle="--")
        #     ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.3)
        #  ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.95, "t="+str(t1)+", "+r"$e_S=$"+str(np.round(error_bound,3)), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=16)
        ax1.set_ylim([0.0, 0.42])
        ax1.set_xlim([92, 108])
        ax1.set_xticks(np.array([92, 96, 100, 104, 108]))
        if(i < 2):
            ax1.set_xticks([])
        if(i == 2):
            ax1.set_xlabel("x")
    plt.tight_layout(pad=0.2, h_pad=0.1)
    fig.savefig(FOLDER+'figs/error_bound_results.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, r2_list, r1_list)):
        max_value = max(np.max(np.abs(pres)), 0.0)
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 1, figsize=(7, 6))
    for i in range(0,3):
        pres = r1_list[i]
        eres = r2_list[i]
        ax1 = axs[i]
        if i == 0:
            ax1.plot(x, eres-pres, "red", linewidth=1.0, linestyle="--", label=r"$D[\hat{e}_1]$")
            ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
            ax1.legend(loc="upper right")  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres-pres, "red", linewidth=1.0, linestyle="--")
            ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        ax1.text(0.01, 0.95, "t="+str(t1s[i]), 
                 transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax1.set_ylim([-global_max, global_max])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('x')
    plt.tight_layout()
    # plt.savefig(FOLDER+"figs/enet_res.png")
    fig.savefig(FOLDER+'figs/enet_res.pdf', format='pdf', dpi=300)
    plt.close()
        

def plot_p_surface(p_net, num=100):
    plt.rcParams['font.size'] = 14
    t1s = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    phat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)

    p_list = []
    for t1 in t1s:
        p_true = p_sol(x, x*0+t1)
        p_list.append(p_true)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.7, label=r"$\hat{p}$")
    z_max = 1.5*np.max(np.abs(phat))
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x*0 + t1
        if i == 0:
            ax.plot(x, t1_monte, p_list[i], color="black", label=r"$p$")
        else:
            ax.plot(x, t1_monte, p_list[i], color="black")

    # ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("PDF")
    # ax.legend()
    # # y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    # # ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    # x_ticks = np.array([90, 95, 100, 105, 110])  # Example y-tick positions
    # ax.set_xticks(x_ticks)  # Set the positions of the y-ticks
    # ax.view_init(35, -125)
    # plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.08)
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("    PDF")
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=14)
    x_ticks = np.array([90, 95, 100, 105, 110])
    y_ticks = np.array([0, 1, 2, 3, 4, 5])  # Example y-tick positions
    ax.set_xticks(x_ticks)  # Set the positions of the y-ticks
    ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    ax.set_zticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4]))
    ax.view_init(20, -60)
    plt.subplots_adjust(left=0.00, right=0.90, top=1.0, bottom=0.0)
    fig.savefig(FOLDER+'figs/phat_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1_surface(p_net, e1_net, num=100):
    plt.rcParams['font.size'] = 14
    t1s = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    e1hat = e1_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)
    
    e1_list = []
    x_monte = x.reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = p_sol(x, x*0+t1).reshape(-1, 1)
        pt_t1_monte = Variable(torch.from_numpy(x_monte*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x_monte, pt_t1_monte).data.cpu().numpy()
        e1 = p_monte - p_hat
        e1_list.append(e1)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1hat, cmap='viridis', alpha=0.7, label=r"$\hat{e}_1$")
    # z_max = 1.2*np.max(np.abs(e1hat))
    # ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="black", s=0.02, label='Data Points')
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x_monte*0 + t1
        if(i == 0):
            ax.plot(x_monte, t1_monte, e1_list[i], color="black", label=r"$e_1$")
        else:
            ax.plot(x_monte, t1_monte, e1_list[i], color="black")
    # Set z-ticks to scientific notation
    ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.zaxis.get_major_formatter().set_powerlimits((-2, 2))  # Use scientific notation if value is outside this range
    # y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    # ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    # ax.legend()
    # ax.set_xlabel("x")
    # ax.set_ylabel("t")
    # ax.set_zlabel("Error")
    # # ax.view_init(35, -125)
    # plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.08)
    # # plt.show()
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel(" e")
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=14)
    x_ticks = np.array([90, 95, 100, 105, 110])
    y_ticks = np.array([0, 1, 2, 3, 4, 5])  # Example y-tick positions
    ax.set_xticks(x_ticks)  # Set the positions of the y-ticks
    ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    ax.view_init(20, -60)
    plt.subplots_adjust(left=0.00, right=0.90, top=1.0, bottom=0.0)
    fig.savefig(FOLDER+'figs/e1hat_surface_plot.pdf', format='pdf', dpi=300)


def plot_train_loss(path_1, path_2):
    loss_history_1 = np.load(path_1)
    min_loss_1 = min(loss_history_1)
    loss_history_2 = np.load(path_2)
    min_loss_2 = min(loss_history_2)
    # print(loss_history_2)
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
    fig.savefig(FOLDER+'figs/train_loss.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


# [edited gap metric]
def show_table(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    # t1s = np.arange(t0, T_end+0.01, 0.01)
    t1s = np.linspace(t0, T_end, num=100)
    a1_list = []
    gap_list = []
    e1_list = []
    eS_ratio_list = []
    for i in range(len(t1s)):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_sol(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_list.append(max(abs(e1))[0])
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        eS = max(abs(e1_hat))*2
        eS = np.round(eS,3)
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        a1_list.append(a1[0])
        gap_list.append((eS - e1_list[i])/np.max(np.abs(p)))
        eS_ratio = eS/np.max(np.abs(p))
        eS_ratio_list.append(eS_ratio)
    gap_metric = np.min(np.array(gap_list))
    # if(np.min(np.array(gap_list))<=0):
    #     gap_metric = np.min(np.array(gap_list))
    # else:
    #     gap_metric = np.max(np.array(gap_list))
    print("[info] max a1: " +str(np.max(np.array(a1_list))) + ", avg a1:" + str(np.mean(np.array(a1_list))))
    print("[info] min gap: " +str(np.min(np.array(gap_list))) + ", max gap:" + str(np.max(np.array(gap_list))))
    print("[info] max eS_ratio: " +str(np.max(np.array(eS_ratio_list))) + ", avg eS_ratio:" + str(np.mean(np.array(eS_ratio_list))))
    

def main():
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    p_net = Net().to(device)
    p_net.apply(init_weights)
    e1_net = E1Net().to(device)
    e1_net.apply(init_weights)

    optimizer = torch.optim.Adam(p_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    p0_max = get_p0_max()
    print("p0_max: ", p0_max)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, p0_max, iterations=10000); print("[p_net train complete]")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,0): ", max_abs_e1_ti)

    e1_net.scale = max_abs_e1_ti
    optimizer = torch.optim.Adam(e1_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_ti, iterations=10000); print("[e1_net train complete]")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_net_results(p_net, e1_net)

    show_table(p_net, e1_net)
    plot_p_surface(p_net)
    plot_e1_surface(p_net, e1_net)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy",
                    FOLDER+"output/e1_net_train_loss.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()


