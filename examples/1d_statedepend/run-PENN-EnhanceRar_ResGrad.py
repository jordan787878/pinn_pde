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

# PENN + (Enhance) RAR & Residual Gradient Loss
# Log

FOLDER = "exp1/run-PENN-EnhanceRar_ResGrad/"

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
S = 500000
RAR_THRESHOLD = 1e-3
TRAIN_LOSS_THRESHOLD = 1e-4

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


# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# p_net
class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 64
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(16,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x,t], axis=1)
        d = 8; n = 100
        for i in range(int(d/2)):
            w_i = (1/pow(n, 2*i/d))
            x_sin_i = torch.sin(w_i*x)
            x_cos_i = torch.cos(w_i*x)
            inputs = torch.cat([inputs, x_sin_i, x_cos_i], axis=1)
        for i in range(int(d/2)):
            w_i = (1/pow(n, 2*i/d))
            t_sin_i = torch.sin(w_i*t)
            t_cos_i = torch.cos(w_i*t)
            inputs = torch.cat([inputs, t_sin_i, t_cos_i], axis=1)
        inputs = inputs[:, 2:]
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
    # t_init = (torch.ones(int(0.1*batch_size), 1, requires_grad=True) * t0).to(device)
    # t = torch.cat((t, t_init), dim=0)
    x = (torch.rand(len(t), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    t_g = (torch.rand(int(1.0*batch_size), 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    # t_g_init = (torch.ones(int(0.1*batch_size), 1, requires_grad=True) * t0).to(device)
    # t_g = torch.cat((t_g, t_g_init), dim=0)
    x_g = (torch.rand(len(t_g), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    S = 100000
    FLAG = False

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        u_bc = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = res_func(x, t, p_net)/max_abs_p_ti
        mse_res = mse_cost_function(res_out, all_zeros)

        res_g = res_func(x_g, t_g, p_net)/max_abs_p_ti
        res_x = torch.autograd.grad(res_g, x_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        res_t = torch.autograd.grad(res_g, t_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        mse_norm_res_input = torch.mean(res_x**2 + res_t**2)

        # loss
        loss = (mse_u + mse_res + mse_norm_res_input)/3.0
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("save epoch:", epoch, ", loss:", loss.data, ", ic:",mse_u.data, ", res:",mse_res.data, 
                  ", res freq:", mse_norm_res_input.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # Termination
        if(loss.data < TRAIN_LOSS_THRESHOLD):
            print("save epoch:", epoch, ", loss:", loss.data, ", ic:",mse_u.data, ", res:",mse_res.data, 
                  ", res freq:", mse_norm_res_input.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            return

        # RAR
        if (epoch%100 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            # _t_init = (torch.ones(100, 1, requires_grad=True) * t0).to(device); t_RAR = torch.cat((t, _t_init), dim=0)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
            res_RAR = res_func(x_RAR, t_RAR, p_net)/max_abs_p_ti
            res_x = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_t = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            mean_res_error = torch.mean(torch.abs(res_RAR))
            mean_res_g     = torch.mean(torch.abs(res_x) + torch.abs(res_t))
            print("... RAR mean res: ", mean_res_error.data)
            if(mean_res_error > RAR_THRESHOLD):
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)
            print("... RAR mean res grad: ", mean_res_g.data)
            if(mean_res_g > RAR_THRESHOLD):
                max_abs_res_g, max_index = torch.max(torch.abs(res_x)+torch.abs(res_t), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x_g = torch.cat((x_g, x_max), dim=0)
                t_g = torch.cat((t_g, t_max), dim=0)
                print("... RES grad add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_g.data)
            FLAG = False

        if (epoch%1000 == 0):
            print(epoch,"Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()



def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data)
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

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
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

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
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
        self.hidden_layer1 = (nn.Linear(16,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
    def forward(self, x, t):
        inputs = torch.cat([x,t], axis=1)
        d = 8; n = 100
        for i in range(int(d/2)):
            w_i = (1/pow(n, 2*i/d))
            x_sin_i = torch.sin(w_i*x)
            x_cos_i = torch.cos(w_i*x)
            inputs = torch.cat([inputs, x_sin_i, x_cos_i], axis=1)
        for i in range(int(d/2)):
            w_i = (1/pow(n, 2*i/d))
            t_sin_i = torch.sin(w_i*t)
            t_cos_i = torch.cos(w_i*t)
            inputs = torch.cat([inputs, t_sin_i, t_cos_i], axis=1)
        inputs = inputs[:, 2:]
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


def train_e1_net(e1_net, optimizer, scheduler1, mse_cost_function, p_net, max_abs_e1_x_0, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    PATH_LOSS = FOLDER+"output/e1_net_train_loss.npy"
    iterations_per_decay = 1000

    x_bc = (torch.rand(int(batch_size), 1)*(x_hig-x_low)+x_low).to(device)
    t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)

    t = (torch.rand(int(0.9*batch_size), 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    # t_init = (torch.ones(int(0.1*batch_size), 1, requires_grad=True) * t0).to(device)
    # t = torch.cat((t, t_init), dim=0)
    x = (torch.rand(len(t), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    t_g = (torch.rand(int(0.9*batch_size), 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    # t_g_init = (torch.ones(int(0.1*batch_size), 1, requires_grad=True) * t0).to(device)
    # t_g = torch.cat((t_g, t_g_init), dim=0)
    x_g = (torch.rand(len(t_g), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    S = 100000
    FLAG = False

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p_bc = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out/max_abs_e1_x_0, u_bc/max_abs_e1_x_0)

        # Loss based on residual
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e1_res_func(x, t, e1_net, p_net)/max_abs_e1_x_0
        mse_res = mse_cost_function(res_out, all_zeros)

        res_g = e1_res_func(x_g, t_g, e1_net, p_net)/max_abs_e1_x_0
        res_x = torch.autograd.grad(res_g, x_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        res_t = torch.autograd.grad(res_g, t_g, grad_outputs=torch.ones_like(res_g), create_graph=True)[0]
        mse_norm_res_input = torch.mean(res_x**2 + res_t**2)

        # Combining the loss functions
        loss = (mse_u + mse_res + mse_norm_res_input)/3.0
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data,
                  ",res freq:", mse_norm_res_input.data
                )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
            FLAG = True

        # Termination
        if(loss.data < TRAIN_LOSS_THRESHOLD):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data,
                  ",res freq:", mse_norm_res_input.data
                )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            return

        if (epoch%1000 == 0):
            print(epoch,"Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler1.step()

        # RAR
        if (epoch%100 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            # _t_init = (torch.ones(100, 1, requires_grad=True) * t0).to(device); t_RAR = torch.cat((t, _t_init), dim=0)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
            res_RAR = e1_res_func(x_RAR, t_RAR, e1_net, p_net)/max_abs_e1_x_0
            res_x = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_t = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            mean_res_error = torch.mean(torch.abs(res_RAR))
            mean_res_g     = torch.mean(torch.abs(res_x) + torch.abs(res_t))
            print("... RAR mean res: ", mean_res_error.data)
            if(mean_res_error > RAR_THRESHOLD):
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)
            print("... RAR mean res grad: ", mean_res_g.data)
            if(mean_res_g > RAR_THRESHOLD):
                max_abs_res_g, max_index = torch.max(torch.abs(res_x)+torch.abs(res_t), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x_g = torch.cat((x_g, x_max), dim=0)
                t_g = torch.cat((t_g, t_max), dim=0)
                print("... RES grad add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_g.data)
            FLAG = False


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best epoch: ", epoch, ", loss:", loss.data)
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
    t1s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
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
        r2 = e1_res_func(pt_x, pt_t1, e1_net, p_net).data.cpu().numpy()
        r2_list.append(r2)

    fig, axs = plt.subplots(6, 1, figsize=(8, 12))
    for i, (t1, ax1, e1, e1_hat, alpha_1) in enumerate(zip(t1s, axs, e1_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))[0]
        if i == 0:
            ax1.plot(x, e1, "black", label=r"$e_s$")
            ax1.plot(x, e1_hat, "blue", linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.2, label=r"$e_L$")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, e1, "black")
            ax1.plot(x, e1_hat, "blue", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.2)
        ax1.set_ylim([-1.2*error_bound, 1.2*error_bound])

        # print out
        # if(t1 == 0.0):
        #     print("t1=",t1, ", a1=", alpha_1_exact)
        print("t1=",t1, ", a1 [Monte]=", alpha_1)
        # grid
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.95, "t="+str(t1)+", "+r"$\alpha_1=$"+str(np.round(alpha_1[0],2)), transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/e1net_result.png")
    plt.close()

    fig, axs = plt.subplots(6, 1, figsize=(8, 6))
    for i, (t1, ax1, p, p_hat, e1_hat, alpha_1) in enumerate(zip(t1s, axs, p_monte_list, p_hat_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))[0]
        if i == 0:
            ax1.plot(x, p, "blue", label=r"$p_s$")
            ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
            # ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.2, label=r"e_L")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, p, "blue")
            ax1.plot(x, p_hat, "red", linestyle="--")
        #     ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.2)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.95, "t="+str(t1)+", "+r"$\alpha_1=$"+str(np.round(alpha_1[0],2))+"\n"+r"$e_L=$"+str(np.round(error_bound,2)), 
                 transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax1.set_ylim([0.0, 0.42])
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/uni_error_bound.png")
    plt.close()

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    for i, (t1, ax1, r2) in enumerate(zip(t1s, axs, r2_list)):
        if i == 0:
            ax1.plot(x, r2, "r--", label=r"$r_2$")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, r2, "r--")
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/e1net_residual.png")
    plt.close()
        

def main():
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    p_net = Net().to(device)
    p_net.apply(init_weights)
    optimizer = torch.optim.Adam(p_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    p0_max = get_p0_max()
    print("p0_max: ", p0_max)
    start_time = time.time()
    train_p_net(p_net, optimizer, scheduler, mse_cost_function, p0_max, iterations=10000); print("[p_net train complete]")
    end_time = time.time()
    time_train_pnet = end_time - start_time
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,0): ", max_abs_e1_ti)

    e1_net = E1Net(scale=max_abs_e1_ti).to(device)
    e1_net.apply(init_weights)
    optimizer = torch.optim.Adam(e1_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    start_time = time.time()
    train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_ti, iterations=30000); print("[e1_net train complete]")
    end_time = time.time()
    time_train_e1net = end_time - start_time
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_net_results(p_net, e1_net)

    print(f"train pnet time: {time_train_pnet:.4f} seconds")
    print(f"train e1net time: {time_train_e1net:.4f} seconds")


if __name__ == "__main__":
    main()


