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

# Log
# Best, w_regular for res e grad = 1e-4, p with softplus activation

FOLDER = "exp1/run-PE/"

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
S = 10000
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


def p_res_func(x, t, out, verbose=False):
    p = out[:,0].reshape(-1,1)
    p_x = torch.autograd.grad(out[:,0], x, grad_outputs=torch.ones_like(out[:,0]), create_graph=True)[0]
    p_t = torch.autograd.grad(out[:, 0], t, grad_outputs=torch.ones_like(out[:, 0]), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    Lp = a*p + (a*x+d)*p_x - 0.5*b*b*(2*p + x*x*p_xx)
    residual = p_t + Lp
    if(verbose):
        print(p_x.shape, p_t.shape, p_xx.shape, residual.shape)
    return residual


def e_res_func(x, t, out, verbose=False):
    e = out[:,1].reshape(-1,1)
    e_x = torch.autograd.grad(out[:,1], x, grad_outputs=torch.ones_like(out[:,1]), create_graph=True)[0]
    e_t = torch.autograd.grad(out[:,1], t, grad_outputs=torch.ones_like(out[:,1]), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    Le = a*e + (a*x+d)*e_x - 0.5*b*b*(2*e + x*x*e_xx)
    residual = e_t + Le
    if(verbose):
        print(e_x.shape, e_t.shape, e_xx.shape, residual.shape)
    return residual


# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    

class PENet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(PENet, self).__init__()
        self.hidden_layer1 = (nn.Linear(64,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,2*neurons))
        self.p_layer1 =  (nn.Linear(neurons,neurons))
        self.p_layer2 =  (nn.Linear(neurons,neurons))
        self.p_out =  (nn.Linear(neurons,1))
        self.e_layer1 =  (nn.Linear(neurons,neurons))
        self.e_layer2 =  (nn.Linear(neurons,neurons))
        self.e_out =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
        self.softplus = nn.Softplus()
    def forward(self, x, t):
        inputs = torch.cat([x,t], axis=1)
        d = 32; n = 100
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
        layer1_out = self.activation(self.hidden_layer1(inputs))
        layer2_out = self.activation(self.hidden_layer2(layer1_out))
        layer3_out = self.activation(self.hidden_layer3(layer2_out))
        layer4_out = self.activation(self.hidden_layer4(layer3_out))
        layer5_out = self.activation(self.hidden_layer5(layer4_out))
        p = self.activation(self.p_layer1(layer5_out[:,:50]))
        p = self.activation(self.p_layer2(p))
        p = F.softplus(self.p_out(p))
        e = self.activation(self.e_layer1(layer5_out[:,50:]))
        e = self.activation(self.e_layer2(e))
        e = (self.e_out(e))
        output = torch.cat([p,e], axis=1)
        # Create new tensor with Softplus applied to the first output
        # output_with_softplus = torch.cat([self.softplus(output[:,0].reshape(-1,1)), output[:,1].reshape(-1,1)], axis=1)
        return output
    

def train_model(net, optimizer, scheduler, mse_cost_function, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/net.pt"
    PATH_LOSS = FOLDER+"output/net_train_loss.npy"
    iterations_per_decay = 1000

    # x_bc = (torch.rand(batch_size, 1)*(x_hig-x_low)+x_low).to(device)
    # x_l = (torch.ones(1, 1) * x_low).to(device)
    # x_r = (torch.ones(1, 1) * x_hig).to(device)
    # x_bc = torch.cat((x_bc, x_l, x_r), dim=0)
    _mean = torch.tensor([100.0])
    _covariance_matrix = torch.tensor([[1.0]])
    mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)
    x_bc = (torch.rand(int(0.5*batch_size), 1)*(x_hig-x_low)+x_low).to(device)
    x_bc_normal = mvn.sample((int(0.5*batch_size),)).to(device)
    x_bc_normal = torch.clamp(x_bc_normal, min=x_low, max=x_hig)
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)

    x = (torch.rand(batch_size, 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
    x_ll = (torch.ones(100, 1, requires_grad=True) * x_low).to(device)
    x_rr = (torch.ones(100, 1, requires_grad=True) * x_hig).to(device)
    x = torch.cat((x, x_ll, x_rr), dim=0)
    t = (torch.rand(len(x), 1, requires_grad=True)*(T_end-t0)   +t0).to(device)

    FLAG = 0
    w_regular = 1.0

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p0 = p_init(x_bc.detach().numpy())
        p0 = Variable(torch.from_numpy(p0).float(), requires_grad=False).to(device)
        normalize_p = torch.max(torch.abs(p0), dim=0)[0]
        normalize_p = normalize_p.detach()
        net_out_0 = net(x_bc, t_bc)
        p0_hat = net_out_0[:, 0].reshape(-1,1)
        p0_diff = (p0-p0_hat)/normalize_p
        mse_ic = torch.mean(p0_diff**2)

        normalize = torch.max(torch.abs(p0-p0_hat), dim=0)[0]
        normalize = normalize.detach()
        e0_hat = net_out_0[:, 1].reshape(-1,1)
        mse_ic_e = mse_cost_function( (p0-p0_hat)/normalize, e0_hat/normalize)

        # Loss based on PDE
        net_out = net(x, t)

        p_res = (p_res_func(x, t, net_out))/normalize_p
        mse_res_p = torch.mean(p_res**2)
        p_res_x = torch.autograd.grad(p_res, x, grad_outputs=torch.ones_like(p_res), create_graph=True)[0]
        p_res_t = torch.autograd.grad(p_res, t, grad_outputs=torch.ones_like(p_res), create_graph=True)[0]
        mse_p_res_grad = torch.mean(p_res_x**2 + p_res_t**2)

        e_res = (e_res_func(x, t, net_out) + p_res)/normalize
        mse_res_e = torch.mean(e_res**2)
        e_res_x = torch.autograd.grad(e_res, x, grad_outputs=torch.ones_like(e_res), create_graph=True)[0]
        e_res_t = torch.autograd.grad(e_res, t, grad_outputs=torch.ones_like(e_res), create_graph=True)[0]
        mse_e_res_grad = torch.mean(e_res_x**2 + e_res_t**2)

        # print(p_res.shape, p_res[0]); print(e_res.shape, e_res[0])

        # loss function
        loss = mse_ic + mse_res_p + w_regular*mse_p_res_grad + mse_ic_e+  mse_res_e + w_regular*mse_e_res_grad
        # loss = max(mse_ic + mse_res_p + mse_p_res_grad, mse_ic_e+  mse_res_e + mse_e_res_grad)
        loss_history.append(loss.data)

        if ((epoch+1)%100 == 0):
            print(epoch+1," Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("save epoch:", epoch, ", loss:", loss.data, 
                  ",ic p:",mse_ic.data, 
                  ",res p:", mse_res_p.data,
                  ",res p grad:", mse_p_res_grad.data,
                  ",ic e:",mse_ic_e.data, 
                  ",res e:", mse_res_e.data,
                  ",res e grad:", mse_e_res_grad.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data
            FLAG = FLAG + 1

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        # RAR
        if (FLAG > 10):
            print("Sample N: ", x_bc.shape, x.shape)
            x_RAR = (torch.rand(S, n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
            x_lll = (torch.ones(100, 1, requires_grad=True) * x_low).to(device)
            x_rrr = (torch.ones(100, 1, requires_grad=True) * x_hig).to(device)
            x_RAR = torch.cat((x_RAR, x_lll, x_rrr), dim=0)
            t_RAR = (torch.rand(len(x_RAR), 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)

            net_out_RAR = net(x_RAR, t_RAR)
            p_res_RAR = p_res_func(x_RAR, t_RAR, net_out_RAR)/normalize_p
            e_res_RAR = (e_res_func(x_RAR, t_RAR, net_out_RAR) + p_res_RAR)/normalize
            max_abs_res, max_index = torch.topk(torch.abs(p_res_RAR), 5, dim=0)
            x_max = x_RAR[max_index].reshape(-1,1)
            t_max = t_RAR[max_index].reshape(-1,1)
            x = torch.cat((x, x_max), dim=0)
            t = torch.cat((t, t_max), dim=0)
            print("... RES add [x,t]:", x_max[0].data, t_max[0].data, ". max res value: ", max_abs_res[0].data)
            
            max_abs_res, max_index = torch.topk(torch.abs(e_res_RAR), 5, dim=0)
            x_max = x_RAR[max_index].reshape(-1,1)
            t_max = t_RAR[max_index].reshape(-1,1)
            x = torch.cat((x, x_max), dim=0)
            t = torch.cat((t, t_max), dim=0)
            print("... RES add [x,t]:", x_max[0].data, t_max[0].data, ". max res value: ", max_abs_res[0].data)
            
            FLAG = 0


def pos_p_net_train(net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data)
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    # print(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/net_loss_history.png")
    plt.close()
    return net


def show_net_results(net):
    # plt.figure(figsize=(8,6))
    # markers = ['o', 's', 'D', '^', 'v', '<']
    # for i in range(len(t1s)):
    #     t1 = t1s[i]
    #     p_true = p_sol(x, t1)
    #     plt.plot(x,p_true,marker=markers[i], markersize=3, label="t="+str(t1))
    # plt.tight_layout()
    # plt.legend(loc="upper right")
    # plt.grid(linewidth=0.5)
    # plt.savefig(FOLDER+"figs/p_sol.png")

    x = np.linspace(x_low, x_hig, num=100).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    # Determine global min and max for y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []

    for t1 in t1s:
        p_monte = p_sol(x, t1).reshape(-1,1)
        p_monte_list.append(p_monte)
        current_min = p_monte.min()
        current_max = p_monte.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        net_out = net(pt_x, pt_t1).data.cpu().numpy()
        phat = net_out[:,0].reshape(-1,1)
        ehat = net_out[:,1].reshape(-1,1)
        p_hat_list.append(phat)
        e1_list.append(p_monte-phat)
        e1_hat_list.append(ehat)
    
    limit_margin = 0.1

    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i, (p_monte, p_hat, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, e1_hat_list)):
        if i == 0:
            ax1 = axs[0,0]
        if i == 1:
            ax1 = axs[1,0]
        if i == 2:
            ax1 = axs[2,0]
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
        eL = 2.0 * np.max(np.abs(e1_hat))
        ax1.plot(x, p_monte, "blue", label=r"$p$")
        ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL, color="green", alpha=0.2)
        ax1.set_ylim(global_min-limit_margin, global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/net_result_1.png")
    plt.close()

    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i, (e1, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        eL = 2.0 * np.max(np.abs(e1_hat))
        if i == 0:
            ax1 = axs[0,0]
        if i == 1:
            ax1 = axs[1,0]
        if i == 2:
            ax1 = axs[2,0]
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
        print("eL: ", np.round(eL,3))
        print("a1: ", np.max(np.abs(e1-e1_hat))/np.max(np.abs(e1_hat)))
        if i == 0:
            ax1.plot(x, e1, "blue", label=r"$e$")
            ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}$")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+eL, y2=0*x.reshape(-1)-eL, color="green", alpha=0.2)
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, e1, "blue")
            ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+eL, y2=0*x.reshape(-1)-eL, color="green", alpha=0.2)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/net_result_2.png")
    plt.close()
        

def main():
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    model = PENet().to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    start_time = time.time()
    train_model(model, optimizer, scheduler, mse_cost_function, iterations=10000); print("[p_net train complete]")
    end_time = time.time()
    time_train = end_time - start_time
    model = pos_p_net_train(model, PATH=FOLDER+"output/net.pt", PATH_LOSS=FOLDER+"output/net_train_loss.npy"); model.eval()
    show_net_results(model)

    print(f"train net time: {time_train:.4f} seconds")


if __name__ == "__main__":
    main()


