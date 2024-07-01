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

# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
FOLDER = "exp4/tmp/"
DATA_FOLDER = "data/exp4/"
device = "cpu"
print(device)
# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

n_d = 1
mu = -2
std = 0.5
a = -0.1
b = 0.1
c = 0.5
d = 0.5
e = 0.8
x_low = -6
x_hig = 6

t0 = 0
T_end = 5
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def f_sde(x):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = a*np.power(x, 3) + b*np.power(x, 2) + c*x + d
        except RuntimeWarning as e:
            print(f"Warning occurred at x = {x}: {e}")
            result = np.nan  # or handle the error in another way
    return result


def p_init(x):
  return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def test_p_init():
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    p = p_init(x)
    return max(abs(p))[0]


def p_sol_monte(t1=T_end, linespace_num=100, stat_sample=10000):
    n_d = 1
    dtt = 0.01
    t_span = np.arange(t0, t1, dtt)
    num_steps = len(t_span)
    
     # Initialize arrays
    X_last = np.random.normal(mu, std, stat_sample)
    bins_x1 = np.linspace(x_low, x_hig, num=linespace_num)

    # Vectorized simulation of the SDE
    for step in tqdm(range(1, num_steps + 1), desc="Simulating samples"):
        dW = np.random.normal(0, np.sqrt(dtt), stat_sample)
        X_new = X_last + f_sde(X_last) * dtt + e * dW
        X_last = X_new

    bins_x1 = np.linspace(x_low, x_hig, num=linespace_num)
    # Digitize v to find which bin each value falls into for both dimensions
    bin_indices_x1 = np.digitize(X_last, bins_x1) - 1
    # Initialize the frequency array
    frequency = np.zeros((len(bins_x1) - 1, 1))

    # Count the occurrences in each 2D bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        if 0 <= bin_indices_x1[i] < frequency.shape[0]:
            frequency[bin_indices_x1[i], :] += 1

    # Normalize the frequency to get the proportion
    frequency = frequency / stat_sample
    dx = bins_x1[1]-bins_x1[0]
    frequency = frequency/(dx**n_d)

    # Calculate the midpoints for bins
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2

    return midpoints_x1, frequency

    # x1 = X[0,:]
    # dx_size = 101
    # x_points = np.linspace(x_low, x_hig, num=dx_size)
    # dx = x_points[1]-x_points[0]
    # bins_1D = 0.5 * (x_points[:-1] + x_points[1:]).reshape(-1)
    # grid = 0*bins_1D
    # inds1 = np.digitize(x1, bins_1D)
    # for i in range(len(inds1)):
    #     if(x1[i]<= x_hig and x1[i]>= x_low and inds1[i] > 0):
    #         grid[inds1[i]-1] += (1.0/stat_sample)
    # # convert to pdf
    # grid = grid/(dx**n_d)
    # return bins_1D, grid, dx


def res_func(x,t, net, verbose=False):
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*a*x*x + 2*b*x + c)*p + (a*x*x*x + b*x*x + c*x + d)*p_x - 0.5*e*e*p_xx
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
        neurons = 30
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.hidden_layer8 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        # normalization to [0-1]
        # _x = 2.0*(x-x_low)/(x_hig-x_low) - 1.0
        # _t = 2.0*(t-t0)/(T_end-t0) - 1.0
        # inputs = torch.cat([_x,_t],axis=1)
        inputs = torch.cat([x, t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        layer6_out = F.softplus((self.hidden_layer6(layer5_out)))
        layer7_out = F.softplus((self.hidden_layer7(layer6_out)))
        layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
        output = F.softplus( self.output_layer(layer8_out) )
        return output


def train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_abs_p_ti, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/p_net.pt"
    PATH_LOSS = FOLDER+"output/p_net_train_loss.npy"
    iterations_per_decay = 1000

    # Define the mean and covariance matrix
    _mean = torch.tensor([-2.0])
    _covariance_matrix = torch.tensor([[0.5]])
    mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    # Normalized space-time points
    # space-time points for BC
    x_bc = (torch.rand(250, 1)*(x_hig-x_low)+x_low).to(device)
    x_bc_normal = mvn.sample((250,)).to(device)
    x_bc_normal = torch.clamp(x_bc_normal, min=x_low, max=x_hig)
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)

    t = (torch.rand(1000, 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    t_init = (torch.ones(100, 1, requires_grad=True) * t0).to(device)
    t = torch.cat((t, t_init), dim=0)
    x = (torch.rand(len(t), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

    S = 100000
    FLAG = False

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        u_bc = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = res_func(x, t, p_net)/max_abs_p_ti
        mse_res = mse_cost_function(res_out, all_zeros)

        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        # norm_res_input = torch.norm(res_input/max_abs_p_ti, dim=1).view(-1,1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # <Baseline>
        loss = mse_u + mse_res + mse_norm_res_input

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("save epoch:", epoch, ", loss:", loss.data, ", ic:",mse_u.data, ", res:",mse_res.data, 
                  ", res freq:", mse_norm_res_input.data,
                  #",l-inf ic:", linf_u.data, ",l-inf res:",linf_res.data, ",D res:", linf_res_x.data, linf_res_t.data, mse_res_x.data, mse_res_t.data,
                  #"res freq:", num_zero_res_x_smooth.data)
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        loss_history.append(loss.data)

        # RAR
        if (epoch%1000 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            _t_init = (torch.ones(100, 1, requires_grad=True) * t0).to(device)
            t_RAR = torch.cat((t, _t_init), dim=0)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
            # t0_RAR = 0.0*t_RAR + t0
            # ic_hat_RAR = p_net(x_RAR, t0_RAR)
            # p_bc_RAR = p_init(x_RAR.detach().numpy())
            # ic_RAR = Variable(torch.from_numpy(p_bc_RAR).float(), requires_grad=False).to(device)
            # mean_ic_error = torch.mean(torch.abs(ic_RAR/max_abs_p_ti - ic_hat_RAR/max_abs_p_ti))
            # print("RAR mean IC: ", mean_ic_error.data)
            # if(mean_ic_error > 5e-3):
            #     max_abs_ic, max_index = torch.max(torch.abs(ic_RAR/max_abs_p_ti - ic_hat_RAR/max_abs_p_ti), dim=0)
            #     x_max = x_RAR[max_index]
            #     t_max = t0_RAR[max_index]
            #     x_bc = torch.cat((x_bc, x_max), dim=0)
            #     t_bc = torch.cat((t_bc, t_max), dim=0)
            #     print("... IC add [x,t]:", x_max.data, t_max.data, ". max ic value: ", max_abs_ic.data)
            #     FLAG = False
            res_RAR = res_func(x_RAR, t_RAR, p_net)/max_abs_p_ti
            mean_res_error = torch.mean(torch.abs(res_RAR))
            print("... RAR mean res: ", mean_res_error.data)
            if(mean_res_error > 0.0):
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)
                FLAG = False
            # res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_input_RAR = torch.cat([res_x_RAR, res_t_RAR], axis=1)
            # norm_res_input_RAR = torch.norm(res_input_RAR, dim=1).view(-1,1)
            # mean_res_input_error = torch.mean(norm_res_input)
            # print("RAR mean res input: ", mean_res_input_error.data)
            # if(mean_res_input_error > 5e-3):
            #    max_abs_res_input, max_index = torch.max(norm_res_input_RAR, dim=0)
            #    x_max = x_RAR[max_index]
            #    t_max = t_RAR[max_index]
            #    x = torch.cat((x, x_max), dim=0)
            #    t = torch.cat((t, t_max), dim=0)
            #    print("... RES_INPUT add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)
            #    FLAG = False

        if (epoch%1000 == 0):
            print(epoch,"Traning Loss:",loss.data, 
                #   ", Time domain:", min(t).data, ",", max(t).data,
                #   ", X domain:", min(x[:,0]).data, ",", max(x[:,0]).data
                    )
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
    keys = p_net.state_dict().keys()
    for k in keys:
        l2_norm = torch.norm(p_net.state_dict()[k], p=2)
        print(f"L2 norm of {k} : {l2_norm.item()}")
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
    x_monte = np.load(DATA_FOLDER + "xsim.npy").reshape(-1,1)
    x = x_monte
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_ti = Variable(torch.from_numpy(x*0+t0).float(), requires_grad=True).to(device)
    p     = p_init(x)
    p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    e1 = p - p_hat
    max_abs_e1_ti = max(abs(e1))[0]

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    # Determine global min and max for y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    p_monte_list = []
    p_hat_list = []
    e1_list = []
    limit_margin = 0.1
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
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
            ax1.plot(x_monte, p_monte, "blue", label=r"$p$")
            ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x_monte, p_monte, "blue")
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
        neurons = 30
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.hidden_layer7 = (nn.Linear(neurons,neurons))
        self.hidden_layer8 = (nn.Linear(neurons,neurons))
        self.hidden_layer9 = (nn.Linear(neurons,neurons))
        self.hidden_layer10 = (nn.Linear(neurons,neurons))
        self.hidden_layer11 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        layer6_out = self.activation((self.hidden_layer6(layer5_out)))
        layer7_out = self.activation((self.hidden_layer7(layer6_out)))
        layer8_out = self.activation((self.hidden_layer8(layer7_out)))
        layer9_out = self.activation((self.hidden_layer9(layer8_out)))
        layer10_out = self.activation((self.hidden_layer10(layer9_out)))
        layer11_out = self.activation((self.hidden_layer11(layer10_out)))
        output = self.output_layer(layer11_out)
        output = self.scale * output
        return output
    

def e1_res_func(x, t, e1_net, p_net, verbose=False):
    net_out = e1_net(x,t)
    net_out_x = torch.autograd.grad(net_out, x, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_t = torch.autograd.grad(net_out, t, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_xx = torch.autograd.grad(net_out_x, x, grad_outputs=torch.ones_like(net_out_x), create_graph=True)[0]
    p_res = res_func(x, t, p_net)
    residual = net_out_t + (3*a*x*x + 2*b*x + c)*net_out + (a*x*x*x + b*x*x + c*x + d)*net_out_x - 0.5*e*e*net_out_xx + p_res
    return residual


# def test_e1_res(e1_net, p_net):
#     batch_size = 5
#     x_collocation = np.random.uniform(low=1.0, high=3.0, size=(batch_size,1))
#     t_collocation = T_end*np.ones((batch_size,1))
#     all_zeros = np.zeros((batch_size,1))
#     pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
#     pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
#     f_out = e1_res_func(pt_x_collocation, pt_t_collocation, e1_net, p_net, verbose=True) # output of f(x,t)
# def augment_tensors(x_tensor, t_tensor):
#     # Get the size of the input tensors
#     M = x_tensor.size(0)
#     N = t_tensor.size(0)
#     augmented_x_tensor = x_tensor.repeat(N, 1).view(N*M, 1)
#     augmented_t_tensor = t_tensor.repeat_interleave(M).view(N*M, 1)
#     return augmented_x_tensor, augmented_t_tensor


def train_e1_net(e1_net, optimizer, scheduler1, mse_cost_function, p_net, max_abs_e1_x_0, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    PATH_LOSS = FOLDER+"output/e1_net_train_loss.npy"
    iterations_per_decay = 1000

    # Define the mean and covariance matrix
    _mean = torch.tensor([-2.0])
    _covariance_matrix = torch.tensor([[0.5]])
    mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    # Normalized space-time points
    # space-time points for BC
    x_bc = (torch.rand(250, 1)*(x_hig-x_low)+x_low).to(device)
    x_bc_normal = mvn.sample((250,)).to(device)
    x_bc_normal = torch.clamp(x_bc_normal, min=x_low, max=x_hig)
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)

    t = (torch.rand(1000, 1, requires_grad=True)*(T_end-t0)   +t0).to(device)
    t_init = (torch.ones(100, 1, requires_grad=True) * t0).to(device)
    t = torch.cat((t, t_init), dim=0)
    x = (torch.rand(len(t), 1, requires_grad=True)*(x_hig-x_low)+x_low).to(device)

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

        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)
        
        # Combining the loss functions
        loss = mse_u + mse_res + mse_norm_res_input

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data,
                  ",res freq:", mse_norm_res_input.data
                  # , l_inf_res.data,# quotient_max.data,
                  # ",a1(t0):", alpha1_t0.data, "error(t0):",error1_t0.data)
                )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
            FLAG = True

        loss_history.append(loss.data)

        if (epoch%1000 == 0):
            print(epoch,"Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler1.step()

        # RAR
        if (epoch%1000 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)  *(T_end-t0)   +t0).to(device)
            _t_init = (torch.ones(100, 1, requires_grad=True) * t0).to(device)
            t_RAR = torch.cat((t, _t_init), dim=0)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True)*(x_hig-x_low)+x_low).to(device)
            # t0_RAR = 0.0*t_RAR + t0
            # ic_hat_RAR = e1_net(x_RAR, t0_RAR)/max_abs_e1_x_0
            # p_bc_RAR = p_init(x_RAR.detach().numpy())
            # p_bc_RAR = Variable(torch.from_numpy(p_bc_RAR).float(), requires_grad=False).to(device)
            # phat_bc_RAR = p_net(x_RAR, t0_RAR)
            # ic_RAR = (p_bc_RAR - phat_bc_RAR)/max_abs_e1_x_0
            # mean_ic_error = torch.mean(torch.abs(ic_RAR - ic_hat_RAR))
            # print("RAR mean IC: ", mean_ic_error.data)
            # if(mean_ic_error > 5e-3):
            #    max_abs_ic, max_index = torch.max(torch.abs(ic_RAR - ic_hat_RAR), dim=0)
            #    x_max = x_RAR[max_index]
            #    t_max = t0_RAR[max_index]
            #    x_bc = torch.cat((x_bc, x_max), dim=0)
            #    t_bc = torch.cat((t_bc, t_max), dim=0)
            #    print("... IC add [x,t]:", x_max.data, t_max.data, ". max ic value: ", max_abs_ic.data)
            #    FLAG = False
            res_RAR = e1_res_func(x_RAR, t_RAR, e1_net, p_net)/max_abs_e1_x_0
            mean_res_error = torch.mean(torch.abs(res_RAR))
            print("... RAR mean res: ", mean_res_error.data)
            if(mean_res_error > 0.0):
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)
                FLAG = False
            # res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_input_RAR = torch.cat([res_x_RAR, res_t_RAR], axis=1)
            # norm_res_input_RAR = torch.norm(res_input_RAR, dim=1).view(-1,1)
            # mean_res_input_error = torch.mean(norm_res_input)
            # print("RAR mean res input: ", mean_res_input_error.data)
            # if(mean_res_input_error > 5e-3):
            #     max_abs_res_input, max_index = torch.max(norm_res_input_RAR, dim=0)
            #     x_max = x_RAR[max_index]
            #     t_max = t_RAR[max_index]
            #     x = torch.cat((x, x_max), dim=0)
            #     t = torch.cat((t, t_max), dim=0)
            #     print("... RES_INPUT add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best epoch: ", epoch, ", loss:", loss.data)
    # see training result
    keys = e1_net.state_dict().keys()
    for k in keys:
        l2_norm = torch.norm(e1_net.state_dict()[k], p=2)
        print(f"L2 norm of {k} : {l2_norm.item()}")
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
    x_monte = np.load(DATA_FOLDER + "xsim.npy").reshape(-1,1)
    x = x_monte
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    p0 = p_init(x)

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
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
        p_monte = np.load(DATA_FOLDER + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
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
        if(t1 == 0.0):
            e1_exact = p0 - p_hat
            alpha_1_exact = max(abs(e1_exact - e1_hat)) / max(abs(e1_hat))

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    for i, (t1, ax1, e1, e1_hat, alpha_1) in enumerate(zip(t1s, axs, e1_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))
        if i == 0:
            ax1.plot(x, e1_exact, "black", label=r"$e_1$")
            ax1.plot(x, e1, "blue", label=r"$e_1$ [Monte]")
            ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, e1, "blue")
            ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
        
        # print out
        if(t1 == 0.0):
            print("t1=",t1, ", a1=", alpha_1_exact)
        print("t1=",t1, ", a1 [Monte]=", alpha_1)
        # grid
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.98, "t="+str(t1)+", "+r"$\alpha_1=$"+str(np.round(alpha_1,2)), transform=ax1.transAxes, verticalalignment='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/e1net_result.png")
    plt.close()

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    for i, (t1, ax1, p, p_hat, e1_hat, alpha_1) in enumerate(zip(t1s, axs, p_monte_list, p_hat_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))
        if i == 0:
            ax1.plot(x, p, "blue", label=r"$p$")
            ax1.plot(x, p_hat, "red", linestyle="--", label=r"$\hat{p}$")
            # ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, p, "blue")
            ax1.plot(x, p_hat, "red", linestyle="--")
        #     ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.98, "t="+str(t1)+", "+r"$\alpha_1=$"+str(np.round(alpha_1,2))+"\n"+r"$B_e=$"+str(np.round(error_bound,2)), transform=ax1.transAxes, verticalalignment='top', fontsize=8)
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
    

def plot_p_monte():
    plt.figure()
    for t1 in t1s:
        x_sim = np.load(DATA_FOLDER+"xsim.npy")
        p_sim = np.load(DATA_FOLDER+"psim_t"+str(t1)+".npy")
        plt.plot(x_sim, p_sim, label="t="+str(t1))
    plt.grid()
    plt.legend()
    plt.savefig(FOLDER+"figs/p_sol_monte.png")
    print("save fig to "+FOLDER+"figs/p_sol_monte.png")
    plt.close()
        


def main():

    FLAG_GENERATE_DATA = True
    N = 5.0
    if(FLAG_GENERATE_DATA):
        for t1 in t1s:
            x_sim, _ = p_sol_monte(t1=t1, linespace_num=100, stat_sample=1)
            p_sim = 0.0*(x_sim.reshape(-1,1))
            for i in range(5):
                _ , _p_sim = p_sol_monte(t1=t1, linespace_num=100, stat_sample=1000)
            print(p_sim.shape, _p_sim.shape)
            p_sim += (_p_sim/N)
            # _x_sim, _p_sim = p_sol_monte(t1=t1, linespace_num=100, stat_sample=1000000000)
            # p_sim = 0.5*(p_sim + _p_sim)
            np.save(DATA_FOLDER+"psim_t"+str(t1)+".npy", p_sim)
            np.save(DATA_FOLDER+"xsim.npy", x_sim)
    # Plot generated data
    plot_p_monte()

    return

    max_pi = test_p_init()
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    p_net = Net().to(device)
    p_net.apply(init_weights)
    optimizer = torch.optim.Adam(p_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_pi, iterations=30000); print("[p_net train complete]")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,0):", max_abs_e1_ti)

    e1_net = E1Net(scale=max_abs_e1_ti).to(device)
    e1_net.apply(init_weights)
    optimizer = torch.optim.Adam(e1_net.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_ti, iterations=100000); print("[e1_net train complete]")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_net_results(p_net, e1_net)

    print("[complete rational nonlinear]")


if __name__ == "__main__":
    main()
