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

# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
FOLDER = "exp2/tmp/"
device = "cpu"
print(device)

n_d = 1
mu = -2
std = 1
a = -0.1
b = 0.1
c = 0.0
d = 1
e = 1
x_low = -8
x_hig = 8

t0 = 0
T_end = 6
# dt = 0.1
t1s = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]

DATA_FOLDER = "data/exp2/"
# X_sim = np.load("data/exp1/X_sim.npy")
# P_sim_t01 = np.load("data/exp1/P_sim_t01.npy")
# P_sim_t05 = np.load("data/exp1/P_sim_t05.npy")
# P_sim_t10 = np.load("data/exp1/P_sim_t10.npy")
# P_sim_t11 = np.load("data/exp1/P_sim_t11.npy")
# P_sim_t15 = np.load("data/exp1/P_sim_t15.npy")
# P_sim_t20 = np.load("data/exp1/P_sim_t20.npy")


def f_sde(x):
  return a*(x**3) + b*(x**2) + c*x + d


def p_init(x):
  return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def test_p_init():
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    p = p_init(x)
    return max(abs(p))[0]


def p_sol_monte(stat_sample=10000, t1=T_end):
    n_d = 1
    dtt = 0.01
    t_span = np.arange(t0, t1, dtt)
    X = np.zeros((1, stat_sample))
    for i in range(0, stat_sample):
        x = np.random.normal(mu, std)
        for t in t_span:
            w = np.random.normal(0, np.sqrt(dtt))
            x = x + f_sde(x)*dtt + e*w
        X[:,i] = x
    x1 = X[0,:]
    dx_size = 101
    x_points = np.linspace(x_low, x_hig, num=dx_size)
    dx = x_points[1]-x_points[0]
    bins_1D = 0.5 * (x_points[:-1] + x_points[1:]).reshape(-1)
    grid = 0*bins_1D
    inds1 = np.digitize(x1, bins_1D)
    for i in range(len(inds1)):
        if(x1[i]<= x_hig and x1[i]>= x_low and inds1[i] > 0):
            grid[inds1[i]-1] += (1.0/stat_sample)
    # convert to pdf
    grid = grid/(dx**n_d)
    return bins_1D, grid, dx


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


# p_net
class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 100
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        # output = torch.tanh(output)
        return output


def count_approx_zero_elements(tensor, epsilon=1e-3):
    """
    Approximate the count of elements in the tensor that are close to zero within epsilon distance.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, 1).
        epsilon (float): Distance from zero to consider.

    Returns:
        torch.Tensor: A tensor containing a single value which is the approximate count.
    """
    # Calculate the distance of each element from zero
    distances = torch.abs(tensor)
    # Apply a smooth indicator function (sigmoid)
    soft_indicators = torch.sigmoid((epsilon - distances) * 10000)  # scale factor 100000 to make transition sharp
    # Sum the soft indicators to get the approximate count
    approx_count = torch.sum(soft_indicators)
    return approx_count


def train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_abs_p_ti, iterations=40000):
    batch_size = 200
    min_loss = np.inf
    loss_history = []
    iterations_per_decay = 1000
    PATH = FOLDER+"output/p_net.pt"
    PATH_LOSS = FOLDER+"output/p_net_train_loss.npy"

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)
        u_bc = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        x = (torch.rand(len(t), 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = res_func(x, t, p_net)/max_abs_p_ti
        mse_res = mse_cost_function(res_out, all_zeros)

        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input/max_abs_p_ti, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # <Baseline>
        loss = mse_u + mse_res + mse_norm_res_input

        # Save the min loss model
        if(loss.data < min_loss):
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

        loss_history.append(loss.data)
        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data, 
                      ", Time domain:", min(t).data, ",", max(t).data,
                      ", X domain:", min(x[:,0]).data, ",", max(x[:,0]).data)
                np.save(PATH_LOSS, np.array(loss_history))


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

    # Plot each subplot with the shared y-axis limits and add grid
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
    # Plot each subplot with the shared y-axis limits
    for i, (ax1, res) in enumerate(zip(axs, res_list)):
        if i == 0:
            ax1.plot(x, res, "red", linestyle="--", label=r"$r_1$")
            ax1.set_ylim(global_min, global_max)
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
        neurons = 100
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
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
    
    
class E1NetFourier(nn.Module):
    def __init__(self, degree=3, scale=1.0):
        super(E1NetFourier, self).__init__()
        neurons = 100
        self.degree = degree
        self.scale = scale
        self.hidden_layer1 = nn.Linear(2, neurons)
        self.hidden_layer2 = nn.Linear(neurons, neurons)
        self.hidden_layer3 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 3 * self.degree + 1)
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)  # Final shape (batch_size, 2)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        H1 = self.output_layer(layer3_out)
        # Extract H1_0
        H1_0 = H1[:, 0].unsqueeze(1)  # shape: (batch_size, 1)
        # Extract H1_k, H1_2k-1, H1_2k for k = 1 to degree
        H1_sin_cos = H1[:, 1:].view(-1, self.degree, 3)  # shape: (batch_size, degree, 3)
        # H1_1, H1_2, ..., H1_{3*degree-2}
        H1_sin_amplitude = H1_sin_cos[:, :, 0]  # shape: (batch_size, degree)
        # H1_2, H1_5, ..., H1_{3*degree-1}
        H1_frequency = H1_sin_cos[:, :, 1]  # shape: (batch_size, degree)
        # H1_3, H1_6, ..., H1_{3*degree}
        H1_cos_amplitude = H1_sin_cos[:, :, 2]  # shape: (batch_size, degree)
        # Compute the sine and cosine terms
        sin_terms = H1_sin_amplitude * torch.sin(H1_frequency * x)
        cos_terms = H1_cos_amplitude * torch.cos(H1_frequency * x)
        # Sum the sine and cosine terms
        sin_cos_sum = (sin_terms + cos_terms).sum(dim=1, keepdim=True)
        # Compute the output
        output = H1_0 / (self.degree + 1)
        output += sin_cos_sum / (self.degree + 1)
        output = output*self.scale
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)


class E1NetFourierFix(nn.Module):
    def __init__(self, scale=1.0):
        neurons = 60
        self.scale = scale
        super(E1NetFourierFix, self).__init__()
        self.hidden_layer1 = nn.Linear(2, neurons)
        self.hidden_layer2 = nn.Linear(neurons, neurons)
        self.hidden_layer3 = nn.Linear(neurons, neurons)
        self.hidden_layer4 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 11*2)  # Change output size to 5
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        H = self.output_layer(layer4_out)
        # Calculating the final output based on the specified formula
        H1 = H[:,0].view(-1,1) \
            + H[:,1].view(-1,1) * torch.sin(0.25*x) +   H[:,2].view(-1,1) * torch.cos(0.25*x) \
            + H[:,3].view(-1,1) * torch.sin(0.5*x) +   H[:,4].view(-1,1) * torch.cos(0.5*x) \
            + H[:,5].view(-1,1) * torch.sin(x) +   H[:,6].view(-1,1) * torch.cos(x) \
            + H[:,7].view(-1,1) * torch.sin(2*x) + H[:,8].view(-1,1) * torch.cos(2*x) \
            + H[:,9].view(-1,1) * torch.sin(4*x) + H[:,10].view(-1,1) * torch.cos(4*x)
        H2 = H[:,11].view(-1,1) \
            + H[:,12].view(-1,1) * torch.sin(0.25*t) +   H[:,13].view(-1,1) * torch.cos(0.25*t) \
            + H[:,14].view(-1,1) * torch.sin(0.5*t) +   H[:,15].view(-1,1) * torch.cos(0.5*t) \
            + H[:,16].view(-1,1) * torch.sin(t) +   H[:,17].view(-1,1) * torch.cos(t) \
            + H[:,18].view(-1,1) * torch.sin(2*t) + H[:,19].view(-1,1) * torch.cos(2*t) \
            + H[:,20].view(-1,1) * torch.sin(4*t) + H[:,21].view(-1,1) * torch.cos(4*t)
        output = self.scale * H1 * H2  # Ensure output shape is (batch_size, 1)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)


class E1NetFourierEmbed(nn.Module):
    def __init__(self, m=10, sigma=1.0, constant=1.0):
        super(E1NetFourierEmbed, self).__init__()
        self.m = m
        self.sigma = sigma
        self.constant = constant
        # Sample bm vectors from a normal distribution
        self.b = torch.randn((2, m), requires_grad=False) * sigma
        neurons = 32
        self.hidden_layer1 = nn.Linear(2 * m, neurons)
        self.hidden_layer2 = nn.Linear(neurons, neurons)
        self.hidden_layer3 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 1)
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x, t):
        # Concatenate inputs x and t along the second dimension
        v = torch.cat((x, t), dim=1)
        # Apply Fourier feature encoding
        gamma_v = self.fourier_feature_encoding(v)
        # Pass through MLP
        layer1_out = torch.tanh(self.hidden_layer1(gamma_v))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out)
        return output*self.constant
    def fourier_feature_encoding(self, v):
        # Compute the dot product b*v and apply sin and cos
        bv = torch.matmul(v, self.b)
        sin_bv = torch.sin(bv)
        cos_bv = torch.cos(bv)
        # Concatenate sin and cos features
        gamma_v = torch.cat((sin_bv, cos_bv), dim=1)
        return gamma_v
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)


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


def train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_x_0, iterations=40000):
    batch_size = 400
    iterations_per_decay = 1000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    PATH_LOSS = FOLDER+"output/e1_net_train_loss.npy"

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t_bc = 0*x_bc + t0
        p_bc = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out/max_abs_e1_x_0, u_bc/max_abs_e1_x_0)

        # Loss based on residual
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        x = (torch.rand(len(t), 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e1_res_func(x, t, e1_net, p_net)/max_abs_e1_x_0
        mse_res = mse_cost_function(res_out, all_zeros)

        # res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_input = torch.cat([res_x, res_t], axis=1)
        # norm_res_input = torch.norm(res_input/max_abs_e1_x_0, dim=1).view(-1,1)
        # mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)
        
        # Combining the loss functions
        loss = mse_u + mse_res #+ mse_norm_res_input

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data,
                  #",res freq:", mse_norm_res_input.data
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

        loss_history.append(loss.data)
        loss.backward() 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
                np.save(PATH_LOSS, np.array(loss_history))


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

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    # Determine global min and max for y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    alpha_1_list = []
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

    fig, axs = plt.subplots(6, 1, figsize=(6, 8))
    for i, (t1, ax1, e1, e1_hat, alpha_1) in enumerate(zip(t1s, axs, e1_list, e1_hat_list, alpha_1_list)):
        error_bound = 2.0 * max(abs(e1_hat))
        if i == 0:
            ax1.plot(x, e1, "blue", label=r"$e_1$")
            ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}_1$")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
            ax1.legend(loc="upper right")
        else:
            ax1.plot(x, e1, "blue")
            ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=0*x.reshape(-1)+error_bound, y2=0*x.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
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
            # ax1.legend(loc="upper right")
        else:
            ax1.plot(x, p, "blue")
            ax1.plot(x, p_hat, "red", linestyle="--")
        #     ax1.plot(x, e1_hat, "red", linestyle="--")
            ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+error_bound, y2=p_hat.reshape(-1)-error_bound, color="green", alpha=0.3, label="Error Bound(t)")
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Add thin grid lines
        # Add text to the left top corner
        ax1.text(0.01, 0.98, "t="+str(t1)+", "+r"$\alpha_1=$"+str(np.round(alpha_1,2)), transform=ax1.transAxes, verticalalignment='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/uni_error_bound.png")
    plt.close()



def show_uniform_bound(p_net, e1_net, max_abs_e1_x_0):
    global x_low, x_hig, t0, X_sim, P_sim_t10, P_sim_t15, P_sim_t20
    x = np.arange(x_low, x_hig+0.1, 0.1).reshape(-1,1)
    x_sim_ = X_sim
    t1 = 2.0
    P_sim = P_sim_t20
    T0 = 0*x + t0
    T1 = 0*x + t1
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
    p_i = p_init(x)
    p_f = P_sim_t20

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact1 = P_sim
    e1_exact_0 = p_i - p_approx0
    # e1_net
    e1_0 = e1_net(pt_x, pt_T0).data.cpu().numpy()
    e1_1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
    e_bound_1 = max(abs(e1_1))*2
    print("Error bound:", e_bound_1)
    print("error(t0):", max(abs(e1_exact_0-e1_0))/ max(abs(e1_exact_0)))
    print("alpha1(t0):", max(abs(e1_exact_0-e1_0))/ max(abs(e1_0)))
    

    # plot unform error bound in p space at t
    plt.figure(figsize=(8,6))
    plt.plot(x, p_i, "black", linewidth=0.5, alpha=0.5, label="$p_i$")
    plt.plot(x_sim_, p_f, "red", linewidth=0.5, alpha=0.5, label="$p_f$ [Monte]")
    plt.plot(x, p_approx1, "b", linewidth=0.5, label=r"$\hat{p}(t)$")
    plt.scatter(x_sim_, p_exact1, s=2, color="blue", marker="*", label="$p(t)$ [Monte]")
    plt.fill_between(x.reshape(-1), y1=p_approx1.reshape(-1)+e_bound_1, y2=p_approx1.reshape(-1)-e_bound_1, color="blue", alpha=0.1, label="Error Bound(t)")
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('y')
    textstr = "t="+str(t1)+", Error Bound="+str(np.round(e_bound_1[0],3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.savefig(FOLDER+"figs/uniform_error_bound_t="+str(t1)+".png")
    plt.close()

    # plot unform error bound in error space at t
    plt.figure(figsize=(8,6))
    x_sim_ = x_sim_.reshape(-1,1)
    pt_x_sim = Variable(torch.from_numpy(x_sim_).float(), requires_grad=False).to(device)
    pt_t_sim = 0*pt_x_sim + t1
    e1_t = e1_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
    error_t = p_exact1.reshape(-1,1) - p_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
    alpha1_t = max(abs(error_t - e1_t))/ max(abs(e1_t))
    print("alpha1(t):", alpha1_t)
    plt.fill_between(x.reshape(-1), y1=0*p_approx1.reshape(-1)+e_bound_1, y2=0*p_approx1.reshape(-1)-e_bound_1, color="blue", alpha=0.1, label="Error Bound(t)")
    plt.scatter(x_sim_, error_t, s=5, color="blue", marker="*", label="$e_1(t)$ [Monte]")
    plt.plot(x, e1_1, "b--", label=r"$\hat{e}_1(t)$")
    plt.plot([x_low, x_hig], [-e_bound_1, -e_bound_1], "black")
    plt.plot([x_low, x_hig], [e_bound_1, e_bound_1], "black")
    plt.plot([x_low, x_hig], [0, 0], "k:", linewidth=0.5)
    textstr = "t="+str(t1)+", Error Bound="+str(np.round(e_bound_1[0],3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('error')
    plt.savefig(FOLDER+"figs/uniform_error_bound(zoom-in)_t="+str(t1)+".png")
    plt.close()
    # plot e1 approximation
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(x, e1_exact_0, "b--")
    # plt.plot(x, e1_0, "b")
    # plt.plot([x_low, x_hig], [0,0], color="black", linewidth=0.5, linestyle=":")
    # plt.subplot(2,1,2)
    # plt.plot(x, e1_1, "r")
    # plt.scatter(x_sim_, error_t, s=5, color="r", marker="*", label="$e_1(t)$ [Monte]")
    # plt.plot([x_low, x_hig], [0,0], color="black", linewidth=0.5, linestyle=":")
    # plt.savefig("figs/e1net_approx_ti_t")
    # plt.close()


def plot_p_monte():
    plt.figure()
    for t1 in t1s:
        x_sim = np.load(DATA_FOLDER+"xsim.npy")
        p_sim = np.load(DATA_FOLDER+"psim_t"+str(t1)+".npy")
        plt.plot(x_sim, p_sim, label="t="+str(t1))
    plt.grid()
    plt.legend()
    plt.show()
        


def main():

    FLAG_GENERATE_DATA = False
    dx_sim = 0.0
    if(FLAG_GENERATE_DATA):
        for t1 in t1s:
            x_sim, p_sim, dx = p_sol_monte(stat_sample=200000, t1=t1)
            np.save(DATA_FOLDER+"psim_t"+str(t1)+".npy", p_sim)
            np.save(DATA_FOLDER+"xsim.npy", x_sim)
            dx_sim = dx
        # Plot generated data
        plot_p_monte()

    max_pi = test_p_init()
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    p_net = Net().to(device)
    p_net.apply(init_weights)
    optimizer = torch.optim.Adam(p_net.parameters())
    # optimizer = torch.optim.Adamax(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_pi, iterations=80000); print("[p_net train complete]")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,0):", max_abs_e1_ti)

    e1_net = E1Net(scale=max_abs_e1_ti).to(device)
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    # # optimizer = torch.optim.Adamax(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_ti, iterations=100000); print("[e1_net train complete]")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_net_results(p_net, e1_net)

    print("[complete rational nonlinear]")


if __name__ == "__main__":
    main()