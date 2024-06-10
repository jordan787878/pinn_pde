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
FOLDER = "exp/tmp/"
device = "cpu"
print(device)

mu = -2
std = 1
a = -0.1
b = 0.1
c = 0.0
d = 1
e = 1
x_low = -5
x_hig = 5

t0 = 0
T_end = 2
dt = 0.1

X_sim = np.load("data/exp1/X_sim.npy")
P_sim_t01 = np.load("data/exp1/P_sim_t01.npy")
P_sim_t05 = np.load("data/exp1/P_sim_t05.npy")
P_sim_t10 = np.load("data/exp1/P_sim_t10.npy")
P_sim_t11 = np.load("data/exp1/P_sim_t11.npy")
P_sim_t15 = np.load("data/exp1/P_sim_t15.npy")
P_sim_t20 = np.load("data/exp1/P_sim_t20.npy")


# def f_sde(x):
#   global a, b, c, d
#   return a*(x**3) + b*(x**2) + c*x + d


def p_init(x):
  global mu, std
  return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def res_func(x,t, net, verbose=False):
    global a, b, c, d, e
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*a*x*x + 2*b*x + c)*p + (a*x*x*x + b*x*x + c*x + d)*p_x - 0.5*e*e*p_xx
    return residual


# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


# p_net
class Net(nn.Module):
    def __init__(self):
        neurons = 32
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus(self.hidden_layer1(inputs))
        layer2_out = F.softplus(self.hidden_layer2(layer1_out))
        layer3_out = F.softplus(self.hidden_layer3(layer2_out))
        layer4_out = F.softplus(self.hidden_layer4(layer3_out))
        layer5_out = F.softplus(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        # output = torch.square(output)
        output = F.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)

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

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, PATH, PATH_LOSS, iterations=40000):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    min_loss = np.inf
    loss_history = []
    dx_train = 0.01
    dt_train = 0.01
    dt_step = 0.01
    iterations_per_decay = 1000

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        # x_quad = torch.arange(x_low, x_hig+dx_train, dx_train, dtype=torch.float, requires_grad=True).view(-1, 1).to(device)
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        # x_bc = torch.cat((x_quad, x_bc), dim=0)
        t_bc = (torch.ones(len(x_bc), 1) * t0).to(device)
        u_bc = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out, u_bc)
        linf_u = linf_loss(net_bc_out, u_bc)/ torch.max(torch.abs(net_bc_out))

        # Loss based on PDE
        # t_quad = torch.arange(t0, T_end+dt_train, dt_train, dtype=torch.float, requires_grad=True).view(-1, 1).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        # t = torch.cat((t_quad, t, t+dt_step, t+dt_step*2), dim=0)
        x = (torch.rand(len(t), 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        # t = torch.cat((t, t_bc), dim=0)
        # x = torch.cat((x, x_bc), dim=0)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = res_func(x, t, p_net)
        mse_f = mse_cost_function(f_out, all_zeros)
        linf_res = linf_loss(f_out, all_zeros)

        res_x = torch.autograd.grad(f_out, x, grad_outputs=torch.ones_like(f_out), create_graph=True)[0]
        res_t = torch.autograd.grad(f_out, t, grad_outputs=torch.ones_like(f_out), create_graph=True)[0]
        mse_res_x = mse_cost_function(res_x, all_zeros)
        mse_res_t = mse_cost_function(res_t, all_zeros)
        linf_res_x = linf_loss(res_x, all_zeros)
        linf_res_t = linf_loss(res_t, all_zeros)

        # <Baseline>
        # loss = torch.max(linf_u, linf_u*0+0.05) + linf_res + 1e-4*linf_res_x + 1e-4*linf_res_t
        # <Mse>
        loss = mse_u + mse_f
        # <Linf>
        # loss = linf_u + linf_res
        # <LinfDerivative>
        # loss = linf_u + linf_res + linf_res_x + linf_res_t
        # <LinfDerivativeWeighted>
        # loss = linf_u + linf_res + 1e-4*linf_res_x + 1e-4*linf_res_t
        # <LinfDerivativeWeighted02>
        # loss = linf_u + linf_res + 1e-2*linf_res_x + 1e-2*linf_res_t
        # <LinfDerivativeSatIC>
        # loss = torch.max(linf_u, linf_u*0+0.05) + linf_res + linf_res_x + linf_res_t
        # <LinfDerivativeSatICWeighted>
        # loss = torch.max(linf_u, linf_u*0+0.05) + linf_res + 1e-4*linf_res_x + 1e-4*linf_res_t
        # <superp>
        # loss = mse_u + mse_f + 1e-5*(linf_res_x + linf_res_t)
        # <linfresfreq>
        num_zero_res_x_smooth = count_approx_zero_elements(res_x)
        # loss = torch.max(linf_u, linf_u*0+0.05) + torch.max(linf_res, linf_res*0+0.01) + 1e-4*(num_zero_res_x_smooth)/batch_size + 1e-5*linf_res_t

        # Save the min loss model
        if(loss.data < min_loss):
            print("save epoch:", epoch, ", loss:", loss.data, ", ic:",mse_u.data, ", res:",mse_f.data, ",l-inf ic:",
                  linf_u.data, ",l-inf res:",linf_res.data, ",D res:", linf_res_x.data, linf_res_t.data, mse_res_x.data, mse_res_t.data,
                  "res freq:", num_zero_res_x_smooth.data)
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
                print(epoch,"Traning Loss:",loss.data)
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
    global x_low, x_hig, X_sim, P_sim_t20, t0
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 2
    P_sim = P_sim_t20
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_init(x)
    p_exact1 = P_sim

    plt.figure()
    plt.plot(x, p_exact0,  "k:", linewidth=5, label="$p(t_i)$")
    plt.plot(x, p_approx0, "k--", label=r"$\hat{p}(t_i)$")
    plt.plot(X_sim, p_exact1,  "r:", linewidth=5, label="$p(t_f)$ [Monte]")
    plt.plot(x, p_approx1, "r--", label=r"$\hat{p}(t_f)$")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.savefig(FOLDER+"figs/pnet_approx.png")
    plt.close()

    res_0 = res_func(pt_x, pt_T0, p_net)
    res_1 = res_func(pt_x, pt_T1, p_net)
    plt.figure()
    plt.plot(x, res_0.detach().numpy(), "black", label="$r_1(t_i)$")
    plt.plot(x, res_1.detach().numpy(), "red", label="$r_1(t_f)$")
    plt.plot([x_low, x_hig], [0,0], "black")
    plt.legend()
    plt.savefig(FOLDER+"figs/pnet_resdiual.png")
    plt.close()

    max_abs_e1_x_0 = max(abs(p_exact0-p_approx0))
    print(max_abs_e1_x_0[0])

    return max_abs_e1_x_0[0]


class E1Net(nn.Module):
    def __init__(self, scale=1.0):
        neurons = 100
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh((self.hidden_layer1(inputs)))
        layer2_out = torch.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = torch.tanh((self.hidden_layer3(layer2_out)))
        layer4_out = torch.tanh((self.hidden_layer4(layer3_out)))
        layer5_out = torch.tanh((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = self.scale*output
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)
    
    
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
    global a, b, c, d, e
    net_out = e1_net(x,t)
    net_out_x = torch.autograd.grad(net_out, x, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_t = torch.autograd.grad(net_out, t, grad_outputs=torch.ones_like(net_out), create_graph=True)[0]
    net_out_xx = torch.autograd.grad(net_out_x, x, grad_outputs=torch.ones_like(net_out_x), create_graph=True)[0]
    p_res = res_func(x, t, p_net)
    residual = net_out_t + (3*a*x*x + 2*b*x + c)*net_out + (a*x*x*x + b*x*x + c*x + d)*net_out_x - 0.5*e*e*net_out_xx + p_res
    return residual


def test_e1_res(e1_net, p_net):
    batch_size = 5
    x_collocation = np.random.uniform(low=1.0, high=3.0, size=(batch_size,1))
    t_collocation = T_end*np.ones((batch_size,1))
    all_zeros = np.zeros((batch_size,1))
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    f_out = e1_res_func(pt_x_collocation, pt_t_collocation, e1_net, p_net, verbose=True) # output of f(x,t)


def augment_tensors(x_tensor, t_tensor):
    # Get the size of the input tensors
    M = x_tensor.size(0)
    N = t_tensor.size(0)
    augmented_x_tensor = x_tensor.repeat(N, 1).view(N*M, 1)
    augmented_t_tensor = t_tensor.repeat_interleave(M).view(N*M, 1)
    return augmented_x_tensor, augmented_t_tensor


def train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_x_0, PATH, PATH_LOSS, iterations=40000):
    global x_low, x_hig, t0, T_end
    batch_size = 200
    iterations_per_decay = 1000
    min_loss = np.inf
    loss_history = []
    
    dx_train = 0.1
    dt_train = 0.1
    dt_step = 0.02
    x_mar = 0
    t_mar = 0
    normalized = 1.0

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_quad = torch.arange(x_low-x_mar, x_hig+dx_train+x_mar, dx_train, dtype=torch.float, requires_grad=True).view(-1, 1).to(device)
        x_bc = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        x_bc = torch.cat((x_quad, x_bc), dim=0)
        t_bc = 0*x_bc + t0
        p_bc = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on residual
        t_quad = torch.arange(t0, T_end+dt_train, dt_train, dtype=torch.float, requires_grad=True).view(-1, 1).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0 + 2*t_mar) + t0-t_mar).to(device)
        t = torch.cat((t_quad, t, t+dt_step, t+dt_step*2), dim=0)
        x = (torch.rand(len(t), 1, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        # t = torch.cat((t, t_bc), dim=0)
        # x = torch.cat((x, x_bc), dim=0)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = e1_res_func(x, t, e1_net, p_net)
        # e1_out_for_res = e1_net(x, t)*max_abs_e1_x_0
        mse_res = mse_cost_function(f_out, all_zeros)
        l_inf_res = linf_loss(f_out, all_zeros)

        # Loss based on time integral of residual
        # t = (torch.rand(10, 1, requires_grad=True) * (T_end - t0 + 2*t_mar) + t0-t_mar).to(device)
        # t = torch.cat((t_quad, t), dim=0)
        # # x_quad = torch.arange(x_low-x_mar, x_hig+3+x_mar, 3, dtype=torch.float, requires_grad=True).view(-1, 1).to(device)
        # aug_x, aug_t = augment_tensors(x_quad, t)
        # res_aug = e1_res_func(aug_x, aug_t, e1_net, p_net, max_abs_e1_x_0).view(t.size(0), x_quad.size(0))
        # e1_aug = e1_net(aug_x, aug_t).view(t.size(0), x_quad.size(0))
        # res_aug_max = torch.max(torch.abs(res_aug), dim=1)
        # e1_aug_max =  torch.max(torch.abs(e1_aug), dim=1)
        # quotient = res_aug_max.values/e1_aug_max.values
        # quotient_max = torch.max(quotient)  #; print(quotient.shape, torch.max(quotient))

        # Loss based on alpha_1(t0)
        x = x_bc
        t = 0*x + t0
        p = p_init(x.detach().numpy())
        p = Variable(torch.from_numpy(p).float(), requires_grad=False).to(device)
        phat = p_net(x, t)
        e1 = p - phat
        e1_hat = e1_net(x, t)
        error1_t0 = linf_loss(e1_hat, e1) / max_abs_e1_x_0
        alpha1_t0 = linf_loss(e1_hat, e1)/ torch.max(torch.abs(e1_hat))
        
        # Combining the loss functions
        loss = normalized*mse_u/max_abs_e1_x_0 + normalized*mse_res/max_abs_e1_x_0
        # [NEW] denominator for l-inf-res
        # loss = normalized*(alpha1_t0 + error1_t0 + l_inf_res/max_abs_e1_x_0 + 1e-3*quotient_max )
        # loss = normalized*torch.max(alpha1_t0, 0*alpha1_t0+0.1) + normalized*torch.max(error1_t0, 0*alpha1_t0+0.1) + 1e-1*(normalized*l_inf_res/max_abs_e1_x_0 + 1e-4*normalized*quotient_max)
        # loss = normalized*max(alpha1_t0, error1_t0, l_inf_res) + 1e-2*quotient_max
        # loss = max(max(alpha1_t0, 0.1), max(error1_t0, 0.1), max(l_inf_res, 0.001), max(quotient_max, 0.1))
        # loss = normalized*(max(alpha1_t0, 0.1) + max(error1_t0, 0.1) + l_inf_res)
        # loss = normalized*(alpha1_t0+error1_t0+mse_res/max_abs_e1_x_0)
        # <tmp>
        # loss = torch.max(error1_t0, error1_t0*0+0.1) + torch.max(alpha1_t0, alpha1_t0*0+0.1) + mse_res

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data/normalized, ",ic loss:", mse_u.data,
                  ",res:", mse_res.data, l_inf_res.data,# quotient_max.data,
                  ",a1(t0):", alpha1_t0.data, "error(t0):",error1_t0.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
            # print("debug", max(abs(e1-e1_hat)).data, max_abs_e1_x_0)
            # print("e1hat_t0,", x[0].data, t[0].data, e1_hat[0].data)

        loss_history.append(loss.data)
        loss.backward() 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data/normalized)
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
    # print(loss_history)
    print("save e1net loss history")
    return e1_net


def show_e1_net(p_net, e1_net, max_abs_e1_x_0):
    a1_t0 = 0.0095
    global x_low, x_hig, t0, X_sim, P_sim_t01, P_sim_t05, P_sim_t10, P_sim_t15, P_sim_t20
    x = np.arange(x_low, x_hig+0.1, 0.1).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    x_sim_ = X_sim
    t1s = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    plt.figure(figsize=(10,8))
    plt.subplot(3,2,1)
    p_exact_t = p_init(x).reshape(-1,1)
    T1 = 0*x + 0
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
    p_approx_t = p_net(pt_x, pt_T1).data.cpu().numpy()
    e1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
    plt.plot(x, p_exact_t-p_approx_t, color="blue", marker='o', markersize=2, linewidth=0.3, label="$e_1$")
    plt.plot(x, e1, linestyle="-", linewidth=1, color="blue", label=r"$\hat{e}_1$")
    plt.plot(x_sim_, x_sim_*0.0, color="black", linestyle=":", linewidth=0.3)
    plt.text(0.98, 0.02, "t=0"+ r"$, \alpha_1=$"+str(np.round(a1_t0,2)), transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right')
    plt.legend(loc='upper right')

    for t1 in t1s:
        if t1 == 0.1:
            p_exact_t = P_sim_t01
            plt.subplot(3,2,2)
        if t1 == 0.5:
            p_exact_t = P_sim_t05
            plt.subplot(3,2,3)
        if t1 == 1.0:
            p_exact_t = P_sim_t10
            plt.subplot(3,2,4)
        if t1 == 1.5:
            p_exact_t = P_sim_t15
            plt.subplot(3,2,5)
            plt.xlabel('x')
        if t1 == 2.0:
            p_exact_t = P_sim_t20
            plt.subplot(3,2,6)
            plt.xlabel('x')
        T1 = 0*x + t1
        # pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
        pt_x_sim = Variable(torch.from_numpy(x_sim_.reshape(-1,1)).float(), requires_grad=False).to(device)
        pt_t_sim = 0*pt_x_sim + t1
        error_t = p_exact_t.reshape(-1,1) - p_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
        e1 = e1_net(pt_x_sim, pt_t_sim).data.cpu().numpy()
        line1, = plt.plot(x_sim_, e1, linestyle="-", linewidth=1, color="blue", label="Approx")
        plt.plot(x_sim_, error_t, color=line1.get_color(), marker='o', markersize=2, linewidth=0.3, label="True")
        plt.plot(x_sim_, x_sim_*0.0, color="black", linestyle=":", linewidth=0.3)
        a1 = np.array(max(abs(error_t - e1))/ max(abs(e1)))
        plt.text(0.98, 0.02, "t="+str(t1)+ r"$, \alpha_1=$"+str(np.round(a1[0],2)), transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right')

        # plt.ylabel("a1:"+str(np.round(a1[0],2)))
    plt.savefig(FOLDER+"figs/e1net_result")
    plt.close()

    # plt.figure()
    # t = np.linspace(0.0, 2.0, 100).reshape(-1,1)
    # pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
    # for x_eval in [-5, 0, 5]:
    #     x = x_eval *np.ones((100,1)).reshape(-1,1)
    #     pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    #     e1_res_x = e1_res_func(pt_x, pt_t, e1_net, p_net, max_abs_e1_x_0).data.cpu().numpy()
    #     cum_sum = np.cumsum(e1_res_x)
    #     plt.plot(t, cum_sum, label="x="+str(x_eval))
    # plt.title("time integral of r2(x,t) at x")
    # plt.xlabel('t')
    # plt.legend(loc='upper right')
    # plt.show()


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

# Run this next time: 06.01
def main():
    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    # create p_net
    p_net = Net().to(device)
    optimizer = torch.optim.Adam(p_net.parameters())
    # optimizer = torch.optim.Adamax(p_net.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_p_net(p_net, optimizer, scheduler, mse_cost_function, 
                 PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy", iterations=200000); print("[p_net train complete]")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_x_0 = show_p_net_results(p_net)
    print("max abs e1(x,0):", max_abs_e1_x_0)



    # create e1_net
    e1_net = E1Net(scale=max_abs_e1_x_0).to(device)
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adamax(e1_net.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_x_0, 
                 PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy", iterations=80000); print("[e1_net train complete]")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_uniform_bound(p_net, e1_net, max_abs_e1_x_0)
    show_e1_net(p_net, e1_net, max_abs_e1_x_0)

    print("[complete rational nonlinear]")


if __name__ == "__main__":
    main()