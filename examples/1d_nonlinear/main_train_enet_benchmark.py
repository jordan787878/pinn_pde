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
from scipy.optimize import curve_fit
import argparse

FOLDER = "exp1/main_benchmark/"
DATA_FOLDER = "exp1/data/"

device = "cpu"

# Set a fixed seed for reproducibility
seed = 0

n_d = 1
mu = -2.0
std = 0.5
const_a = -0.1
const_b = 0.1
const_c = 0.5
const_d = 0.5
const_e = 0.8
x_low = -6
x_hig = 6

t0 = 0.0
T_end = 5.0
t1s = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

datas = ["data1/"]
enet_terminate = 5e-5
MAX_EPOCHS = 100000


def p_init(x):
    return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def p_init_torch(x):
    # Ensure x is a torch tensor
    x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Compute the Gaussian function
    exponent = -0.5 * ((x - mu) / std) ** 2
    normalization = std * torch.sqrt(torch.tensor(2 * torch.pi))
    result = torch.exp(exponent) / normalization

    return result


def p_res_func(x, t, pnet, verbose=False):
    p = pnet(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*const_a*x*x + 2*const_b*x + const_c)*p \
                   + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*p_x \
                   - 0.5*const_e*const_e*p_xx
    if(verbose):
        print(p_xx[0:10,:]) #; print(p_x.shape, p_t.shape, p_xx.shape, residual.shape)
    return residual


def e_res_func(x, t, e_net, p_net, verbose=False):
    e1_out = e_net(x, t)[:, 0].view(-1, 1)
    p_res = p_res_func(x, t, p_net)
    e_x = torch.autograd.grad(e1_out, x, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_t = torch.autograd.grad(e1_out, t, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x,  x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    residual = p_res + e_t + (3*const_a*x*x + 2*const_b*x + const_c)*e1_out \
                     + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*e_x \
                     - 0.5*const_e*const_e*e_xx
    if(verbose):
        print(e_x.shape, e_t.shape, e_xx.shape, residual.shape)
    return residual


def Diff_e_func(x, t, e_net, verbose=False):
    e1_out = e_net(x, t)[:, 0].view(-1, 1)
    e_x = torch.autograd.grad(e1_out, x, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_t = torch.autograd.grad(e1_out, t, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x,  x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    residual =  e_t + (3*const_a*x*x + 2*const_b*x + const_c)*e1_out \
                    + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*e_x \
                    - 0.5*const_e*const_e*e_xx
    if(verbose):
        print(e_x.shape, e_t.shape, e_xx.shape, residual.shape)
    return residual


def init_weights(m):
    if isinstance(m, nn.Linear):
        # print("init")
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.normal_(m.weight, std=0.01)
        # print(m.weight)
        m.bias.data.fill_(0.01)
# def init_weights_He(model):
#     """Initialize weights of the model using He initialization."""
#     for layer in model.modules():
#         if isinstance(layer, nn.Linear):
#             # He initialization: Uniform distribution
#             nn.init.normal_(layer.weight, std=0.01)


# class PNet(nn.Module):
#     def __init__(self, scale=1.0):
#         super(PNet, self).__init__()
#         self.scale = scale
#         num_hidden_layers=20
#         neurons=20
#         # List to hold layers
#         layers = []
#         # Input layer
#         layers.append(nn.Linear(2, neurons))
#         # Hidden layers
#         for _ in range(num_hidden_layers):
#             layers.append(nn.Linear(neurons, neurons))
#         # Output layer
#         layers.append(nn.Linear(neurons, 1))
#         # Register all layers
#         self.layers = nn.ModuleList(layers)
#     def forward(self, x, t):
#         inputs = torch.cat([x, t], dim=1)
#         out = inputs
#         for layer in self.layers[:-1]:  # Apply hidden layers
#             out = F.softplus(layer(out))
#         out = self.layers[-1](out)  # Apply output layer
#         return F.softplus(out)
 
class PNet(nn.Module):
    def __init__(self, scale=1.0):
        super(PNet, self).__init__()
        self.scale = scale
        num_hidden_layers=20
        neurons=20
        # List to hold layers
        layers = []
        # Input layer
        layers.append(nn.Linear(2, neurons))
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
        # Output layer
        layers.append(nn.Linear(neurons, 1))
        # Register all layers
        self.layers = nn.ModuleList(layers)
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = inputs
        for layer in self.layers[:-1]:  # Apply hidden layers
            out = F.softplus(layer(out))
        out = self.layers[-1](out)  # Apply output layer
        return F.softplus(out)



# class ENet(nn.Module):
#     def __init__(self, scale=1.0): 
#         neurons = 50
#         self.scale = scale
#         super(ENet, self).__init__()
#         self.hidden_layer1 = (nn.Linear(2,neurons))
#         self.hidden_layer2 = (nn.Linear(neurons,neurons))
#         self.hidden_layer3 = (nn.Linear(neurons,neurons))
#         self.hidden_layer4 = (nn.Linear(neurons,neurons))
#         self.hidden_layer5 = (nn.Linear(neurons,neurons))
#         self.hidden_layer6 = (nn.Linear(neurons,neurons))
#         self.hidden_layer7 = (nn.Linear(neurons,neurons))
#         self.hidden_layer8 = (nn.Linear(neurons,neurons))
#         self.hidden_layer9 = (nn.Linear(neurons,neurons))
#         self.output_layer =  (nn.Linear(neurons,1))
#     def forward(self, x, t):
#         inputs = torch.cat([x,t], axis=1)
#         layer1_out = F.softplus((self.hidden_layer1(inputs)))
#         layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
#         layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
#         layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
#         layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
#         layer6_out = F.softplus((self.hidden_layer6(layer5_out)))
#         layer7_out = F.softplus((self.hidden_layer7(layer6_out)))
#         layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
#         layer9_out = F.softplus((self.hidden_layer9(layer8_out)))
#         output = self.scale * (self.output_layer(layer9_out))
#         return output


class ENet(nn.Module):
    def __init__(self, scale=1.0):
        super(ENet, self).__init__()
        self.scale = scale
        num_hidden_layers=30
        neurons=40
        # Define a list to hold the layers
        layers = []
        # Input layer
        layers.append(nn.Linear(2, neurons))
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
        # Output layer
        layers.append(nn.Linear(neurons, 1))
        # Register all layers as a ModuleList
        self.layers = nn.ModuleList(layers)
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = inputs
        for layer in self.layers[:-1]:  # Apply hidden layers
            out = F.softplus(layer(out))
        out = self.layers[-1](out)  # Apply output layer
        return self.scale * out 


def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def get_e1_normalize(pnet):
    x = np.linspace(x_low, x_hig, num=200, endpoint=True).reshape(-1,1)
    p0_true = p_init(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t = pt_x*0.0 + t0
    p0_hat = pnet(pt_x, pt_t).data.cpu().numpy()
    e0_true = p0_true - p0_hat
    return np.max(np.abs(e0_true))


def train_pnet_model(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global x_low, x_hig, t0, T_end
    PATH = FOLDER+"output/p_net.pth"
    ti = t0; tf = T_end
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    x_mar = 0.0
    
    # space-time points for BC
    x_bc = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t_bc = (torch.ones(len(x_bc), 1, requires_grad=True) * ti).to(device)
    
    # space-time points for RES
    # x = (torch.rand(3500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    # t = (torch.rand(2500, 1, requires_grad=True)   * (tf - ti) + ti).to(device)
    # _t_init = (torch.ones(500, 1, requires_grad=True) * ti).to(device)
    # _t_end =  (torch.ones(500, 1, requires_grad=True) * tf).to(device)
    # t = torch.cat((t, _t_init, _t_end), dim=0)
    _x = np.linspace(x_low, x_hig, num=70, endpoint=True)
    _t = np.linspace(ti, tf, num=70, endpoint=True)
    _xx, _tt = np.meshgrid(_x, _t)
    x = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device)
    t = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device)

    max_abs_p_ti = p_net.scale

    # RAR
    weight_regular = 1.0
    S = 10000
    FLAG = False

    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # IC Loss
        p0 = p_init_torch(x_bc)
        p0_target = p0.clone().detach().numpy()
        p0_target = Variable(torch.from_numpy(p0_target).float(), requires_grad=False).to(device)
        p0_hat =  p_net(x_bc, t_bc)
        # e0     = p0 - p0_hat
        # e0_x = torch.autograd.grad(e0, x_bc, grad_outputs=torch.ones_like(e0), create_graph=True)[0]
        all_zeros = torch.zeros((len(x_bc),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_u = mse_cost_function(p0_hat/max_abs_p_ti, p0_target/max_abs_p_ti)
        # mse_u_grad = torch.mean((e0_x/max_abs_p_ti)**2)

        # PDE Loss
        res_out = p_res_func(x, t, p_net)/max_abs_p_ti
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Res grad Loss
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        mse_res_g = torch.mean(res_x**2 +res_t**2)

        # Loss Function
        loss = mse_u + mse_res + weight_regular*mse_res_g

        # RAR
        if (epoch%100 == 0 and FLAG):
            # t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            # _t_init_RAR = (torch.ones(500, 1, requires_grad=True) * ti).to(device)
            # _t_end_RAR = (torch.ones(500, 1, requires_grad=True) * tf).to(device)
            # t_RAR = torch.cat((t_RAR, _t_init_RAR, _t_end_RAR), dim=0)
            # x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            # x_RAR = torch.clamp(x_RAR, min=x_low, max=x_hig)

            quad_number = random.randint(10,30)
            _x = np.linspace(x_low, x_hig, num=quad_number, endpoint=True)
            _t = np.linspace(ti, tf, num=quad_number, endpoint=True)
            _xx, _tt = np.meshgrid(_x, _t)
            x_quad = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device)
            t_quad = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = torch.cat((t_RAR, t_quad), dim=0)
            x_RAR = torch.cat((x_RAR, x_quad), dim=0)

            t0_RAR = 0.0*t_RAR.clone() + t0
            x0_RAR = x_RAR.clone()

            p0_RAR = p_init_torch(x0_RAR)
            p0_hat_RAR = p_net(x0_RAR, t0_RAR)
            e0_RAR = p0_RAR - p0_hat_RAR
            mean_e0_RAR = torch.mean(torch.abs(e0_RAR))
            if(mean_e0_RAR > 0.0):
                max_abs_e0, max_index = torch.max(torch.abs(e0_RAR), dim=0)
                x_max = x0_RAR[max_index]
                t_max = t0_RAR[max_index]
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... Ic add [x,t]:", x_max.data, t_max.data, max_abs_e0.data)
                FLAG = False

            res_RAR = p_res_func(x_RAR, t_RAR, p_net)
            mean_res_RAR = torch.mean(torch.abs(res_RAR))
            print("mean res RAR:", mean_res_RAR.data)
            if(mean_res_RAR > 0.0):
                # Find the index of the maximum absolute value in res_RAR
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                # Get the corresponding x_RAR and t_RAR vectors
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                # Append x_max and t_max to x and t
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... Res add [x,t]:", x_max.data, t_max.data, max_abs_res.data)
                FLAG = False

            res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            mean_res_g_RAR = torch.mean(torch.abs(res_x_RAR)+torch.abs(res_t_RAR))
            print("RAR mean res input: ", mean_res_g_RAR.data)
            if(mean_res_g_RAR > 0.0):
                max_abs_res_input, max_index = torch.max(res_x_RAR**2+ res_t_RAR**2, dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... Res Grad add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)
                FLAG = False

        loss_history.append(loss.data)
        
        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            min_loss = loss.data
            FLAG = True
            training_time = time.time() - start_time
            print("pnet saved epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, ",res:", mse_res.data,
                  "res input:", mse_res_g.data,
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "p_net",
                    'train_time': training_time,
                }, PATH)
            np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))

        # Terminate training 
        if(loss.data < pnet_terminate):
            training_time = time.time() - start_time
            print("pnet saved epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, ",res:", mse_res.data,
                  "res input:", mse_res_g.data,
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "p_net",
                    'train_time': training_time,
                }, PATH)
            np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))
            return

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def train_enet_model(p_net, e1_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global x_low, x_hig, t0, T_end
    ti = t0; tf = T_end
    min_loss = np.inf
    loss_history = []
    iterations_per_decay = 1000
    PATH = FOLDER+"output/e1_net.pth"
    x_mar = 0.0

    x_bc = (torch.rand(500, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t_bc = (torch.ones(len(x_bc), 1, requires_grad=True) * ti).to(device)
    
    # x = (torch.rand(3500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    # t = (torch.rand(2500, 1, requires_grad=True)   * (tf - ti) + ti).to(device)
    # _t_init = (torch.ones(500, 1, requires_grad=True) * ti).to(device)
    # _t_end =  (torch.ones(500, 1, requires_grad=True) * tf).to(device)
    # t = torch.cat((t, _t_init, _t_end), dim=0)
    _x = np.linspace(x_low, x_hig, num=30, endpoint=True)
    _t = np.linspace(ti, tf, num=30, endpoint=True)
    _xx, _tt = np.meshgrid(_x, _t)
    x = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device)
    t = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device)
    x_rand = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t_rand = (torch.rand(1000, 1  , requires_grad=True) * (tf - ti) + ti).to(device)
    x = torch.cat((x, x_rand), dim=0)
    t = torch.cat((t, t_rand), dim=0)
    
    weight_reg = 1.0
    FLAG = False
    S = 10000
    normalize = e1_net.scale

    # Store alpha data (prepare)
    Nsample_list = []
    Alpha_list = []
    Alpha_mean_list = []
    Loss_1_list = []
    Loss_2_list = []
    E1_list = []
    x_data = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_data = Variable(torch.from_numpy(x_data).float(), requires_grad=False).to(device)
    for t1 in t1s:
        p_data = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_t_data = Variable(torch.from_numpy(x_data*0+t1).float(), requires_grad=True).to(device)
        phat = p_net(pt_x_data, pt_t_data).data.cpu().numpy() # change tensor to numpy
        e1 = p_data - phat
        E1_list.append(e1)

    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        p0 = p_init_torch(x_bc)
        p0_hat  =  p_net(x_bc, t_bc)
        e10     = p0 - p0_hat
        e10_target = e10.clone().detach().numpy()
        e10_target = Variable(torch.from_numpy(e10_target).float(), requires_grad=False).to(device)
        e10_hat = e1_net(x_bc, t_bc)[:,0].view(-1,1)
        mse_e1_ic = mse_cost_function(e10_hat/normalize, e10/normalize)
        
        # using detached p_net
        diff_e = Diff_e_func(x, t, e1_net)
        diff_e_target = -p_res_func(x, t, p_net).detach().numpy()
        diff_e_target = Variable(torch.from_numpy(diff_e_target).float(), requires_grad=False).to(device)
        mse_e1_res = mse_cost_function(diff_e/normalize, diff_e_target/normalize)
        # using no detached p_net
        # all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        # e1_res_out = e_res_func(x, t, e1_net, p_net)/normalize
        # mse_e1_res = mse_cost_function(e1_res_out, all_zeros)

        # e1_res_out = e_res_func(x, t, e1_net, p_net)/normalize
        # res_x = torch.autograd.grad(e1_res_out, x, grad_outputs=torch.ones_like(e1_res_out), create_graph=True)[0]
        # res_t = torch.autograd.grad(e1_res_out, t, grad_outputs=torch.ones_like(e1_res_out), create_graph=True)[0]
        # mse_res_grad = torch.mean(res_x**2+res_t**2)
    
        # Combining the loss functions
        loss = mse_e1_ic + mse_e1_res #+ weight_reg*mse_res_grad

        # RAR
        if (epoch%100 == 0 and FLAG):
            quad_number = random.randint(10,30)
            _x = np.linspace(x_low, x_hig, num=quad_number, endpoint=True)
            _t = np.linspace(ti, tf, num=quad_number, endpoint=True)
            _xx, _tt = np.meshgrid(_x, _t)
            x_quad = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device)
            t_quad = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device)

            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            # _t_init_RAR = (torch.ones(500, 1, requires_grad=True) * ti).to(device)
            # _t_end_RAR = (torch.ones(500, 1, requires_grad=True) * tf).to(device)
            # t_RAR = torch.cat((t_RAR, _t_init_RAR, _t_end_RAR), dim=0)
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = torch.cat((t_RAR, t_quad), dim=0)
            x_RAR = torch.cat((x_RAR, x_quad), dim=0)
            # x_RAR = torch.clamp(x_RAR, min=x_low, max=x_hig)

            # t0_RAR = 0.0*t_RAR.clone() + t0
            # x0_RAR = x_RAR.clone()
            # p0_RAR = p_init_torch(x0_RAR)
            # p0_hat_RAR = p_net(x0_RAR, t0_RAR)
            # e10_RAR = p0_RAR - p0_hat_RAR
            # e10_hat_RAR = e1_net(x0_RAR, t0_RAR)
            # mean_e0_RAR = torch.mean(torch.abs(e10_RAR/normalize-e10_hat_RAR/normalize))
            # if(mean_e0_RAR > 0.0):
            #     max_abs_e0, max_index = torch.max(torch.abs(e10_RAR/normalize-e10_hat_RAR/normalize), dim=0)
            #     x_max = x0_RAR[max_index]
            #     t_max = t0_RAR[max_index]
            #     x_bc = torch.cat((x_bc, x_max), dim=0)
            #     t_bc = torch.cat((t_bc, t_max), dim=0)
            #     print("... Ic add [x,t]:", x_max.data, t_max.data, max_abs_e0.data)

            res_RAR = e_res_func(x_RAR, t_RAR, e1_net, p_net)/normalize
            mean_res_error = torch.mean(torch.abs(res_RAR))
            print("RAR mean res: ", mean_res_error.data)
            if(mean_res_error > 0.0):
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... Res add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)

            # res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # mean_res_grad = torch.mean(torch.abs(res_x_RAR)+torch.abs(res_t_RAR))
            # print("RAR mean res input: ", mean_res_grad.data)
            # if(mean_res_grad > 0.0):
            #     max_abs_res_input, max_index = torch.max(res_x_RAR**2+ res_t_RAR**2, dim=0)
            #     x_max = x_RAR[max_index].view(-1,1)
            #     t_max = t_RAR[max_index].view(-1,1)
            #     x = torch.cat((x, x_max), dim=0)
            #     t = torch.cat((t, t_max), dim=0)
            #     print("... Res Grad add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)

            FLAG = False

        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            min_loss = loss.data
            FLAG = True
            training_time = time.time() - start_time
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  "ic:", mse_e1_ic.data,
                  "res:", mse_e1_res.data,
                #   "grad:", mse_res_grad.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "e1_net",
                    'train_time': training_time,
                    }, PATH)
            np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))
            # Store alpha data (calculate data)
            Nsample_i = x_bc.shape[0] + x.shape[0]
            Loss_1_i    = (loss).data.cpu().numpy().item()
            Loss_2_i    = (0.0*loss).data.cpu().numpy().item() ###
            alpha_over_time = []
            for i in range(len(t1s)):
                t1 = t1s[i]
                pt_t_data = Variable(torch.from_numpy(x_data*0+t1).float(), requires_grad=True).to(device)
                ehat = e1_net(pt_x_data, pt_t_data).data.cpu().numpy()
                e1 = E1_list[i]
                alpha = np.max(np.abs(e1-ehat))/ np.max(np.abs(ehat))
                alpha = np.round(alpha, 3)
                alpha_over_time.append(alpha)
            max_alpha = np.max(alpha_over_time)
            Nsample_list.append(Nsample_i)
            Loss_1_list.append(Loss_1_i)
            Loss_2_list.append(Loss_2_i)
            Alpha_list.append(max_alpha)
            Alpha_mean_list.append(np.mean(alpha_over_time))
            # Store alpha data (write data)
            np.save(FOLDER+"output/e1_Nsample_list.npy", np.array(Nsample_list))
            np.save(FOLDER+"output/e1_Loss_1_list.npy", np.array(Loss_1_list))
            np.save(FOLDER+"output/e1_Loss_2_list.npy", np.array(Loss_2_list))
            np.save(FOLDER+"output/e1_Alpha_list.npy", np.array(Alpha_list))
            np.save(FOLDER+"output/e1_Alpha_mean_list.npy", np.array(Alpha_mean_list))

        # Terminate training 
        if(loss.data < enet_terminate):
            training_time = time.time() - start_time
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "e1_net",
                    'train_time': training_time,
                    }, PATH)
            np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))
            # Store alpha data (calculate data)
            Nsample_i = x_bc.shape[0] + x.shape[0]
            Loss_1_i    = (loss).data.cpu().numpy().item()
            Loss_2_i    = (0.0*loss).data.cpu().numpy().item() ###
            alpha_over_time = []
            for i in range(len(t1s)):
                t1 = t1s[i]
                pt_t_data = Variable(torch.from_numpy(x_data*0+t1).float(), requires_grad=True).to(device)
                ehat = e1_net(pt_x_data, pt_t_data).data.cpu().numpy()
                e1 = E1_list[i]
                alpha = np.max(np.abs(e1-ehat))/ np.max(np.abs(ehat))
                alpha = np.round(alpha, 3)
                alpha_over_time.append(alpha)
            max_alpha = np.max(alpha_over_time)
            Nsample_list.append(Nsample_i)
            Loss_1_list.append(Loss_1_i)
            Loss_2_list.append(Loss_2_i)
            Alpha_list.append(max_alpha)
            Alpha_mean_list.append(np.mean(alpha_over_time))
            # Store alpha data (write data)
            np.save(FOLDER+"output/e1_Nsample_list.npy", np.array(Nsample_list))
            np.save(FOLDER+"output/e1_Loss_1_list.npy", np.array(Loss_1_list))
            np.save(FOLDER+"output/e1_Loss_2_list.npy", np.array(Loss_2_list))
            np.save(FOLDER+"output/e1_Alpha_list.npy", np.array(Alpha_list))
            np.save(FOLDER+"output/e1_Alpha_mean_list.npy", np.array(Alpha_mean_list))
            return

        loss.backward(retain_graph=True) 
        optimizer.step()
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def load_trained_model(net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(checkpoint['label'] + " best epoch   : ", epoch, ", loss:", loss.data)
    print(checkpoint['label'] + " training time: ", checkpoint['train_time'])
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/"+checkpoint['label']+"_loss_history.png")
    plt.close()
    return net


def show_results(pnet, enet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    limit_margin = 0.0

    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    e2_list = []
    e2_hat_list = []
    p_res_list = []
    e_res_list = []
    e2_res_list = []
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        # if(t1 == 0.0): p_monte = p_init(x)
        p_monte_list.append(p_monte)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        phat = pnet(pt_x, pt_t1)
        ehat = enet(pt_x, pt_t1)[:,0].view(-1,1)
        #e2hat = enet(pt_x, pt_t1)[:,1].view(-1,1)
        pres = p_res_func(pt_x, pt_t1, pnet).data.cpu().numpy()
        eres = e_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()
        # e2res = e2_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()
        
        phat = phat.data.cpu().numpy() # change tensor to numpy
        ehat = ehat.data.cpu().numpy() # change tensor to numpy
        #e2hat = e2hat.data.cpu().numpy()
        p_hat_list.append(phat)
        e1 = p_monte - phat
        e1_list.append(e1)
        e1_hat_list.append(ehat)
        # e2 = e1 - ehat
        # e2_list.append(e2)
        # e2_hat_list.append(e2hat)

        p_res_list.append(pres)
        e_res_list.append(eres)
        # e2_res_list.append(e2res)

    global_max = float('-inf')
    for i, (phat) in enumerate(zip(p_hat_list)):
        max_value = np.max(np.abs(phat))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(6, 6))
    for i, (p_monte, p_hat, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, e1_hat_list)):
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("PDF")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("PDF")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("PDF")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat)); eL = np.round(eL, 3)

        ax1.plot(x, p_monte, "black", linewidth = 1.0, label=r"$p$")
        ax1.plot(x, p_hat, "red", linewidth = 1.0, linestyle="--", label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL, 
                            color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")

        ax1.set_xlim([x_low, x_hig])
        ax1.set_ylim(0.0, global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $e_S:$ "+str(eL), 
                  transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/phat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        max_value = max(np.max(np.abs(e1_true)), np.max(np.abs(e1_hat)))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(6, 6))
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("Error")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("Error")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("Error")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat))
        eL = np.round(eL, 3)
        alpha = np.max(np.abs(e1_true-e1_hat))/ np.max(np.abs(e1_hat))
        alpha = np.round(alpha, 3)
        print("eL: ", np.round(eL, 3), "\t alpha: ", np.round(alpha,3))

        ax1.plot(x, e1_true, "black", linewidth=1.0, label=r"$e_1$")
        ax1.plot(x, e1_hat,  "red", linewidth=1.0, linestyle="--", label=r"$\hat{e}_1$")
        ax1.fill_between(x.reshape(-1), y1=0.0*p_hat.reshape(-1)+eL, y2=0.0*p_hat.reshape(-1)-eL, 
                         color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")

        ax1.set_xlim([x_low, x_hig])
        # ax1.set_ylim(-(1.5*eL), 1.5*eL)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $\alpha_1:$ "+str(alpha), 
                  transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/e1hat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (eres, pres) in enumerate(zip(e_res_list, p_res_list)):
        max_value = max(np.max(np.abs(pres/pnet.scale)), 0.0)
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for i in range(0,6):
        pres = p_res_list[i]
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("Error")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("Error")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("Error")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        if i == 0:
            ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--", label=r"$r_1$")
            #ax1.plot(x, pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        ax1.set_ylim([-global_max, global_max])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(FOLDER+"figs/enet_res.png")
    fig.savefig(FOLDER+'figs/p_res.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, e_res_list, p_res_list)):
        max_value = max(np.max(np.abs(eres/enet.scale)), 0.0)
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for i in range(0,6):
        pres = p_res_list[i]
        eres = e_res_list[i]/enet.scale
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("Error")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("Error")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("Error")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        if i == 0:
            ax1.plot(x, eres, "red", linewidth=1.0, linestyle="--", label=r"$D[\hat{e}_1]$")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres, "red", linewidth=1.0, linestyle="--")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        ax1.set_ylim([-global_max, global_max])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(FOLDER+"figs/enet_res.png")
    fig.savefig(FOLDER+'figs/e1_res.pdf', format='pdf', dpi=300)
    plt.close()

    # fig, axs = plt.subplots(3, 2, figsize=(6, 6))
    # for i, (e2_true, e2_hat) in enumerate(zip(e2_list, e2_hat_list)):
    #     if i == 0:
    #         ax1 = axs[0,0]
    #         ax1.set_ylabel("Error")
    #     if i == 1:
    #         ax1 = axs[1,0]
    #         ax1.set_ylabel("Error")
    #     if i == 2:
    #         ax1 = axs[2,0]
    #         ax1.set_xlabel("x")
    #         ax1.set_ylabel("Error")
    #     if i == 3:
    #         ax1 = axs[0,1]
    #     if i == 4:
    #         ax1 = axs[1,1]
    #     if i == 5:
    #         ax1 = axs[2,1]
    #         ax1.set_xlabel("x")
    #     ax1.plot(x, e2_true, "black", linewidth=1.0, label=r"$e_2$")
    #     ax1.plot(x, e2_hat,  "red", linewidth=1.0, linestyle="--", label=r"$\hat{e}_2$")
    #     if i == 0:
    #         ax1.legend(loc="upper right")
    #     ax1.set_xlim([-4.5, 4.5])
    #     ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    # plt.tight_layout()
    # fig.savefig(FOLDER+'figs/e2hat.pdf', format='pdf', dpi=300)
    # plt.close()

    # fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    # for i in range(0,6):
    #     e2res = e2_res_list[i]
    #     if i == 0:
    #         ax1 = axs[0,0]
    #         ax1.set_ylabel("Error")
    #     if i == 1:
    #         ax1 = axs[1,0]
    #         ax1.set_ylabel("Error")
    #     if i == 2:
    #         ax1 = axs[2,0]
    #         ax1.set_xlabel("x")
    #         ax1.set_ylabel("Error")
    #     if i == 3:
    #         ax1 = axs[0,1]
    #     if i == 4:
    #         ax1 = axs[1,1]
    #     if i == 5:
    #         ax1 = axs[2,1]
    #         ax1.set_xlabel("x")
    #     if i == 0:
    #         ax1.plot(x, e2res, "red", linewidth=1.0, linestyle="--", label=r"$r_3$")
    #         # ax1.plot(x, -eres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
    #         ax1.legend()  # Add legend only to the first subplot
    #     else:
    #         ax1.plot(x, e2res, "red", linewidth=1.0, linestyle="--")
    #         # ax1.plot(x, -eres, "black", linewidth=1.0, linestyle="-")
    #     ax1.set_ylim([-global_max, global_max])
    #     ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    # plt.tight_layout()
    # # plt.savefig(FOLDER+"figs/enet_res.png")
    # fig.savefig(FOLDER+'figs/e2_res.pdf', format='pdf', dpi=300)
    # plt.close()


def plot_p_monte():
    markers = ['o', 's', 'D', '^', 'v', '<']
    plt.figure(figsize=(8,6))
    for i in range(len(t1s)):
        t1 = t1s[i]
        x_sim = np.load(DATA_FOLDER+ datas[0] + "xsim.npy")
        p_sim = np.load(DATA_FOLDER+ datas[0] + "psim_t"+str(t1)+".npy")
        line, = plt.plot(x_sim, p_sim, linewidth=1.0)
        
        # Plot markers at specific intervals
        interval = 5  # Marker interval
        plt.plot(x_sim[::interval], p_sim[::interval], markers[i], markersize=3, color=line.get_color(), label="t="+str(t1))  # 'o' marker style


    plt.grid(linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/p_sol_monte.png")
    print("save fig to "+FOLDER+"figs/p_sol_monte.png")
    plt.close()


def plot_p_surface(p_net, num=100):
    # x_samples = np.load(FOLDER+"output/p_xsamples.npy")
    # t_samples = np.load(FOLDER+"output/p_tsamples.npy")
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    phat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)

    p_list = []
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        p_list.append(p_monte)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.8, label=r"$\hat{p}$")
    # z_max = 1.5*np.max(np.abs(phat))
    # ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="black", s=0.02, label='Data Points')
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x_monte*0 + t1
        if i == 0:
            ax.plot(x_monte, t1_monte, p_list[i], color="black", label=r"$p_s$")
        else:
            ax.plot(x_monte, t1_monte, p_list[i], color="black")

    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel('PDF')
    ax.legend()
    ax.view_init(20, -60)
    # y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    # ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    # Adjust layout manually
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/phat_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1_surface(p_net, e1_net, num=100):
    # x_samples = np.load(FOLDER+"output/e1_xsamples.npy")
    # t_samples = np.load(FOLDER+"output/e1_tsamples.npy")
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    e1hat = e1_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)
    
    e1_list = []
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_t1_monte = Variable(torch.from_numpy(x_monte*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x_monte, pt_t1_monte).data.cpu().numpy()
        e1 = p_monte - p_hat
        e1_list.append(e1)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1hat, cmap='viridis', alpha=0.6, label=r"$\hat{e}_1$")
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

    # ax.set_zlim([-z_max, z_max])
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("Error")
    ax.view_init(20, -60)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/e1hat_surface_plot.pdf', format='pdf', dpi=300)


def plot_pres_surface(p_net, num=100):
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pres = p_res_func(pt_x, pt_t, p_net).data.cpu().numpy().reshape(num, -1)

    p_list = []
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        p_list.append(p_monte)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, pres, cmap='viridis', alpha=0.8, label=r"$\hat{p}$")
    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel('r')
    ax.legend()
    ax.view_init(20, -60)
    # y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    # ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/pres_surface_plot.pdf', format='pdf', dpi=300)


def plot_train_loss(path_1, path_2):
    loss_history_1 = np.load(path_1)
    min_loss_1 = min(loss_history_1)
    loss_history_2 = np.load(path_2)
    min_loss_2 = min(loss_history_2)
    print(loss_history_2)
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


def fit_function(x, C1, C2):
    return C1 * np.sqrt(x) + C2


def plot_alpha_data():
    data_folder = "exp1/main_store_alpha/"
    max_alpha_to_fit = np.inf
    max_total_loss_to_fit = np.inf
    num_runs = 2

    plt.figure()
    X = []
    Y = []
    for i in range(0,1):
        data_folder_seedi = data_folder + "seed" + str(i) + "/"
        Nsample_array = np.load(data_folder_seedi+"output/e1_Nsample_list.npy")
        Loss_1_array = np.load(data_folder_seedi+"output/e1_Loss_1_list.npy")
        Loss_2_array = np.load(data_folder_seedi+"output/e1_Loss_2_list.npy")
        Alpha_array = np.load(data_folder_seedi+"output/e1_Alpha_mean_list.npy")

        mask_1 = Alpha_array < max_alpha_to_fit
        mask_2 = (Loss_1_array+Loss_2_array) < max_total_loss_to_fit
        mask = mask_1 & mask_2

        Loss_1_array = Loss_1_array[mask]
        Loss_2_array = Loss_2_array[mask]
        Alpha_array = Alpha_array[mask]
        x_data = Loss_1_array + Loss_2_array
        y_data = Alpha_array
        plt.plot(x_data, y_data, marker='o', linestyle='None', markersize=2, label="run seed"+str(i))
        X.append(x_data)
        Y.append(y_data)
    # Fit the model
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    lower_bounds = [0, 0]
    upper_bounds = [np.inf, np.inf]
    popt, pcov = curve_fit(fit_function, X, Y, bounds=(lower_bounds, upper_bounds))
    C1, C2 = popt
    print("fit line constants:", C1, C2)
    xs = np.linspace(np.min(X), np.max(X), num=100)
    ys = C1 * np.sqrt(xs) + C2
    plt.plot(xs, ys, color="black", linewidth=0.5, linestyle="--", label="fit line")

    plt.grid(linewidth=0.5)
    plt.xlabel("num. integral training loss")
    plt.ylabel(r"max $\alpha_1$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/alpha_vs_loss.pdf', format='pdf', dpi=300)
    plt.close()


def main():
    global seed
    global FOLDER
    print("seed: ", seed, ", folder:", FOLDER)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # plot_p_monte()
    mse_cost_function = torch.nn.MSELoss()
    
    p_model = PNet().to(device)
    e_model = ENet().to(device) 
    e_model.apply(init_weights)
    optimizer_e_model = torch.optim.Adam(e_model.parameters())
    scheduler_e_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_e_model, gamma=0.95)
    
    p_model.scale = get_p_normalize()
    p_model = load_trained_model(p_model, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    p_model.eval()

    e_model.scale = get_e1_normalize(p_model)
    print("enet scale: ", e_model.scale)
    train_enet_model(p_model, e_model, optimizer_e_model, scheduler_e_model, mse_cost_function, iterations=MAX_EPOCHS); print("[e1_net train complete]")
    e_model = load_trained_model(e_model, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")
    e_model.eval()
    show_results(p_model, e_model)

    # plot_alpha_data()
    # plot_p_surface(p_model)
    # plot_e1_surface(p_model, e_model)
    # plot_train_loss(FOLDER+"output/p_net_train_loss.npy", FOLDER+"output/e1_net_train_loss.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass seed as a command-line argument")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    args = parser.parse_args()
    
    # Modify the global variable
    seed = args.seed
    FOLDER = FOLDER + "seed" + str(seed) + "/"

    main()
