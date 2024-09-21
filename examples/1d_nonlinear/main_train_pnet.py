import torch
import torch.nn as nn
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

FOLDER = "exp1/main_alphas/seed-test/"
DATA_FOLDER = "exp1/data/"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set default tensor type to CUDA tensors
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
# device = "cpu"
print("train on: ", device)

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
pnet_terminate = 5e-5
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


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class PNet(nn.Module):
    def __init__(self, scale=1.0):
        super(PNet, self).__init__()
        self.scale = scale
        num_hidden_layers=5
        neurons=50
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
        out_1 = self.layers[0](out)
        out = F.softplus(out_1)   # First hidden layer
        for layer in self.layers[1:-1]:         # Apply hidden layers
            out_l = layer(out)
            out = F.softplus(out_l)
        out = self.layers[-1](out)              # Apply output layer (last + first hidden layers output)
        return F.softplus(out)
  

def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def train_pnet_model(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global x_low, x_hig, t0, T_end
    PATH = FOLDER+"output/p_net.pth"
    ti = t0; tf = T_end
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    x_mar = 0.0
    
    _x = np.linspace(x_low, x_hig, num=20, endpoint=True)
    _t = np.linspace(ti, tf, num=20, endpoint=True)

    # space-time points for BC
    x_bc = (torch.rand(200, n_d, requires_grad=True) * (x_hig - x_low) + x_low) 
    x_bc_quad = Variable(torch.from_numpy(_x.reshape(-1,1)).float(), requires_grad=True).to(device) 
    x_bc = torch.cat((x_bc, x_bc_quad), dim=0)
    t_bc = (torch.ones(len(x_bc), 1, requires_grad=True) * ti) 
    
    _xx, _tt = np.meshgrid(_x, _t)
    x = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device) 
    t = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device) 
    x_rand = (torch.rand(600, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device) 
    t_rand = (torch.rand(600, 1  , requires_grad=True) * (tf - ti) + ti).to(device) 
    x = torch.cat((x, x_rand), dim=0).to(device)
    t = torch.cat((t, t_rand), dim=0).to(device)

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
        p0_target = p0.detach()
        p0_hat =  p_net(x_bc, t_bc)
        all_zeros = torch.zeros((len(x_bc),1), dtype=torch.float32, requires_grad=False).to(device) 
        mse_u = mse_cost_function(p0_hat/max_abs_p_ti, p0_target/max_abs_p_ti)

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
        loss_history.append(loss.item())
        
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

        # RAR
        if (epoch%100 == 0 and FLAG):
            quad_number = random.randint(10,30)
            _x = np.linspace(x_low, x_hig, num=quad_number, endpoint=True)
            _t = np.linspace(ti, tf, num=quad_number, endpoint=True)
            _xx, _tt = np.meshgrid(_x, _t)
            x_quad = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device) 
            t_quad = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device) 
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device) 
            x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device) 
            t_RAR = torch.cat((t_RAR, t_quad), dim=0).to(device)
            x_RAR = torch.cat((x_RAR, x_quad), dim=0).to(device)

            t0_RAR = 0.0*t_RAR.clone() + t0
            x0_RAR = x_RAR.clone()
            p0_RAR = p_init_torch(x0_RAR)
            p0_hat_RAR = p_net(x0_RAR, t0_RAR)
            e0_RAR = p0_RAR - p0_hat_RAR
            if(True):
                max_abs_e0, max_index = torch.max(torch.abs(e0_RAR), dim=0)
                x_max = x0_RAR[max_index]
                t_max = t0_RAR[max_index]
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... Ic add [x,t]:", x_max.data, t_max.data, max_abs_e0.data)

            res_RAR = p_res_func(x_RAR, t_RAR, p_net)/p_net.scale
            if(True):
                # Find the index of the maximum absolute value in res_RAR
                max_abs_res, max_index = torch.max((res_RAR)**2, dim=0)
                # Get the corresponding x_RAR and t_RAR vectors
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                # Append x_max and t_max to x and t
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                debug_value = (p_res_func(x_max, t_max, p_net)/p_net.scale)**2
                print("... Res add [x,t]:", x_max.data, t_max.data, max_abs_res.data, debug_value.data)
            FLAG = False

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

        if (epoch) % 1000 == 0:
            print(epoch, " loss: ", loss.data)
        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # Exponential learning rate decay
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


def main():
    global seed
    global FOLDER
    print("seed: ", seed, ", folder:", FOLDER)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mse_cost_function = torch.nn.MSELoss()
    
    p_model = PNet().to(device) 
    p_model.apply(init_weights_He)
    optimizer_p_model = torch.optim.Adam(p_model.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_p_model, gamma=0.95)
    
    p_model.scale = get_p_normalize()
    train_pnet_model(p_model, optimizer_p_model, scheduler_p_model, mse_cost_function, iterations=MAX_EPOCHS); print("[p_net train complete]")
    torch.cuda.empty_cache()
    # time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass seed as a command-line argument")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    args = parser.parse_args()
    
    # Modify the global variable
    seed = args.seed

    main()
