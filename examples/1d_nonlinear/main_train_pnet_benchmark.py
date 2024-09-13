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

device = "cpu"; print(device)

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
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class PNet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(PNet, self).__init__()
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
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        layer6_out = F.softplus((self.hidden_layer6(layer5_out)))
        layer7_out = F.softplus((self.hidden_layer7(layer6_out)))
        layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
        output = F.softplus(self.output_layer(layer8_out))
        return output
  

def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


# def get_e1_normalize(pnet):
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
    x_bc = (torch.rand(500, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t_bc = (torch.ones(len(x_bc), 1, requires_grad=True) * ti).to(device)
    
    # space-time points for RES
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

            # p0_RAR = p_init_torch(x0_RAR)
            # p0_hat_RAR = p_net(x0_RAR, t0_RAR)
            # e0_RAR = p0_RAR - p0_hat_RAR
            # mean_e0_RAR = torch.mean(torch.abs(e0_RAR))
            # if(mean_e0_RAR > 0.0):
            #     max_abs_e0, max_index = torch.max(torch.abs(e0_RAR), dim=0)
            #     x_max = x0_RAR[max_index]
            #     t_max = t0_RAR[max_index]
            #     x_bc = torch.cat((x_bc, x_max), dim=0)
            #     t_bc = torch.cat((t_bc, t_max), dim=0)
            #     print("... Ic add [x,t]:", x_max.data, t_max.data, max_abs_e0.data)
            #     FLAG = False

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

            # res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # mean_res_g_RAR = torch.mean(torch.abs(res_x_RAR)+torch.abs(res_t_RAR))
            # print("RAR mean res input: ", mean_res_g_RAR.data)
            # if(mean_res_g_RAR > 0.0):
            #     max_abs_res_input, max_index = torch.max(res_x_RAR**2+ res_t_RAR**2, dim=0)
            #     x_max = x_RAR[max_index]
            #     t_max = t_RAR[max_index]
            #     x = torch.cat((x, x_max), dim=0)
            #     t = torch.cat((t, t_max), dim=0)
            #     print("... Res Grad add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)
            #     FLAG = False

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
    print("seed: ", seed)
    print("folder:", FOLDER)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # plot_p_monte()
    mse_cost_function = torch.nn.MSELoss()
    
    p_model = PNet().to(device)
    p_model.apply(init_weights_He)
    optimizer_p_model = torch.optim.Adam(p_model.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_p_model, gamma=0.95)
    
    p_model.scale = get_p_normalize()
    train_pnet_model(p_model, optimizer_p_model, scheduler_p_model, mse_cost_function, iterations=MAX_EPOCHS); print("[p_net train complete]")

    # sleep for 10 seconds
    time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass seed as a command-line argument")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    args = parser.parse_args()
    
    # Modify the global variable
    seed = args.seed
    FOLDER = FOLDER + "seed" + str(seed) + "/"

    main()
