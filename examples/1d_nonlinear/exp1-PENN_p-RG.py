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

FOLDER = "exp1/exp1-PENN_p-RG/"
DATA_FOLDER = "exp1/data/"

device = "cpu"; print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

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
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

datas = ["data1/"]
batch_size = 500
S = 10000
pnet_terminate = 1e-4
enet_terminate = 1e-4


def p_init(x):
    return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


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


def e_res_func(x, t, e1_net, p_net, verbose=False):
    e1_out = e1_net(x, t)
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class PNet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        self.normalize = 1.0
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
        output = F.softplus( self.output_layer(layer8_out) )
        return output


class ENet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(ENet, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
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
    ti = t0; tf = T_end
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    x_mar = 0.0

    # Define the mean and covariance matrix
    mean = torch.tensor([-2.0])
    covariance_matrix = torch.tensor([[0.5]])
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    # space-time points for BC
    x_bc = (torch.rand(600, n_d) * (x_hig - x_low) + x_low).to(device); 
    x_bc_normal = mvn.sample((600,))
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    # space-time points for RES
    x = (torch.rand(2500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    t = (torch.rand(2500, 1, requires_grad=True) *   (tf - ti) + ti).to(device)

    max_abs_p_ti = p_net.normalize

    # RAR
    S = 100000
    FLAG = False
    
    PATH = FOLDER+"output/p_net.pth"

    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p0 = p_init(x_bc.detach().numpy())
        u_bc = Variable(torch.from_numpy(p0).float(), requires_grad=False).to(device)
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        res_out = p_res_func(x, t, p_net)/max_abs_p_ti
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Frequnecy Loss
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + mse_res + mse_norm_res_input

        # RAR
        if (epoch%500 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            res_RAR = p_res_func(x_RAR, t_RAR, p_net)/max_abs_p_ti
            mean_res_RAR = torch.mean(res_RAR**2)
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
                print("... add [x,t]:", x_max.data, t_max.data, max_abs_res.data)
            res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_input_RAR = torch.cat([res_x_RAR, res_t_RAR], axis=1)
            norm_res_input_RAR = torch.norm(res_input_RAR, dim=1).view(-1,1)
            if(torch.mean(norm_res_input) > 0.0):
                max_abs_res_input, max_index = torch.max(norm_res_input_RAR, dim=0)
                # Get the corresponding x_RAR and t_RAR vectors
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                # Append x_max and t_max to x and t
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... add [x,t]:", x_max.data, t_max.data, max_abs_res_input.data)
            FLAG = False

        loss_history.append(loss.data)
        
        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                  "res input:", mse_norm_res_input.data,
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "p_net",
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # terminate training 
        if(loss.data < pnet_terminate):
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                  "res input:", mse_norm_res_input.data,
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "p_net",
                    }, PATH)
            return

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
                np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))

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

    # Define the mean and covariance matrix
    mean = torch.tensor([-2.0])
    covariance_matrix = torch.tensor([[0.5]])
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    # space-time points for BC
    x_bc = (torch.rand(600, n_d) * (x_hig - x_low) + x_low).to(device)
    x_bc_normal = mvn.sample((600,))
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(2500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    t = (torch.rand(2500, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
    FLAG = False
    S = 100000
    max_abs_e1_ti = e1_net.scale

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p0 = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p0).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = (p_bc - phat_bc)/max_abs_e1_ti
        net_bc_out = e1_net(x_bc, t_bc)/max_abs_e1_ti
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e_res_func(x, t, e1_net, p_net)/max_abs_e1_ti
        mse_res = mse_cost_function(res_out, all_zeros)

        # Frequnecy Loss
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)
        
        # Combining the loss functions
        loss = mse_u + mse_res + mse_norm_res_input

        if (epoch%1000 == 0):
            print(epoch,"Traning Loss:",loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                  ",res freq:", mse_norm_res_input.data
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "e1_net",
                    }, PATH)
            min_loss = loss.data 
            FLAG = True

        # RAR
        if (epoch%1000 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            
            ic_hat_RAR = e1_net(x_RAR, t0_RAR)/max_abs_e1_ti
            p0_RAR = p_init(x_RAR.detach().numpy())
            p_bc_RAR = Variable(torch.from_numpy(p0_RAR).float(), requires_grad=False).to(device)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            ic_RAR = (p_bc_RAR - phat_bc_RAR)/max_abs_e1_ti
            mean_ic_error = torch.mean(torch.abs(ic_RAR - ic_hat_RAR))
            print("RAR mean IC: ", mean_ic_error.data)
            if(mean_ic_error > 5e-3):
                max_abs_ic, max_index = torch.max(torch.abs(ic_RAR - ic_hat_RAR), dim=0)
                x_max = x_RAR[max_index].clone().detach()
                t_max = t0_RAR[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... IC add [x,t]:", x_max.data, t_max.data, ". max ic value: ", max_abs_ic.data)
                FLAG = False

            res_RAR = e_res_func(x_RAR, t_RAR, e1_net, p_net)/max_abs_e1_ti
            mean_res_error = torch.mean(torch.abs(res_RAR))
            print("RAR mean res: ", mean_res_error.data)
            if(mean_res_error > 5e-3):
                max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res.data)
                FLAG = False

            res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            res_input_RAR = torch.cat([res_x_RAR, res_t_RAR], axis=1)
            norm_res_input_RAR = torch.norm(res_input_RAR, dim=1).view(-1,1)
            mean_res_input_error = torch.mean(norm_res_input)
            print("RAR mean res input: ", mean_res_input_error.data)
            if(mean_res_input_error > 5e-3):
                max_abs_res_input, max_index = torch.max(norm_res_input_RAR, dim=0)
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RES_INPUT add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)
                FLAG = False

        loss_history.append(loss.data)
        loss.backward(retain_graph=True) 
        optimizer.step()
        
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))


def load_trained_model(net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(checkpoint['label'] + " best epoch: ", epoch, ", loss:", loss.data)
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
    p_res_list = []
    e_res_list = []

    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        # if(t1 == 0.0): p_monte = p_init(x)
        p_monte_list.append(p_monte)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        phat = pnet(pt_x, pt_t1)
        ehat = enet(pt_x, pt_t1)
        pres = p_res_func(pt_x, pt_t1, pnet).data.cpu().numpy()
        eres = e_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()
        
        phat = phat.data.cpu().numpy() # change tensor to numpy
        ehat = ehat.data.cpu().numpy() # change tensor to numpy
        p_hat_list.append(phat)
        e1 = p_monte - phat
        e1_list.append(e1)
        e1_hat_list.append(ehat)
        p_res_list.append(pres)
        e_res_list.append(eres)

    global_max = float('-inf')
    for i, (phat) in enumerate(zip(p_hat_list)):
        max_value = np.max(np.abs(phat))
        global_max = max(global_max, max_value)
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
        ax1.set_ylim(0.0, global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_result.png")
    plt.close()

    global_max = float('-inf')
    for i, (pres) in enumerate(zip(p_res_list)):
        max_value = np.max(np.abs(pres))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i in range(0,6):
        pres = p_res_list[i]
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
        if i == 0:
            ax1.plot(x, pres, "red", linestyle="--", label=r"$r_1$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, pres, "red", linestyle="--")
        ax1.set_ylim([-global_max, global_max])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_res.png")
    plt.close()

    global_max = float('-inf')
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        max_value = max(np.max(np.abs(e1_true)), 0.0)
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
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
        alpha = np.max(np.abs(e1_true-e1_hat))/ np.max(np.abs(e1_hat))
        print("eL: ", np.round(eL, 3), "\t alpha: ", np.round(alpha,3))
        ax1.plot(x, e1_true, "blue", label=r"$e_1$")
        ax1.plot(x, e1_hat, "red", linestyle="--", label=r"$\hat{e}_1$")
        ax1.fill_between(x.reshape(-1), y1=0.0*p_hat.reshape(-1)+eL, y2=0.0*p_hat.reshape(-1)-eL, color="green", alpha=0.2)
        # ax1.set_ylim(-(global_max+limit_margin), global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/enet_result.png")
    plt.close()

    global_max = float('-inf')
    for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, e_res_list, p_res_list)):
        max_value = max(np.max(np.abs(eres)), 0.0)
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    for i in range(0,6):
        pres = p_res_list[i]
        eres = e_res_list[i]
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
        if i == 0:
            ax1.plot(x, eres, "red", linestyle="--", label=r"$r_2$")
            # ax1.plot(x, -pres, "blue", linestyle="-")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres, "red", linestyle="--")
            # ax1.plot(x, -pres, "blue", linestyle="-")
        ax1.set_ylim([-global_max, global_max])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/enet_res.png")
    plt.close()


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
        

def main():
    plot_p_monte()
    mse_cost_function = torch.nn.MSELoss()
    
    p_model = PNet().to(device)
    e_model = ENet().to(device)
    p_model.apply(init_weights)
    e_model.apply(init_weights)
    optimizer_p_model = torch.optim.Adam(p_model.parameters())
    optimizer_e_model = torch.optim.Adam(e_model.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_p_model, gamma=0.95)
    scheduler_e_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_e_model, gamma=0.95)
    
    p_model.normalize = get_p_normalize()
    start_time = time.time()
    # train_pnet_model(p_model, optimizer_p_model, scheduler_p_model, mse_cost_function, iterations=40000); print("[p_net train complete]")
    end_time = time.time()
    time_train_p = end_time - start_time
    p_model = load_trained_model(p_model, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    p_model.eval()

    e_model.scale = get_e1_normalize(p_model)
    print("enet scale: ", e_model.scale)
    start_time = time.time()
    train_enet_model(p_model, e_model, optimizer_e_model, scheduler_e_model, mse_cost_function, iterations=100000); print("[e1_net train complete]")
    end_time = time.time()
    time_train_e = end_time - start_time
    e_model = load_trained_model(e_model, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")
    e_model.eval()
    show_results(p_model, e_model)

    print(f"train pnet time: {time_train_p:.4f} seconds")
    print(f"train enet time: {time_train_e:.4f} seconds")


if __name__ == "__main__":
    main()



# Log

### PENN Softplus ENet ###
# (venv) chko1829@UCB-X1Q6HY3GQC 1d_nonlinear % python exp1-PENN_p-RG.py
# cpu
# save fig to exp1/exp1-PENN_p-RG/figs/p_sol_monte.png
# epoch: 0 ,loss: tensor(76.2332) ,ic: tensor(2.0900) ,res: tensor(62.2887) ,res g: tensor(11.8545)
# epoch: 1 ,loss: tensor(60.6372) ,ic: tensor(1.6696) ,res: tensor(49.5199) ,res g: tensor(9.4478)
# epoch: 2 ,loss: tensor(47.6442) ,ic: tensor(1.3190) ,res: tensor(38.8828) ,res g: tensor(7.4424)
# epoch: 3 ,loss: tensor(36.9818) ,ic: tensor(1.0312) ,res: tensor(30.1554) ,res g: tensor(5.7953)
# epoch: 4 ,loss: tensor(28.3594) ,ic: tensor(0.7982) ,res: tensor(23.1002) ,res g: tensor(4.4610)
# epoch: 5 ,loss: tensor(21.4946) ,ic: tensor(0.6127) ,res: tensor(17.4861) ,res g: tensor(3.3958)
# epoch: 6 ,loss: tensor(16.1175) ,ic: tensor(0.4672) ,res: tensor(13.0919) ,res g: tensor(2.5584)
# epoch: 7 ,loss: tensor(11.9740) ,ic: tensor(0.3549) ,res: tensor(9.7086) ,res g: tensor(1.9105)
# epoch: 8 ,loss: tensor(8.8306) ,ic: tensor(0.2696) ,res: tensor(7.1444) ,res g: tensor(1.4166)
# epoch: 9 ,loss: tensor(6.4799) ,ic: tensor(0.2058) ,res: tensor(5.2288) ,res g: tensor(1.0453)
# [pnet train complete]
# pnet best epoch:  19719 , loss: tensor(3.9051e-05)
# enet scale:  0.016813713809397668
# epoch: 0 ,loss: tensor(8.2147) ,ic: tensor(0.2951) ,res: tensor(6.5774) ,res g: tensor(1.3422)
# epoch: 1 ,loss: tensor(3.3424) ,ic: tensor(0.1701) ,res: tensor(2.5958) ,res g: tensor(0.5766)
# epoch: 2 ,loss: tensor(0.7596) ,ic: tensor(0.1114) ,res: tensor(0.4818) ,res g: tensor(0.1664)
# epoch: 3 ,loss: tensor(0.2453) ,ic: tensor(0.1114) ,res: tensor(0.0557) ,res g: tensor(0.0782)
# epoch: 11 ,loss: tensor(0.1907) ,ic: tensor(0.1064) ,res: tensor(0.0153) ,res g: tensor(0.0690)
# epoch: 49 ,loss: tensor(0.1711) ,ic: tensor(0.1052) ,res: tensor(0.0062) ,res g: tensor(0.0597)
# enet best epoch:  26508 , loss: tensor(0.0041)
# eL:  0.021 	 alpha:  0.165
# eL:  0.021 	 alpha:  0.274
# eL:  0.019 	 alpha:  0.323
# eL:  0.016 	 alpha:  0.269
# eL:  0.013 	 alpha:  0.315
# eL:  0.01 	 alpha:  0.79
# train pnet time: 0.0000 seconds
# train enet time: 0.0000 seconds
# (venv) chko1829@UCB-X1Q6HY3GQC 1d_nonlinear % python qv.py
# cpu
# save fig to exp1/exp1-PENN_p-RG/figs/p_sol_monte.png
# pnet best epoch:  19719 , loss: tensor(3.9051e-05)
# eL:  1.024 	 alpha:  1.0
# eL:  1.022 	 alpha:  1.0
# eL:  1.048 	 alpha:  1.001
# eL:  1.084 	 alpha:  1.002
# eL:  1.106 	 alpha:  1.002
# eL:  1.111 	 alpha:  0.999
# enet scale:  0.016813713809397668
# enet best epoch:  60993 , loss: tensor(0.0030)
# eL:  0.021 	 alpha:  0.169
# eL:  0.021 	 alpha:  0.271
# eL:  0.018 	 alpha:  0.284
# eL:  0.015 	 alpha:  0.222
# eL:  0.011 	 alpha:  0.541
# eL:  0.008 	 alpha:  1.453

### NN Tanh ENet ### ---> ENet threshold = 1e-4 (with IC and Res Loss only)
# (pre-trained) pnet: pnet best epoch:  19719 , loss: tensor(3.9051e-05), stored as pre-pnet.pt
# enet best epoch:  31239 , loss: tensor(0.0002)
# eL:  0.02 	 alpha:  0.061
# eL:  0.018 	 alpha:  0.053
# eL:  0.013 	 alpha:  0.137
# eL:  0.011 	 alpha:  0.28
# eL:  0.01 	 alpha:  0.359
# eL:  0.009 	 alpha:  0.444
#
# enet best epoch:  35906 , loss: tensor(0.0002)
# eL:  0.02 	 alpha:  0.07
# eL:  0.018 	 alpha:  0.059
# eL:  0.013 	 alpha:  0.135
# eL:  0.011 	 alpha:  0.268
# eL:  0.011 	 alpha:  0.26
# eL:  0.01 	 alpha:  0.324
#
# enet best epoch:  40149 , loss: tensor(0.0001)
# eL:  0.02 	 alpha:  0.071
# eL:  0.018 	 alpha:  0.066
# eL:  0.013 	 alpha:  0.134
# eL:  0.012 	 alpha:  0.262
# eL:  0.011 	 alpha:  0.23
# eL:  0.01 	 alpha:  0.26
#
# enet best epoch:  48472 , loss: tensor(0.0001)
# eL:  0.019 	 alpha:  0.075
# eL:  0.017 	 alpha:  0.054
# eL:  0.013 	 alpha:  0.14
# eL:  0.012 	 alpha:  0.264
# eL:  0.011 	 alpha:  0.219
# eL:  0.011 	 alpha:  0.14
#
# enet best epoch:  54271 , loss: tensor(0.0001)
# eL:  0.019 	 alpha:  0.075
# eL:  0.017 	 alpha:  0.058
# eL:  0.013 	 alpha:  0.14
# eL:  0.012 	 alpha:  0.261
# eL:  0.012 	 alpha:  0.209
# eL:  0.012 	 alpha:  0.097
#
# enet best epoch:  82053 , loss: tensor(0.0001)
# eL:  0.019 	 alpha:  0.072
# eL:  0.017 	 alpha:  0.052
# eL:  0.013 	 alpha:  0.163
# eL:  0.012 	 alpha:  0.248
# eL:  0.014 	 alpha:  0.175
# eL:  0.015 	 alpha:  0.178
#
# enet best epoch:  94124 , loss: tensor(9.7544e-05)
# eL:  0.019 	 alpha:  0.071
# eL:  0.017 	 alpha:  0.052
# eL:  0.013 	 alpha:  0.167
# eL:  0.012 	 alpha:  0.249
# eL:  0.014 	 alpha:  0.171
# eL:  0.015 	 alpha:  0.213

### PENN+Softplus Pnet & NN+Tanh ENet (seed 0) ###
# pnet best epoch:  12777 , loss: tensor(4.0000e-05)
# enet best epoch:  92383 , loss: tensor(0.0001)
# eL:  0.035 	 alpha:  0.033
# eL:  0.018 	 alpha:  0.076
# eL:  0.016 	 alpha:  0.137
# eL:  0.02 	 alpha:  0.192
# eL:  0.026 	 alpha:  0.255
# eL:  0.025 	 alpha:  0.441
# train pnet time: 659.7231 seconds
# train enet time: 1300.6258 seconds

### PENN+Softplus Pnet & NN+Tanh ENet (seed 1) ### FAIL ---> rerun this with enet terminate to 4e-5, epoch to 300k
# ---> did not work, so use the pre-trained pnet (seed 1), and train the enet (seed 1) as baseline.
# pnet best epoch:  11322 , loss: tensor(3.9956e-05)
# enet best epoch:  83829 , loss: tensor(1.0000e-04)
# eL:  0.037 	 alpha:  0.032
# eL:  0.008 	 alpha:  0.236
# eL:  0.008 	 alpha:  0.597
# eL:  0.006 	 alpha:  1.097
# eL:  0.007 	 alpha:  1.133
# eL:  0.01 	 alpha:  1.086
# train pnet time: 593.1258 seconds
# train enet time: 1049.3910 seconds

### PENN+Softplus Pnet & NN+Tanh ENet (seed 1) FAIL Baseline ###
# pnet best epoch:  11322 , loss: tensor(3.9956e-05)
# enet best epoch:  154394 , loss: tensor(0.0001)
# eL:  0.037 	 alpha:  0.031
# eL:  0.008 	 alpha:  0.135
# eL:  0.007 	 alpha:  0.503
# eL:  0.004 	 alpha:  0.918
# eL:  0.003 	 alpha:  1.17
# eL:  0.003 	 alpha:  1.455
# train pnet time: 0.0000 seconds
# train enet time: 4210.3268 seconds

### PENN+Softplus (pre-trained by seed 1) & PENN+Tanh ENet ###
# ---> Fail
# enet best epoch:  84819 , loss: tensor(3.9998e-05)
# eL:  0.037 	 alpha:  0.03
# eL:  0.008 	 alpha:  0.166
# eL:  0.007 	 alpha:  0.461
# eL:  0.004 	 alpha:  1.037
# eL:  0.004 	 alpha:  0.845
# eL:  0.002 	 alpha:  1.378
# train pnet time: 0.0000 seconds
# train enet time: 2007.6325 seconds

### PENN+Softplus (pre-trained by seed 1) & PENN+Softplus ENet ###
# --> Fail
# enet best epoch:  101957 , loss: tensor(5.4436e-05)
# eL:  0.015 	 alpha:  0.091
# eL:  0.008 	 alpha:  0.169
# eL:  0.008 	 alpha:  0.73
# eL:  0.01 	 alpha:  1.164
# eL:  0.013 	 alpha:  1.127
# eL:  0.015 	 alpha:  1.046

### PENN+Softplus (pre-trained by seed 1) & PENN+Tanh ENet, with weight_res x10 ###
# --> Fail

### PENN+Softplus (pre-trained by seed 1) & PENN+Tanh ENet, with res_grad los ###
# --> Hard to Converge
# enet best epoch:  27446 , loss: tensor(0.0026)
# eL:  0.015 	 alpha:  0.205
# eL:  0.007 	 alpha:  1.03
# eL:  0.01 	 alpha:  1.175
# eL:  0.018 	 alpha:  0.909
# eL:  0.023 	 alpha:  0.915
# eL:  0.026 	 alpha:  0.98

### Rerun seed 1 Pnet with 1e-4 termination & PENN+Tanh 100 neurons Enet ###
# pnet best epoch:  6907 , loss: tensor(9.9811e-05)

### seed 1, PENN+Softplus 30 neurons Pnet & PENN+Softplus 100 neurons Enet ###
### N_ic=500, N_r=2000, p_terminate=5e-5, e_terminate=4e-5 ###
# --> Success
# enet best epoch:  42537 , loss: tensor(0.0002)
# eL:  0.022 	 alpha:  0.152
# eL:  0.021 	 alpha:  0.333
# eL:  0.019 	 alpha:  0.6
# eL:  0.017 	 alpha:  0.721
# eL:  0.018 	 alpha:  0.761
# eL:  0.018 	 alpha:  0.77
# enet best epoch:  53853 , loss: tensor(0.0001)
# eL:  0.022 	 alpha:  0.153
# eL:  0.021 	 alpha:  0.34
# eL:  0.018 	 alpha:  0.624
# eL:  0.017 	 alpha:  0.73
# eL:  0.018 	 alpha:  0.718
# eL:  0.019 	 alpha:  0.671

### Test if NN+Tanh ENet, with only IC and Res Loss works? (with more data, N_r=2000) ###
# --> Fail, stuck at loss=4.4336e-05
# enet best epoch:  166192 , loss: tensor(4.4336e-05)
# eL:  0.023 	 alpha:  0.055
# eL:  0.027 	 alpha:  0.046
# eL:  0.028 	 alpha:  0.152
# eL:  0.024 	 alpha:  0.371
# eL:  0.02 	 alpha:  0.819
# eL:  0.014 	 alpha:  1.614

### Test if PENN+Tanh ENet, with only IC and Res Loss works? (with more data, N_r=2000) ###
# --> Success
# enet best epoch:  29662 , loss: tensor(3.9968e-05)
# eL:  0.06 	 alpha:  0.019
# eL:  0.024 	 alpha:  0.149
# eL:  0.024 	 alpha:  0.225
# eL:  0.024 	 alpha:  0.334
# eL:  0.025 	 alpha:  0.479
# eL:  0.023 	 alpha:  0.466
# train enet time: 1370.8514 seconds