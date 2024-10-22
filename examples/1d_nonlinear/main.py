import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter
import random
import warnings
import time
import argparse

FOLDER = "exp1/main/"
DATA_FOLDER = "exp1/data/"
TRAIN_FLAG = False

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
t1s = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

datas = ["data1/"]
S = 30000
pnet_terminate = 1e-4
enet_terminate = 1e-5


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


def diff_e(x, t, e1_net):
    e1_out = e1_net(x, t)
    e_x = torch.autograd.grad(e1_out, x, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_t = torch.autograd.grad(e1_out, t, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x,  x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    diff_e1net = e_t + (3*const_a*x*x + 2*const_b*x + const_c)*e1_out \
                     + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*e_x \
                     - 0.5*const_e*const_e*e_xx
    return diff_e1net


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
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        output = F.softplus( self.output_layer(layer3_out) )
        return output


class E1Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Softplus()
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        layer6_out = self.activation((self.hidden_layer6(layer5_out)))
        output = self.output_layer(layer6_out)
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
    PATH = FOLDER+"output/p_net.pth"

    # space-time points for BC
    x_bc = (torch.rand(1000, n_d) * (x_hig - x_low) + x_low).to(device); #print(min(x_bc[:,0]), max(x_bc[:,0]), min(x_bc[:,1]), max(x_bc[:,1]))
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t = (torch.rand(1000, 1, requires_grad=True) *   (tf - ti) + ti).to(device)

    normalize = p_net.scale

    # RAR
    FLAG = False

    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p0 = p_init_torch(x_bc).detach()
        p0_hat = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(p0_hat/normalize, p0/normalize)

        # Loss based on PDE
        res_out = p_res_func(x, t, p_net)/normalize
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Frequnecy Loss
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + 5.0*(mse_res + mse_norm_res_input)
        loss_history.append(loss.data)

        # Terminate training 
        if(loss.data < pnet_terminate):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                  "res grad:", mse_norm_res_input.data,
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "p_net",
                    'train_time': train_time,
                    }, PATH)
            np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))
            return

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                  "res grad:", mse_norm_res_input.data,
                 )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "p_net",
                    'train_time': train_time,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            max_ic_error = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR))/normalize
            print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > 5e-3):
                # max_abs_ic, max_index = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR), dim=0)
                max_abs_ic, max_index = torch.topk(torch.abs(phat_bc_RAR.squeeze() - p_bc_RAR.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t0_RAR[max_index].clone()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                # print("... IC add [x,t]:", x_max.data, t_max.data, max_abs_ic.data)

            res_RAR = p_res_func(x_RAR, t_RAR, p_net)/normalize
            max_res_RAR = torch.max(torch.abs(res_RAR))
            print("RAR max RES:", max_res_RAR.data)
            if(max_res_RAR > 5e-3):
                # max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                max_abs_res, max_index = torch.topk(torch.abs(res_RAR.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t_RAR[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                # print("... add [x,t]:", x_max.data, t_max.data, max_abs_res.data)
            np.save(FOLDER+"output/p_xsamples.npy", x.clone().data.cpu().numpy())
            np.save(FOLDER+"output/p_tsamples.npy", t.clone().data.cpu().numpy())
            FLAG = False

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))


def train_enet_model(p_net, e1_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global x_low, x_hig, t0, T_end
    ti = t0; tf = T_end
    min_loss = np.inf
    loss_history = []
    iterations_per_decay = 1000
    PATH = FOLDER+"output/e1_net.pth"

    # space-time points for BC
    x_bc = (torch.rand(1000, n_d) * (x_hig - x_low) + x_low).to(device);
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
    t = (torch.rand(1000, 1, requires_grad=True) *   (tf - ti) + ti).to(device)

    normalize = e1_net.scale
    
    # RAR
    FLAG = False

    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p0 = p_init_torch(x_bc).detach()
        p0_hat = p_net(x_bc, t_bc).to(device)
        e0 = (p0 - p0_hat).detach()
        e0_hat = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(e0_hat/normalize, e0/normalize)

        # Loss based on PDE
        diff_ehat = diff_e(x, t, e1_net)
        diff_e_target = -p_res_func(x, t, p_net).detach()
        mse_res = mse_cost_function(diff_ehat/normalize, diff_e_target/normalize)

        # Frequnecy Loss
        res_out = e_res_func(x, t, e1_net, p_net)/normalize
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_grad = torch.cat([res_x, res_t], axis=1)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        norm_res_grad = torch.norm(res_grad, dim=1).view(-1,1)
        mse_res_grad = mse_cost_function(norm_res_grad, all_zeros)
        
        # Combining the loss functions
        loss = mse_u + 5.0*(mse_res + mse_res_grad)
        loss_history.append(loss.item())

        # Terminaion
        if(loss.data < enet_terminate):
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                  ",res grad:", mse_res_grad.data
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "e1_net",
                    'train_time': time.time() - start_time
                    }, PATH)
            np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))
            return

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                  ",res grad:", mse_res_grad.data
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label': "e1_net",
                    'train_time': time.time() - start_time
                    }, PATH)
            min_loss = loss.data 
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            ic_hat_RAR = e1_net(x_RAR, t0_RAR)/normalize
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            ic_RAR = (p_bc_RAR - phat_bc_RAR)/normalize
            max_ic_error = torch.max(torch.abs(ic_RAR - ic_hat_RAR))
            print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > 5e-3):
                # max_abs_ic, max_index = torch.max(torch.abs(ic_RAR - ic_hat_RAR), dim=0)
                ic_diff = ic_RAR - ic_hat_RAR
                max_abs_ic, max_index = torch.topk(torch.abs(ic_diff.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t0_RAR[max_index].clone()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                # print("... IC add [x,t]:", x_max.data, t_max.data, max_abs_ic.data)

            res_RAR = e_res_func(x_RAR, t_RAR, e1_net, p_net)/normalize
            max_res_error = torch.max(torch.abs(res_RAR))
            print("RAR max RES: ", max_res_error.data)
            if(max_res_error > 5e-3):
                # max_abs_res, max_index = torch.max(torch.abs(res_RAR), dim=0)
                max_abs_res, max_index = torch.topk(torch.abs(res_RAR.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t_RAR[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                # print("... RES add [x,t]:", x_max.data, t_max.data, max_abs_res.data)
            np.save(FOLDER+"output/e1_xsamples.npy", x.clone().data.cpu().numpy())
            np.save(FOLDER+"output/e1_tsamples.npy", t.clone().data.cpu().numpy())
            FLAG = False

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
    print(checkpoint['label'] + " best epoch: ", epoch, ", loss:", loss.data, ", train time: ", checkpoint['train_time'])
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


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
       self.format = "%1.1f"


def show_results_old(pnet, enet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    limit_margin = 0.0

    # Create a ScalarFormatter object
    formatter = ScalarFormatter()
    formatter.set_scientific(True)

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
    plt.tight_layout(pad=0.3, h_pad=0.3)
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
        ax1.set_ylim(-(global_max), global_max)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $\alpha_1:$ "+str(alpha), 
                  transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        # Set y-axis to scientific notation
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax1.yaxis.set_major_formatter(yScalarFormatter)
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
            ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--", label=r"$D[\hat{e}_1]$")
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
        max_value = max(np.max(np.abs(eres/enet.scale)), 0.0)**2
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
            ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--", label=r"$r_2^2$")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        # ax1.set_ylim([0, global_max])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(FOLDER+"figs/enet_res.png")
    fig.savefig(FOLDER+'figs/e1_res.pdf', format='pdf', dpi=300)
    plt.close()


def show_results(pnet, enet):
    plt.rcParams['font.size'] = 18
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    limit_margin = 0.1

    # Create a ScalarFormatter object
    formatter = ScalarFormatter()
    formatter.set_scientific(True)

    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    e2_list = []
    e2_hat_list = []
    p_res_list = []
    e_res_list = []
    e2_res_list = []
    t1s = [1.0, 3.0, 5.0]
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
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    for i, (p_monte, p_hat, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, e1_hat_list)):
        # if i == 0:
        #     ax1 = axs[0,0]
        #     ax1.set_ylabel("PDF")
        # if i == 1:
        #     ax1 = axs[1,0]
        #     ax1.set_ylabel("PDF")
        # if i == 2:
        #     ax1 = axs[2,0]
        #     ax1.set_xlabel("x")
        #     ax1.set_ylabel("PDF")
        # if i == 3:
        #     ax1 = axs[0,1]
        # if i == 4:
        #     ax1 = axs[1,1]
        # if i == 5:
        #     ax1 = axs[2,1]
        #     ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat)); eL = np.round(eL, 3)
        ax1 = axs[i]
        ax1.plot(x, p_monte, "black", linewidth = 1.0, label=r"$p$")
        ax1.plot(x, p_hat, "red", linewidth = 1.0, linestyle="--", label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL, 
                            color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")
        if i < 2:
            ax1.set_xticks([])
        if i == 2:
            ax1.set_xlabel("x")
        ax1.set_xlim([x_low, x_hig])
        ax1.set_ylim(0.0, global_max+limit_margin)
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $e_S:$ "+str(eL), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        # Set y-axis to scientific notation
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout(pad=0.2, h_pad=0.1)
    fig.savefig(FOLDER+'figs/phat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        max_value = max(np.max(np.abs(e1_true)), np.max(np.abs(e1_hat)))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        # if i == 0:
        #     ax1 = axs[0,0]
        #     ax1.set_ylabel("Error")
        # if i == 1:
        #     ax1 = axs[1,0]
        #     ax1.set_ylabel("Error")
        # if i == 2:
        #     ax1 = axs[2,0]
        #     ax1.set_xlabel("x")
        #     ax1.set_ylabel("Error")
        # if i == 3:
        #     ax1 = axs[0,1]
        # if i == 4:
        #     ax1 = axs[1,1]
        # if i == 5:
        #     ax1 = axs[2,1]
        #     ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat))
        eL = np.round(eL, 3)
        alpha = np.max(np.abs(e1_true-e1_hat))/ np.max(np.abs(e1_hat))
        alpha = np.round(alpha, 3)
        print("eL: ", np.round(eL, 3), "\t alpha: ", np.round(alpha,3))
        ax1 = axs[i]
        ax1.plot(x, e1_true, "black", linewidth=1.0, label=r"$e_1$")
        ax1.plot(x, e1_hat,  "red", linewidth=1.0, linestyle="--", label=r"$\hat{e}_1$")
        ax1.fill_between(x.reshape(-1), y1=0.0*p_hat.reshape(-1)+eL, y2=0.0*p_hat.reshape(-1)-eL, 
                         color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")
        if i < 2:
            ax1.set_xticks([])
        if i == 2:
            ax1.set_xlabel("x")

        ax1.set_xlim([x_low, x_hig])
        ax1.set_ylim(-(1.5*eL), 1.5*eL)
        # ax1.set_ylim(-(global_max), global_max)
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $\alpha_1:$ "+str(alpha), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        # Set y-axis to scientific notation
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout(pad=0.2, h_pad=0.1)
    fig.savefig(FOLDER+'figs/e1hat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    # global_max = float('-inf')
    # for i, (eres, pres) in enumerate(zip(e_res_list, p_res_list)):
    #     max_value = max(np.max(np.abs(pres/pnet.scale)), 0.0)
    #     global_max = max(global_max, max_value)
    # fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    # for i in range(0,6):
    #     pres = p_res_list[i]
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
    #         ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--", label=r"$D[\hat{e}_1]$")
    #         #ax1.plot(x, pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
    #         ax1.legend()  # Add legend only to the first subplot
    #     else:
    #         ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--")
    #         #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
    #     ax1.set_ylim([-global_max, global_max])
    #     ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    # plt.tight_layout()
    # # plt.savefig(FOLDER+"figs/enet_res.png")
    # fig.savefig(FOLDER+'figs/p_res.pdf', format='pdf', dpi=300)
    # plt.close()

    # global_max = float('-inf')
    # for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, e_res_list, p_res_list)):
    #     max_value = max(np.max(np.abs(eres/enet.scale)), 0.0)**2
    #     global_max = max(global_max, max_value)
    # fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    # for i in range(0,6):
    #     pres = p_res_list[i]
    #     eres = e_res_list[i]/enet.scale
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
    #         ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--", label=r"$r_2^2$")
    #         #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
    #         ax1.legend()  # Add legend only to the first subplot
    #     else:
    #         ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--")
    #         #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
    #     # ax1.set_ylim([0, global_max])
    #     ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    # plt.tight_layout()
    # # plt.savefig(FOLDER+"figs/enet_res.png")
    # fig.savefig(FOLDER+'figs/e1_res.pdf', format='pdf', dpi=300)
    # plt.close()


def show_table(pnet, enet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    gap_list = []
    a1_list = []
    eS_ratio_list = []
    t1s = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        phat = pnet(pt_x, pt_t1)
        ehat = enet(pt_x, pt_t1)[:,0].view(-1,1)
        phat = phat.data.cpu().numpy() # change tensor to numpy
        ehat = ehat.data.cpu().numpy() # change tensor to numpy
        e1 = p_monte - phat
        a1 = np.max(np.abs(e1 - ehat))/np.max(np.abs(ehat))
        a1_list.append(a1)
        eS = 2.0*np.max(np.abs(ehat))
        eS_ratio = eS/np.max(np.abs(p_monte))
        gap = (eS - np.max(np.abs(e1)))/np.max(np.abs(p_monte))
        gap_list.append(gap)
        eS_ratio_list.append(eS_ratio)
    
    print("[info] max a1: " +str(np.max(np.array(a1_list))) + ", avg a1:" + str(np.mean(np.array(a1_list))))
    print("[info] min gap: " +str(np.min(np.array(gap_list))) + ", max gap:" + str(np.max(np.array(gap_list))))
    print("[info] max eS_ratio: " +str(np.max(np.array(eS_ratio_list))) + ", avg eS_ratio:" + str(np.mean(np.array(eS_ratio_list))))
        


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
    plt.rcParams['font.size'] = 18
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

    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("  PDF")
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=18)
    y_ticks = np.array([0, 1, 2, 3, 4, 5])  # Example y-tick positions
    ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    ax.view_init(20, -60)
    plt.subplots_adjust(left=0.00, right=0.90, top=1.0, bottom=0.0)
    fig.savefig(FOLDER+'figs/phat_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1_surface(p_net, e1_net, num=100):
    plt.rcParams['font.size'] = 18
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

    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel(" e")
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=18)
    y_ticks = np.array([0, 1, 2, 3, 4, 5])  # Example y-tick positions
    ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    ax.view_init(20, -60)
    plt.subplots_adjust(left=0.00, right=0.90, top=1.0, bottom=0.0)
    fig.savefig(FOLDER+'figs/e1hat_surface_plot.pdf', format='pdf', dpi=300)


def plot_pres_surface(p_net, num=100):
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pres = p_res_func(pt_x, pt_t, p_net).data.cpu().numpy().reshape(num, -1)/p_net.scale

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, pres**2, cmap='Greys', alpha=0.8, label=r"$r_1^2$")
    x_samples = np.load(FOLDER+"output/p_xsamples.npy")
    t_samples = np.load(FOLDER+"output/p_tsamples.npy")
    z_max = 1.0*np.max(pres**2)
    ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="o", color="blue", s=2.0, alpha=0.05, label='sample points')
    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel(r"$r_1^2$")
    # ax.legend()
    ax.view_init(40, -60)
    zScalarFormatter = ScalarFormatterClass(useMathText=True)
    zScalarFormatter.set_powerlimits((0,0))
    ax.zaxis.set_major_formatter(zScalarFormatter)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/pres_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1res_surface(p_net, e1_net, num=100):
    x = np.linspace(x_low, x_hig, num=num, endpoint=True)
    t = np.linspace(t0, T_end, num=num, endpoint=True)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    e1res = e_res_func(pt_x, pt_t, e1_net, p_net).data.cpu().numpy().reshape(num, -1)
    e1res = e1res/e1_net.scale

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1res**2, cmap='Greys', alpha=0.8, label=r"$r_2^2$")
    x_samples = np.load(FOLDER+"output/e1_xsamples.npy")
    t_samples = np.load(FOLDER+"output/e1_tsamples.npy")
    z_max = 1.0*np.max(e1res**2)
    ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="o", color="blue", s=2.0, alpha=0.05, label='sample points')
    # ax.scatter(x_samples[-1], t_samples[-1], z_max, marker="x", color="red", s=10.0, label='Max Points')
    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel(r"$r_2^2$")
    ax.legend()
    ax.view_init(40, -60)
    zScalarFormatter = ScalarFormatterClass(useMathText=True)
    zScalarFormatter.set_powerlimits((0,0))
    ax.zaxis.set_major_formatter(zScalarFormatter)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/e1res_surface_plot.pdf', format='pdf', dpi=300)


def plot_train_loss(path_1, path_2):
    loss_history_1 = np.load(path_1)
    min_loss_1 = min(loss_history_1)
    loss_history_2 = np.load(path_2)
    min_loss_2 = min(loss_history_2)
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


def main():
    # plot_p_monte()
    mse_cost_function = torch.nn.MSELoss()
    
    p_model = PNet().to(device)
    e_model = E1Net().to(device)
    p_model.apply(init_weights_He)
    e_model.apply(init_weights_He)
    optimizer_p_model = torch.optim.Adam(p_model.parameters())
    optimizer_e_model = torch.optim.Adam(e_model.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_p_model, gamma=0.95)
    scheduler_e_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_e_model, gamma=0.95)
    
    p_model.scale = get_p_normalize()
    if(TRAIN_FLAG):
        train_pnet_model(p_model, optimizer_p_model, scheduler_p_model, mse_cost_function, iterations=15000); print("[p_net train complete]")
    p_model = load_trained_model(p_model, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    p_model.eval()

    e_model.scale = get_e1_normalize(p_model)
    if(TRAIN_FLAG):
        train_enet_model(p_model, e_model, optimizer_e_model, scheduler_e_model, mse_cost_function, iterations=25000); print("[e1_net train complete]")
    e_model = load_trained_model(e_model, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")
    e_model.eval()

    show_table(p_model, e_model)
    show_results(p_model, e_model)
    plot_p_surface(p_model)
    plot_pres_surface(p_model)
    plot_e1_surface(p_model, e_model)
    plot_e1res_surface(p_model, e_model)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy", 
                    FOLDER+"output/e1_net_train_loss.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()