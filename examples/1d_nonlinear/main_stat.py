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
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import norm

FOLDER = "exp1/main_stat/"
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

datas = ["10+8_samples/"]
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
        self.activation = nn.GELU()
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
        loss = mse_u + 5.0*mse_res + 5.0*mse_norm_res_input
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

    # Save intermediate enet
    save_count = 1
    save_loss = 10.0

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
        loss = mse_u + 5.0*mse_res #+ 5.0*mse_res_grad
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
            a1_data = compute_stat(p_net, e1_net)
            np.savez(FOLDER+'output_data/a1_data_'+str(save_count)+'.npz', loss=loss.item(), a1_data=a1_data)
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

        # save intermediate results
        if(loss.item() < 0.75*save_loss):
            a1_data = compute_stat(p_net, e1_net)
            np.savez(FOLDER+'output_data/a1_data_'+str(save_count)+'.npz', loss=loss.item(), a1_data=a1_data)
            save_count = save_count + 1
            save_loss = loss.item()

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

    a1_data = compute_stat(p_net, e1_net)
    np.savez(FOLDER+'output_data/a1_data_'+str(save_count)+'.npz', loss=loss.item(), a1_data=a1_data)
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
            ax1.legend(loc="upper right", fontsize=16)

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


def gaussian_mixture_pdf_1d(x, weights, means, covariances):
    """
    Computes the probability density function (PDF) of a 1D Gaussian Mixture Model (GMM)
    over an array of x values.

    Parameters:
    - x: np.ndarray, shape (num_points,), the points at which to evaluate the PDF
    - weights: np.ndarray, shape (N,), mixture weights (sum to 1)
    - means: np.ndarray, shape (N,), means of the Gaussians
    - covariances: np.ndarray, shape (N,), variances (not covariance matrices in 1D)

    Returns:
    - np.ndarray: The total PDF values at each x point (shape: (num_points,))
    """
    N = len(weights)  # Number of Gaussian components
    total_pdf = np.zeros_like(x)  # Initialize the PDF array
    for k in range(N):
        # Compute 1D Gaussian PDF for each component
        pdf_k = norm.pdf(x, loc=means[k], scale=np.sqrt(covariances[k]))  
        total_pdf += weights[k] * pdf_k  # Weighted sum
    return total_pdf


def show_results(pnet, enet):
    plt.rcParams['font.size'] = 18
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    dx = x[1] - x[0]
    print(x.shape)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    limit_margin = 0.1

    # load gmm data
    gmm = np.load(DATA_FOLDER+"others/1dnonlinear_gmm.npz")
    gmm_means = gmm["arr1"]
    gmm_covs  = gmm["arr2"]
    gmm_weights = gmm["arr3"]

    # Create a ScalarFormatter object
    formatter = ScalarFormatter()
    formatter.set_scientific(True)

    p_monte_list = []
    p_hat_list = []
    p_gmm_list = []
    e1_list = []
    e1_hat_list = []
    e2_list = []
    e2_hat_list = []
    p_res_list = []
    e_res_list = []
    e2_res_list = []
    t1s = [0.0, 3.0, 5.0]
    for i, t1 in enumerate(t1s):
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        # if(t1 == 0.0): p_monte = p_init(x)
        p_monte_list.append(p_monte)
        p_gmm = gaussian_mixture_pdf_1d(x, gmm_weights, gmm_means[:,i], gmm_covs[:,i]) # constant gmm_weights over time
        p_gmm_list.append(p_gmm)
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

    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 20,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 20,              # Title font size
    "axes.labelsize": 20,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 20,             # Legend text size
    "legend.title_fontsize": 20        # Legend title size (if you use legend titles)
    })

    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    N_dataset = 5
    palette = sns.color_palette("muted", N_dataset)
    global_max = float('-inf')
    for i, (phat) in enumerate(zip(p_hat_list)):
        max_value = np.max(np.abs(phat))
        global_max = max(global_max, max_value)
    for i, (p_monte, p_hat, p_gmm, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, p_gmm_list, e1_hat_list)):
        eL = 2.0 * np.max(np.abs(e1_hat)); eL = np.round(eL, 3)
        ax1 = axs[i]
        ax1.plot(x, p_monte, color="black", linewidth = 2.0, label=r"$p$")
        ax1.plot(x, p_hat, color="red", linewidth = 1.5, linestyle="--", label=r"$\hat{p}$")
        # ax1.plot(x[::6], p_hat[::6], color=palette[1], linewidth = 0.0, linestyle="-", marker="o", markersize=5, label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL, 
                         color=palette[2], alpha=0.5, label=r"$B_1$")
        ax1.plot(x, p_gmm, color="blue", linewidth = 1.5, linestyle="--", label=r"$\tilde{p}_{GM}$")
        # ax1.plot(x[::6], p_gmm[::6], color=palette[4], linewidth = 0.0, linestyle="-", marker="x", markersize=8, label=r"$\tilde{p}_{GM}$")
        print("[check] mean: ", np.sum(x*p_monte)*dx, np.sum(x*p_gmm)*dx)
        if i == 0:
            ax1.legend(loc="upper right", ncol=2, fontsize=24)
        if i == 2:
            ax1.set_xlabel("x")
        ax1.set_xlim([-5, 5])
        ax1.set_ylim(0.0, global_max+limit_margin)
        ax1.set_ylabel('PDF')
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
    for ax in axs.flat:
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    plt.tight_layout(pad=0.2, h_pad=0.1)
    # plt.tight_layout()
    fig.savefig(FOLDER+'figs/phat_eS.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
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
    fig.savefig(FOLDER+'figs/e1hat_eS.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
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
    a1_list = np.array(a1_list)
    gap_list = np.array(gap_list)
    norm_B2_list = eS_ratio_list
    print("[info] max a1: ", np.round(np.max(a1_list),2), ", avg a1:", np.round(np.mean(a1_list),2), ", std a1:", np.round(np.std(a1_list),3))
    print("[info] min gap: ", np.round(np.min(gap_list),3), ", avg gap:", np.round(np.mean(gap_list),3))
    print("[info] avg B2_norm: ", np.round(np.mean(norm_B2_list),2), ", std B2_norm:", np.round(np.std(norm_B2_list),3))


def compute_stat(pnet, enet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    a1_list = []
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
    a1_list = np.array(a1_list)
    return(a1_list)


# Paper: Fig.1(a)
def plot_a1_data(data_folder=FOLDER, number_of_data=12):
    # Get the last 3 colors from the "hls" palette with 8 colors
    # colors = sns.color_palette("hls", 8)[-3:]  # Indexes 5, 6, 7 (last three)   

    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 22,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 22,              # Title font size
    "axes.labelsize": 22,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 20,             # Legend text size
    "legend.title_fontsize": 20        # Legend title size (if you use legend titles)
    })

    fig, axs = plt.subplots(1, 1, figsize=(7, 6))
    width = 0.01
    # load data
    for i in range(1, number_of_data+1):
        data_i = np.load(data_folder+'output_data/a1_data_'+str(i)+'.npz')
        loss = data_i['loss']
        a1_data   = data_i['a1_data']
        axs.scatter(loss, np.max(a1_data), s=15, color="blue")
        axs.plot([loss-width, loss+width], [np.mean(a1_data), np.mean(a1_data)], color="blue", linewidth=1)
        axs.add_patch(Rectangle((loss-width, np.mean(a1_data)-np.std(a1_data)), 2*width, 2*np.std(a1_data),
             edgecolor = 'blue',
             facecolor = 'none',
             fill=False,
             lw=1))
    # axs.legend(loc="upper right", framealpha=0.3)
    axs.set_xlabel('Train loss')
    axs.set_ylabel('$a_1$')
    axs.set_ylim([0, 4])
    axs.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    # plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.show()
    # plt.tight_layout()
    # plt.savefig(FOLDER+'figs/1dou_compare_error_bounds.pdf', format='pdf', dpi=300)
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
    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 18,                   # Base font size for non-math text
    # "figure.autolayout": True,
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 18,              # Title font size
    "axes.labelsize": 18,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 18,             # Legend text size
    "legend.title_fontsize": 18        # Legend title size (if you use legend titles)
    })

    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    phat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)

    p_list = []
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        p_list.append(p_monte)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.8, label=r"$\hat{p}$")
    # z_max = 1.5*np.max(np.abs(phat))
    # ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="black", s=0.02, label='Data Points')
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x_monte*0 + t1
        if i == 0:
            ax.plot(x_monte, t1_monte, p_list[i], color="black", label=r"$p$")
        else:
            ax.plot(x_monte, t1_monte, p_list[i], color="black")

    ax.view_init(20, -50)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.text2D(0.94, 0.77, "PDF", transform=ax.transAxes)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc='lower right', bbox_to_anchor=(0.6, 0.60), fontsize=28)  
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/1dnl_phatsurface.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def plot_e1_surface(p_net, e1_net, num=100):
    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 18,                   # Base font size for non-math text
    # "figure.autolayout": True,
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 18,              # Title font size
    "axes.labelsize": 18,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 18,             # Legend text size
    "legend.title_fontsize": 18        # Legend title size (if you use legend titles)
    })

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
    t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_t1_monte = Variable(torch.from_numpy(x_monte*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x_monte, pt_t1_monte).data.cpu().numpy()
        e1 = p_monte - p_hat
        e1_list.append(e1)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1hat, cmap='inferno', alpha=0.6, label=r"$\hat{e}_1$")
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x_monte*0 + t1
        if(i == 0):
            ax.plot(x_monte, t1_monte, e1_list[i], color="black", label=r"$e_1$")
        else:
            ax.plot(x_monte, t1_monte, e1_list[i], color="black")

    ax.view_init(20, -50)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.text2D(0.94, 0.77, "Error", transform=ax.transAxes)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc='lower right', bbox_to_anchor=(0.5, 0.65), fontsize=28)  
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/1dnl_e1hatsurface.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


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

    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 22,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 22,              # Title font size
    "axes.labelsize": 22,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 20,             # Legend text size
    "legend.title_fontsize": 20        # Legend title size (if you use legend titles)
    })
    # Get the last 3 colors from the "hls" palette with 8 colors
    colors = sns.color_palette("muted", 2)

    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    axs[0].plot(np.arange(len(loss_history_1)), loss_history_1, color="black", linewidth=1.0)
    axs[0].set_ylim([min_loss_1, 10*min_loss_1])
    axs[1].plot(np.arange(len(loss_history_2)), loss_history_2, color="black", linewidth=1.0)
    axs[1].set_ylim([min_loss_2, 10*min_loss_2])
    axs[0].set_xlabel("iterations")
    axs[1].set_xlabel("iterations")
    axs[0].set_ylabel("train loss: "+r"$\hat{p}$")
    axs[1].set_ylabel("train loss: "+r"$\hat{e}_1$")
    axs[0].grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    axs[1].grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/1dnl_trainloss.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
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
    if(False):
        train_pnet_model(p_model, optimizer_p_model, scheduler_p_model, mse_cost_function, iterations=20000); print("[p_net train complete]")
    p_model = load_trained_model(p_model, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    p_model.eval()

    e_model.scale = get_e1_normalize(p_model)
    if(TRAIN_FLAG):
        train_enet_model(p_model, e_model, optimizer_e_model, scheduler_e_model, mse_cost_function, iterations=20000); print("[e1_net train complete]")
    e_model = load_trained_model(e_model, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")
    e_model.eval()

    show_table(p_model, e_model)
    show_results(p_model, e_model)
    # plot_a1_data()
    plot_p_surface(p_model)
    # plot_pres_surface(p_model)
    plot_e1_surface(p_model, e_model)
    # plot_e1res_surface(p_model, e_model)
    # plot_train_loss(FOLDER+"output/p_net_train_loss.npy", 
    #                FOLDER+"output/e1_net_train_loss.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()