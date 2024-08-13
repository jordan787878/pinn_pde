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
batch_size = 1000
S = 10000


def p_init(x):
    return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def p_res_func(x, t, out, verbose=False):
    p = out
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*const_a*x*x + 2*const_b*x + const_c)*p \
                   + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*p_x \
                   - 0.5*const_e*const_e*p_xx
    if(verbose):
        print(p_xx[0:10,:])
        # print(p_x.shape, p_t.shape, p_xx.shape, residual.shape)
    return residual


def e_res_func(x, t, out, verbose=False):
    e1_out = out
    e_x = torch.autograd.grad(e1_out, x, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_t = torch.autograd.grad(e1_out, t, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x,  x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    residual = e_t + (3*const_a*x*x + 2*const_b*x + const_c)*e1_out \
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
  
 

def get_e1_normalize(pnet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    p0_true = p_init(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t = pt_x*0.0 + t0
    p0_hat = pnet(pt_x, pt_t).data.cpu().numpy()
    e0_true = p0_true - p0_hat
    return np.max(np.abs(e0_true))


def train_pnet_model(net, optimizer, scheduler, mse_cost_function, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/pnet.pt"
    PATH_LOSS = FOLDER+"output/pnet_train_loss.npy"
    iterations_per_decay = 1000

    margin_t = 0.05*5
    margin_x = 0.05*12
    x_bc = (torch.rand(batch_size, 1)*(x_hig-x_low+2*margin_x)+x_low-margin_x).to(device)
    x_bc = torch.clamp(x_bc, min=x_low, max=x_hig)
    t_bc = (torch.ones(batch_size, 1) * t0).to(device)

    t = (torch.rand(batch_size, 1, requires_grad=True)*(T_end-t0+2*margin_t)    +t0-margin_t).to(device)
    t = torch.clamp(t, min=t0, max=T_end)
    x = (torch.rand(len(t),     1, requires_grad=True)*(x_hig-x_low+2*margin_x) +x_low-margin_x).to(device)
    x = torch.clamp(x, min=x_low, max=x_hig)

    FLAG = 0
    w_regular = 1.0

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # p init loss
        p0 = p_init(x_bc.detach().numpy())
        p0 = Variable(torch.from_numpy(p0).float(), requires_grad=False).to(device)
        p0_hat = net(x_bc, t_bc)
        mse_ic_p = mse_cost_function(p0_hat, p0)

        # p residual loss
        net_out = net(x, t)
        p_res = p_res_func(x, t, net_out, verbose=False)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res_p = mse_cost_function(p_res, all_zeros)
        res_x = torch.autograd.grad((p_res), x, grad_outputs=torch.ones_like(p_res), create_graph=True)[0]
        res_t = torch.autograd.grad((p_res), t, grad_outputs=torch.ones_like(p_res), create_graph=True)[0]
        mse_res_p_g = torch.mean(res_x**2 + res_t**2)

        loss = (mse_ic_p + mse_res_p + w_regular*mse_res_p_g)
        loss_history.append(loss.data)

        if ((epoch+1)%1000 == 0):
            print(epoch+1," Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("epoch:", epoch, ",loss:", loss.data, 
                  ",ic:",mse_ic_p.data, 
                  ",res:", mse_res_p.data,
                  ",res g:", mse_res_p_g.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label':"pnet",
                    }, PATH)
            min_loss = loss.data
            FLAG = FLAG + 1

        if(loss.data < 0.0):
            print("epoch:", epoch, ",loss:", loss.data, 
                  ",ic:",mse_ic_p.data, 
                  ",res:", mse_res_p.data,
                  ",res g:", mse_res_p_g.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label':"pnet",
                    }, PATH)
            return

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        # RAR
        if ((epoch+1)%500==0 and FLAG >= 3):
            # sample points
            print("Sample N (p,e): ", x.shape)
            t_RAR = (torch.rand(S, 1, requires_grad=True)*(T_end-t0+2*margin_t)    +t0-margin_t).to(device)
            t_RAR = torch.clamp(t_RAR, min=t0, max=T_end)
            x_RAR = (torch.rand(S, 1, requires_grad=True)*(x_hig-x_low+2*margin_x) +x_low-margin_x).to(device)
            x_RAR = torch.clamp(x_RAR, min=x_low, max=x_hig)
            
            net_out_RAR = net(x_RAR, t_RAR)
            p_res_RAR = p_res_func(x_RAR, t_RAR, net_out_RAR)

            # add to p residual samples
            max_abs_res, max_index = torch.topk(torch.abs(p_res_RAR), 10, dim=0)
            x_max = x_RAR[max_index].reshape(-1,1)
            t_max = t_RAR[max_index].reshape(-1,1)
            x = torch.cat((x, x_max), dim=0)
            t = torch.cat((t, t_max), dim=0)
            print("... RES add [x,t]:", x_max[0].data, t_max[0].data, ". max res value: ", max_abs_res[0].data)
            FLAG = 0


def train_enet_model(pnet, enet, optimizer, scheduler, mse_cost_function, iterations=40000):
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/enet.pt"
    PATH_LOSS = FOLDER+"output/enet_train_loss.npy"
    iterations_per_decay = 1000

    margin_t = 0.05*5
    margin_x = 0.05*12
    x_bc = (torch.rand(batch_size, 1)*(x_hig-x_low+2*margin_x)+x_low-margin_x).to(device)
    x_bc = torch.clamp(x_bc, min=x_low, max=x_hig)
    t_bc = (torch.ones(batch_size, 1) * t0).to(device)

    t = (torch.rand(batch_size, 1, requires_grad=True)*(T_end-t0+2*margin_t)    +t0-margin_t).to(device)
    t = torch.clamp(t, min=t0, max=T_end)
    x = (torch.rand(len(t),     1, requires_grad=True)*(x_hig-x_low+2*margin_x) +x_low-margin_x).to(device)
    x = torch.clamp(x, min=x_low, max=x_hig)

    FLAG = 0
    w_regular = 1.0

    for epoch in range(iterations):
        optimizer.zero_grad()

        # get e(x,0) true
        p0 = p_init(x_bc.detach().numpy())
        p0 = Variable(torch.from_numpy(p0).float(), requires_grad=False).to(device)
        p0_hat = pnet(x_bc, t_bc).detach().numpy()
        p0_hat = Variable(torch.from_numpy(p0_hat).float(), requires_grad=False).to(device)
        e0_true = p0 - p0_hat

        normalize = enet.scale
        e0_hat = enet(x_bc, t_bc)
        mse_ic_e = mse_cost_function(e0_hat/normalize, e0_true/normalize) 
        e_res = (e_res_func(x, t, enet(x,t)) + p_res_func(x, t, pnet(x,t)))/normalize
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res_e = mse_cost_function(e_res, all_zeros)
        # e_res = e_res_func(x, t, enet(x, t))
        # e_res_target = -p_res_func(x, t, pnet(x,t)).data.cpu().numpy()
        # e_res_target = Variable(torch.from_numpy(e_res_target).float(), requires_grad=False).to(device)
        # mse_res_e = mse_cost_function(e_res/normalize, e_res_target/normalize)

        #_e_res = (e_res + p_res_func(x, t, pnet(x,t)))/normalize
        _e_res = e_res
        _res_x = torch.autograd.grad((_e_res), x, grad_outputs=torch.ones_like(_e_res), create_graph=True)[0]
        _res_t = torch.autograd.grad((_e_res), t, grad_outputs=torch.ones_like(_e_res), create_graph=True)[0]
        res_input = torch.cat([_res_x, _res_t], axis=1)
        norm_res_g = torch.norm(res_input, dim=1).view(-1,1)
        mse_res_e_g = mse_cost_function(norm_res_g, all_zeros)
        # mse_res_e_g = torch.mean(_res_x**2 + _res_t**2)

        loss = (mse_ic_e + w_regular*mse_res_e + w_regular*mse_res_e_g)
        loss_history.append(loss.data)

        if ((epoch+1)%1000 == 0):
            print(epoch+1," Traning Loss:",loss.data)
            np.save(PATH_LOSS, np.array(loss_history))

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("epoch:", epoch, ",loss:", loss.data, 
                  ",ic:",mse_ic_e.data, 
                  ",res:", mse_res_e.data,
                  ",res g:", mse_res_e_g.data,
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': enet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'label':"enet",
                    }, PATH)
            min_loss = loss.data
            FLAG = FLAG + 1

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        # RAR
        if ((epoch+1)%500==0 and FLAG >= 2):
            # sample points
            print("Sample N (p,e): ", x.shape)
            t_RAR = (torch.rand(S, 1, requires_grad=True)*(T_end-t0+2*margin_t)    +t0-margin_t).to(device)
            t_RAR = torch.clamp(t_RAR, min=t0, max=T_end)
            x_RAR = (torch.rand(S, 1, requires_grad=True)*(x_hig-x_low+2*margin_x) +x_low-margin_x).to(device)
            x_RAR = torch.clamp(x_RAR, min=x_low, max=x_hig)
            # t_RAR, x_RAR = generate_RAR_samples()

            e_res_RAR = e_res_func(x_RAR, t_RAR, enet(x_RAR,t_RAR)) + p_res_func(x_RAR, t_RAR, pnet(x_RAR,t_RAR))
            # add to e residual samples
            max_abs_res, max_index = torch.topk(torch.abs(e_res_RAR), 10, dim=0)
            x_max = x_RAR[max_index].reshape(-1,1)
            t_max = t_RAR[max_index].reshape(-1,1)
            x = torch.cat((x, x_max), dim=0)
            t = torch.cat((t, t_max), dim=0)
            print("... RES add [x,t]:", x_max.data[0], t_max[0].data, ". max res value: ", max_abs_res[0].data)
            FLAG = 0


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
        pres = p_res_func(pt_x, pt_t1, phat).data.cpu().numpy()
        eres = e_res_func(pt_x, pt_t1, ehat).data.cpu().numpy() + pres
        
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
        ax1.set_ylim(-(global_max+limit_margin), global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/enet_result.png")
    plt.close()

    global_max = float('-inf')
    for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, e_res_list, p_res_list)):
        max_value = max(np.max(np.abs(pres)), 0.0)
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
            ax1.plot(x, eres-pres, "red", linestyle="--", label=r"$r_2$")
            ax1.plot(x, -pres, "blue", linestyle="-")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres-pres, "red", linestyle="--")
            ax1.plot(x, -pres, "blue", linestyle="-")
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
    mse_cost_function = torch.nn.MSELoss()
    
    p_model = PNet().to(device)
    e_model = ENet().to(device)
    p_model.apply(init_weights)
    e_model.apply(init_weights)
    optimizer_p_model = torch.optim.Adam(p_model.parameters())
    optimizer_e_model = torch.optim.Adam(e_model.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_p_model, gamma=0.95)
    scheduler_e_model = torch.optim.lr_scheduler.ExponentialLR(optimizer_e_model, gamma=0.95)
    
    p_model = load_trained_model(p_model, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    p_model.eval()

    e_model.scale = get_e1_normalize(p_model)
    print("enet scale: ", e_model.scale)
    e_model = load_trained_model(e_model, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")
    e_model.eval()
    show_results(p_model, e_model)


if __name__ == "__main__":
    main()
