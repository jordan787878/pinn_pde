import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter
import time

FOLDER = ""

device = "cpu"
print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

x0 = 1
x_low = -6
x_hig = 6
alpha = 0.2
D = 0.2

t0 = 1
T_end = 3
dt = 0.1

def p_exact(x,t):
  return np.sqrt(alpha/(2*np.pi*D*(1-np.exp(-2*alpha*t)))) * np.exp(-1*alpha*(x-x0*np.exp(-alpha*t))**2/(2*D*(1-np.exp(-2*alpha*t))))


def p_init(x):
  return p_exact(x, t0)


def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def test_p_init():
  x = np.arange(x_low,x_hig,0.01).reshape(-1,1)
  p0 = p_init(x)
  print(np.sum(p0)*0.01)
  plt.plot(x,p0, "b", label="t0")
  p1 = p_exact(x, T_end)
  print(np.sum(p1)*0.01)
  plt.plot(x,p1, "r", label="T_end")
  plt.xlabel("x")
  plt.ylabel("pdf")
  plt.legend()


def res_func(x,t, net, verbose=False):
    global D
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t - alpha*(p_x*x + p) - D*p_xx
    if(verbose):
      print(p)
      print(residual)
      print(residual.shape)
    return residual


# p_net
class Net(nn.Module):
    def __init__(self):
        neurons = 32
        self.normalize = 1.0
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus(self.hidden_layer1(inputs))
        layer2_out = F.softplus(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out)
        output = F.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def train_p_net(p_net, optimizer, mse_cost_function, scheduler):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 2000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/p_net.pt"
    normalize = p_net.normalize
    iterations_per_decay = 1000

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        u_bc = p_init(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = res_func(x, t, p_net)/normalize
        mse_f = mse_cost_function(f_out, all_zeros)

        loss = mse_u + mse_f

        loss_history.append(loss.data)

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        # Save the min loss model
        if(loss.data < min_loss):
            print("save epoch:", epoch, ", loss:", loss.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data
        # with torch.autograd.no_grad():
        #     if (epoch%4000 == 0):
        #         print(epoch,"Traning Loss:",loss.data)
    np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))


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
    print(len(loss_history))
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/pnet_loss_history.png")
    plt.close()
    return p_net


def get_e1_normalize(p_net):
    global x_low, x_hig, X_sim, P_sim_t20, t0
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 2
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_exact0 = p_init(x)
    max_abs_e1_x_0 = max(abs(p_exact0-p_approx0))

    return max_abs_e1_x_0[0]


def plot_p_net_results(p_net):
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = [1.0, 2.0, 3.0]
    colors = ["black","blue","green"]
    
    plt.figure()
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        plt.plot(x, p, color=colors[i], linestyle="-", linewidth=1.0, label=r"$p$(x,"+str(t1)+")")
        plt.plot(x, phat, color=colors[i], linestyle="--", linewidth=1.0, label=r"$\hat{p}$(x,"+str(t1)+")")
    plt.legend()
    plt.grid(linewidth=0.5)
    plt.savefig(FOLDER+"figs/phat_result.png")
    plt.close()
    


class E1Net(nn.Module):
    def __init__(self, scale=1.0):
        neurons = 32
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.tanh((self.hidden_layer1(inputs)))
        layer2_out = F.tanh((self.hidden_layer2(layer1_out)))
        output = self.scale * self.output_layer(layer2_out)
        return output


def e1_res_func(x, t, e1_net, p_net, verbose=False):
    global D
    e = e1_net(x,t)
    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    p_res = res_func(x, t, p_net)
    residual = e_t - alpha*(e_x*x + e) - D*e_xx + p_res
    return residual


def train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0, scheduler):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 2000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    iterations_per_decay = 1000
    
    x_mar = 0
    t_mar = 0

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        t_bc = 0*x_bc + t0
        p_bc = p_init(x_bc.detach().numpy())
        p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out, u_bc)
        mse_u = mse_u/max_abs_e1_x_0

        # Loss based on PDE
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0 + 2*t_mar) + t0-t_mar).to(device)
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = e1_res_func(x, t, e1_net, p_net)
        mse_res = mse_cost_function(f_out, all_zeros)/max_abs_e1_x_0
        
        # Combining the loss functions
        loss = mse_u + mse_res
        loss_history.append(loss.data)

        loss.backward() 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data) # , ",a1(t0):", alpha1_t0.data, ",ic loss:", mse_u.data, ",res:" , mse_res_t0.data, mse_res.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
    np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))


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
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/e1net_loss_history.png")
    plt.close()
    return e1_net

# construct artificial e2hat 
e2_hat_mag = 5e-3
e2_hat_freq = 5
e2_hat_drift = 1e-5

def plot_tight_error_bounds(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = [1.5, 2.0, 3.0]
    colors = ["black","black","black"]

    fig, axs = plt.subplots(3, 1, figsize=(7, 6))
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x) + e2_hat_drift
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        a2 = max(abs(e2-e2_hat))/max(abs(e2_hat))
        axs[i].plot(x, e2, color=colors[i], linestyle="-", linewidth=1.0, label=r"$e_2$")
        axs[i].plot(x, e2_hat, linestyle="--", color = colors[i], linewidth=1.0, label=r"$\hat{e}_2$")
        axs[i].grid(linewidth=0.5)
        axs[i].legend(loc="upper right")
        # Add text to the left top corner
        axs[i].text(0.01, 0.98, "t="+str(t1), transform=axs[i].transAxes, verticalalignment='top', fontsize=8)
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/e2hat_result.pdf', format='pdf', dpi=300)
    plt.close()
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x) + e2_hat_drift
        r_21 = max(abs(e2_hat))/ max(abs(e1_hat))
        eB = max(abs(e1_hat))*(1/(1-r_21))
        eB = np.round(eB, 4)
        eL = 2*max(abs(e1_hat))[0]
        eL = np.round(eL,3)
        if i == 0:
            axs[i].plot(x, p, color=colors[i], linestyle="-", linewidth=1.0, label=r"$p$")
            axs[i].plot(x, phat, linestyle="--", color = colors[i], linewidth=1.0, label=r"$\hat{p}$")
            axs[i].fill_between(x.reshape(-1), y1=phat.reshape(-1)+eL, y2=phat.reshape(-1)-eL, color="green", alpha=0.3, label=r"$e_S$")
            axs[i].legend(loc="upper right")
        else:
            axs[i].plot(x, p, color=colors[i], linestyle="-", linewidth=1.0)
            axs[i].plot(x, phat, linestyle="--", color = colors[i], linewidth=1.0)
            axs[i].fill_between(x.reshape(-1), y1=phat.reshape(-1)+eL, y2=phat.reshape(-1)-eL, color="green", alpha=0.3)
        axs[i].set_ylim([0, 0.75])
        axs[i].grid(linewidth=0.5)
        axs[i].text(0.01, 0.98, "t="+str(t1)+", "+r"$e_S=$"+str(eL), transform=axs[i].transAxes, verticalalignment='top', fontsize=8)
        axs[i].set_xlim([-3,3])
        axs[i].set_ylabel("PDF")
    axs[2].set_xlabel("x")
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/error_bounds_result.pdf', format='pdf', dpi=300)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x) + e2_hat_drift
        r_21 = max(abs(e2_hat))/ max(abs(e1_hat))
        eB = max(abs(e1_hat))*(1/(1-r_21))
        eB = np.round(eB,3)
        eL = 2*max(abs(e1_hat))
        eL = np.round(eL,3)
        print("Error Bounds: tight, special", eB, eL)
        if i == 0:
            axs[i].plot(x, e1, color=colors[i], linestyle="-", linewidth=1.0, label=r"$e_1$")
            axs[i].plot(x, e1_hat, linestyle="--", color = colors[i], linewidth=1.0, label=r"$\hat{e}_1$")
            axs[i].plot(x, x*0+eB, linestyle=":", color = colors[i], linewidth=1.0, label=r"$e_B$")
            axs[i].plot(x, x*0-eB, linestyle=":", color = colors[i], linewidth=1.0)
            axs[i].fill_between(x.reshape(-1), y1=0*phat.reshape(-1)+eL, y2=0*phat.reshape(-1)-eL, color="green", alpha=0.3, label=r"$e_S$")
            axs[i].legend(loc="upper right")
        else:
            axs[i].plot(x, e1, color=colors[i], linestyle="-", linewidth=1.0)
            axs[i].plot(x, e1_hat, linestyle="--", color = colors[i], linewidth=1.0)
            axs[i].plot(x, x*0+eB, linestyle=":", color = colors[i], linewidth=1.0)
            axs[i].plot(x, x*0-eB, linestyle=":", color = colors[i], linewidth=1.0)
            axs[i].fill_between(x.reshape(-1), y1=0*phat.reshape(-1)+eL, y2=0*phat.reshape(-1)-eL, color="green", alpha=0.3, label=r"$e_S$")
        axs[i].grid(linewidth=0.5)
        axs[i].text(0.01, 0.95, "t="+str(t1)+", "+r"$e_B=$"+str(eB[0])+", "+r"$e_S=$"+str(eL[0]),
                    transform=axs[i].transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        axs[i].set_xlim([-3,3])
        axs[i].set_ylim([-2*eL, 2*eL])
    axs[2].set_xlabel('x')
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/e1hat_result.pdf', format='pdf', dpi=300)
    plt.close()


def plot_alphas(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = np.arange(1.0, 3.0+0.01, 0.01)

    a1_list = []
    a2_list = []
    e1_list = []
    eS_list = []
    eB_list = []

    for i in range(len(t1s)):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_list.append(max(abs(e1))[0])
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x) + e2_hat_drift
        r_21 = max(abs(e2_hat))/ max(abs(e1_hat))
        eB = max(abs(e1_hat))*(1/(1-r_21))
        eB = np.round(eB,3)
        eB_list.append(eB)
        eS = max(abs(e1_hat))*2
        eS = np.round(eS,3)
        eS_list.append(eS)
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        a2 = max(abs(e2-e2_hat))/max(abs(e2_hat))
        a1_list.append(a1[0])
        a2_list.append(a2[0])
        # print(a1, a2, (1-a1), a2, a1*a1, a2*(1+a2))

    # print("a1_list", a1_list)
    a1_list = np.array(a1_list)
    a2_list = np.array(a2_list)
    cond_1 = (1-a1_list)
    y_1 = a2_list
    cond_2 = a1_list**2
    y_2 = a2_list*(1+a2_list)

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    axs[0,0].plot(t1s, eB_list, color="black", linestyle="-", linewidth=1.0, label=r"$e_B$")
    axs[0,0].plot(t1s, eS_list, color="black", linestyle=":", linewidth=1.0, label=r"$e_S$")
    axs[0,0].plot(t1s, e1_list, color="black", linestyle="--", linewidth=1.0, label=r"$\max|e|$")
    axs[0,0].legend(loc="upper right")
    axs[0,0].grid(linewidth=0.5)

    axs[0,1].plot(t1s, a1_list, color="black", linestyle="--", linewidth=1.0, label=r"$\alpha_1$")
    axs[0,1].plot(t1s, t1s*0+1.0, color="black", linestyle="-", linewidth=1.0,)
    axs[0,1].legend(loc="lower right")
    axs[0,1].grid(linewidth=0.5)
    axs[0,1].text(0.01, 0.98, "condition: " + r"$\alpha_1(t)<1$", 
                  transform=axs[0,1].transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
    axs[1,0].plot(t1s, cond_1, color="black", linewidth=1.0, linestyle="-", label=r"$1-\alpha_1$")
    axs[1,0].plot(t1s, y_1, color="black", linewidth=1.0, linestyle="--", label=r"$\alpha_2$")
    axs[1,0].set_xlabel('t')
    axs[1,0].legend(loc="lower right")
    axs[1,0].grid(linewidth=0.5)
    axs[1,0].text(0.01, 0.98, "condition: "+r"$\alpha_2 < 1-\alpha_1$", 
                  transform=axs[1,0].transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    axs[1,1].plot(t1s, cond_2, color="black", linewidth=1.0, linestyle="-", label=r"$\alpha_1^2$")
    axs[1,1].plot(t1s, y_2, color="black", linewidth=1.0, linestyle="--", label=r"$\alpha_2(1+\alpha_2)$")
    axs[1,1].set_xlabel('t')
    axs[1,1].legend(loc="lower right")
    axs[1,1].grid(linewidth=0.5)
    axs[1,1].text(0.01, 0.98, "condition: "+r"$\alpha_2(1+\alpha_2) <\alpha_1^2$", 
                  transform=axs[1,1].transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/error_and_conditions.png")
    plt.close()
    # plt.show()


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


def plot_p_surface(p_net, num=100):
    t1s = [1.0, 2.0, 3.0]
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    phat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)

    p_list = []
    for t1 in t1s:
        p_true = p_exact(x, x*0+t1)
        p_list.append(p_true)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.7, label=r"$\hat{p}$")
    z_max = 1.5*np.max(np.abs(phat))
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x*0 + t1
        if i == 0:
            ax.plot(x, t1_monte, p_list[i], color="black", label=r"$p$")
        else:
            ax.plot(x, t1_monte, p_list[i], color="black")

    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("PDF")
    ax.legend()
    y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    ax.view_init(20, -50)
    plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/phat_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1_surface(p_net, e1_net, num=100):
    t1s = [1.0, 2.0, 3.0]
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    e1hat = e1_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)
    
    e1_list = []
    x_monte = x.reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = p_exact(x, x*0+t1).reshape(-1, 1)
        pt_t1_monte = Variable(torch.from_numpy(x_monte*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x_monte, pt_t1_monte).data.cpu().numpy()
        e1 = p_monte - p_hat
        e1_list.append(e1)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1hat, cmap='viridis', alpha=0.7, label=r"$\hat{e}_1$")
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
    y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("Error")
    ax.view_init(20, -50)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/e1hat_surface_plot.pdf', format='pdf', dpi=300)


def main():
    # create p_net
    p_net = Net()
    p_net = p_net.to(device)
    p_net.normalize = get_p_normalize()
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    start_time = time.time()
    # train_p_net(p_net, optimizer, mse_cost_function, scheduler_p_model); print("p_net train complete")
    time_train_p = time.time() - start_time
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    max_abs_e1_x_0 = get_e1_normalize(p_net)

    # create e1_net
    e1_net = E1Net(scale=max_abs_e1_x_0)
    e1_net = e1_net.to(device)
    optimizer = torch.optim.Adam(e1_net.parameters())
    scheduler_e1_model = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    start_time = time.time()
    # train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0, scheduler_e1_model); print("e1_net train complete")
    time_train_e = time.time() - start_time
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")

    plot_tight_error_bounds(p_net, e1_net)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy", FOLDER+"output/e1_net_train_loss.npy")
    plot_alphas(p_net, e1_net)
    plot_p_surface(p_net)
    plot_e1_surface(p_net, e1_net)

    print("[complete 1d OU]")
    print(f"train pnet time: {time_train_p:.4f} seconds")
    print(f"train enet time: {time_train_e:.4f} seconds")


if __name__ == "__main__":
    main()


# Log
# train pnet time: 6.5799 seconds
# train enet time: 8.9959 seconds