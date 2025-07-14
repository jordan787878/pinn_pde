import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import random
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter
import time
import argparse
import seaborn as sns

FOLDER = ""
TRAIN_FLAG = False

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


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


# p_net
class Net(nn.Module):
    def __init__(self):
        neurons = 32
        self.scale = 1.0
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus(self.hidden_layer1(inputs))
        layer2_out = F.softplus(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out)
        output = F.softplus(output)
        return output


def train_p_net(p_net, optimizer, mse_cost_function, scheduler):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 2000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/p_net.pt"
    normalize = p_net.scale
    iterations_per_decay = 1000
    start_time = time.time()

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
        f_out = res_func(x, t, p_net)
        mse_f = mse_cost_function(f_out/normalize, all_zeros)

        loss = mse_u + 2.0*mse_f
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("save epoch:", epoch, ", loss:", loss.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': time.time() - start_time
                    }, PATH)
            min_loss = loss.data

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

    np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best pnet epoch: ", epoch, ", loss:", loss.data, "train time:", checkpoint['train_time'])
    # see training result
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
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
        plt.plot(x, phat, color="red", linestyle="--", linewidth=1.0, label=r"$\hat{p}$(x,"+str(t1)+")")
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
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
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


def diff_e1(x, t, e1_net):
    global D
    e = e1_net(x,t)
    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    return e_t - alpha*(e_x*x + e) - D*e_xx


def train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0, scheduler):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 4000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    iterations_per_decay = 1000
    normalize = max_abs_e1_x_0
    start_time = time.time()

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
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # Loss based on PDE
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        diff_e1_hat = diff_e1(x, t, e1_net)
        diff_e1_target = -res_func(x, t, p_net)
        mse_res = mse_cost_function(diff_e1_hat/normalize, diff_e1_target/normalize)
        
        # Combining the loss functions
        loss = mse_u + 2.0*mse_res
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("e1net epoch:", epoch, ",loss:", loss.data, ",ic loss:", mse_u.data, ",res:", mse_res.data) # , ",a1(t0):", alpha1_t0.data, ",ic loss:", mse_u.data, ",res:" , mse_res_t0.data, mse_res.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': time.time() - start_time
                    }, PATH)
            min_loss = loss.data 
        
        loss.backward() 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

    np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best e1net epoch: ", epoch, ", loss:", loss.data, "train time:", checkpoint['train_time'])
    # see training result
    # keys = e1_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(e1_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
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
e2_hat_mag = 2e-2
e2_hat_freq = 5
e2_hat_drift = 1e-4


def plot_tight_error_bounds(p_net, e1_net):
    plt.rcParams['font.size'] = 18
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = [1.5, 2.0, 3.0]
    colors = ["black","black","black"]

    fig, axs = plt.subplots(3, 1, figsize=(6, 6))
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
        axs[i].plot(x, e2_hat, linestyle="--", color = "red", linewidth=1.0, label=r"$\hat{e}_2$")
        axs[i].grid(linewidth=0.5)
        axs[i].legend(loc="upper right")
        # Add text to the left top corner
        axs[i].text(0.01, 0.98, "t="+str(t1), transform=axs[i].transAxes, verticalalignment='top', fontsize=8)
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/e2hat_result.pdf', format='pdf', dpi=300)
    plt.close()
    
    # paper
    colors = sns.color_palette("Set2", 3)
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

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
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
        axs[i].plot(x, p, color="black", linestyle="-", linewidth=2.0, label=r"$p$")
        axs[i].plot(x, phat, linestyle="--", color="red", linewidth=2.0, label=r"$\hat{p}$")
        axs[i].fill_between(x.reshape(-1), y1=phat.reshape(-1)+eL, y2=phat.reshape(-1)-eL, color=colors[0], alpha=0.5, label=r"$B_1$")
        axs[i].fill_between(x.reshape(-1), y1=phat.reshape(-1)+eB, y2=phat.reshape(-1)-eB, color=colors[1], alpha=0.5, label=r"$B_2$")
        if i == 0:
            axs[i].legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.25))
        axs[i].set_ylim([0, 0.75])
        axs[i].set_xlim([-2.5,3])
        axs[i].text(0.01, 0.98, "t="+str(t1), transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        axs[i].set_ylabel("PDF")
    axs[2].set_xlabel(r"$x$")
    # Format y-ticks to display two decimal places for all subplots
    for ax in axs.flat:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # Add dotted grid to each subplot
    for ax in axs.flat:
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/1dl_errorboundsresult.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
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
        print("Error Bounds: 2nd, 1st", eB, eL)
        if i == 0:
            axs[i].plot(x, e1, color=colors[i], linestyle="-", linewidth=1.0, label=r"$e_1$")
            axs[i].plot(x, e1_hat, linestyle="--", color = "red", linewidth=1.0, label=r"$\hat{e}_1$")
            axs[i].plot(x, x*0+eB, linestyle=":", color = "blue", linewidth=1.0, label=r"$e_B$")
            axs[i].plot(x, x*0-eB, linestyle=":", color = "blue", linewidth=1.0)
            axs[i].fill_between(x.reshape(-1), y1=0*phat.reshape(-1)+eL, y2=0*phat.reshape(-1)-eL, color="green", alpha=0.3, label=r"$e_S$")
            axs[i].legend(loc="lower left", ncol=2, fontsize=16, framealpha=0.6)
        else:
            axs[i].plot(x, e1, color=colors[i], linestyle="-", linewidth=1.0)
            axs[i].plot(x, e1_hat, linestyle="--", color = "red", linewidth=1.0)
            axs[i].plot(x, x*0+eB, linestyle=":", color = "blue", linewidth=1.0)
            axs[i].plot(x, x*0-eB, linestyle=":", color = "blue", linewidth=1.0)
            axs[i].fill_between(x.reshape(-1), y1=0*phat.reshape(-1)+eL, y2=0*phat.reshape(-1)-eL, color="green", alpha=0.3, label=r"$e_S$")
        axs[i].text(0.01, 0.95, "t="+str(t1)+", "+r"$e_B=$"+str(eB[0])+", "+r"$e_S=$"+str(eL[0]),
                    transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        axs[i].set_xlim([-3,3])
        axs[i].set_ylim([-2*eL, 2*eL])
        axs[i].set_ylabel("e")
        if i < 2:
            axs[i].set_xticks([])
    axs[2].set_xlabel('x')
    plt.tight_layout(pad=0.2, h_pad=0.2)
    fig.savefig(FOLDER+'figs/e1hat_result.pdf', format='pdf', dpi=300)
    plt.close()


# Paper: Fig.1(a)
def plot_compare_error_bounds(t1s, B2, B1, e1):
    # Get the last 3 colors from the "hls" palette with 8 colors
    # colors = sns.color_palette("hls", 8)[-3:]  # Indexes 5, 6, 7 (last three)   
    colors = sns.color_palette("muted", 3)
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
    e1 = np.ravel(e1)  # Flatten the array if it's 2D
    B2 = np.ravel(B2)  # Flatten the array if it's 2D
    B1 = np.ravel(B1)  # Flatten the array if it's 2D
    fig, axs = plt.subplots(1, 1, figsize=(5, 6))
    axs.plot(t1s, e1, color="black", linestyle="--", linewidth=2.0, label='$max_x|e_1(x,t)|$')
    axs.plot(t1s, B2, color=colors[1], linestyle="-", linewidth=2.0, label='$B_2(t)$')
    axs.plot(t1s, B1, color=colors[0], linestyle="-", linewidth=2.0, label='$B_1(t)$')
    axs.fill_between(t1s, e1, B2, color=colors[1], alpha=0.1)
    axs.fill_between(t1s, B2, B1, color=colors[0], alpha=0.1)
    axs.legend(loc="upper right", framealpha=0.3)
    axs.set_xlabel('$t$')
    axs.set_ylabel('Value$(t)$')
    axs.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/1dou_compare_error_bounds.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

# Paper: Fig.1(b)
def plot_conditions(t, a, b, threshold_1, threshold_2, lhs_condition3, rhs_condition3):
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
    colors = sns.color_palette("muted", 3)
    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    # Condition 1: a(t) < 1
    axs[0].fill_between(t, a, threshold_1, where=(a < 1), color=colors[0], alpha=0.1)
    axs[0].plot(t, t*0+threshold_1, color=colors[0], linestyle='--', label='1')
    axs[0].plot(t, a, label=r'$\alpha_1(t$)', color=colors[0])
    axs[0].set_ylabel('Value$(t)$')
    axs[0].legend(ncol=2, loc="upper left", framealpha=0.6)
    # Condition 2: b(t) < 1 - a(t)
    axs[1].fill_between(t, b, threshold_2, where=(b < threshold_2), color=colors[1], alpha=0.1)
    axs[1].plot(t, threshold_2, '--', color=colors[1], label=r'$1 - \alpha_1(t)$')
    axs[1].plot(t, b, label=r'$\alpha_2(t)$', color=colors[1])
    axs[1].set_ylabel('Value$(t)$')
    axs[1].legend(ncol=2, loc="upper left", framealpha=0.6)
    # Condition 3: b(t)(1 + b(t)) < a(t)^2
    axs[2].fill_between(t, lhs_condition3, rhs_condition3, where=(lhs_condition3 < rhs_condition3), color=colors[2], alpha=0.1)
    axs[2].plot(t, rhs_condition3, '--', label=r'$\alpha_1(t)^2$', color=colors[2])
    axs[2].plot(t, lhs_condition3, label=r'$\alpha_2(t)(1 + \alpha_2(t))$', color=colors[2])
    axs[2].set_xlabel('$t$')
    axs[2].set_ylabel('Value$(t)$')
    axs[2].legend(ncol=2, loc="upper left", framealpha=0.6)
    # Format y-ticks to display two decimal places for all subplots
    for ax in axs.flat:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Add dotted grid to each subplot
    for ax in axs.flat:
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dotted grid
    plt.tight_layout()
    plt.savefig('figs/1dou_conditions.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    # plt.show()


def plot_alphas(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = np.arange(1.0, 3.0+0.01, 0.01)

    a1_list = []
    a2_list = []
    e1_list = []
    eS_list = []
    eB_list = []
    gap_list = []
    e_ratio = []
    eS_ratio_list = []

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
        gap = (eS - e1_list[i])/np.max(np.abs(p))
        gap_list.append(gap)
        e_ratio.append(eS/eB)
        eS_ratio = eS/np.max(np.abs(p))
        eS_ratio_list.append(eS_ratio)
        # print(a1, a2, (1-a1), a2, a1*a1, a2*(1+a2))

    a1_list = np.array(a1_list)
    gap_list = np.array(gap_list)
    norm_B2_list = eS_ratio_list
    print("[info] max a1: ", np.round(np.max(a1_list),2), ", avg a1:", np.round(np.mean(a1_list),2), ", std a1:", np.round(np.std(a1_list),3))
    print("[info] min gap: ", np.round(np.min(gap_list),3), ", avg gap:", np.round(np.mean(gap_list),3))
    print("[info] avg B2_norm: ", np.round(np.mean(norm_B2_list),2), ", std B2_norm:", np.round(np.std(norm_B2_list),3))
    a2_list = np.array(a2_list)
    cond_1 = (1-a1_list)
    y_1 = a2_list
    cond_2 = a1_list**2
    y_2 = a2_list*(1+a2_list)

    plot_compare_error_bounds(t1s, eB_list, eS_list, e1_list)
    plot_conditions(t1s, a1_list, a2_list, 1, 1-a1_list, a2_list*(1+a2_list), a1_list**2)


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
    fig.savefig(FOLDER+'figs/1dl_trainloss.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
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

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.8, label=r"$\hat{p}$")
    z_max = 1.5*np.max(np.abs(phat))
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x*0 + t1
        if i == 0:
            ax.plot(x, t1_monte, p_list[i], color="black", label=r"$p$")
        else:
            ax.plot(x, t1_monte, p_list[i], color="black")
    ax.view_init(20, -50)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.text2D(0.94, 0.77, "PDF", transform=ax.transAxes)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60))  
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/1dl_phatsurface.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
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

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1hat, cmap='inferno', alpha=0.8, label=r"$\hat{e}_1$")
    # z_max = 1.2*np.max(np.abs(e1hat))
    # ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="black", s=0.02, label='Data Points')
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
    ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60))  
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/1dl_e1hatsurface.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def main():
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    p_net = Net().to(device)
    p_net.apply(init_weights)
    e1_net = E1Net().to(device)
    e1_net.apply(init_weights)
    p_net.scale = get_p_normalize()
    optimizer = torch.optim.Adam(p_net.parameters())
    scheduler_p_model = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, mse_cost_function, scheduler_p_model); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    max_abs_e1_x_0 = get_e1_normalize(p_net)

    e1_net.scale = max_abs_e1_x_0
    optimizer = torch.optim.Adam(e1_net.parameters())
    scheduler_e1_model = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0, scheduler_e1_model); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")

    plot_tight_error_bounds(p_net, e1_net)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy", FOLDER+"output/e1_net_train_loss.npy")
    plot_alphas(p_net, e1_net)
    plot_p_surface(p_net)
    plot_e1_surface(p_net, e1_net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()
