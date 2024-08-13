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

FOLDER = "exp1/"

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
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = nn.functional.softplus(self.hidden_layer1(inputs))
        layer2_out = nn.functional.softplus(self.hidden_layer2(layer1_out))
        layer3_out = nn.functional.softplus(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out)
        output = nn.functional.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def train_p_net(p_net, optimizer, mse_cost_function):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 1000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/p_net.pt"
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        u_bc = p_init(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = res_func(x, t, p_net)
        mse_f = mse_cost_function(f_out, all_zeros)

        loss = mse_u + mse_f

        loss_history.append(loss.data)

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

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


def show_p_net_results(p_net):
    global x_low, x_hig, X_sim, P_sim_t20, t0
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 2
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    # pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    # p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_init(x)
    # p_exact1 = p_exact(x, T1)
    # plt.figure()
    # plt.plot(x, p_approx0, "b--", linewidth=0.5, label="Approx t")
    # plt.plot(x, p_exact0,  "b", linewidth=0.5, label="True t: ")
    # plt.plot(x, p_approx1, "r--", linewidth=0.5, label="Approx t")
    # plt.plot(x, p_exact1,  "r", linewidth=0.5, label="True t: ")
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('pdf')
    # plt.savefig(FOLDER+"figs/pnet_approx.png")
    # plt.close()
    # res_0 = res_func(pt_x, pt_T0, p_net)
    # res_1 = res_func(pt_x, pt_T1, p_net)
    # plt.figure()
    # plt.plot(x, res_0.detach().numpy(), "blue", alpha=0.3, label="res t0")
    # plt.plot(x, res_1.detach().numpy(), "red", alpha=0.3, label="res t1")
    # plt.plot([x_low, x_hig], [0,0], "black")
    # plt.legend()
    # plt.savefig(FOLDER+"figs/pnet_resdiual.png")
    # plt.close()

    max_abs_e1_x_0 = max(abs(p_exact0-p_approx0))
    print(max_abs_e1_x_0[0])

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
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh((self.hidden_layer1(inputs)))
        layer2_out = torch.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = torch.tanh((self.hidden_layer3(layer2_out)))
        output = self.scale * self.output_layer(layer3_out)
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


def train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    iterations = 1000
    # iterations = 20000
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/e1_net.pt"
    
    x_mar = 0
    t_mar = 0

    reg_mse_res_t0 = 0.0
    reg_mse_res    = 0.5 #(maybe increasing this further)
    reg_alpha1_t0  = 0.0 #(the order of targeted loss of ic)

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
        loss = (1-reg_alpha1_t0-reg_mse_res_t0-reg_mse_res)*mse_u  + reg_mse_res*mse_res
        loss_history.append(loss.data)

        loss.backward() 
        optimizer.step()

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



def plot_tight_error_bounds(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = [1.5, 2.0, 3.0]
    colors = ["black","blue","green"]

    e2_hat_mag = 1e-3
    e2_hat_freq = 5

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x)
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        a2 = max(abs(e2-e2_hat))/max(abs(e2_hat))
        axs[i].plot(x, e2, color=colors[i], linestyle="-", linewidth=1.0, label=r"$e_2$")
        axs[i].plot(x, e2_hat, linestyle="--", color = colors[i], linewidth=1.0, label=r"$\hat{e}_2$")
        axs[i].grid(linewidth=0.5)
        axs[i].legend(loc="upper right")
        # Add text to the left top corner
        axs[i].text(0.01, 0.98, "t="+str(t1), transform=axs[i].transAxes, verticalalignment='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/e2hat_result.png")
    plt.close()
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x)
        r_21 = max(abs(e2_hat))/ max(abs(e1_hat))
        eB = max(abs(e1_hat))*(1/(1-r_21))
        eB = np.round(eB, 4)
        eL = 2*max(abs(e1_hat))
        eL = np.round(eL,4)
        axs[i].plot(x, p, color=colors[i], linestyle="-", linewidth=1.0, label=r"$p$")
        axs[i].plot(x, phat, linestyle="--", color = colors[i], linewidth=1.0, label=r"$\hat{p}$")
        axs[i].fill_between(x.reshape(-1), y1=phat.reshape(-1)+eL, y2=phat.reshape(-1)-eL, color=colors[i], alpha=0.1, label=r"$e_B$")
        # axs[i].fill_between(x.reshape(-1), y1=phat.reshape(-1)+eL, y2=phat.reshape(-1)-eL, color=colors[i], alpha=0.5, label=r"$e_L$")
        axs[i].set_ylim([0, 0.75])
        axs[i].grid(linewidth=0.5)
        axs[i].legend(loc="upper right")
        # Add text to the left top corner
        axs[i].text(0.01, 0.98, "t="+str(t1)+","+r"$e_L=$"+str(eL), transform=axs[i].transAxes, verticalalignment='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/error_bounds_result.png")
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    for i in range(3):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x)
        r_21 = max(abs(e2_hat))/ max(abs(e1_hat))
        eB = max(abs(e1_hat))*(1/(1-r_21))
        eB = np.round(eB,3)
        eL = 2*max(abs(e1_hat))
        eL = np.round(eL,3)
        print(eB, eL)
        axs[i].plot(x, e1, color=colors[i], linestyle="-", linewidth=1.0, label=r"$e_1$")
        axs[i].plot(x, e1_hat, linestyle="--", color = colors[i], linewidth=1.0, label=r"$\hat{e}_1$")
        axs[i].plot(x, x*0+eB, linestyle=":", color = colors[i], linewidth=2.0, label=r"$e_B$")
        axs[i].plot(x, x*0-eB, linestyle=":", color = colors[i], linewidth=2.0)
        axs[i].fill_between(x.reshape(-1), y1=0*phat.reshape(-1)+eL, y2=0*phat.reshape(-1)-eL, color=colors[i], alpha=0.1, label=r"$e_L$")
        # axs[i].fill_between(x.reshape(-1), y1=0*phat.reshape(-1)+eL, y2=0*phat.reshape(-1)-eL, color=colors[i], alpha=0.3, label=r"$e_L$")
        axs[i].grid(linewidth=0.5)
        axs[i].legend(loc="upper right")
        # Add text to the left top corner
        axs[i].text(0.01, 0.95, "t="+str(t1)+", "+r"$e_B=$"+str(eB[0])+", "+r"$e_L=$"+str(eL[0]),
                    transform=axs[i].transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/e1hat_result.png")
    plt.close()


def plot_alphas(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = np.arange(1.0, 3.0+0.01, 0.01)

    a1_list = []
    a2_list = []
    e1_list = []
    eB_list = []

    e2_hat_mag = 1e-3
    e2_hat_freq = 5

    for i in range(len(t1s)):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_exact(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_list.append(max(abs(e1))[0])
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        e2 = e1 - e1_hat
        e2_hat = e2 + e2_hat_mag*max(abs(e2))*np.sin(e2_hat_freq*x)
        r_21 = max(abs(e2_hat))/ max(abs(e1_hat))
        eB = max(abs(e1_hat))*(1/(1-r_21))
        eB = np.round(eB,3)
        eB_list.append(eB)
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        a2 = max(abs(e2-e2_hat))/max(abs(e2_hat))
        a1_list.append(a1[0])
        a2_list.append(a2[0])
        # print(a1, a2, (1-a1), a2, a1*a1, a2*(1+a2))

    print("a1_list", a1_list)
    a1_list = np.array(a1_list)
    a2_list = np.array(a2_list)
    cond_1 = (1-a1_list)
    y_1 = a2_list
    cond_2 = a1_list**2
    y_2 = a2_list*(1+a2_list)

    # fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    # axs.plot(t1s, eL_list, color="black", linestyle=":", label=r"$e_L$")
    # axs.plot(t1s, e1_list, color="black", linestyle="-", label=r"$\max|e|$")
    # axs.set_xlabel('t')
    # axs.legend()
    # axs.grid(linewidth=0.5)
    # plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    axs[0].plot(t1s, eB_list, color="black", linestyle=":", label=r"$e_B$")
    axs[0].plot(t1s, e1_list, color="black", linestyle="-", label=r"$\max|e|$")
    axs[0].legend(loc="upper right")
    axs[0].grid(linewidth=0.5)
    
    axs[1].plot(t1s, cond_1, color="black", linestyle=":", label=r"$1-\alpha_1$")
    axs[1].plot(t1s, y_1, color="black", linestyle="-", label=r"$\alpha_2$")
    axs[1].legend(loc="upper right")
    axs[1].grid(linewidth=0.5)
    axs[1].text(0.01, 0.98, "condition: "+r"$\alpha_2 < 1-\alpha_1$", transform=axs[1].transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    axs[2].plot(t1s, cond_2, color="black", linestyle=":", label=r"$\alpha_1^2$")
    axs[2].plot(t1s, y_2, color="black", linestyle="-", label=r"$\alpha_2(1+\alpha_2)$")
    axs[2].set_xlabel('t')
    axs[2].legend(loc="upper right")
    axs[2].grid(linewidth=0.5)
    axs[2].text(0.01, 0.98, "condition: "+r"$\alpha_2(1+\alpha_2) \leq \alpha_1^2$", transform=axs[2].transAxes, verticalalignment='top', fontsize=8,
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

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(np.arange(len(loss_history_1)), loss_history_1, "black")
    axs[1].plot(np.arange(len(loss_history_2)), loss_history_2, "blue")
    axs[0].grid(linewidth=0.5)
    axs[1].grid(linewidth=0.5)
    axs[1].set_label("epochs")
    axs[0].set_ylabel("train loss: "+r"$\hat{p}$")
    axs[1].set_ylabel("train loss: "+r"$\hat{e}_1$")
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/train_loss.png")
    plt.close()
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.savefig(FOLDER+"figs/pnet_loss_history.png")
    # plt.close()



def main():
    # create p_net
    p_net = Net()
    p_net = p_net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters())
    # train_p_net(p_net, optimizer, mse_cost_function); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    max_abs_e1_x_0 = show_p_net_results(p_net)
    # plot_p_net_results(p_net)

    # create e1_net
    e1_net = E1Net(scale=max_abs_e1_x_0)
    e1_net = e1_net.to(device)
    optimizer = torch.optim.Adam(e1_net.parameters())
    # train_e1_net(e1_net, optimizer, mse_cost_function, p_net, max_abs_e1_x_0); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")

    plot_tight_error_bounds(p_net, e1_net)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy", FOLDER+"output/e1_net_train_loss.npy")
    plot_alphas(p_net, e1_net)



    print("[complete 1d OU]")

if __name__ == "__main__":
    main()