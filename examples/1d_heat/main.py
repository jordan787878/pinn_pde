import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, ScalarFormatter
import time


FOLDER = "exp1/main/"

# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = "cpu"
print(device)

x_low = -1
x_hig = 1

t0 = 0
T_end = 1

def p_init(x):
    return -np.sin(np.pi*x)


def p_sol(x,t):
    return -np.sin(np.pi*x)*np.exp(-np.pi**2*t)


def p_init_torch(x):
    pi = torch.tensor(torch.pi, device=device)  # Ensure this is a tensor on the same device as x
    return -torch.sin(pi*x)


def res_func(x,t, net, verbose=False):
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    Lp = -p_xx
    residual = p_t + Lp
    return residual.to(device)


# p_net
class Net(nn.Module):
    def __init__(self):
        neurons = 64
        super(Net, self).__init__()
        self.hidden_layer1 = spectral_norm(nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        output = self.output_layer(layer3_out)
        # output = nn.functional.softplus(output)
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def train_p_net(p_net, optimizer, mse_cost_function, iterations=40000):
    global x_low, x_hig, t0, T_end
    batch_size = 500
    min_loss = np.inf
    loss_history = []
    PATH = FOLDER+"output/p_net.pt"
    reg_bc = 1

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        u_bc = p_init_torch(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        f_out = res_func(x, t, p_net)
        mse_f = mse_cost_function(f_out, all_zeros)

        x_bc = (torch.ones(batch_size, 1) * x_low).to(device)
        net_bc_out = p_net(x_bc, t)
        mse_bc1 = mse_cost_function(net_bc_out, all_zeros)
        x_bc = (torch.ones(batch_size, 1) * x_hig).to(device)
        net_bc_out = p_net(x_bc, t)
        mse_bc2 = mse_cost_function(net_bc_out, all_zeros)
        mse_bc = 0.5*(mse_bc1 + mse_bc2)

        loss = mse_u + mse_f + reg_bc*mse_bc

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
        loss_history.append(loss.data)
        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))


def pos_p_net_train(p_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    p_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("pnet best epoch: ", epoch, ", loss:", loss.data)
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


def show_p_net_results(p_net):
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1 = 0.2
    T0 = 0*x + t0
    T1 = 0*x + t1

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_T0 = Variable(torch.from_numpy(T0).float(), requires_grad=True).to(device)
    pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)

    p_approx0 = p_net(pt_x, pt_T0).data.cpu().numpy()
    p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
    p_exact0 = p_init(x)
    p_exact1 = p_sol(x,T1)

    e1 = p_exact0 - p_approx0
    max_abs_e1_t0 = max(abs(p_exact0 - p_approx0))[0]
    # print(max_abs_e1_t0)

    plt.figure()
    plt.plot(x, p_exact0,  "k--", label=r"$p(t_i)$")
    plt.plot(x, p_approx0, "k", label=r"$\hat{p}(t_i)$")
    plt.plot(x, p_exact1,  "r--", label=r"$p(t_f)$")
    plt.plot(x, p_approx1, "r", label=r"$\hat{p}(t_f)$")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.savefig(FOLDER+"figs/pnet_approx.png")
    plt.close()

    res_0 = res_func(pt_x, pt_T0, p_net)
    res_1 = res_func(pt_x, pt_T1, p_net)
    plt.figure()
    plt.plot(x, res_0.detach().numpy(), "black", label=r"$r_1(t_i)$")
    plt.plot(x, res_1.detach().numpy(), "red", label=r"$r_1(t_f)$")
    plt.plot([x_low, x_hig], [0,0], "black")
    plt.legend()
    plt.savefig(FOLDER+"figs/pnet_resdiual.png")
    plt.close()
    
    return max_abs_e1_t0


class E1Net(nn.Module):
    def __init__(self, scale=1.0):
        neurons = 100
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        # Initialize weights with random values
        self.initialize_weights()
    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = torch.tanh((self.hidden_layer1(inputs)))
        layer2_out = torch.tanh((self.hidden_layer2(layer1_out)))
        layer3_out = torch.tanh((self.hidden_layer3(layer2_out)))
        layer4_out = torch.tanh((self.hidden_layer4(layer3_out)))
        layer5_out = torch.tanh((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = output*self.scale
        return output
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights with random values using a normal distribution
                init.xavier_uniform_(module.weight)


def e1_res_func(x, t, e1_net, p_net, verbose=False):
    e1 = e1_net(x,t)
    e_x = torch.autograd.grad(e1, x, grad_outputs=torch.ones_like(e1), create_graph=True)[0]
    e_t = torch.autograd.grad(e1, t, grad_outputs=torch.ones_like(e1), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    Le = -e_xx
    p_res = res_func(x, t, p_net)
    residual = e_t + Le + p_res
    if(verbose):
      print("e1net output: ", e1)
      print(residual)
      print(p_res)
    return residual


def test_e1_res(e1_net, p_net):
    batch_size = 5
    x_collocation = np.random.uniform(low=1.0, high=3.0, size=(batch_size,1))
    t_collocation = T_end*np.ones((batch_size,1))
    all_zeros = np.zeros((batch_size,1))
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    f_out = e1_res_func(pt_x_collocation, pt_t_collocation, e1_net, p_net, verbose=True) # output of f(x,t)


def train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_t0, iterations=40000):
    batch_size = 500
    min_loss = np.inf
    loss_history = []
    reg_bc = 1.0
    iterations_per_decay = 1000
    PATH = FOLDER+"output/e1_net.pt"

    # dt_train = 0.1
    # k = 5
    # reg_alpha1_t0 = 1e-4

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * t0).to(device)
        p_bc = p_init_torch(x_bc)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
        t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e1_res_func(x, t, e1_net, p_net)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Loss based on boundary conditions: randomly between -1 and 1
        choices = torch.tensor([x_low, x_hig])
        rand_indices = torch.randint(0, 2, (batch_size, 1))
        x_bc = choices[rand_indices]
        t_bc = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
        p_bc = 0*t_bc # p(x,t) = 0, x in boundary, known values
        p_net_bc = p_net(x_bc, t_bc)
        e1_bc = p_bc - p_net_bc
        e1_net_bc = e1_net(x_bc, t_bc)
        mse_bc = mse_cost_function(e1_net_bc, e1_bc)

        # for i in range(1,k+1):
        #     pt_t_k = t + (i)*dt_train
        #     f_out_k = e1_res_func(x, t, e1_net, p_net)
        #     mse_f_k = mse_cost_function(f_out_k, all_zeros)
        #     mse_f = mse_f_k + (mse_f_k/k)
        # # Loss based on alpha_1(t0)
        # x = torch.arange(x_low, x_hig, 0.01).view(-1, 1).to(device)
        # t = 0*x
        # e1hat_t0 = e1_net(x, t)
        # p_t0 = p_init_torch(x)
        # phat_t0 = p_net(x,t)
        # e1_t0 = p_t0 - phat_t0
        # alpha1_t0 = max(abs(e1_t0-e1hat_t0))/max(abs(e1hat_t0))
        
        # Combining the loss functions
        loss = (mse_u + mse_res + reg_bc*mse_bc)/max_abs_e1_t0

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  ",ic:",mse_u.data/max_abs_e1_t0, 
                  ",res:", mse_res.data/max_abs_e1_t0, 
                  ",bc:", mse_bc.data/max_abs_e1_t0)
             #, ", alpha1(t0):", alpha1_t0.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data 
        loss_history.append(loss.data)
        loss.backward() 
        optimizer.step()
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best epoch: ", epoch, ", loss:", loss.data)
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


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
       self.format = "%1.1f"


def show_e1_results(p_net, e1_net):
    plt.rcParams['font.size'] = 18
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1s = [0.2, 0.6, 1.0]
    j = 1
    for i in range(3):
        t1 = t1s[i]
        plt.subplot(3,1,j)
        T1 = 0*x + t1
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
        p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
        p_exact1 = p_sol(x, t1)
        e1_exact_1 = p_exact1 - p_approx1   
        e1_1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
        error_bound = max(abs(e1_1))*2
        error_bound = error_bound[0]
        a1 = np.max(np.abs(e1_exact_1-e1_1)) / np.max(np.abs(e1_1))
        ax1 = axs[i]
        ax1.plot(x, e1_exact_1, color="black", linewidth=1.0, linestyle="-", label="$e$")
        ax1.plot(x, e1_1, "red", linewidth=1.0, linestyle="--", label=r"$\hat{e}_1$")
        ax1.fill_between(x.reshape(-1), y1=0.0*p_approx1.reshape(-1)+error_bound, 
                         y2=0.0*p_approx1.reshape(-1)-error_bound, 
                         color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")
        if i < 2:
            ax1.set_xticks([])
        if i == 2:
            ax1.set_xlabel("x")
        ax1.set_xlim([x_low, x_hig])
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $\alpha_1:$ "+str(np.round(a1, 3)), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        # Set y-axis to scientific notation
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout(pad=0.3, h_pad=0.3)
    plt.savefig(FOLDER+"figs/e1net_result.pdf", format='pdf', dpi=300)
    plt.close()


def show_uniform_bound(p_net, e1_net):
    plt.rcParams['font.size'] = 18
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    global x_low, x_hig, t0, T_end
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    t1s = [0.2, 0.6, 1.0]
    j = 1
    for i in range(3):
        t1 = t1s[i]
        plt.subplot(3,1,j)
        T1 = 0*x + t1
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_T1 = Variable(torch.from_numpy(T1).float(), requires_grad=True).to(device)
        p_approx1 = p_net(pt_x, pt_T1).data.cpu().numpy()
        p_exact1 = p_sol(x, t1)
        # e1_exact_1 = p_exact1 - p_approx1   
        e1_1 = e1_net(pt_x, pt_T1).data.cpu().numpy()
        error_bound = max(abs(e1_1))*2
        error_bound = error_bound[0]
        ax1 = axs[i]
        ax1.plot(x, p_exact1, color="black", linewidth=1.0, linestyle="-", label="$u$")
        ax1.plot(x, p_approx1, "red", linewidth=1.0, linestyle="--", label=r"$\hat{u}$")
        ax1.fill_between(x.reshape(-1), y1=p_approx1.reshape(-1)+error_bound, y2=p_approx1.reshape(-1)-error_bound, 
                         color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")
        if i < 2:
            ax1.set_xticks([])
        if i == 2:
            ax1.set_xlabel("x")
        ax1.set_xlim([x_low, x_hig])
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $e_S:$ "+str(np.round(error_bound,4)), 
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        # Set y-axis to scientific notation
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout(pad=0.3, h_pad=0.3)
    plt.savefig(FOLDER+"figs/uniform_error_bound.pdf", format='pdf', dpi=300)
    plt.close()


def plot_p_surface(p_net, num=100):
    plt.rcParams['font.size'] = 18
    t1s = [0.0, 0.5, 1.0]
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    phat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)

    p_list = []
    for t1 in t1s:
        p_true = p_sol(x, x*0+t1)
        p_list.append(p_true)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.7, label=r"$\hat{u}$")
    z_max = 1.5*np.max(np.abs(phat))
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x*0 + t1
        if i == 0:
            ax.plot(x, t1_monte, p_list[i], color="black", label=r"$u$")
        else:
            ax.plot(x, t1_monte, p_list[i], color="black")

    ax.set_xlabel("x"); ax.set_ylabel("t"); #ax.set_zlabel("PDF")
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85), fontsize=18)
    x_ticks = np.array([-1, 0, 1])  # Example y-tick positions
    ax.set_xticks(x_ticks)  # Set the positions of the y-ticks
    ax.view_init(20, 150)
    plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.08)
    # plt.show()
    fig.savefig(FOLDER+'figs/phat_surface_plot.pdf', format='pdf', dpi=300)
    plt.close()


def plot_e1_surface(p_net, e1_net, num=100):
    plt.rcParams['font.size'] = 18
    t1s = [0.0, 0.5, 1.0]
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
        p_monte = p_sol(x, x*0+t1).reshape(-1, 1)
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
            ax.plot(x_monte, t1_monte, e1_list[i], color="black", label=r"$e$")
        else:
            ax.plot(x_monte, t1_monte, e1_list[i], color="black")
    # Set z-ticks to scientific notation
    ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.zaxis.get_major_formatter().set_powerlimits((-2, 2))  # Use scientific notation if value is outside this range
    x_ticks = np.array([-1, 0, 1])  # Example y-tick positions
    ax.set_xticks(x_ticks)  # Set the positions of the y-ticks
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85), fontsize=18)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("Error")
    ax.view_init(20, 150)
    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/e1hat_surface_plot.pdf', format='pdf', dpi=300)
    plt.close()


def plot_train_loss(path_1, path_2):
    loss_history_1 = np.load(path_1)
    min_loss_1 = min(loss_history_1)
    loss_history_2 = np.load(path_2)
    min_loss_2 = min(loss_history_2)
    # print(loss_history_2)
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


def show_enet_res(p_net, e1_net):
    t1s = [0.2, 0.6, 1.0]
    x = np.linspace(x_low, x_hig, num=100).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    r1_list = []
    r2_list = []
    for t1 in t1s:
        pt_t1 = Variable(torch.from_numpy(0*x + t1).float(), requires_grad=True).to(device)
        r1 = res_func(pt_x, pt_t1, p_net).data.cpu().numpy()
        r1_list.append(r1)
        r2 = e1_res_func(pt_x, pt_t1, e1_net, p_net).data.cpu().numpy()
        r2_list.append(r2)
    fig, axs = plt.subplots(3, 1, figsize=(7, 6))
    for i in range(0,3):
        pres = r1_list[i]
        eres = r2_list[i]
        ax1 = axs[i]
        if i == 0:
            ax1.plot(x, eres-pres, "red", linewidth=1.0, linestyle="--", label=r"$D[\hat{e}_1]$")
            ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{u}]$")
            ax1.legend(loc="upper right")  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres-pres, "red", linewidth=1.0, linestyle="--")
            ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        ax1.text(0.01, 0.95, "t="+str(t1s[i]), 
                 transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('x')
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/enet_res.pdf', format='pdf', dpi=300)
    plt.close()


def show_table(p_net, e1_net):
    x = np.arange(x_low, x_hig+0.005, 0.005).reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = np.arange(t0, T_end+0.01, 0.01)
    a1_list = []
    gap_list = []
    e1_list = []
    eS_ratio_list = []
    for i in range(len(t1s)):
        t1 = t1s[i]
        pt_t1 = Variable(torch.from_numpy(0*x+t1).float(), requires_grad=True).to(device)
        p = p_sol(x, x*0+t1)
        phat = p_net(pt_x, pt_t1).data.cpu().numpy()
        e1 = p - phat
        e1_list.append(max(abs(e1))[0])
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy()
        eS = max(abs(e1_hat))*2
        eS = np.round(eS,3)
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        a1_list.append(a1[0])
        gap_list.append((eS - e1_list[i])/max(abs(p)))
        eS_ratio_list.append(eS/ max(abs(p)) )
    print("[info] max a1: " +str(np.max(np.array(a1_list))) + ", avg a1:" + str(np.mean(np.array(a1_list))))
    print("[info] min gap: " +str(np.min(np.array(gap_list))) + ", max gap:" + str(np.max(np.array(gap_list))))
    print("[info] max eS_ratio: " +str(np.max(np.array(eS_ratio_list))) + ", avg eS_ratio:" + str(np.mean(np.array(eS_ratio_list))))


def main():
    # create p_net
    p_net = Net()
    p_net = p_net.to(device)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters())
    start_time = time.time()
    # train_p_net(p_net, optimizer, mse_cost_function, iterations=10000); print("p_net train complete")
    time_train_p = time.time() - start_time
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abs_e1_t0 = show_p_net_results(p_net)
    print("max(abs(e1(x,0))):", max_abs_e1_t0)

    # create e1_net
    e1_net = E1Net(scale=max_abs_e1_t0)
    e1_net = e1_net.to(device)
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3);   # test_e1_res(e1_net, p_net)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    start_time = time.time()
    # train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_t0, iterations=10000); print("e1_net train complete")
    time_train_e = time.time() - start_time
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_results(p_net, e1_net)
    show_uniform_bound(p_net, e1_net)
    show_enet_res(p_net, e1_net)

    print(f"train pnet time: {time_train_p:.4f} seconds")
    print(f"train enet time: {time_train_e:.4f} seconds")
    plot_p_surface(p_net)
    plot_e1_surface(p_net, e1_net)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy",
                    FOLDER+"output/e1_net_train_loss.npy")
    show_table(p_net, e1_net)


if __name__ == "__main__":
    main()