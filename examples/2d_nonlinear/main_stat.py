import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, ScalarFormatter
from scipy.stats import multivariate_normal
from scipy.linalg import expm
from tqdm import tqdm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
import time
import argparse
import seaborn as sns

FOLDER = "exp1/main_stat/"
FOLDER_DATA = "exp1/data/"
FOLDER_INTERMED = "../meta/data_2dpendulum/"
device = "cpu"
print(device)

pi = np.pi
n_d = 2
mu_0 = np.array([pi*(0.5), 0.0]).reshape(2,)
cov_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
B = np.array([[0.5, 0.0],[0.0, 0.5]])
x_low = -3.0*pi
x_hig =  3.0*pi
ti = 0.0
tf = 5.0
g = 9.8
l = 9.8
t1s = [1.0, 2.0, 3.0, 4.0, 5.0]

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

TRAIN_FLAG = False

# x1: theta
# x2: d(theta)/dt
def f_sde(x):
    x1 = x[0]
    x2 = x[1]
    dx1dt = x2
    dx2dt = -g*np.sin(x1)/l
    return np.array([dx1dt, dx2dt]).reshape(2,)


def p_init(x):
    pdf_func = multivariate_normal(mean=mu_0, cov=cov_0)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1)
    return pdf_eval


def p_init_torch(x):
    # Ensure input x is a torch tensor
    if not isinstance(x, torch.Tensor):
        raise ValueError("Input x must be a torch tensor")
    mu_0_torch = torch.tensor(mu_0, dtype=torch.float32)
    cov_0_torch = torch.tensor(cov_0, dtype=torch.float32)
    # Define the multivariate normal distribution
    m = torch.distributions.MultivariateNormal(loc=mu_0_torch, covariance_matrix=cov_0_torch)
    # Evaluate the PDF at each point in x
    pdf_eval = m.log_prob(x).exp().reshape(-1, 1)  # Convert log-prob to prob
    return pdf_eval


def p_sol_monte(t1=ti, linespace_num=50, stat_sample=50000):
    start_time = time.time()
    dtt = 0.01
    t_span = np.arange(ti, t1, dtt)
    mean = mu_0
    cov = cov_0
    # X = np.random.multivariate_normal(mean, cov, stat_sample).T
    X = np.zeros((n_d, stat_sample))
    for i in tqdm(range(stat_sample), desc="Processing samples"):
        x = np.random.multivariate_normal(mean, cov).T
        for t in t_span:
            w1 = np.random.normal(0, np.sqrt(dtt))
            w2 = np.random.normal(0, np.sqrt(dtt))
            w = np.array([w1, w2]).reshape(2,)
            x = x + f_sde(x)*dtt + np.matmul(B, w)
        X[:,i] = x
    
    # Define bins as the edges of x1 and x2
    bins_x1 = np.linspace(x_low, x_hig, num=linespace_num)
    bins_x2 = np.linspace(x_low, x_hig, num=linespace_num)

    # Digitize v to find which bin each value falls into for both dimensions
    bin_indices_x1 = np.digitize(X[0, :], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[1, :], bins_x2) - 1  

    # Initialize the frequency array
    frequency_2d = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1))

    # Count the occurrences in each 2D bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        if 0 <= bin_indices_x1[i] < frequency_2d.shape[0] and 0 <= bin_indices_x2[i] < frequency_2d.shape[1]:
            frequency_2d[bin_indices_x2[i], bin_indices_x1[i]] += 1

    # Normalize the frequency to get the proportion
    frequency_2d = frequency_2d / stat_sample
    dx = bins_x1[1]-bins_x1[0]
    frequency_2d = frequency_2d/(dx**n_d)

    # Calculate the midpoints for bins
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    X_grid, Y_grid = np.meshgrid(midpoints_x1, midpoints_x2)
    comp_time = time.time() - start_time
    print("MC time (sec): ", comp_time)
    return X_grid, Y_grid, frequency_2d, midpoints_x1


def test_p_sol_monte(stat_sample=10000):
    linspace_num = 100
    j = 0
    for t1 in t1s:
        print("generate p sol Monte for t="+str(t1))
        x_sim_grid, y_sim_grid, p_sim_grid, x_points = p_sol_monte(t1=t1, linespace_num=linspace_num, stat_sample=stat_sample)
        if(j == 0):
            np.save(FOLDER_DATA+"x_sim_grid.npy", x_sim_grid)
            np.save(FOLDER_DATA+"y_sim_grid.npy", y_sim_grid)
            np.save(FOLDER_DATA+"x_points.npy", x_points)
        np.save(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy", p_sim_grid)
        j = j + 1


def test_p_init():
    sample_size = 100
    x1s = np.linspace(x_low, x_hig, num=sample_size)
    x2s = np.linspace(x_low, x_hig, num=sample_size)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    p_exact0 = p_init(x)
    return max(abs(p_exact0))[0]


def res_func(x, t, p_net, verbose=False):
    # f1 = x2
    # f2 = -g*sin(x1)/l
    B_torch = torch.tensor(B, dtype=torch.float32, requires_grad=True)
    p = p_net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x1 = p_x[:,0].view(-1,1)
    p_x2 = p_x[:,1].view(-1,1)
    x1 = x[:,0].view(-1, 1)
    x2 = x[:,1].view(-1, 1)

    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(p_x.size(1)):
        grad2 = torch.autograd.grad(p_x[:, i], x, grad_outputs=torch.ones_like(p_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    p_xx = torch.stack(hessian, dim=-1)
    p_x1x1 = p_xx[:, 0, 0].view(-1, 1)
    p_x2x2 = p_xx[:, 1, 1].view(-1, 1)

    f1 = torch.reshape(x2, (-1,1))
    f2 = torch.reshape(-g*torch.sin(x1)/l, (-1,1))

    # f1_x1 = torch.reshape(torch.autograd.grad(f1, x1, grad_outputs=torch.ones_like(f1), create_graph=True)[0], (-1,1))
    # f2_x2 = torch.reshape(torch.autograd.grad(f2, x2, grad_outputs=torch.ones_like(f2), create_graph=True)[0], (-1,1))
    f1_x1 = (0.0*x1).view(-1,1)
    f2_x2 = (0.0*x2).view(-1,1)

    Lp = p_x1*f1 + p*f1_x1 + p_x2*f2 + p*f2_x2 - 0.5*(B_torch[0,0]*B_torch[0,0]*p_x1x1 + B_torch[1,1]*B_torch[1,1]*p_x2x2)
    residual = p_t + Lp
    if(verbose):
      print(f1_x1.shape)
      print("residual: ", residual, residual.shape)
    return residual


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        output = F.softplus( self.output_layer(layer3_out) )
        return output
                

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_abs_p_ti, iterations=40000):
    global x_low, x_hig, ti, tf
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    x_mar = 0.0

    # space-time points for BC
    x_bc = (torch.rand(1000, n_d) * (x_hig - x_low) + x_low).to(device); #print(min(x_bc[:,0]), max(x_bc[:,0]), min(x_bc[:,1]), max(x_bc[:,1]))
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    t = (torch.rand(1000, 1, requires_grad=True) *   (tf - ti) + ti).to(device)

    # RAR
    S = 30000
    FLAG = False
    
    PATH = FOLDER+"output/p_net.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        u_bc = p_init_torch(x_bc).detach()
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        res_out = res_func(x, t, p_net)/max_abs_p_ti
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Frequnecy Loss
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + 5.0*mse_res + 5.0*mse_norm_res_input
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                   ",res_freq:", mse_norm_res_input.data
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    'train_time': train_time,
                    }, PATH)
            min_loss = loss.data
            FLAG = True

        # RAR
        if (epoch%100 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            max_ic_error = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR))/max_abs_p_ti
            print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > 5e-3):
                # max_abs_ic, max_index = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR), dim=0)
                max_abs_ic, max_index = torch.topk(torch.abs(phat_bc_RAR.squeeze() - p_bc_RAR.squeeze()), 5)
                x_max = x_RAR[max_index,:].clone()
                t_max = t0_RAR[max_index].clone()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                # print("... IC add [x,t]:", x_max.data, t_max.data, max_abs_ic.data)

            res_RAR = res_func(x_RAR, t_RAR, p_net)/max_abs_p_ti
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
            FLAG = False

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
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
    print("pnet best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("pnet loss")
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/pnet_loss_history.pdf', format='pdf', dpi=300)
    plt.close()
    return p_net


def show_p_net_results(p_net):

    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 18,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 18,              # Title font size
    "axes.labelsize": 18,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 18,             # Legend text size
    "legend.title_fontsize": 18        # Legend title size (if you use legend titles)
    })

    max_abe_e1_ti = np.inf
    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    # plot p_net vs p
    all_p = []
    for t1 in t1s:
        p = np.load(FOLDER_DATA + "p_sim_grid" + str(t1) + ".npy")
        all_p.append(p)
    all_p = np.concatenate(all_p)  # Combine all p values
    vmin = np.min(all_p)
    vmax = np.max(all_p)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 6, width_ratios=[1]*5 + [0.05], wspace=0.2)
    # Create subplot grid (2x5) for the plots
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(5)]
    # Create a subplot for the colorbar spanning the height of the grid
    cax = fig.add_subplot(gs[:, -1])
   
    # Define ticks
    ticks_values = [-2*np.pi, 0, 2*np.pi]
    ticks_labels = [r'$-2\pi$', r'$0$', r'$2\pi$']
    for i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        if(i <= 4):
            t1 = t1s[i]
            p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
            cp = ax.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower',
                           vmin=vmin, vmax=vmax)
            ax.set_title("t="+str(t1))
            if(i == 0):
                ax.set_ylabel(r"$\omega$")
                ax.set_yticks(ticks_values, ticks_labels, fontsize=14)
        else:
            t1 = t1s[i-5]
            pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
            p_hat = p_net(pt_x, pt_t1)
            p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
            cp = ax.imshow(p_hat_numpy, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower',
                            vmin=vmin, vmax=vmax)
            ax.set_xlabel(r"$\theta$")
            ax.set_xticks(ticks_values, ticks_labels, fontsize=14)
            if(i == 5):
                ax.set_ylabel(r"$\omega$")
                ax.set_yticks(ticks_values, ticks_labels, fontsize=14)
    # Add the colorbar to the colorbar subplot
    cbar = fig.colorbar(cp, cax=cax, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # Add a box with text at the top-left corner of the figure
    fig.text(0.01, 0.87, r"$p(x,t)$", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    fig.text(0.01, 0.47, r"$\hat{p}(x,t)$", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    # ax.text2D(0.94, 0.77, "PDF", transform=ax.transAxes)
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plt.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.1, top=0.87, wspace=0.4, hspace=0.1)
    plt.savefig(FOLDER+'figs/2dpend_phatresult.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # fig.subplots_adjust(left=0.07, right=0.92, bottom=0.1, top=0.9, wspace=0.4, hspace=0.1)
    # fig.savefig(FOLDER+'figs/p_vs_phat.pdf', format='pdf', dpi=300)
    # plt.close()

    # plot pnet residual
    fig = plt.figure(figsize=(10, 6))
    j = 0
    for t1 in t1s:
        if (j < 5):
            ax = fig.add_subplot(2, 3, j+1, projection="3d")
            pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
            res_out = res_func(pt_x, pt_t1, p_net)
            res_numpy = res_out.data.cpu().numpy().reshape((sample_size, sample_size))
            surf1 = ax.plot_surface(x1, x2, res_numpy, cmap='viridis')
            ax.set_xlabel(r"$\theta$", fontsize=8)
            ax.set_ylabel(r"$\omega$", fontsize=8)
            ax.set_zlabel(r"$r_1$", fontsize=8)
            ax.set_title("t="+str(t1))
        j = j + 1
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5)
    fig.savefig(FOLDER+'figs/pnet_resdiual.pdf', format='pdf', dpi=300)
    plt.close()


def get_e1net_scale(p_net):
    x1s = np.linspace(x_low, x_hig, num=200, endpoint=True)
    x2s = np.linspace(x_low, x_hig, num=200, endpoint=True)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])
    p0 = p_init(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_ti = Variable(torch.from_numpy(x[:,0]*0+ti).float(), requires_grad=True).view(-1,1).to(device)
    p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    e1 = p0 - p_hat
    max_abe_e1_ti = max(abs(e1))[0]
    return max_abe_e1_ti


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
        inputs = torch.cat([x,t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        layer6_out = self.activation((self.hidden_layer6(layer5_out)))
        output = self.output_layer(layer6_out)
        output = self.scale * output
        return output


def e1_res_func(x, t, e1_net, p_net, verbose=False):
    p_res = res_func(x, t, p_net)

    B_torch = torch.tensor(B, dtype=torch.float32, requires_grad=True)
    net = e1_net(x,t)
    net_x = torch.autograd.grad(net, x, grad_outputs=torch.ones_like(net), create_graph=True)[0]
    net_t = torch.autograd.grad(net, t, grad_outputs=torch.ones_like(net), create_graph=True)[0]
    net_x1 = net_x[:,0].view(-1,1)
    net_x2 = net_x[:,1].view(-1,1)
    x1 = x[:,0].view(-1, 1)
    x2 = x[:,1].view(-1, 1)

    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(net_x.size(1)):
        grad2 = torch.autograd.grad(net_x[:, i], x, grad_outputs=torch.ones_like(net_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    net_xx = torch.stack(hessian, dim=-1)
    net_x1x1 = net_xx[:, 0, 0].view(-1, 1)
    net_x2x2 = net_xx[:, 1, 1].view(-1, 1)

    f1 = torch.reshape(x2, (-1,1))
    f2 = torch.reshape(-g*torch.sin(x1)/l, (-1,1))

    # f1_x1 = torch.reshape(torch.autograd.grad(f1, x1, grad_outputs=torch.ones_like(f1), create_graph=True)[0], (-1,1))
    # f2_x2 = torch.reshape(torch.autograd.grad(f2, x2, grad_outputs=torch.ones_like(f2), create_graph=True)[0], (-1,1))
    f1_x1 = (0.0*x1).view(-1,1)
    f2_x2 = (0.0*x2).view(-1,1)

    Lnet = net_x1*f1 + net*f1_x1 + net_x2*f2 + net*f2_x2 - 0.5*(B_torch[0,0]*B_torch[0,0]*net_x1x1 + B_torch[1,1]*B_torch[1,1]*net_x2x2)
    residual = net_t + Lnet + p_res

    if(verbose):
      print("residual: ", residual, residual.shape)
    return residual


def diff_e1(x, t, e1_net, verbose=False):
    B_torch = torch.tensor(B, dtype=torch.float32, requires_grad=True)
    net = e1_net(x,t)
    net_x = torch.autograd.grad(net, x, grad_outputs=torch.ones_like(net), create_graph=True)[0]
    net_t = torch.autograd.grad(net, t, grad_outputs=torch.ones_like(net), create_graph=True)[0]
    net_x1 = net_x[:,0].view(-1,1)
    net_x2 = net_x[:,1].view(-1,1)
    x1 = x[:,0].view(-1, 1)
    x2 = x[:,1].view(-1, 1)

    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(net_x.size(1)):
        grad2 = torch.autograd.grad(net_x[:, i], x, grad_outputs=torch.ones_like(net_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    net_xx = torch.stack(hessian, dim=-1)
    net_x1x1 = net_xx[:, 0, 0].view(-1, 1)
    net_x2x2 = net_xx[:, 1, 1].view(-1, 1)

    f1 = torch.reshape(x2, (-1,1))
    f2 = torch.reshape(-g*torch.sin(x1)/l, (-1,1))

    # f1_x1 = torch.reshape(torch.autograd.grad(f1, x1, grad_outputs=torch.ones_like(f1), create_graph=True)[0], (-1,1))
    # f2_x2 = torch.reshape(torch.autograd.grad(f2, x2, grad_outputs=torch.ones_like(f2), create_graph=True)[0], (-1,1))
    f1_x1 = (0.0*x1).view(-1,1)
    f2_x2 = (0.0*x2).view(-1,1)

    Lnet = net_x1*f1 + net*f1_x1 + net_x2*f2 + net*f2_x2 - 0.5*(B_torch[0,0]*B_torch[0,0]*net_x1x1 + B_torch[1,1]*B_torch[1,1]*net_x2x2)
    diff_e1 = net_t + Lnet

    return diff_e1


def train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_ti, iterations=40000):
    min_loss = np.inf
    loss_history = []
    iterations_per_decay = 1000
    PATH = FOLDER+"output/e1_net.pth"
    x_mar = 0.0

    # space-time points for BC
    x_bc = (torch.rand(1000, n_d) * (x_hig - x_low) + x_low).to(device); #print(min(x_bc[:,0]), max(x_bc[:,0]), min(x_bc[:,1]), max(x_bc[:,1]))
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    t = (torch.rand(1000, 1, requires_grad=True) *   (tf - ti) + ti).to(device)

    S = 30000
    FLAG = False

    # Save intermediate enet
    save_count = 1
    save_loss = 10.0

    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p_bc = p_init_torch(x_bc)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = (p_bc - phat_bc)/max_abs_e1_ti
        net_bc_out = e1_net(x_bc, t_bc)/max_abs_e1_ti
        mse_u = mse_cost_function(net_bc_out, u_bc.detach())

        # Loss based on PDE
        diff_e1_hat = diff_e1(x, t, e1_net)
        diff_e1_target = -res_func(x, t, p_net).detach()
        mse_res = mse_cost_function(diff_e1_hat/max_abs_e1_ti, diff_e1_target/max_abs_e1_ti)

        # Frequnecy Loss
        res_out = e1_res_func(x, t, e1_net, p_net)/max_abs_e1_ti
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = torch.mean(norm_res_input**2)
        
        # Combining the loss functions
        loss = mse_u + 5.0*mse_res
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
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
                    'train_time': train_time,
                    }, PATH)
            min_loss = loss.data 
            FLAG = True

        # save intermediate results
        if(loss.item() < 0.75*save_loss):
            a1_data = compute_stat(p_net, e1_net)
            np.savez(FOLDER_INTERMED+'a1_data_'+str(save_count)+'.npz', loss=loss.item(), a1_data=a1_data)
            save_count = save_count + 1
            save_loss = loss.item()


        # RAR
        if (epoch%100 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            ic_hat_RAR = e1_net(x_RAR, t0_RAR)/max_abs_e1_ti
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            ic_RAR = (p_bc_RAR - phat_bc_RAR)/max_abs_e1_ti
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

            res_RAR = e1_res_func(x_RAR, t_RAR, e1_net, p_net)/max_abs_e1_ti
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
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    a1_data = compute_stat(p_net, e1_net)
    np.savez(FOLDER_INTERMED+'a1_data_'+str(save_count)+'.npz', loss=loss.item(), a1_data=a1_data)
    np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))


def pos_e1_net_train(e1_net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    e1_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("e1net best epoch: ", epoch, ", loss:", loss.data, ", train time:", checkpoint['train_time'])
    # keys = e1_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(e1_net.state_dict()[k], p=2)
    #     # print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("e1net loss")
    plt.savefig(FOLDER+'figs/e1net_loss_history.pdf', format='pdf', dpi=300)
    plt.close()
    return e1_net


def show_e1_net_results(p_net, e1_net):
    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 18,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 18,              # Title font size
    "axes.labelsize": 18,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 18,             # Legend text size
    "legend.title_fontsize": 18        # Legend title size (if you use legend titles)
    })

    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    # plot e1net vs e1
    e1_all = []
    e1hat_all = []
    for t1 in t1s:
        p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p_hat = p_net(pt_x, pt_t1)
        p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat_numpy
        e1_all.append(e1)
        e1_hat_numpy = e1_net(pt_x, pt_t1).detach().numpy().reshape((sample_size, sample_size))
        e1hat_all.append(e1_hat_numpy)
    e1_all = np.concatenate(e1_all)
    vmin = np.min(e1_all)
    vmax = np.max(e1_all)
    B1_max = 2.0*np.max(np.abs(e1hat_all))

    # Define ticks
    ticks_values = [-2*np.pi, 0, 2*np.pi]
    ticks_labels = [r'$-2\pi$', r'$0$', r'$2\pi$']

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 6, width_ratios=[1]*5 + [0.05], wspace=0.2)
    # Create subplot grid (2x5) for the plots
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(5)]
    # Create a subplot for the colorbar spanning the height of the grid
    cax = fig.add_subplot(gs[:, -1])
    for i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        if(i <= 4):
            t1 = t1s[i]
            p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
            pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
            p_hat = p_net(pt_x, pt_t1)
            p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
            e1 = p - p_hat_numpy
            cp = ax.imshow(e1, extent=[x_low, x_hig, x_low, x_hig], cmap='inferno', aspect='equal', origin='lower',
                           vmin=vmin, vmax=vmax)
            ax.set_title("t="+str(t1))
            if(i == 0):
                ax.set_ylabel(r"$\omega$")
                ax.set_yticks(ticks_values, ticks_labels, fontsize=14)
        else:
            t1 = t1s[i-5]
            p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
            pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
            p_hat = p_net(pt_x, pt_t1)
            p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
            e1 = p - p_hat_numpy
            pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
            e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
            cp = ax.imshow(e1_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='inferno', aspect='equal', origin='lower',
                            vmin=vmin, vmax=vmax)
            alpha = max(abs(e1.reshape(-1,1) - e1_hat.reshape(-1,1))) / max(abs(e1_hat.reshape(-1,1)))
            alpha = alpha[0]
            print("t: ",t1, ", a1: {:.3f}".format(alpha))
            ax.set_xlabel(r"$\theta$")
            ax.set_xticks(ticks_values, ticks_labels, fontsize=14)
            ax.set_title(r"$\alpha_1=$"+str(np.round(alpha,2)))
            if(i == 5):
                ax.set_ylabel(r"$\omega$")
                ax.set_yticks(ticks_values, ticks_labels, fontsize=14)
                
    # Add the colorbar to the colorbar subplot
    cbar = fig.colorbar(cp, cax=cax, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # Add a box with text at the top-left corner of the figure
    fig.text(0.01, 0.87, r"$e_1(x,t)$", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    fig.text(0.01, 0.47, r"$\hat{e}_1(x,t)$", bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.1, top=0.87, wspace=0.4, hspace=0.1)
    plt.savefig(FOLDER+'figs/2dpend_e1hatresult.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # Paper: error bound plot
    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 14,                   # Base font size for non-math text
    
    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols
    
    # Title and label sizes
    "axes.titlesize": 14,              # Title font size
    "axes.labelsize": 14,              # Axis label font size
    
    # Legend settings
    "legend.fontsize": 14,             # Legend text size
    "legend.title_fontsize": 14        # Legend title size (if you use legend titles)
    })
    N_dataset = 3
    palette = sns.color_palette("dark", N_dataset)
    grid_points_struct = load_gridpoints_from_monte()
    grid_points = grid_points_struct[-1]
    dx1 = grid_points_struct[0][1] - grid_points_struct[0][0]
    dx2 = grid_points_struct[2][1] - grid_points_struct[2][0]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]

    num_stride = 6
    fig, axs = plt.subplots(1, 5, figsize=(16, 12), subplot_kw={'projection': '3d'})
    for i in range(len(t1s)):
        ax = axs[i]
        t_eval = t1s[i]
        pdf_true = np.load(FOLDER_DATA+"p_sim_grid"+str(t_eval)+".npy")
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_eval)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
        e1 = pdf_true - pdf_nn.reshape(x1_grid.shape)
        e1_hat = e1_net(grid_points_tensor, t_tensor).data.cpu().numpy()
        B1 = 2*np.max(np.abs(e1_hat.ravel()))
        ax.plot_wireframe(x1_grid, x2_grid, np.abs(e1), 
                      color=palette[0], linewidth=0.7, alpha=1.0, 
                      rstride=num_stride, cstride=num_stride, label=r"$|e_1|$")
        ax.plot_wireframe(x1_grid, x2_grid, np.abs(e1_hat.reshape(x1_grid.shape)), 
                        color=palette[1], linewidth=0.7, alpha=1.0, 
                        rstride=num_stride, cstride=num_stride, label=r"$|\hat{e}_1|$")
        ax.plot_surface(x1_grid, x2_grid, e1*0.0 + B1, color=palette[2], alpha=0.3, label=r"$B_1$")
        ax.view_init(22,-30)
        # ax.set_title("t="+str(t_eval))
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.set_ylabel(r"$\omega$", fontsize=14)
        if(i == 0):
            ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60), fontsize=16)  
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # ax.set_xlabel(r'$\theta$')
        # ax.set_ylabel(r'$w$')
        ax.text2D(0.98, 0.75, "Error", transform=ax.transAxes)
        ax.text2D(0.45, 0.90,  "t="+str(t_eval), transform=ax.transAxes)
        ax.set_yticks(ticks_values, ticks_labels, fontsize=14)
        ax.set_xticks(ticks_values, ticks_labels, fontsize=14)
        ax.set_zlim([0,1.05*B1_max])
        # ax.set_zticks(np.linspace(np.min(np.abs(e1)), B1, num=5))  
        # ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60))  
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.9, wspace=0.25, hspace=0.3)
    plt.savefig(FOLDER+'figs/2dpend_errorbound.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


    # ax.plot_wireframe(x1_grid, x2_grid, np.abs(e1), 
    #                   color=palette[0], linewidth=0.7, alpha=1.0, 
    #                   rstride=num_stride, cstride=num_stride, label=r"$|e_1|$")
    # ax.plot_wireframe(x1_grid, x2_grid, np.abs(e1_hat.reshape(x1_grid.shape)), 
    #                   color=palette[1], linewidth=0.7, alpha=1.0, 
    #                   rstride=num_stride, cstride=num_stride, label=r"$|\hat{e}_1|$")
    # ax.plot_surface(x1_grid, x2_grid, e1*0.0 + B1, color=palette[2], alpha=0.3, label=r"$B_1$")
    # ax.view_init(22,-30)
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.set_xlabel(r'$\theta$')
    # ax.set_ylabel(r'$w$')
    # # ax.set_zlabel("|Error|", labelpad=15)  # Increase label padding for spacing
    # ax.text2D(0.94, 0.77, "Error", transform=ax.transAxes)
    # ax.set_zticks(np.linspace(np.min(np.abs(e1)), B1, num=5))  
    # ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60)) 
    # # ax.zaxis.set_label_coords(0.5, 5.1)  # Move label slightly above the axis
    # # ax.zaxis.label.set_rotation(0)  # Rotate label to horizontal    
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(FOLDER+'figs/2dnl_e.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    # plt.close()

    # plot special error bound
    # fig = plt.figure(figsize=(10, 6))
    # j = 0
    # for t1 in t1s:
    #     if (j < 5):
    #         ax = fig.add_subplot(2, 3, j+1, projection="3d")
    #         p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
    #         pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    #         p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
    #         e1 = p - p_hat
    #         e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
    #         error_bound = max(abs(e1_hat.reshape(-1,1)))*2
    #         error_bound = error_bound[0]
    #         ax.plot_surface(x1, x2, abs(e1), cmap='viridis', vmin=0.0, vmax=0.008)
    #         ax.plot_surface(x1, x2, e1_hat*0+error_bound, color="green", alpha=0.3)
    #         ax.set_zlim([0, 0.02])
    #         ax.set_xlabel(r"$\theta$", fontsize=8)
    #         ax.set_ylabel(r"$\omega$", fontsize=8)
    #         ax.set_zlabel(r"$|e|$", fontsize=8)
    #         print("t: ",t1, ", max|e|: {:.3f}".format(np.max(np.abs(e1))), ", e_S: {:.3f}".format(error_bound))
    #         ax.set_title("t="+str(t1)+", "+r"$e_S=$"+str(np.round(error_bound,3)))
    #     j = j + 1
    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5)
    # plt.savefig(FOLDER+'figs/special_error_bound.pdf', format='pdf', dpi=300)
    # plt.close()


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
    fig.savefig(FOLDER+'figs/2dpend_trainloss.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
       self.format = "%1.1f"


def plot_results_at_one_time(t1, p_net, e1_net):
    plt.rcParams['font.size'] = 14
    # Load data
    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
    pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    p_hat = p_net(pt_x, pt_t1)
    p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
    e1 = p - p_hat_numpy
    pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
    alpha = max(abs(e1.reshape(-1,1) - e1_hat.reshape(-1,1))) / max(abs(e1_hat.reshape(-1,1)))
    alpha = alpha[0]
    alpha = np.round(alpha, 2)

    p_all = []
    p_all.append(p)
    p_all.append(p_hat_numpy)
    p_all = np.concatenate(p_all)
    pmin = np.min(p_all)
    pmax = np.max(p_all)

    e_all = []
    e_all.append(e1)
    e_all = np.concatenate(e_all)
    emin = np.min(e_all)
    emax = np.max(e_all)

    data1 = p
    data2 = p_hat_numpy
    data3 = e1
    data4 = e1_hat

    # Create the figure and 1x5 grid of subplots using gridspec to adjust layout
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 2])

    # Create axes for the subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4], projection='3d')

    # First two subplots: 2D imshow with a common vertical colorbar
    im1 = ax1.imshow(data1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', origin="lower")
    im2 = ax2.imshow(data2, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', origin="lower")
    ax1.set_xticks(np.array([-9, -5, 0, 5, 9]))
    ax1.set_yticks(np.array([-9, -5, 0, 5, 9]))
    ax2.set_xticks(np.array([-9, -5, 0, 5, 9]))
    ax2.set_yticks([])
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax2.set_xlabel(r"$x_1$")
    ax1.text(0.05, 0.95, r"$p$", 
             transform=ax1.transAxes, verticalalignment='top', fontsize=16,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax2.text(0.05, 0.95, r"$\hat{p}$", 
             transform=ax2.transAxes, verticalalignment='top', fontsize=16,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    # Create a single colorbar for the first two subplots and adjust its location
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', fraction=0.018, pad=0.2)
    cbar.set_label(r'Colorbar for $p$ and $\hat{p}$', fontsize=12)

    # Next two subplots: 2D imshow with a common horizontal colorbar
    im3 = ax3.imshow(data3, extent=[x_low, x_hig, x_low, x_hig], cmap='plasma', origin="lower")
    im4 = ax4.imshow(data4, extent=[x_low, x_hig, x_low, x_hig], cmap='plasma', origin="lower")
    ax3.set_xticks(np.array([-9, -5, 0, 5, 9]))
    ax4.set_xticks(np.array([-9, -5, 0, 5, 9]))
    ax3.set_yticks([])
    ax4.set_yticks([])
    ax3.set_xlabel(r"$x_1$")
    ax4.set_xlabel(r"$x_1$")
    ax3.text(0.05, 0.95, r"$e_1$", 
             transform=ax3.transAxes, verticalalignment='top', fontsize=16,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax4.text(0.05, 0.95, r"$\hat{e}_1$", 
             transform=ax4.transAxes, verticalalignment='top', fontsize=16,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    # Create a single colorbar for the third and fourth subplots and adjust its location
    cbar2 = fig.colorbar(im3, ax=[ax3, ax4], orientation='horizontal', fraction=0.018, pad=0.2)
    cbar2.set_label(r'Colorbar for $e_1$ and $\hat{e}_1$', fontsize=12)

    # Fifth subplot: 3D surface plot
    ax5.plot_surface(x1, x2, e1, cmap='plasma', label=r"$e_1$")
    eS = 2.0 * np.max(np.abs(e1_hat))
    eS = np.round(eS, 3)
    ax5.plot_surface(x1, x2, e1_hat*0.0 + eS, color="green", alpha=0.3, label=r"$e_S$")
    ax5.plot_surface(x1, x2, e1_hat*0.0 - eS, color="green", alpha=0.3)
    ax5.set_zlabel("  e")
    ax5.set_xticks(np.array([-9, -5, 0, 5, 9]))
    ax5.set_yticks(np.array([-9, -5, 0, 5, 9]))
    ax5.set_xlabel(r"$x_1$")
    ax5.set_ylabel(r"$x_2$")
    ax5.legend(loc="upper left")

    # Add a common title above the first four subplots
    common_txt = "t="+str(t1) + ", " + r"$\alpha_1=$"+str(alpha) + ", " + r"$e_S=$" + str(eS)
    fig.text(0.4, 0.8, common_txt, ha='center', fontsize=16)

    plt.savefig(FOLDER+'figs/special_error_bound_t1.pdf', format='pdf', dpi=300)
    plt.close()

    # # Create a figure with a specified size
    # fig = plt.figure(figsize=(9, 5))
    
    # # Create a GridSpec for a 2x2 layout for 2D plots and 1x1 for 3D plot
    # gs = GridSpec(2, 3, width_ratios=[0.8, 0.9, 1.7])  # Adjust width ratios to make 3D plot larger

    # # 2D Subplots
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax4 = fig.add_subplot(gs[1, 1])

    # # Plot p
    # im1 = ax1.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', 
    #                  origin='lower', vmin=pmin, vmax=pmax)
    # ax1.text(0.05, 0.95, r"$p$", 
    #          transform=ax1.transAxes, verticalalignment='top', fontsize=16,
    #          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    # ax1.set_xticks([])
    # ax1.set_yticks(np.array([-9, -5, 0, 5, 9]))
    # ax1.set_ylabel(r"$x_2$")

    # # Plot p_hat
    # im2 = ax2.imshow(p_hat_numpy, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', 
    #                  origin='lower', vmin=pmin, vmax=pmax)
    # ax2.text(0.05, 0.95, r"$\hat{p}$", 
    #          transform=ax2.transAxes, verticalalignment='top', fontsize=16,
    #          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # # Add colorbar for p_hat
    # cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    # zScalarFormatter = ScalarFormatterClass(useMathText=True)
    # zScalarFormatter.set_powerlimits((0, 0))
    # cbar2.ax.yaxis.set_major_formatter(zScalarFormatter)
    # # cbar2.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # # cbar2.set_label('Value')
    
    # # Plot e1
    # im3 = ax3.imshow(e1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', 
    #                  origin='lower', vmin=emin, vmax=emax)
    # ax3.text(0.05, 0.95, r"$e$", 
    #          transform=ax3.transAxes, verticalalignment='top', fontsize=16,
    #          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    # ax3.set_xticks(np.array([-9, -5, 0, 5, 9]))
    # ax3.set_yticks(np.array([-9, -5, 0, 5, 9]))
    # ax3.set_ylabel(r"$x_2$")
    # ax3.set_xlabel(r"$x_1$")

    # # Plot e1_hat
    # im4 = ax4.imshow(e1_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', 
    #                  origin='lower', vmin=emin, vmax=emax)
    # ax4.text(0.05, 0.95, r"$\hat{e}_1$", 
    #          transform=ax4.transAxes, verticalalignment='top', fontsize=16,
    #          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    # ax4.set_yticks([])
    # ax4.set_xticks(np.array([-9, -5, 0, 5, 9]))
    # ax4.set_xlabel(r"$x_1$")

    # # Add colorbar for e1_hat
    # cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical', fraction=0.046, pad=0.04)
    # zScalarFormatter = ScalarFormatterClass(useMathText=True)
    # zScalarFormatter.set_powerlimits((0, 0))
    # cbar4.ax.yaxis.set_major_formatter(zScalarFormatter)
    # # cbar4.set_label('Error')

    # # 3D Surface Plot
    # ax5 = fig.add_subplot(gs[:, 2], projection='3d')  # Use the entire right column for 3D plot
    # ax5.plot_surface(x1, x2, e1, cmap='viridis', vmin=emin, vmax=emax)
    # eS = 2.0 * np.max(np.abs(e1_hat))
    # eS = np.round(eS, 3)
    # ax5.plot_surface(x1, x2, e1_hat*0.0 + eS, color="green", alpha=0.3)
    # ax5.plot_surface(x1, x2, e1_hat*0.0 - eS, color="green", alpha=0.3)
    # ax5.set_zlabel("e")
    # ax5.set_xticks(np.array([-9, -5, 0, 5, 9]))
    # ax5.set_yticks(np.array([-9, -5, 0, 5, 9]))
    # zScalarFormatter = ScalarFormatterClass(useMathText=True)
    # zScalarFormatter.set_powerlimits((0, 0))
    # ax5.zaxis.set_major_formatter(zScalarFormatter)
    # ax5.set_ylabel(r"$x_2$")
    # ax5.set_xlabel(r"$x_1$")
    # ax5.set_title("t="+str(t1) + ", " + r"$\alpha_1=$"+str(alpha) + ", " + r"$e_S=$" + str(eS))

    # plt.subplots_adjust(left=0.08, right=0.9, top=0.90, bottom=0.1)
    # plt.savefig(FOLDER+'figs/special_error_bound_t1.pdf', format='pdf', dpi=300)
    # plt.close()


def show_table(p_net, e1_net):
    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = [1.0, 2.0, 3.0, 4.0, 5.0]
    a1_list = []
    gap_list = []
    eS_ratio_list = []
    for t1 in t1s:
        p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        eS = max(abs(e1_hat.reshape(-1,1)))*2
        eS = eS[0]
        a1 = max(abs(e1.reshape(-1,1) - e1_hat.reshape(-1,1))) / max(abs(e1_hat.reshape(-1,1)))
        a1 = a1[0]
        a1_list.append(a1)
        e1max = max(abs(e1.reshape(-1,1)))
        gap = (eS - e1max)/ max(abs(p.reshape(-1,1)))
        gap_list.append(gap)
        eS_ratio_list.append(eS/ max(abs(p.reshape(-1,1))) )
    a1_list = np.array(a1_list)
    gap_list = np.array(gap_list)
    norm_B2_list = eS_ratio_list
    print("[info] max a1: ", np.round(np.max(a1_list),2), ", avg a1:", np.round(np.mean(a1_list),2), ", std a1:", np.round(np.std(a1_list),3))
    print("[info] min gap: ", np.round(np.min(gap_list),3), ", avg gap:", np.round(np.mean(gap_list),3))
    print("[info] avg B2_norm: ", np.round(np.mean(norm_B2_list),2), ", std B2_norm:", np.round(np.std(norm_B2_list),3))


def compute_stat(p_net, e1_net):
    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    t1s = [1.0, 2.0, 3.0, 4.0, 5.0]
    a1_list = []
    for t1 in t1s:
        p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        a1 = max(abs(e1.reshape(-1,1) - e1_hat.reshape(-1,1))) / max(abs(e1_hat.reshape(-1,1)))
        a1 = a1[0]
        a1_list.append(a1)
    a1_list = np.array(a1_list)
    return a1_list


def load_gridpoints_from_monte():
    x1s = np.load(FOLDER_DATA+"/x_points.npy")
    x2s = np.load(FOLDER_DATA+"/x_points.npy")
    x1_grid, x2_grid = np.meshgrid(x1s, x2s) # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel().ravel()]).T
    return [x1s, x1_grid, x2s, x2_grid, grid_points]


def plot_paper(p_net, e1_net, t_eval=1.0):
    grid_points_struct = load_gridpoints_from_monte()
    grid_points = grid_points_struct[-1]
    dx1 = grid_points_struct[0][1] - grid_points_struct[0][0]
    dx2 = grid_points_struct[2][1] - grid_points_struct[2][0]
    x1_grid = grid_points_struct[1]
    x2_grid = grid_points_struct[3]

    pdf_true = np.load(FOLDER_DATA+"p_sim_grid"+str(t_eval)+".npy")
    # print("[check] x1_grid x2_grid x3_grid shape: ", x1_grid.shape, x1_grid.dtype)
    # for t in constants.T_SPAN:
    #     # print("[check] t=",t)
    #     # load true pdf(t)
    #     pdf_true = constants.load_p_sol_monte(t)
    #     # print("[check] pdf_true dtype: ", pdf_true.dtype)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_eval)
    #     # print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy()
    e1 = pdf_true - pdf_nn.reshape(x1_grid.shape)
    e1_hat = e1_net(grid_points_tensor, t_tensor).data.cpu().numpy()
    B1 = 2*np.max(np.abs(e1_hat.ravel()))

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

    N_dataset = 3
    palette = sns.color_palette("dark", N_dataset)
    fig, axs = plt.subplots(1, 1, figsize=(9, 6), subplot_kw={'projection': '3d'})
    ax = axs
    num_stride = 5
    ax.plot_surface(x1_grid, x2_grid, pdf_nn.reshape(x1_grid.shape), cmap='viridis', alpha=0.6, label=r"$\hat{p}$")
    ax.plot_wireframe(x1_grid, x2_grid, pdf_true.reshape(x1_grid.shape), 
                      color="black", linewidth=1.5, alpha=1.0, 
                      rstride=num_stride, cstride=num_stride, label=r"$p$")
    # ax.plot_wireframe(x1_grid, x2_grid, pdf_nn.reshape(x1_grid.shape), 
    #                   color=palette[1], linewidth=0.7, alpha=1.0, 
    #                   rstride=num_stride, cstride=num_stride, label=r"$\hat{p}$")
    ax.view_init(22,-30)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$w$')
    ax.text2D(0.96, 0.72, "PDF", transform=ax.transAxes)
    # Create a dummy plot for the legend
    # dummy_plot = ax.plot([], [], [], color=palette[0], label=r'$p$')
    # Create the legend and place it at specific 2D coordinates (x, y)
    ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.65), fontsize=28) 
    plt.tight_layout()
    # plt.show()  
    plt.savefig(FOLDER+'figs/2dnl_p.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    num_stride = 6
    fig, axs = plt.subplots(1, 1, figsize=(9, 6), subplot_kw={'projection': '3d'})
    ax = axs
    ax.plot_wireframe(x1_grid, x2_grid, np.abs(e1), 
                      color="black", linewidth=1.5, alpha=1.0, 
                      rstride=num_stride, cstride=num_stride, label=r"$|e_1|$")
    ax.plot_surface(x1_grid, x2_grid, np.abs(e1_hat).reshape(x1_grid.shape), cmap='inferno', alpha=0.5, label=r"$|\hat{e}_1|$")
    # ax.plot_wireframe(x1_grid, x2_grid, np.abs(e1_hat.reshape(x1_grid.shape)), 
    #                   color=palette[1], linewidth=0.7, alpha=1.0, 
    #                   rstride=num_stride, cstride=num_stride, label=r"$|\hat{e}_1|$")
    ax.plot_surface(x1_grid, x2_grid, e1*0.0 + B1, color=palette[2], alpha=0.3, label=r"$B_1$")
    ax.view_init(22,-30)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$w$')
    ax.text2D(0.94, 0.77, "Error", transform=ax.transAxes)
    ax.set_zticks(np.linspace(np.min(np.abs(e1)), B1, num=5))  
    ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60), fontsize=28)    
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/2dnl_e.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def main():
    # test_p_sol_monte(stat_sample=20000000)

    mse_cost_function = torch.nn.MSELoss() # Mean squared error

    p_net = Net().to(device)
    p_net.apply(init_weights_He)
    e1_net = E1Net().to(device)
    e1_net.apply(init_weights_He)

    max_pi = test_p_init()
    p_net.scale = max_pi
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_pi, iterations=15000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    print("[load pnet model from: "+FOLDER+"output/p_net.pth]")
    # show_p_net_results(p_net)

    max_abe_e1_ti = get_e1net_scale(p_net)
    e1_net.scale = max_abe_e1_ti
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abe_e1_ti, iterations=40000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    print("[load e1net model from: "+FOLDER+"output/e1_net.pth]")
    show_e1_net_results(p_net ,e1_net)
    show_table(p_net, e1_net)
    plot_train_loss(FOLDER+"output/p_net_train_loss.npy", FOLDER+"output/e1_net_train_loss.npy")
    plot_results_at_one_time(3.0, p_net, e1_net)
    plot_paper(p_net, e1_net, t_eval=3.0)

    if(TRAIN_FLAG == False):
        print("[complete 2d nonlinear, with pre-trained models]")
    else:
        print("[complete 2d nonlinear]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Modify the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()