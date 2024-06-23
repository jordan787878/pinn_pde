import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import multivariate_normal
from scipy.linalg import expm
from tqdm import tqdm
from matplotlib.lines import Line2D
import torch.nn.functional as F

### Try Fixed Sampling? ###


# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
FOLDER = "exp1/RAR/"
FOLDER_DATA = "exp1/data/"
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
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

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


# def p_sol(x, t):
#     exp_At = expm(A*t)
#     mu_t = np.dot(exp_At, mu_0)
#     cov_t = exp_At @ cov_0 @ (exp_At.T)
#     pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
#     pdf_eval = pdf_func.pdf(x).reshape(-1,1)
#     return pdf_eval


def p_sol_monte(t1=ti, linespace_num=50, stat_sample=50000):
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
    # bins_1D = np.linspace(x_low, x_hig, num=linespace_num)
    # dx = bins_1D[1]-bins_1D[0]
    # inds1 = np.digitize(x1, bins_1D)
    # inds2 = np.digitize(x2, bins_1D)
    # grid = np.zeros((len(bins_1D), len(bins_1D)))
    # for i in range(len(inds1)):
    #     if(x1[i]< x_hig and x1[i]> x_low and x2[i]< x_hig and x2[i]> x_low):
    #         grid[inds2[i]-1][inds1[i]-1] += (1.0/stat_sample)
    # # convert statistical prob to density p = pdf*(dx^d)
    # grid = grid/(dx**n_d)
    # X_grid, Y_grid = np.meshgrid(bins_1D, bins_1D)
    X_grid, Y_grid = np.meshgrid(midpoints_x1, midpoints_x2)
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
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # ax1 = axs[0]
    # c = ax1.imshow(p_exact0, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    # ax1.set_xlabel('x1')
    # ax1.set_ylabel('x2')
    # fig.colorbar(c, ax=ax1)
    # p_exact1 = p_sol(x, t=tf).reshape((sample_size, sample_size))
    # ax2 = axs[1]
    # c = ax2.imshow(p_exact1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    # ax2.set_xlabel('x1')
    # ax2.set_ylabel('x2')
    # fig.colorbar(c, ax=ax2)
    # plt.show()
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# p_net
class Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        return output
    
# p_net with residual connection
# class Net(nn.Module):
#     def __init__(self, scale=1.0): 
#         neurons = 32
#         self.scale = scale
#         super(Net, self).__init__()
#         self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
#         self.hidden_layer2 = (nn.Linear(neurons,neurons))
#         self.hidden_layer3 = (nn.Linear(neurons,neurons))
#         self.hidden_layer4 = (nn.Linear(neurons,neurons))
#         self.hidden_layer5 = (nn.Linear(neurons,neurons))
#         self.hidden_layer6 = (nn.Linear(neurons,neurons))
#         self.hidden_layer7 = (nn.Linear(neurons,neurons))
#         self.hidden_layer8 = (nn.Linear(neurons,neurons))
#         self.hidden_layer9 = (nn.Linear(neurons,neurons))
#         self.hidden_layer10 = (nn.Linear(neurons,neurons))
#         self.output_layer =  (nn.Linear(neurons,1))
#     def forward(self, x, t):
#         inputs = torch.cat([x,t],axis=1)
#         layer1_out = F.softplus((self.hidden_layer1(inputs)))
#         layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
#         # layer2_out += layer1_out
#         layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
#         layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
#         # layer4_out += layer3_out
#         layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
#         layer6_out = F.softplus((self.hidden_layer6(layer5_out)))
#         layer7_out = F.softplus((self.hidden_layer7(layer6_out)))
#         layer8_out = F.softplus((self.hidden_layer8(layer7_out)))
#         layer9_out = F.softplus((self.hidden_layer9(layer8_out)))
#         layer10_out = F.softplus((self.hidden_layer10(layer9_out)))
#         output = F.softplus(self.output_layer(layer10_out))
#         return output
                

# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def sigmoid_transform(tensor, epsilon=1e-3):
    """
    Approximate the count of row vectors in the tensor that are close to zero within epsilon distance.
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, 2).
        epsilon (float): Distance from zero to consider.

    Returns:
        torch.Tensor: A tensor containing a single value which is the approximate count.
    """
    # Calculate the norm of each row vector
    distances = torch.norm(tensor, dim=1)
    # Apply a smooth indicator function (sigmoid)
    soft_indicators = torch.sigmoid((epsilon - distances) * 10000)  # scale factor 100000 to make transition sharp
    return soft_indicators


def count_approx_zero_elements(tensor, epsilon=1e-3):
    """
    Approximate the count of row vectors in the tensor that are close to zero within epsilon distance.
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, 2).
        epsilon (float): Distance from zero to consider.
    Returns:
        torch.Tensor: A tensor containing a single value which is the approximate count.
    """
    # Calculate the norm of each row vector
    distances = torch.norm(tensor, dim=1)
    # Apply a smooth indicator function (sigmoid)
    soft_indicators = torch.sigmoid((epsilon - distances) * 10000)  # scale factor 100000 to make transition sharp
    # Sum the soft indicators to get the approximate count
    approx_count = torch.sum(soft_indicators)
    return approx_count


def train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_abs_p_ti, iterations=40000):
    global x_low, x_hig, ti, tf
    batch_size = 600
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    x_mar = 0.0

    # Define the mean and covariance matrix
    mean = torch.tensor([pi*(0.5), 0.0])
    covariance_matrix = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    # space-time points for BC
    x_bc = (torch.rand(batch_size, n_d) * (x_hig - x_low) + x_low).to(device); #print(min(x_bc[:,0]), max(x_bc[:,0]), min(x_bc[:,1]), max(x_bc[:,1]))
    x_bc_normal = mvn.sample((batch_size,))
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(2500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    t = (torch.rand(2500, 1, requires_grad=True) *   (tf - ti) + ti).to(device)

    # RAR
    S = 100000
    
    PATH = FOLDER+"output/p_net.pth"

    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        u_bc = p_init_torch(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)
        linf_u = linf_loss(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        res_out = res_func(x, t, p_net)/max_abs_p_ti
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)
        linf_res = linf_loss(res_out, all_zeros)

        # Frequnecy Loss
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input/max_abs_p_ti, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)
        linf_norm_res_input = linf_loss(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + mse_res + mse_norm_res_input
        # loss = linf_u + 1e-2*(linf_res+ linf_norm_res_input)

        # RAR
        if (epoch%500 == 0):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            res_RAR = res_func(x_RAR, t_RAR, p_net)/max_abs_p_ti
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
            norm_res_input_RAR = torch.norm(res_input_RAR/max_abs_p_ti, dim=1).view(-1,1)
            if(torch.mean(norm_res_input) > 0.0):
                max_abs_res_input, max_index = torch.max(norm_res_input_RAR, dim=0)
                # Get the corresponding x_RAR and t_RAR vectors
                x_max = x_RAR[max_index]
                t_max = t_RAR[max_index]
                # Append x_max and t_max to x and t
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... add [x,t]:", x_max.data, t_max.data, max_abs_res_input.data)


        loss_history.append(loss.data)
        
        # Save the min loss model
        if(loss.data < min_loss):
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                   ",linf-ic:", linf_u.data, ",norm. linf-res:", linf_res.data, 
                   ",res_freq:", mse_norm_res_input.data, linf_norm_res_input.data # , linf_res_t.data #res_cross_zero_metric.data, metric.data
            #       # "FFT: ", largest_frequency_output.data)
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
                np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()



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
    # x_low = -2.0
    # x_hig = 2.0
    max_abe_e1_ti = np.inf
    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_ti = Variable(torch.from_numpy(x[:,0]*0+ti).float(), requires_grad=True).view(-1,1).to(device)
    p0 = p_init(x)
    p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    e1 = p0 - p_hat
    max_abe_e1_ti = max(abs(e1))[0]

    fig, axs = plt.subplots(2, 6, figsize=(18, 6))
    j = 0
    for t1 in t1s:
        p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        axp = axs[0, j]
        if(j == 0):
            p0_monte = p
            ax1 = axs[1, 0]
            ax2 = axs[1, 1]
            print("[DEBUG] Monte Sim Error:", max(abs(p0 - p.reshape(-1,1))) )
            p0 = p0.reshape(sample_size,sample_size)
            e1i = p0 - p
            cp = axp.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
            fig.colorbar(cp, ax=axp)
            cp = ax1.imshow(p0, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
            fig.colorbar(cp, ax=ax1)
            cp = ax2.imshow(e1i, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
            fig.colorbar(cp, ax=ax2)
        else:
            cp = axp.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
            fig.colorbar(cp, ax=axp)
            axp.set_xlabel(r"$\theta$")
            axp.set_ylabel(r"$\omega$")
            axp.set_title("t="+str(t1))
        j = j + 1
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/p_sol_monte.png")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    surf1 = axs[0].plot_wireframe(x1, x2, p0_monte, linewidth=0.5, linestyle="--", color="red", rstride=8, cstride=8, label=r"$p(x,0)$ [Monte]")
    surf2 = axs[0].plot_wireframe(x1, x2, p0, linewidth=0.5, color="blue", rstride=8, cstride=8, label=r"$p(x,0)$ [Exact]")
    axs[0].legend()
    # axs[0].view_init(20, 120)
    surf3 = axs[1].plot_wireframe(x1, x2, e1i, linewidth=0.5, color="blue", rstride=8, cstride=8, label="error of exact and Monte")
    axs[1].legend()
    plt.savefig(FOLDER+"figs/p_monte_vs_exact.png")
    plt.close()

    fig, axs = plt.subplots(1, 6, figsize=(18, 6))
    j = 0
    rstride = 10
    cstride = 10
    for t1 in t1s:
        # p = p_sol(x, t=t1).reshape((sample_size, sample_size))
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p_hat = p_net(pt_x, pt_t1)
        p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
        axp = axs[j]
        cp = axp.imshow(p_hat_numpy, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        # wire1 = axp.plot_wireframe(x1, x2, p, color="blue", alpha=1.0, rstride=rstride, cstride=cstride)
        # wire2 = axp.plot_wireframe(x1, x2, p_hat, color="red", alpha=1.0, rstride=rstride, cstride=cstride)
        # wire1.set_linewidth(0.5)
        # wire2.set_linewidth(0.5)  # Adjust the line width as needed
        # wire2.set_linestyle("--")
        # cp = axp.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        # cphat = axphat.imshow(p_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        fig.colorbar(cp, ax=axp)
        axp.set_xlabel(r"$\theta$")
        axp.set_ylabel(r"$\omega$")
        # fig.colorbar(cphat, ax=axphat)
        # if(j == 0):
        #     axp.set_xlabel(r"$x_1$")
        #     axp.set_ylabel(r"$x_2$")
        # #     axp.set_zlabel(r"$p$")
        # # Add legend
        # if j == 0:
        #     axp.set_zlabel(r"$p$")
            # legend_elements = [
            #     Line2D([0], [0], color='blue', lw=2, label=r"$p$"),
            #     Line2D([0], [0], color='red', lw=2, linestyle='--', label=r"$\hat{p}$")
            # ]
            # axp.legend(handles=legend_elements)
        axp.set_title("t="+str(t1))
        j = j + 1
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_result.png")
    plt.close()

    fig, axs = plt.subplots(1, 6, figsize=(18, 6))
    j = 0
    for t1 in t1s:
        p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        if(j == 0):
            p = p0.reshape(sample_size, sample_size)
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p_hat = p_net(pt_x, pt_t1)
        p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat_numpy
        axp = axs[j]
        cp = axp.imshow(e1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        fig.colorbar(cp, ax=axp)
        axp.set_xlabel(r"$\theta$")
        axp.set_ylabel(r"$\omega$")
        axp.set_title("t="+str(t1))
        j = j + 1
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/pnet_error.png")
    plt.close()

    fig, axs = plt.subplots(1, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
    j = 0
    for t1 in t1s:
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        res_out = res_func(pt_x, pt_t1, p_net)
        res_numpy = res_out.data.cpu().numpy().reshape((sample_size, sample_size))
        ax1 = axs[j]
        surf1 = ax1.plot_surface(x1, x2, res_numpy, cmap='viridis')
        # ax2 = axs[1, j]
        # surf2 = ax2.plot_surface(x1, x2, e1, cmap='viridis')
        # ax2.plot_surface(x1, x2, e1*0, alpha=0.3, color="red")
        # if(j == 0):
        #     ax1.set_zlabel(r"$r_1$")
        #     ax2.set_zlabel(r"$e_1$")
        ax1.set_title("t="+str(t1))
        j = j + 1
    plt.savefig(FOLDER+"figs/pnet_resdiual.png")
    plt.close()

    return max_abe_e1_ti


class E1Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 32
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.Tanh()
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        output = self.output_layer(layer5_out)
        output = self.scale * output
        return output
    
# class E1Net(nn.Module):
#     def __init__(self, scale=1.0): 
#         neurons = 20
#         self.scale = scale
#         super(E1Net, self).__init__()
#         self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
#         self.hidden_layer2 = (nn.Linear(neurons,neurons))
#         self.hidden_layer3 = (nn.Linear(neurons,neurons))
#         self.hidden_layer4 = (nn.Linear(neurons,neurons))
#         self.hidden_layer5 = (nn.Linear(neurons,neurons))
#         self.hidden_layer6 = (nn.Linear(neurons,neurons))
#         self.hidden_layer7 = (nn.Linear(neurons,neurons))
#         self.hidden_layer8 = (nn.Linear(neurons,neurons))
#         self.hidden_layer9 = (nn.Linear(neurons,neurons))
#         self.hidden_layer10 = (nn.Linear(neurons,neurons))
#         self.output_layer =  (nn.Linear(neurons,1))
#     def forward(self, x, t):
#         inputs = torch.cat([x,t],axis=1)
#         layer1_out = torch.tanh((self.hidden_layer1(inputs)))
#         layer2_out = torch.tanh((self.hidden_layer2(layer1_out)))
#         layer3_out = torch.tanh((self.hidden_layer3(layer2_out)))
#         layer4_out = torch.tanh((self.hidden_layer4(layer3_out)))
#         layer5_out = torch.tanh((self.hidden_layer5(layer4_out)))
#         layer6_out = torch.tanh((self.hidden_layer6(layer5_out)))
#         layer7_out = torch.tanh((self.hidden_layer7(layer6_out)))
#         layer8_out = torch.tanh((self.hidden_layer8(layer7_out)))
#         layer9_out = torch.tanh((self.hidden_layer9(layer8_out)))
#         layer10_out = torch.tanh((self.hidden_layer10(layer9_out)))
#         output = self.output_layer(layer10_out)
#         output = self.scale * output
#         return output


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


# def test_e1_res(e1_net, p_net):
#     batch_size = 5
#     x_collocation = np.random.uniform(low=1.0, high=3.0, size=(batch_size,1))
#     t_collocation = T_end*np.ones((batch_size,1))
#     all_zeros = np.zeros((batch_size,1))
#     pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
#     pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
#     f_out = e1_res_func(pt_x_collocation, pt_t_collocation, e1_net, p_net, verbose=True) # output of f(x,t)


def train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abs_e1_ti, iterations=40000):
    min_loss = np.inf
    loss_history = []
    iterations_per_decay = 1000
    PATH = FOLDER+"output/e1_net.pt"
    x_mar = 0.0

    # Define the mean and covariance matrix
    mean = torch.tensor([pi*(0.5), 0.0])
    covariance_matrix = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    # space-time points for BC
    x_bc = (torch.rand(600, n_d) * (x_hig - x_low) + x_low).to(device); #print(min(x_bc[:,0]), max(x_bc[:,0]), min(x_bc[:,1]), max(x_bc[:,1]))
    x_bc_normal = mvn.sample((600,))
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    x = (torch.rand(2500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
    t = (torch.rand(2500, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
    FLAG = False
    S = 100000

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        p_bc = p_init_torch(x_bc)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = (p_bc - phat_bc)/max_abs_e1_ti
        net_bc_out = e1_net(x_bc, t_bc)/max_abs_e1_ti
        mse_u = mse_cost_function(net_bc_out, u_bc)

        # Loss based on PDE
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e1_res_func(x, t, e1_net, p_net)/max_abs_e1_ti
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
                    }, PATH)
            min_loss = loss.data 
            FLAG = True

        # RAR
        if (epoch%1000 == 0 and FLAG):
            x_RAR = (torch.rand(S, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            ic_hat_RAR = e1_net(x_RAR, t0_RAR)/max_abs_e1_ti
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            ic_RAR = (p_bc_RAR - phat_bc_RAR)/max_abs_e1_ti
            mean_ic_error = torch.mean(torch.abs(ic_RAR - ic_hat_RAR))
            print("RAR mean IC: ", mean_ic_error.data)
            if(mean_ic_error > 5e-3):
                max_abs_ic, max_index = torch.max(torch.abs(ic_RAR - ic_hat_RAR), dim=0)
                x_max = x_RAR[max_index]
                t_max = t0_RAR[max_index]
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... IC add [x,t]:", x_max.data, t_max.data, ". max ic value: ", max_abs_ic.data)
                FLAG = False

            res_RAR = e1_res_func(x_RAR, t_RAR, e1_net, p_net)/max_abs_e1_ti
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

        loss_history.append(loss.data)
        loss.backward(retain_graph=True) 
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
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/e1net_loss_history.png")
    plt.close()
    return e1_net


def show_e1_net_results(p_net, e1_net):
    x_points = np.load(FOLDER_DATA+"x_points.npy")
    sample_size = len(x_points)
    x1s = x_points
    x2s = x_points
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    p0 = p_init(x).reshape(sample_size, sample_size)

    fig, axs = plt.subplots(1, 6, figsize=(18, 6))
    j = 0
    for t1 in t1s:
        if(j == 0):
            print("use exact p for t="+str(t1))
            p = p0
            # print("use Monte p for t="+str(t1))
            # p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        else:
            # From Monte Carlo
            print("use Monte p for t="+str(t1))
            p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        ax1 = axs[j]
        cp = ax1.imshow(e1_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        fig.colorbar(cp, ax=ax1)
        ax1.set_xlabel(r"$\theta$")
        ax1.set_ylabel(r"$\omega$")
        ax1.set_title("t="+str(t1))
        alpha = max(abs(e1.reshape(-1,1) - e1_hat.reshape(-1,1))) / max(abs(e1_hat.reshape(-1,1)))
        ax1.set_title("t="+str(t1)+"\n"+r"$\alpha_1=$"+str(np.round(alpha,2)))
        print("t1: ", t1, " , alpha_1: ", alpha)

        j = j + 1
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/e1_result.png")
    plt.close()

    # fig, axs = plt.subplots(1, 6, figsize=(18, 6))
    # j = 0
    # rstride = 10
    # cstride = 10
    # for t1 in t1s:
    #     # p = p_sol(x, t=t1).reshape((sample_size, sample_size))
    #     pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    #     p_hat = p_net(pt_x, pt_t1)
    #     p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
    #     axp = axs[j]
    #     cp = axp.imshow(p_hat_numpy, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    #     # wire1 = axp.plot_wireframe(x1, x2, p, color="blue", alpha=1.0, rstride=rstride, cstride=cstride)
    #     # wire2 = axp.plot_wireframe(x1, x2, p_hat, color="red", alpha=1.0, rstride=rstride, cstride=cstride)
    #     # wire1.set_linewidth(0.5)
    #     # wire2.set_linewidth(0.5)  # Adjust the line width as needed
    #     # wire2.set_linestyle("--")
    #     # cp = axp.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    #     # cphat = axphat.imshow(p_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    #     fig.colorbar(cp, ax=axp)
    #     axp.set_xlabel(r"$\theta$")
    #     axp.set_ylabel(r"$\omega$")
    #     # fig.colorbar(cphat, ax=axphat)
    #     # if(j == 0):
    #     #     axp.set_xlabel(r"$x_1$")
    #     #     axp.set_ylabel(r"$x_2$")
    #     # #     axp.set_zlabel(r"$p$")
    #     # # Add legend
    #     # if j == 0:
    #     #     axp.set_zlabel(r"$p$")
    #         # legend_elements = [
    #         #     Line2D([0], [0], color='blue', lw=2, label=r"$p$"),
    #         #     Line2D([0], [0], color='red', lw=2, linestyle='--', label=r"$\hat{p}$")
    #         # ]
    #         # axp.legend(handles=legend_elements)
    #     axp.set_title("t="+str(t1))
    #     j = j + 1
    # plt.tight_layout()
    # plt.savefig(FOLDER+"figs/pnet_result.png")
    # plt.close()
    # fig, axs = plt.subplots(1, 6, figsize=(18, 6))
    # j = 0
    # for t1 in t1s:
    #     p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
    #     pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    #     p_hat = p_net(pt_x, pt_t1)
    #     p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
    #     e1 = p - p_hat_numpy
    #     axp = axs[j]
    #     cp = axp.imshow(e1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    #     fig.colorbar(cp, ax=axp)
    #     axp.set_xlabel(r"$\theta$")
    #     axp.set_ylabel(r"$\omega$")
    #     axp.set_title("t="+str(t1))
    #     j = j + 1
    # plt.tight_layout()
    # plt.savefig(FOLDER+"figs/pnet_error.png")
    # plt.close()
    # fig, axs = plt.subplots(1, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
    # j = 0
    # for t1 in t1s:
    #     pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    #     res_out = res_func(pt_x, pt_t1, p_net)
    #     res_numpy = res_out.data.cpu().numpy().reshape((sample_size, sample_size))
    #     ax1 = axs[j]
    #     surf1 = ax1.plot_surface(x1, x2, res_numpy, cmap='viridis')
    #     # ax2 = axs[1, j]
    #     # surf2 = ax2.plot_surface(x1, x2, e1, cmap='viridis')
    #     # ax2.plot_surface(x1, x2, e1*0, alpha=0.3, color="red")
    #     # if(j == 0):
    #     #     ax1.set_zlabel(r"$r_1$")
    #     #     ax2.set_zlabel(r"$e_1$")
    #     ax1.set_title("t="+str(t1))
    #     j = j + 1
    # plt.savefig(FOLDER+"figs/pnet_resdiual.png")
    # plt.close()



def main():
    # test_p_sol_monte(stat_sample=100000000)

    max_pi = test_p_init()
    print("max abs p(x,ti):", max_pi)

    p_net = Net(scale=max_pi).to(device)
    p_net.apply(init_weights)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_pi, iterations=80000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abe_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,ti):", max_abe_e1_ti)

    e1_net = E1Net(scale=max_abe_e1_ti).to(device)
    e1_net.apply(init_weights)
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    # train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abe_e1_ti, iterations=100000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_net_results(p_net ,e1_net)

    print("[complete 2d linear]")


if __name__ == "__main__":
    main()