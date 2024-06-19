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
from matplotlib.lines import Line2D

import torch.nn.functional as F


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3)
#         self.activation = nn.Tanh()
#     def forward(self, x, t):
#         x = torch.cat([x,t], axis=1).view(-1,3,1)
#         # Input x has shape (batch_size, 3, 1)
#         x = self.conv1(x)  # Output shape: (batch_size, 1, new_length)
#         x = self.activation(x)
#         # Perform average pooling over the last dimension
#         x = torch.mean(x, dim=2)  # Output shape: (batch_size, 1)
#         return x


# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
FOLDER = "exp/1-mse-resgrad/"
device = "cpu"
print(device)
n_d = 2
mu_0 = np.array([1.0, -1.0]).reshape(2,)
cov_0 = np.array([[0.5, 0.0], [0.0, 0.5]])
A = np.array([[0.0, 1.0],[0.0, 0.0]])
x_low = -4.0
x_hig =  4.0
ti = 0.0
tf = 5.0

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


def p_sol(x, t):
    exp_At = expm(A*t)
    mu_t = np.dot(exp_At, mu_0)
    cov_t = exp_At @ cov_0 @ (exp_At.T)
    pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1)
    return pdf_eval


def test_p_init():
    sample_size = 50
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
    A_torch = torch.tensor(A, dtype=torch.float32, requires_grad=True)
    p = p_net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x1 = torch.reshape(p_x[:,0], (-1,1))
    p_x2 = torch.reshape(p_x[:,1], (-1,1))
    x1 = x[:,0]
    x2 = x[:,1]
    f1 = torch.reshape(A_torch[0,0]*x1+ A_torch[0,1]*x2, (-1,1))
    f2 = torch.reshape(A_torch[1,0]*x1+ A_torch[1,1]*x2, (-1,1))
    f1_x1 = torch.reshape(torch.autograd.grad(f1, x1, grad_outputs=torch.ones_like(f1), create_graph=True)[0], (-1,1))
    f2_x2 = torch.reshape(torch.autograd.grad(f2, x2, grad_outputs=torch.ones_like(f2), create_graph=True)[0], (-1,1))
    Lp = p_x1*f1 + p*f1_x1 + p_x2*f2 + p*f2_x2
    residual = p_t + Lp
    if(verbose):
      print("residual: ", residual, residual.shape)
    return residual


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


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
        # output = torch.tanh(output)
        return output
                

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
    batch_size = 200
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []

    x_mar = 0.0

    # N_ti = 2
    # dt_freq = 0.1
    # dx_freq = 0.1
    # x_quad = torch.arange(x_low, x_hig+dx_freq, dx_freq, dtype=torch.float, requires_grad=True).to(device)
    # x1 = x_quad.repeat_interleave(len(x_quad)).view(-1, 1)
    # x2 = x_quad.repeat(len(x_quad)).view(-1, 1)
    # xs = torch.cat([x1,x2], axis=1)
    # t_quad = torch.arange(ti, tf+dt_freq, dt_freq, dtype=torch.float, requires_grad=True).to(device)
    # x_fix = xs.repeat((len(t_quad),1))
    # t_fix = t_quad.repeat(len(xs)).view(-1,1)

    PATH = FOLDER+"output/p_net.pt"
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero
        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, n_d) * (x_hig - x_low) + x_low).to(device); #print(min(x_bc[:,0]), max(x_bc[:,0]), min(x_bc[:,1]), max(x_bc[:,1]))
        t_bc = (torch.ones(batch_size, 1) * ti).to(device)
        u_bc = p_init_torch(x_bc)
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)
        linf_u = linf_loss(net_bc_out/max_abs_p_ti, u_bc/max_abs_p_ti)

        # Loss based on PDE
        t = (torch.rand(batch_size, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
        # t = torch.cat((t, (torch.ones(N_ti, 1) * ti).to(device) ))
        x = (torch.rand(len(t), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        # x = torch.clamp(x, min=x_low, max=x_hig)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = res_func(x, t, p_net)/max_abs_p_ti
        mse_res = mse_cost_function(res_out, all_zeros)
        linf_res = linf_loss(res_out, all_zeros)

        # Frequnecy Loss
        # x = x_fix
        # t = t_fix
        # res_out = res_func(x, t, p_net)/max_abs_p_ti
        # t = (torch.rand(2000, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
        # # t = torch.cat((t, (torch.ones(N_ti, 1) * ti).to(device) ))
        # x = (torch.rand(len(t), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        # # x = torch.clamp(x, min=x_low, max=x_hig)
        # res_out = res_func(x, t, p_net)/max_abs_p_ti
        res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        res_input = torch.cat([res_x, res_t], axis=1)
        norm_res_input = torch.norm(res_input/max_abs_p_ti, dim=1).view(-1,1)
        # # norm_res_t = torch.norm(res_t/max_abs_p_ti, dim=1).view(-1,1)
        # all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        # # linf_res_t = linf_loss(norm_res_t, all_zeros)
        # linf_res_x = linf_loss(norm_res_x, all_zeros)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)
        # # mse_norm_res_t = mse_cost_function(norm_res_t, all_zeros)
        # # res_count_zero = count_approx_zero_elements(res_out-torch.mean(res_out), epsilon=1e-4)
        # # test = torch.exp(-torch.abs(res_out)) * norm_res_x
        # # test_metric = mse_norm_res_x
        # test_metric = linf_res_x
        # test_metric = res_count_zero/len(t)
        # test_metric = linf_loss(test, all_zeros)
        # test_metric = linf_loss(norm_res_x, all_zeros)
        # test_metric = mse_cost_function(norm_res_x, all_zeros)
        # test_metric = torch.max(norm_res_x)
        # test_metric = torch.mean(test)
        # print("=== DEBUG: ===", test_metric.data)
        # print("=== DEBUG: === res crossing zero: ", res_cross_zero.data)
        # print("=== DEBUG: === exp  metric      : ", res_cross_zero_metric.data)

        # Loss Function
        # <mse>
        loss = mse_u + mse_res #+ mse_norm_res_input
        # loss = torch.max(linf_u, 0*linf_u+0.08) + torch.max(linf_res, 0*linf_res+0.06) + largest_frequency_output
        # loss = torch.max(linf_u, 0*linf_u+0.08) + torch.max(linf_res, 0*linf_res+0.07)
        # <drdx>
        # loss = torch.max(mse_u, mse_u*0+1e-5) + torch.max(mse_res, mse_res*0+1e-5) + 1e-5*test_metric
        # loss = mse_u + mse_res + 1e-2*res_cross_zero_metric
        # loss = torch.max(linf_u, 0*linf_u+0.1) + torch.max(linf_res, 0*linf_res+0.1) + 1e-3*res_cross_zero_metric
        # loss = mse_u + mse_res + 5e-3*( test_metric )
        # loss = test_metric + mse_norm_res_x
        # loss = torch.max(linf_u, linf_u*0+0.01) + torch.max(linf_res, linf_res*0+0.01) + 0.01*test_metric

        loss_history.append(loss.data)
        # Save the min loss model
        if(loss.data < min_loss):
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, ",res:", mse_res.data,
                  ",norm. linf-ic:", linf_u.data, ",norm. linf-res:", linf_res.data, 
                  ",res_freq:", mse_norm_res_input.data# , linf_res_t.data #res_cross_zero_metric.data, metric.data
                  # "FFT: ", largest_frequency_output.data)
                  )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data,
                    }, PATH)
            min_loss = loss.data

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        with torch.autograd.no_grad():
            if (epoch%1000 == 0):
                print(epoch,"Traning Loss:",loss.data)
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
        # termination condition
        # if(linf_u.data <= 0.05 and linf_res <= 0.05 and res_freq.data/batch_size <= 0.05):
        #     np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))
        #     break
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
    sample_size = 200
    x1s = np.linspace(x_low, x_hig, num=sample_size)
    x2s = np.linspace(x_low, x_hig, num=sample_size)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_ti = Variable(torch.from_numpy(x[:,0]*0+ti).float(), requires_grad=True).view(-1,1).to(device)
    p = p_sol(x, t=ti)
    p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    e1 = p - p_hat
    max_abe_e1_ti = max(abs(e1))[0]

    t1s = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
    fig, axs = plt.subplots(1, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
    j = 0
    rstride = 10
    cstride = 10
    for t1 in t1s:
        p = p_sol(x, t=t1).reshape((sample_size, sample_size))
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        # print(pt_x.shape); print(pt_t1.shape)
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        axp = axs[j]
        axp.view_init(elev=20, azim=135)
        wire1 = axp.plot_wireframe(x1, x2, p, color="blue", alpha=1.0, rstride=rstride, cstride=cstride)
        wire2 = axp.plot_wireframe(x1, x2, p_hat, color="red", alpha=1.0, rstride=rstride, cstride=cstride)
        wire1.set_linewidth(0.5)
        wire2.set_linewidth(0.5)  # Adjust the line width as needed
        wire2.set_linestyle("--")
        # cp = axp.imshow(p, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        # cphat = axphat.imshow(p_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        # fig.colorbar(cp, ax=axp)
        # fig.colorbar(cphat, ax=axphat)
        if(j == 0):
            axp.set_xlabel(r"$x_1$")
            axp.set_ylabel(r"$x_2$")
        #     axp.set_zlabel(r"$p$")
        # Add legend
        if j == 0:
            axp.set_zlabel(r"$p$")
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label=r"$p$"),
                Line2D([0], [0], color='red', lw=2, linestyle='--', label=r"$\hat{p}$")
            ]
            axp.legend(handles=legend_elements)
            
        axp.set_title("t="+str(t1))
        j = j + 1
    plt.savefig(FOLDER+"figs/pnet_result.png")
    plt.close()

    fig, axs = plt.subplots(2, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
    j = 0
    for t1 in t1s:
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p = p_sol(x, t=t1).reshape((sample_size, sample_size))
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat
        res_out = res_func(pt_x, pt_t1, p_net)
        res_numpy = res_out.data.cpu().numpy().reshape((sample_size, sample_size))
        ax1 = axs[0, j]
        surf1 = ax1.plot_surface(x1, x2, res_numpy, cmap='viridis')
        ax1.plot_surface(x1, x2, e1*0, alpha=0.3, color="red")
        ax2 = axs[1, j]
        surf2 = ax2.plot_surface(x1, x2, e1, cmap='viridis')
        ax2.plot_surface(x1, x2, e1*0, alpha=0.3, color="red")
        if(j == 0):
            ax1.set_zlabel(r"$r_1$")
            ax2.set_zlabel(r"$e_1$")
        ax1.set_title("t="+str(t1))
        j = j + 1
    plt.savefig(FOLDER+"figs/pnet_resdiual.png")
    plt.close()

    return max_abe_e1_ti


class E1Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 100
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
    A_torch = torch.tensor(A, dtype=torch.float32, requires_grad=True)
    e1 = e1_net(x,t)
    e1_x = torch.autograd.grad(e1, x, grad_outputs=torch.ones_like(e1), create_graph=True)[0]
    e1_t = torch.autograd.grad(e1, t, grad_outputs=torch.ones_like(e1), create_graph=True)[0]
    e1_x1 = e1_x[:,0].view(-1,1)
    e1_x2 = e1_x[:,1].view(-1,1)
    x1 = x[:,0]
    x2 = x[:,1]
    f1 = torch.reshape(A_torch[0,0]*x1+ A_torch[0,1]*x2, (-1,1))
    f2 = torch.reshape(A_torch[1,0]*x1+ A_torch[1,1]*x2, (-1,1))
    f1_x1 = torch.reshape(torch.autograd.grad(f1, x1, grad_outputs=torch.ones_like(f1), create_graph=True)[0], (-1,1))
    f2_x2 = torch.reshape(torch.autograd.grad(f2, x2, grad_outputs=torch.ones_like(f2), create_graph=True)[0], (-1,1))
    Le1 = e1_x1*f1 + e1*f1_x1 + e1_x2*f2 + e1*f2_x2
    residual = e1_t + Le1 + p_res
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
    batch_size = 200
    min_loss = np.inf
    loss_history = []
    iterations_per_decay = 1000
    PATH = FOLDER+"output/e1_net.pt"
    x_mar = 0.0
    # dt_step = 1e-3
    # dt_train = 1e-2
    # N_ti = 2
    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc = (torch.rand(batch_size, n_d) * (x_hig - x_low) + x_low).to(device)
        t_bc = (torch.ones(batch_size, 1) * ti).to(device)
        p_bc = p_init_torch(x_bc)
        phat_bc = p_net(x_bc, t_bc)
        u_bc = p_bc - phat_bc
        net_bc_out = e1_net(x_bc, t_bc)
        mse_u = mse_cost_function(net_bc_out/max_abs_e1_ti, u_bc/max_abs_e1_ti)
        linf_u = linf_loss(net_bc_out/max_abs_e1_ti, u_bc/max_abs_e1_ti)

        # Loss based on PDE
        # t = torch.cat((t, (torch.ones(N_ti, 1) * ti).to(device) ))
        x = (torch.rand(batch_size, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
        # x1 = (torch.rand(batch_size, 1, requires_grad=True) * (3.0  - 1.0)  + 1.0).to(device)
        # x2 = (torch.rand(batch_size, 1, requires_grad=True) * (-1.0 - -3.0) + -3.0).to(device)
        # x = torch.cat((x, torch.cat([x1,x2], axis=1)), dim=0)
        t = (torch.rand(len(x), 1, requires_grad=True) *   (tf - ti) + ti).to(device)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = e1_res_func(x, t, e1_net, p_net)
        mse_res = mse_cost_function(res_out/max_abs_e1_ti, all_zeros)
        linf_res = linf_loss(res_out/max_abs_e1_ti, all_zeros)
        
        # Combining the loss functions
        loss = mse_u + mse_res
        # loss = linf_u + linf_res

        # Save the min loss model
        if(loss.data < min_loss):
            print("e1net best epoch:", epoch, ", loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data)
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
    # x_low = -2.0
    # x_hig = 2.0
    sample_size = 50
    x1s = np.linspace(x_low, x_hig, num=sample_size)
    x2s = np.linspace(x_low, x_hig, num=sample_size)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    t1s = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
    j = 0
    for t1 in t1s:
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p = p_sol(x, t=t1).reshape((sample_size, sample_size))
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        ax1 = axs[0, j]
        ax2 = axs[1, j]
        # Set the desired viewing angle
        # ax1.view_init(elev=30, azim=-20)  # Change the elevation and azimuth to your preference
        # ax2.view_init(elev=30, azim=-20)  # Change the elevation and azimuth to your preference
        ax1.plot_surface(x1, x2, e1, cmap='viridis')
        ax2.plot_surface(x1, x2, e1_hat, cmap='viridis')
        if(j == 0):
            ax1.set_zlabel(r"$e_1$")
            ax1.set_xlabel(r"$x_1$")
            ax1.set_ylabel(r"$x_2$")
            ax2.set_zlabel(r"$\hat{e}_1$")
        ax1.set_title("t="+str(t1))
        e1 = e1.reshape(-1,1)
        e1_hat = e1_hat.reshape(-1,1)
        a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        print(t1, a1)
        ax2.set_xlabel(r"$\alpha_1=$"+str(np.round(a1,2)))
        j = j + 1
    plt.savefig(FOLDER+"figs/e1net_result.png")
    plt.close()


def show_uniform_error_bound(p_net, e1_net):
    # x_low = -2.0
    # x_hig = 2.0
    sample_size = 50
    x1s = np.linspace(x_low, x_hig, num=sample_size)
    x2s = np.linspace(x_low, x_hig, num=sample_size)
    x1, x2 = np.meshgrid(x1s, x2s)
    x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    t1s = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0]
    fig, axs = plt.subplots(1, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
    j = 0
    for t1 in t1s:
        pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
        p = p_sol(x, t=t1).reshape((sample_size, sample_size))
        p_hat = p_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        e1 = p - p_hat
        e1_hat = e1_net(pt_x, pt_t1).data.cpu().numpy().reshape((sample_size, sample_size))
        error_bound = max(abs(e1_hat.reshape(-1,1)))*2
        surf1 = axs[j].plot_surface(x1, x2, abs(e1), cmap='viridis')
        surf1 = axs[j].plot_surface(x1, x2, e1*0+error_bound, alpha=0.3, label=r"B_{e_1}")
        axs[j].set_zlabel(r"$|e_1|$")
        axs[j].set_xlabel(r"$x_1$")
        axs[j].set_ylabel(r"$x_2$")
        axs[j].set_title("t="+str(t1))
        # c = ax1.imshow(e1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        # fig.colorbar(c, ax=ax1)
        # ax2 = axs[1, j]
        # c = ax2.imshow(e1_hat, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
        # fig.colorbar(c, ax=ax2)
        # if(j == 0):
        #     ax1.set_ylabel(r"$e_1$")
        #     ax2.set_ylabel(r"$\hat{e}_1$")
        # ax1.set_title("t="+str(t1))
        # e1 = e1.reshape(-1,1)
        # e1_hat = e1_hat.reshape(-1,1)
        # a1 = max(abs(e1-e1_hat))/max(abs(e1_hat))
        # ax2.set_xlabel(r"$\alpha_1=$"+str(np.round(a1[0],2)))
        j = j + 1
    plt.savefig(FOLDER+"figs/uniform_error_bound.png")
    plt.close()


def main():
    # x = torch.ones(5, 2)
    # norm = torch.norm(x, dim=1)
    # print(x)
    # print(norm)

    max_pi = test_p_init()
    print("max abs p(x,ti):", max_pi)

    p_net = Net(scale=max_pi).to(device)
    p_net.apply(init_weights)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adamax(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    # train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_pi, iterations=60000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abe_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,ti):", max_abe_e1_ti)

    e1_net = E1Net(scale=max_abe_e1_ti).to(device)
    # e1_net = CNN()
    e1_net.apply(init_weights)
    # e1_net = TransformerModel(3, 1).to(device)
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adamax(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abe_e1_ti, iterations=40000); print("e1_net train complete")
    e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    show_e1_net_results(p_net ,e1_net)
    show_uniform_error_bound(p_net, e1_net)

    print("[complete 2d linear]")


if __name__ == "__main__":
    main()