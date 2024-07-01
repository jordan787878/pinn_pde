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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

### Try Fixed Sampling? ###


# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
FOLDER = "exp1/tmp/"
FOLDER_DATA = "exp1/data/"
device = "cpu"
print(device)
pi = np.pi
n_d = 3
mu_0 = np.array([-1.0, -3.0, 0.0]).reshape(n_d,)
cov_0 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.017]])
B = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
x_low = [-4.0, -6.0, -0.5*pi]
x_hig = [ 4.0,  6.0,  1.5*pi]
ti = 0.0
tf = 5.0
g = 9.8
l = 9.8
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def get_u1(t):
    return 1.0


def get_u2(t):
    return 0.8*np.sin(1.2*t)


def get_u2_torch(t):
    return 0.8*torch.sin(1.2*t)



# x1: x
# x2: y
# x3: theta
def f_sde(x, t):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    u1 = get_u1(t)
    u2 = get_u2(t)
    dx1dt = u1*np.cos(x3)
    dx2dt = u1*np.sin(x3)
    dx3dt = u2
    return np.array([dx1dt, dx2dt, dx3dt]).reshape(n_d,)


def test_dynamics():
    dtt = 0.01
    t_span = np.arange(ti, tf, dtt)
    _mean = mu_0
    _cov = cov_0

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')

    x = np.random.multivariate_normal(_mean, _cov).T
    _t_sim = np.round(ti,6)
    for t in t_span:
        if(_t_sim % 0.5 == 0):
            axs.scatter(x[0], x[1], x[2]*180.0/pi)
            print(_t_sim, x[0], x[1], x[2])

        # Generate Brownian motion increments
        w = np.random.normal(0, np.sqrt(dtt), size=n_d)
        # Update x using the SDE discretization
        x = x + f_sde(x,t) * dtt + np.dot(B, w)
        _t_sim = np.round(_t_sim + dtt,6)

    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel(r'$\theta$')
    plt.savefig(FOLDER+"figs/test_dynamics.png")
    plt.close()


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


def p_sol_monte(t1=ti, linespace_num=100, stat_sample=10000):
    dtt = 0.01
    t_span = np.arange(ti, t1, dtt)
    _mean = mu_0
    _cov = cov_0

    # Initialize X to store samples
    X = np.zeros((n_d, stat_sample))

    # Generate samples using Monte Carlo simulation
    for i in tqdm(range(stat_sample), desc="Processing samples"):
        x = np.random.multivariate_normal(_mean, _cov).T
        for t in t_span:
            # Generate Brownian motion increments
            w = np.random.normal(0, np.sqrt(dtt), size=n_d)
            # Update x using the SDE discretization
            x = x + f_sde(x,t) * dtt + np.dot(B, w)
        X[:, i] = x
        # print(x)

    # Define bins as the edges of x1, x2, x3
    bins_x1 = np.linspace(x_low[0], x_hig[0], num=linespace_num)
    bins_x2 = np.linspace(x_low[1], x_hig[1], num=linespace_num)
    bins_x3 = np.linspace(x_low[2], x_hig[2], num=linespace_num)

    # Digitize X to find which bin each value falls into for all dimensions
    bin_indices_x1 = np.digitize(X[0, :], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[1, :], bins_x2) - 1
    bin_indices_x3 = np.digitize(X[2, :], bins_x3) - 1

    # Initialize the frequency array
    frequency_3d = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1, len(bins_x3) - 1))

    # Count the occurrences in each 3D bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        if (0 <= bin_indices_x1[i] < frequency_3d.shape[0] and
            0 <= bin_indices_x2[i] < frequency_3d.shape[1] and
            0 <= bin_indices_x3[i] < frequency_3d.shape[2]):
            # print(bin_indices_x1[i], bin_indices_x2[i], bin_indices_x3[i])
            frequency_3d[bin_indices_x2[i], bin_indices_x1[i], bin_indices_x3[i]] += 1

    # Normalize the frequency to get the proportion
    frequency_3d = frequency_3d / stat_sample
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    dx3 = bins_x3[1] - bins_x3[0]
    frequency_3d = frequency_3d / (dx1*dx2*dx3)

    # Calculate the midpoints for bins
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
    return midpoints_x1, midpoints_x2, midpoints_x3, frequency_3d


def test_p_sol_monte(linspace_num=20, stat_sample=10000):
    j = 0
    for t1 in t1s:
        print("generate p sol Monte for t="+str(t1))
        x1_points, x2_points, x3_points, p_sim_grid = p_sol_monte(t1=t1, linespace_num=linspace_num, stat_sample=stat_sample)
        np.save(FOLDER_DATA+"x1_points.npy", x1_points)
        np.save(FOLDER_DATA+"x2_points.npy", x2_points)
        np.save(FOLDER_DATA+"x3_points.npy", x3_points)
        np.save(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy", p_sim_grid)
        j = j + 1


def plot_p_sol_Monte():
    x1_points = np.load(FOLDER_DATA+"x1_points.npy")
    x2_points = np.load(FOLDER_DATA+"x2_points.npy")
    x3_points = np.load(FOLDER_DATA+"x3_points.npy")
    x1, x2, x3 = np.meshgrid(x1_points, x2_points, x3_points)

    # p_max_global = 0.0
    # for t1 in t1s:
    #     if(t1 == 0.0):
    #         p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy").flatten()
    #         p_max = np.max(p)
    #         p_max_global = max(p_max_global, p_max)
    # print(p_max_global)

    # fig = plt.figure(figsize=(20, 6))
    # axs = [fig.add_subplot(1, len(t1s), i+1, projection='3d') for i in range(len(t1s))]
    # j = 0
    for t1 in t1s:
        if(t1 <= 10):
            print("plot p monte",t1)
            p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy")
            fig = go.Figure(data=go.Volume(
            x=x1.flatten(), y=x2.flatten(), z=x3.flatten(),
            value=p.flatten(),
            # isomin=0.0,
            # isomax=p_max_global,
            opacity=0.1,
            surface_count=11,
            caps= dict(x_show=True, y_show=True, z_show=True, x_fill=1), # with caps (default mode)
            ))

            # Change camera view for a better view of the sides, XZ plane
            name = 'eye = (x:0., y:0., z:2.5)'
            camera = dict(
                        eye=dict(x=0., y=0., z=2.5)
                        )

            fig.update_layout(scene_camera=camera, title="p Monte")
            # (see https://plotly.com/python/v3/3d-camera-controls/)
            # fig.update_layout(
            #    scene_camera = dict(
            #    up=dict(x=0, y=0, z=1),
            #    center=dict(x=0, y=0, z=0),
            #    eye=dict(x=0.1, y=2.5, z=0.1)
            #    ),
            #    title="p Monte"
            #)
            # fig.show()
            fig.write_image(FOLDER+"figs/p_sol_monte_t"+str(t1)+".png")
        


def test_p_init():
    sample_size = 100
    x1s = np.linspace(x_low, x_hig, num=sample_size)
    x2s = np.linspace(x_low, x_hig, num=sample_size)
    x3s = np.linspace(x_low, x_hig, num=sample_size)
    x1, x2, x3 = np.meshgrid(x1s, x2s, x3s)
    x = np.column_stack([x1.ravel(), x2.ravel(), x3.ravel()])#; print(x)
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
    B_torch = torch.tensor(B, dtype=torch.float32, requires_grad=True)
    p = p_net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_x1 = p_x[:,0].view(-1,1)
    p_x2 = p_x[:,1].view(-1,1)
    p_x3 = p_x[:,2].view(-1,1)
    x1 = x[:,0].view(-1, 1)
    x2 = x[:,1].view(-1, 1)
    x3 = x[:,2].view(-1, 1)

    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(p_x.size(1)):
        grad2 = torch.autograd.grad(p_x[:, i], x, grad_outputs=torch.ones_like(p_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    p_xx = torch.stack(hessian, dim=-1)
    p_x1x1 = p_xx[:, 0, 0].view(-1, 1)
    p_x2x2 = p_xx[:, 1, 1].view(-1, 1)
    p_x3x3 = p_xx[:, 2, 2].view(-1, 1)

    u1 = get_u1(t)
    u2 = get_u2_torch(t)
    f1 = u1*torch.cos(x3)
    f2 = u1*torch.sin(x3)
    f3 = u2

    # f1_x1 = torch.reshape(torch.autograd.grad(f1, x1, grad_outputs=torch.ones_like(f1), create_graph=True)[0], (-1,1))
    f1_x1 = (0.0*x1).view(-1,1)
    f2_x2 = (0.0*x2).view(-1,1)
    f3_x3 = (0.0*x3).view(-1,1)

    Lp = p_x1*f1 + p*f1_x1 + p_x2*f2 + p*f2_x2 + p_x3*f3 + p*f3_x3 - 0.5*(B_torch[0,0]*B_torch[0,0]*p_x1x1 + B_torch[1,1]*B_torch[1,1]*p_x2x2 + B_torch[2,2]*B_torch[2,2]*p_x3x3)
    residual = p_t + Lp
    if(verbose):
      print(p_x1)
      print(p_x2)
      print(p_x3)
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
                

# Custom L-infinity loss function
def linf_loss(output, target):
    return torch.max(torch.abs(output - target))


def get_space_samples(batch_size=1000, requires_grad=False):
    _x1 = (torch.rand(batch_size, 1, requires_grad=requires_grad) * (x_hig[0] - x_low[0]) + x_low[0]).to(device)
    _x2 = (torch.rand(batch_size, 1, requires_grad=requires_grad) * (x_hig[1] - x_low[1]) + x_low[1]).to(device)
    _x3 = (torch.rand(batch_size, 1, requires_grad=requires_grad) * (x_hig[2] - x_low[2]) + x_low[2]).to(device)
    # _x1 = (torch.rand(batch_size, 1, requires_grad=requires_grad) ).to(device)
    # _x2 = (torch.rand(batch_size, 1, requires_grad=requires_grad) ).to(device)
    # _x3 = (torch.rand(batch_size, 1, requires_grad=requires_grad) ).to(device)
    _x = torch.cat([_x1, _x2, _x3], axis=1)
    return _x


def train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_abs_p_ti, iterations=40000):
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    mar = 0.5

    # Define the mean and covariance matrix
    mu_0_tensor = torch.tensor(mu_0, dtype=torch.float32, requires_grad=True)
    cov_0_tensor = torch.tensor(cov_0, dtype=torch.float32, requires_grad=True)
    mvn = torch.distributions.MultivariateNormal(mu_0_tensor, cov_0_tensor)
    
    # space-time points for BC
    x_bc = get_space_samples(batch_size=500)
    x_bc_normal = mvn.sample((500,))
    x_bc = torch.cat((x_bc, x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)
    
    # space-time points for RES
    t = (torch.rand(5000, 1, requires_grad=True)*(tf-ti)+ti).to(device)
    t_inits = (torch.ones(100, 1, requires_grad=True) * ti).to(device)
    t = torch.cat((t, t_inits), dim=0)
    x = get_space_samples(batch_size=len(t), requires_grad=True)

    # RAR
    S = 500000
    FLAG = False

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
        norm_res_input = torch.norm(res_input, dim=1).view(-1,1)
        mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)
        linf_norm_res_input = linf_loss(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + mse_res #+ mse_norm_res_input
        # loss = linf_u #+ linf_res #+ linf_norm_res_input

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
                    }, FOLDER+"output/p_net.pth")
            min_loss = loss.data
            FLAG = True

        if (epoch%1000 == 0):
            print(epoch,"Traning Loss:",loss.data)
            np.save(FOLDER+"output/p_net_train_loss.npy", np.array(loss_history))

        loss.backward(retain_graph=True) # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

        # RAR
        if (epoch%500 == 0 and FLAG):
            t_RAR = (torch.rand(S, 1, requires_grad=True)*(tf-ti)+ti).to(device)
            _t_inits = (torch.ones(100, 1, requires_grad=True) * ti).to(device)
            t_RAR = torch.cat((t_RAR, _t_inits), dim=0)
            x_RAR = get_space_samples(batch_size=len(t_RAR), requires_grad=True)
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
                FLAG = False
            # res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
            # res_input_RAR = torch.cat([res_x_RAR, res_t_RAR], axis=1)
            # norm_res_input_RAR = torch.norm(res_input_RAR/max_abs_p_ti, dim=1).view(-1,1)
            # if(torch.mean(norm_res_input) > 0.0):
            #     max_abs_res_input, max_index = torch.max(norm_res_input_RAR, dim=0)
            #     # Get the corresponding x_RAR and t_RAR vectors
            #     x_max = x_RAR[max_index]
            #     t_max = t_RAR[max_index]
            #     # Append x_max and t_max to x and t
            #     x = torch.cat((x, x_max), dim=0)
            #     t = torch.cat((t, t_max), dim=0)
            #     print("... add [x,t]:", x_max.data, t_max.data, max_abs_res_input.data)

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
    max_abe_e1_ti = 1.0

    x1_points = np.load(FOLDER_DATA+"x1_points.npy")
    x2_points = np.load(FOLDER_DATA+"x2_points.npy")
    x3_points = np.load(FOLDER_DATA+"x3_points.npy")
    x1, x2, x3 = np.meshgrid(x1_points, x2_points, x3_points)
    x = np.column_stack([x1.ravel(), x2.ravel(), x3.ravel()])
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    # p_max_global = 0.0
    # for t1 in t1s:
    #     if(t1 <= 2.0):
    #         p = np.load(FOLDER_DATA+"p_sim_grid"+str(t1)+".npy").flatten()
    #         pt_t1 = (pt_x[:,0]*0 + t1).view(-1,1)
    #         print(pt_x.shape, pt_t1.shape)
    #         p_hat = p_net(pt_x, pt_t1).data.cpu().numpy()
    #         p_max = np.max(p_hat)
    #         p_max_global = max(p_max_global, p_max)
    # print(p_max_global)

    # fig = plt.figure(figsize=(20, 6))
    # axs = [fig.add_subplot(1, len(t1s), i+1, projection='3d') for i in range(len(t1s))]
    # j = 0
    for t1 in t1s:
        if(t1 <= 2.0):
            pt_t1 = (pt_x[:,0]*0 + t1).view(-1,1)
            p_hat = p_net(pt_x, pt_t1).data.cpu().numpy()
            # p_hat = res_func(pt_x, pt_t1, p_net).data.cpu().numpy()
            fig = go.Figure(data=go.Volume(
            x=x1.flatten(), y=x2.flatten(), z=x3.flatten(),
            value=p_hat.flatten(),
            # isomin=0.0,
            # isomax=p_max_global,
            opacity=0.1,
            surface_count=11,
            caps= dict(x_show=True, y_show=True, z_show=True, x_fill=1), # with caps (default mode)
            ))
            # Change camera view for a better view of the sides, XZ plane
            # (see https://plotly.com/python/v3/3d-camera-controls/)
            name = 'eye = (x:0., y:0., z:2.5)'
            camera = dict(eye=dict(x=0., y=0., z=2.5))
            fig.update_layout(scene_camera=camera, title="p NN")
            # fig.update_layout(
            #    scene_camera = dict(
            #    up=dict(x=0, y=0, z=1),
            #    center=dict(x=0, y=0, z=0),
            #    eye=dict(x=0.1, y=2.5, z=0.1)
            #    ),
            #    title="p NN"
            # )
            # fig.show()
            fig.write_image(FOLDER+"figs/p_hat_t"+str(t1)+".png")

            # p0 = p_init(x)
            # fig = go.Figure(data=go.Volume(
            # x=x1.flatten(), y=x2.flatten(), z=x3.flatten(),
            # value=p0.flatten(),
            # opacity=0.1,
            # surface_count=11,
            # caps= dict(x_show=True, y_show=True, z_show=True, x_fill=1), # with caps (default mode)
            # ))
            # # Change camera view for a better view of the sides, XZ plane
            # # (see https://plotly.com/python/v3/3d-camera-controls/)
            # fig.update_layout(
            #     scene_camera = dict(
            #     up=dict(x=0, y=0, z=1),
            #     center=dict(x=0, y=0, z=0),
            #     eye=dict(x=0.1, y=2.5, z=0.1)
            #     ),
            #     title="p True"
            # )
            # fig.show()


    # max_abe_e1_ti = np.inf
    # x_points = np.load(FOLDER_DATA+"x_points.npy")
    # sample_size = len(x_points)
    # x1s = x_points
    # x2s = x_points
    # x1, x2 = np.meshgrid(x1s, x2s)
    # x = np.column_stack([x1.ravel(), x2.ravel()])#; print(x)
    # pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    # pt_ti = Variable(torch.from_numpy(x[:,0]*0+ti).float(), requires_grad=True).view(-1,1).to(device)
    # p0 = p_init(x)
    # p_hat = p_net(pt_x, pt_ti).data.cpu().numpy()
    # e1 = p0 - p_hat
    # max_abe_e1_ti = max(abs(e1))[0]

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
    #     fig.colorbar(cp, ax=axp, orientation='horizontal')
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
    #     if(j == 0):
    #         p = p0.reshape(sample_size, sample_size)
    #     pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t1).float(), requires_grad=True).view(-1,1).to(device)
    #     p_hat = p_net(pt_x, pt_t1)
    #     p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
    #     e1 = p - p_hat_numpy
    #     axp = axs[j]
    #     cp = axp.imshow(e1, extent=[x_low, x_hig, x_low, x_hig], cmap='viridis', aspect='equal', origin='lower')
    #     fig.colorbar(cp, ax=axp, orientation='horizontal')
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
        fig.colorbar(cp, ax=ax1, orientation='horizontal')
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

    fig, axs = plt.subplots(1, 6, figsize=(18, 6), subplot_kw={'projection': '3d'})
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
        error_bound = max(abs(e1_hat.reshape(-1,1)))*2
        ax1 = axs[j]
        ax1.plot_surface(x1, x2, abs(e1_hat), cmap='viridis')
        ax1.plot_surface(x1, x2, e1_hat*0+error_bound, color="green", alpha=0.5)
        ax1.set_xlabel(r"$\theta$")
        ax1.set_ylabel(r"$\omega$")
        ax1.set_title("t="+str(t1)+"\n"+r"$B_e=$"+str(np.round(error_bound,2)))
        j = j + 1
    plt.savefig(FOLDER+"figs/uni_error_bound.png")
    plt.close()


def main():
    # test_dynamics()
    # test_p_sol_monte(linspace_num=100, stat_sample=10000000)
    plot_p_sol_Monte()

    max_pi = test_p_init()
    print("max abs p(x,ti):", max_pi)

    p_net = Net(scale=max_pi).to(device)
    p_net.apply(init_weights)
    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adamax(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # train_p_net(p_net, optimizer, scheduler, mse_cost_function, max_pi, iterations=100000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
    max_abe_e1_ti = show_p_net_results(p_net)
    print("max abs e1(x,ti):", max_abe_e1_ti)

    # e1_net = E1Net(scale=max_abe_e1_ti).to(device)
    # e1_net.apply(init_weights)
    # optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    # train_e1_net(e1_net, optimizer, scheduler, mse_cost_function, p_net, max_abe_e1_ti, iterations=2000); print("e1_net train complete")
    # e1_net = pos_e1_net_train(e1_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e1_net.eval()
    # show_e1_net_results(p_net ,e1_net)

    # print("[complete 2d linear]")


if __name__ == "__main__":
    main()
