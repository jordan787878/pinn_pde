import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal
import argparse


DATA_FOLDER = "data/"
device = "cpu"
TRAIN_FLAG = False


pi = np.pi
n_d = 2
a_0 = 8000
lam_0 = 0.0
mu_0 = np.array([a_0, lam_0]).reshape(2,)
cov_0 = np.array([[100.0**2, 0.0], [0.0, (pi/10)**2]])
x1_low = 7000
x1_hig = 9000
x2_low = -pi
x2_hig = pi
ti = 0.0
tf = 2.0*86400.0
mu_gravity = 398600.4418


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


def res_func(x, t, p_net, verbose=False):
    p = p_net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    # p_x1 = p_x[:,0].view(-1,1)
    p_x2 = p_x[:,1].view(-1,1)
    x1 = x[:,0].view(-1, 1)
    # x2 = x[:,1].view(-1, 1)

    mean_motion = torch.sqrt(mu_gravity/x1**3)
    Lp = p_x2*mean_motion
    residual = p_t + Lp
    if(verbose):
      print(x1[0,:].item(), t[0,:].item(), p[0,:].item(), p_x2[0,:].item(), mean_motion[0,:].item(), Lp[0,:].item(), p_t[0,:].item())
    #   print("residual: ", residual, residual.shape)
    return residual


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


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
        _x1 = (x[:,0].view(-1, 1)-a_0)/1000.0
        _x2 = (x[:,1].view(-1, 1))/(pi)
        _t  = t/tf
        inputs = torch.cat([_x1, _x2, _t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        return output
                

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global x_low, x_hig, ti, tf
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    # Define the mean and covariance matrix
    _mean = torch.tensor([a_0, lam_0])
    _covariance_matrix = torch.tensor([[100.0**2, 0.0], [0.0, (pi/10)**2]])
    _mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    _x1_bc = (torch.rand(2000, 1) * (x1_hig - x1_low) + x1_low).to(device)
    _x2_bc = (torch.rand(2000, 1) * (x2_hig - x2_low) + x2_low).to(device)
    _x_bc = torch.cat((_x1_bc, _x2_bc), dim=1)
    _x_bc_normal = _mvn.sample((2000,))
    x_bc = torch.cat((_x_bc, _x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)

    _x1 = (torch.rand(2000, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
    _x2 = (torch.rand(2000, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
    _x = torch.cat((_x1, _x2), dim=1)
    _x_normal = _mvn.sample((2000,))
    x = torch.cat((_x, _x_normal), dim=0)
    t = (torch.rand(len(x), 1, requires_grad=True)  * (tf - ti) + ti).to(device)

    # periodic boundary
    _x2_low_bc = (torch.ones(2000, 1) * x2_low).to(device)
    _x2_hig_bc = (torch.ones(2000, 1) * x2_hig).to(device)
    _x1_period = (torch.rand(2000, 1) * (x1_hig - x1_low) + x1_low).to(device)
    x_period_low = torch.cat((_x1_period, _x2_low_bc), dim=1)
    x_period_hig = torch.cat((_x1_period, _x2_hig_bc), dim=1)
    t_period = (torch.rand(len(x_period_low), 1)  * (tf - ti) + ti).to(device)

    # RAR
    S = 30000
    FLAG = False
    
    PATH = "output/p_net.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # space-time points for BC
        # x1_bc = (torch.rand(1500, 1) * (x1_hig - x1_low) + x1_low).to(device)
        # x2_bc = (torch.rand(1500, 1) * (x2_hig - x2_low) + x2_low).to(device)
        # x_bc = torch.cat((x1_bc, x2_bc), dim=1)
        
        # space-time points for RES
        # x1 = (torch.rand(1500, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
        # x2 = (torch.rand(1500, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
        # x = torch.cat((x1, x2), dim=1)
        # t = (torch.rand(1500, 1, requires_grad=True)  * (tf - ti) + ti).to(device)

        # Loss based on boundary conditions
        u_bc = p_init_torch(x_bc).detach()
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)
        # if(epoch%2000 == 0):
        #     print("[debug]")
        #     fig = plt.figure(figsize=(10, 7))
        #     ax = fig.add_subplot(111, projection='3d')
        #     scatter = ax.scatter(x_bc[:,0].view(-1,1).numpy(), x_bc[:,1].view(-1,1).numpy(), u_bc.numpy(), c='b', marker='o', s=5)
        #     # Add labels
        #     ax.set_xlabel('X Axis Label')
        #     ax.set_ylabel('Y Axis Label')
        #     ax.set_zlabel('Z Axis Label')
        #     ax.set_title('3D Scatter Plot')
        #     plt.show()

        # Loss based on PDE
        res_out = res_func(x, t, p_net, verbose=False)/normalize
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)

        u_period_low = p_net(x_period_low, t_period).to(device)
        u_period_hig = p_net(x_period_hig, t_period).to(device)
        u_period_diff = u_period_low - u_period_hig
        all_zeros = torch.zeros((len(u_period_diff),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_period = mse_cost_function(u_period_diff/normalize, all_zeros)

        # Frequnecy Loss
        # res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_input = torch.cat([res_x, res_t], axis=1)
        # norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
        # mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + tf*(2.0*mse_res) + tf*mse_period
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                  ",period: ", mse_period.data
                #    ",res_freq:", mse_norm_res_input.data
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
            _x1_RAR = (torch.rand(S, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
            _x2_RAR = (torch.rand(S, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
            x_RAR = torch.cat((_x1_RAR, _x2_RAR), dim=1)
            t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
            t0_RAR = 0.0*t_RAR + ti
            p_bc_RAR = p_init_torch(x_RAR)
            phat_bc_RAR = p_net(x_RAR, t0_RAR)
            max_ic_error = torch.max(torch.abs(phat_bc_RAR - p_bc_RAR))/normalize
            # print("RAR max IC: ", max_ic_error.data)
            if(max_ic_error > 5e-3):
                max_abs_ic, max_index = torch.topk(torch.abs(phat_bc_RAR.squeeze() - p_bc_RAR.squeeze()), 10)
                x_max = x_RAR[max_index,:].clone().detach()
                t_max = t0_RAR[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", x_max[0,:].data, t_max[0].item(), max_abs_ic[0].item())
            res_RAR = res_func(x_RAR, t_RAR, p_net)/normalize
            max_res_RAR = torch.max(torch.abs(res_RAR))
            if(max_res_RAR > 0.0):
                max_abs_res, max_index = torch.topk(torch.abs(res_RAR.squeeze()), 10)
                x_max = x_RAR[max_index,:].clone()
                t_max = t_RAR[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", x_max[0,:].data, t_max[0].item(), max_abs_res[0].item())
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()

        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

    np.save("output/p_net_train_loss.npy", np.array(loss_history))


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
    plt.savefig("figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    plt.close()
    return p_net


def get_pnet_output(p_net, x1_grid, x2_grid, t1):
    x1s = torch.from_numpy(x1_grid.reshape(-1,1)).float()
    x2s = torch.from_numpy(x2_grid.reshape(-1,1)).float()
    Nx = len(x1_grid)
    xs = torch.cat((x1s, x2s), dim=1)
    ts = x1s*0.0 + t1
    phat = p_net(xs, ts).detach().numpy().reshape(Nx, Nx)
    return phat


def compare_pdf(x1_grid, x2_grid, pdf, pdf_nn):
    # Determine the color limits
    all_data = np.concatenate((pdf.ravel(), pdf_nn.ravel()))
    vmin = all_data.min()
    vmax = all_data.max()
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
    # First subplot for PDF
    ax1 = axs[0]
    ax1.plot_surface(x1_grid, x2_grid, pdf, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('PDF')
    ax1.set_title('Monte Carlo')
    ax1.set_xlim([x1_low, x1_hig])
    ax1.set_ylim([x2_low, x2_hig])
    ax1.set_zlim([vmin, vmax])
    # ax1.view_init(elev=30, azim=-145)
    # Second subplot for pdf_nn
    ax2 = axs[1]
    ax2.plot_surface(x1_grid, x2_grid, pdf_nn, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)  # Change cmap as needed
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('PDF')
    ax2.set_title('Neural Network')
    ax2.set_xlim([x1_low, x1_hig])
    ax2.set_ylim([x2_low, x2_hig])
    ax2.set_zlim([vmin, vmax])
    # ax2.view_init(elev=30, azim=-145)
    plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x1_grid, x2_grid, pdf-pdf_nn, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('X1')
    # ax.set_ylabel('X2')
    # ax.set_zlabel('Error')
    # ax.set_title('Approximation Error')
    # # ax.view_init(elev=30, azim=-145)
    # plt.show()


def main():
    mse_cost_function = torch.nn.MSELoss()

    p_net = Net().to(device)
    p_net.apply(init_weights_He)

    max_pi = 0.004
    p_net.scale = max_pi
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pth", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()
    print("[load pnet model from: "+ "output/p_net.pth]")

    t1s = [0.0, 3600.0, 7200.0, 86400.0, 2.0*86400.0]
    for t1 in t1s:
        # load data
        x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
        x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
        pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
        pdf_nn = get_pnet_output(p_net, x1_grid, x2_grid, t1)
        compare_pdf(x1_grid, x2_grid, pdf, pdf_nn)
    
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