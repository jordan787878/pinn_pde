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
from monte import pi, n_d, a_0, lam_0, mu_0, cov_0, x1_low, x1_hig, x2_low, x2_hig, ti, tf, t1s, mu_gravity, t_orbit
from test import propagate_guassian


# DATA_FOLDER = "data_combine/"
DATA_FOLDER = "meta_data/exp1/data_combine/"
device = "cpu"
TRAIN_FLAG = False

# Set default float type for PyTorch
torch.set_default_dtype(torch.float32)

# Constants
# pi = np.float32(np.pi)
# n_d = 2
# a_0 = np.float32(8000.0)
# lam_0 = np.float32(0.0)
# mu_0 = np.float32(np.array([a_0, lam_0]).reshape(2,))
# cov_0 = np.float32(np.array([[100.0**2, 0.0], [0.0, (pi/10)**2]]))
# x1_low = np.float32(7000.0)
# x1_hig = np.float32(9000.0)
# x2_low = np.float32(-2.0 * pi)
# x2_hig = np.float32(82) #4 pi
# ti = np.float32(0.0)
# tf = np.float32(24.0 * 3600)  # 3 hr
# t1s = np.float32([0.0, 3600.0, 7200.0, 10800.0, 6.0*3600.0, 9.0*3600.0, 12.0*3600.0, 18.0*3600.0, 24.0*3600.0])
# mu_gravity = np.float32(398600.4418)
# t_orbit = np.float32(2 * pi * np.sqrt(a_0**3 / mu_gravity))
# print("Nominal orbit period: ", t_orbit)


def p_init(x):
    pdf_func = multivariate_normal(mean=mu_0, cov=cov_0)
    pdf_eval = np.float32(pdf_func.pdf(x)).reshape(-1,1)
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


def p_multi_normal(x, mean, cov):
    pdf_func = multivariate_normal(mean=mean, cov=cov)
    pdf_value = np.float32(pdf_func.pdf(x)).reshape(-1,1)
    return pdf_value


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
        neurons = 50
        self.scale = scale
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(n_d+1,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        _x1 = (x[:,0].view(-1, 1)-0.5*(x1_low+x1_hig))/(0.5*(x1_hig-x1_low))
        _x2 = (x[:,1].view(-1, 1)-0.5*(x2_low+x2_hig))/(0.5*(x2_hig-x2_low))
        _t  = t/t_orbit
        inputs = torch.cat([_x1, _x2, _t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        layer4_out = F.softplus((self.hidden_layer4(layer3_out)))
        layer5_out = F.softplus((self.hidden_layer5(layer4_out)))
        output = F.softplus( self.output_layer(layer5_out) )
        output = output*self.scale
        return output
                

def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global ti, tf
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    # Define the mean and covariance matrix
    _mean = torch.tensor([a_0, lam_0], dtype=torch.float32)
    _covariance_matrix = torch.tensor([[100.0**2, 0.0], [0.0, (pi/10)**2]], dtype=torch.float32)
    _mvn = torch.distributions.MultivariateNormal(_mean, _covariance_matrix)

    _x1_bc = (torch.rand(1000, 1) * (x1_hig - x1_low) + x1_low).to(device)
    _x2_bc = (torch.rand(1000, 1) * (x2_hig - x2_low) + x2_low).to(device)
    _x_bc = torch.cat((_x1_bc, _x2_bc), dim=1)
    _x_bc_normal = _mvn.sample((3000,))
    x_bc = torch.cat((_x_bc, _x_bc_normal), dim=0)
    t_bc = (torch.ones(len(x_bc), 1) * ti).to(device)

    _x1 = (torch.rand(1000, 1, requires_grad=True) * (x1_hig - x1_low) + x1_low).to(device)
    _x2 = (torch.rand(1000, 1, requires_grad=True) * (x2_hig - x2_low) + x2_low).to(device)
    _x = torch.cat((_x1, _x2), dim=1)
    _x_normal = _mvn.sample((3000,))
    x = torch.cat((_x, _x_normal), dim=0)
    t = (torch.rand(len(x), 1, requires_grad=True)  * (tf - ti) + ti).to(device)

    # RAR
    S = 30000
    FLAG = False
    
    PATH = "output/p_net.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        u_bc = p_init_torch(x_bc).detach()
        net_bc_out = p_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # #[debug] visualizing the initial distribution samples
        # if(epoch%2000 == 0):
        #     fig = plt.figure(figsize=(10, 7))
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(x_bc[:,0].view(-1,1).numpy(), x_bc[:,1].view(-1,1).numpy(), u_bc.numpy(), c='b', marker='o', s=2)
        #     ax.set_xlabel('X Axis Label')
        #     ax.set_ylabel('Y Axis Label')
        #     ax.set_zlabel('Z Axis Label')
        #     ax.set_title('3D Scatter Plot')
        #     plt.show()

        # Loss based on PDE
        res_out = res_func(x, t, p_net, verbose=False)/normalize
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_out, all_zeros)

        # Frequnecy Loss
        # res_x = torch.autograd.grad(res_out, x, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_t = torch.autograd.grad(res_out, t, grad_outputs=torch.ones_like(res_out), create_graph=True)[0]
        # res_input = torch.cat([res_x, res_t], axis=1)
        # norm_res_input = torch.norm(res_input, dim=1).view(-1,1) ###
        # mse_norm_res_input = mse_cost_function(norm_res_input, all_zeros)

        # Loss Function
        loss = mse_u + t_orbit*(mse_res)
        loss_history.append(loss.data)

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                #   ",res_freq:", mse_norm_res_input.data
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
    # [debug] x1_grid is float64...
    # print(x1_grid.dtype, x1s.dtype)
    # diff_x1_points = x1_grid.reshape(-1,1) - x1s.detach().numpy()
    # plt.figure()
    # plt.plot(np.arange(0,len(diff_x1_points)), diff_x1_points)
    # plt.show()
    return phat


def get_p_linear_guassian(x1_grid, x2_grid, t1):
    _mu_G, _cov_G = propagate_guassian(t1, mu_0, cov_0)
    x1s = x1_grid.reshape(-1,1)
    Nx = len(x1_grid)
    x2s = x2_grid.reshape(-1,1)
    xs = np.hstack((x1s, x2s))
    pdf_xs = p_multi_normal(xs, _mu_G, _cov_G).reshape(Nx, Nx)
    return pdf_xs


def compare_pdf(x1_grid, x2_grid, t1, pdf, pdf_nn, pdf_linear_guassian):
    # Determine the color limits
    all_data = np.concatenate((pdf.ravel(), pdf_nn.ravel(), pdf_linear_guassian.ravel()))
    vmin = all_data.min()
    vmax = all_data.max()
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), subplot_kw={'projection': '3d'})
    # First subplot for PDF
    ax1 = axs[0]
    ax1.plot_surface(x1_grid, x2_grid, pdf, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('PDF')
    ax1.set_title('Monte Carlo, t1='+str(t1))
    ax1.set_xlim([x1_low, x1_hig])
    ax1.set_ylim([x2_low, x2_hig])
    ax1.set_zlim([vmin, vmax])
    # ax1.view_init(elev=0, azim=0)
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
    # ax2.view_init(elev=0, azim=0)
    # Third subplot for pdf_linear_guassian
    ax3 = axs[2]
    ax3.plot_surface(x1_grid, x2_grid, pdf_linear_guassian, cmap='viridis', edgecolor='none', vmin=vmin, vmax=vmax)  # Change cmap as needed
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.set_zlabel('PDF')
    ax3.set_title('Linear Guassian')
    ax3.set_xlim([x1_low, x1_hig])
    ax3.set_ylim([x2_low, x2_hig])
    ax3.set_zlim([vmin, vmax])
    plt.show()

    # Error plot NN vs Linear Guassian
    e_nn = pdf - pdf_nn
    e_linear_guassian = pdf - pdf_linear_guassian
    # Create a single 3D subplot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the neural network error wireframe
    ax.plot_wireframe(x1_grid, x2_grid, e_nn, color='red', 
                      linestyle="--", linewidth=1.0, label='NN Error')
    # Plot the linear Gaussian error wireframe
    ax.plot_wireframe(x1_grid, x2_grid, e_linear_guassian, color='blue', 
                      linestyle="-", linewidth=0.5, label='Linear Gaussian Error')
    # Set labels and title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Error')
    ax.set_title('Error Comparison: Neural Network vs Linear Gaussian, t='+str(t1))
    # Optionally, add a legend
    ax.legend()
    plt.show()

    # Error of Monte vs Analytical
    # [This shows that the NN and Linear Guassian are more accurate then Monte Carlo...]
    if(t1 == 0.0):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        xs = np.hstack((x1_grid.reshape(-1,1), x2_grid.reshape(-1,1)))
        print("[debug] monte vs analytical at t=0:", xs.shape)
        pdf_true = p_init(xs).reshape(len(x1_grid), len(x1_grid))
        print(pdf_true.shape)
        ax.plot_wireframe(x1_grid, x2_grid, pdf_true - pdf, color='black', 
                          linestyle="-", linewidth=0.5, label='Monte Error')
        ax.plot_wireframe(x1_grid, x2_grid, pdf_true - pdf_nn, color='r', 
                          linestyle="--", linewidth=0.5, label='NN Error')
        ax.plot_wireframe(x1_grid, x2_grid, pdf_true - pdf_linear_guassian, color='b', 
                          linestyle=":", linewidth=0.5, label='Linear Gaussian Error')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Error')
        ax.set_title("Initial Error of Monte Carlo ... ")
        ax.legend()
        plt.show()


def compute_mean_cov(x1_grid, x2_grid, pdf, debug_text):
    print("### "+debug_text+" ###")
    # Numerical integration using the trapezoidal rule
    x1_range = x1_grid[0,:]
    x2_range = x2_grid[:,0]
    dx1 = x1_range[1] - x1_range[0]
    dx2 = x2_range[1] - x2_range[0]
    x1_grid = x1_grid.reshape(-1,)
    x2_grid = x2_grid.reshape(-1,)
    pdf = pdf.reshape(-1,)

    # [testing]
    # x_test = np.hstack((x1_grid.reshape(-1,1), x2_grid.reshape(-1,1)))
    # print(x_test.shape)
    # pdf = p_init(x_test).reshape(-1,)
    
    # [testing] renormalization of pdf
    # Essential part to ensure that the cov(x) computation is correct
    if(debug_text == "pdf_nn"):
        p_sum = np.sum(pdf) * dx1 * dx2
        if(p_sum > 1.0):
            print("[debug] pdf_nn has p_sum="+str(p_sum)+", do renomralization.")
            pdf = pdf/p_sum

    output_data = np.stack((x1_grid, x2_grid, pdf))
    csv_filename = "output/output"+debug_text+".csv"
    np.savetxt(csv_filename, output_data.T, delimiter=",", header="x1,x2,pdf", comments="", fmt="%f")
    print("[debug] dtype of xgrid and pdf: ", x1_grid.dtype, pdf.dtype)

    mean_x1 = np.sum(x1_grid * pdf) * dx1 * dx2
    mean_x2 = np.sum(x2_grid * pdf) * dx1 * dx2
    print("mean: {:.3f}, {:.3f}".format(mean_x1, mean_x2))

    # E_XX = np.sum(x1_grid * x1_grid * pdf) * dx1 * dx2
    # cov_x1_x1 = E_XX - mean_x1 * mean_x1
    # print("[debug] cov(x1x1) using E[(x)^2] - mux^2 ", cov_x1_x1)
    cov_x1_x1 = np.sum((x1_grid - mean_x1)**2 * pdf) * dx1 * dx2
    # print("[debug] cov(x1x1) using E[(x-mux)^2] ", cov_x1_x1)

    # E_YY = np.sum(x2_grid * x2_grid * pdf) * dx1 * dx2
    # cov_x2_x2 = E_YY - mean_x2 * mean_x2
    # print("[debug] cov(x2x2) using E[(x)^2] - mux^2 ", cov_x2_x2)
    cov_x2_x2 = np.sum((x2_grid - mean_x2)**2 * pdf) * dx1 * dx2
    # print("[debug] cov(x2x2) using E[(x-mux)^2] ", cov_x2_x2)

    # E_XY = np.sum(x1_grid * x2_grid * pdf) * dx1 * dx2
    # cov_x1_x2 = E_XY - mean_x1 * mean_x2
    # print("[debug] cov(x1x2) using E[(x1x2)^2] - mux1.mux2^2 ", cov_x1_x2)
    cov_x1_x2 = np.sum((x1_grid - mean_x1)*(x2_grid - mean_x2) * pdf) * dx1 * dx2
    # print("[debug] cov(x1x2) using E[(x1-mux1)(x2-mux2)] ", cov_x1_x2)
    cov_matrix = np.zeros((2,2))
    cov_matrix[0,0] = cov_x1_x1
    cov_matrix[0,1] = cov_x1_x2
    cov_matrix[1,0] = cov_x1_x2
    cov_matrix[1,1] = cov_x2_x2
    print("cov: ")
    for row in cov_matrix:
        print("  ".join("{:.3f}".format(x) for x in row))

    # [testing] skewness (normal distribution has skewness 0)
    skew_x1 = np.sum( ((x1_grid-mean_x1)/(cov_x1_x1**0.5))**3 * pdf) * dx1 * dx2
    skew_x2 = np.sum( ((x2_grid-mean_x2)/(cov_x2_x2**0.5))**3 * pdf) * dx1 * dx2
    print("skew     X1={:.3f}, X2={:.3f}".format(skew_x1, skew_x2))

    # [testing] kurtosis (normal distribution has kurtoisis 3)
    kurt_x1 = np.sum( ((x1_grid-mean_x1)/(cov_x1_x1**0.5))**4 * pdf) * dx1 * dx2
    kurt_x2 = np.sum( ((x2_grid-mean_x2)/(cov_x2_x2**0.5))**4 * pdf) * dx1 * dx2
    print("kurtosis X1={:.3f}, X2={:.3f}".format(kurt_x1, kurt_x2))


def main():
    mse_cost_function = torch.nn.MSELoss()

    p_net = Net().to(device)
    p_net.apply(init_weights_He)

    max_pi = 0.0016 # set this before training
    p_net.scale = max_pi
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=100000); print("p_net train complete")
    p_net = pos_p_net_train(p_net, PATH="output/p_net.pth", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()
    print("[load pnet model from: "+ "output/p_net.pth]")

    t1s = [0.0]
    for t1 in t1s:
        print("t1: ", t1)
        # load data
        x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
        x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
        print("[debug] x1_grid shape: ", x1_grid.shape)
        pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
        pdf_nn = get_pnet_output(p_net, x1_grid, x2_grid, t1)
        pdf_linear_guassian = get_p_linear_guassian(x1_grid, x2_grid, t1)
        compute_mean_cov(x1_grid, x2_grid, pdf, "pdf")
        compute_mean_cov(x1_grid, x2_grid, pdf_nn, "pdf_nn")
        compute_mean_cov(x1_grid, x2_grid, pdf_linear_guassian, "pdf_linear_guassian")
        compare_pdf(x1_grid, x2_grid, t1, pdf, pdf_nn, pdf_linear_guassian)
    
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