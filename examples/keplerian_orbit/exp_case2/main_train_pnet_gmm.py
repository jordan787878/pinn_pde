"""
train a p_nn that is parameterized by GMM

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata
import argparse
from constants import Case2_4D_Constants
from monte import p_init, get_p_init_max
import torch.distributions as D
from main_train_pnet import PNet



DATA_FOLDER = "data/"
device = "cpu"
TRAIN_FLAG = False
constants = Case2_4D_Constants()

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class PNet_GMM(nn.Module):
    global constants
    def __init__(self, input_dim_t=1, output_dim_x=4, hidden_dim=64, verbose=False):
        """
        Constructs a neural network p that for each t outputs a normalized pdf p(x,t)
        over x in R^4 by outputting the parameters of a multivariate Gaussian.
        
        Args:
            input_dim_t (int): Dimension of time input t.
            output_dim_x (int): Dimension of x (here 4).
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(PNet_GMM, self).__init__()
        self.input_dim_t = input_dim_t
        self.output_dim_x = output_dim_x  # here x ∈ ℝ⁴
        self.verbose = verbose

        # Two hidden layers
        self.fc1 = nn.Linear(input_dim_t, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # The final output layer produces:
        #   - output_dim_x numbers for the mean vector, and
        #   - output_dim_x*(output_dim_x+1)//2 numbers for the lower-triangular part of L.
        final_out_dim = output_dim_x + (output_dim_x * (output_dim_x + 1)) // 2
        self.fc_out = nn.Linear(hidden_dim, final_out_dim)
    
    def forward(self, x_input, t):
        """
        Args:
            t (Tensor): Time input of shape [batch, input_dim_t].
        
        Returns:
            dist (torch.distributions.MultivariateNormal): A Gaussian distribution 
                with parameters depending on t.
        """
        # Pass through the network
        h = F.gelu(self.fc1(t))
        h = F.gelu(self.fc2(h))
        out = self.fc_out(h)  # shape: [batch, final_out_dim]
        
        batch_size = out.shape[0]
        n = self.output_dim_x
        
        # Extract the mean vector (first n outputs)
        mean = out[:, :n]  # shape: [batch, 4]
        for i in range(4):
            mean[:,i] = mean[:,i] + constants.N_MEAN_I[i]
        
        # The remaining outputs parameterize the lower-triangular matrix L
        cov_params = out[:, n:]  # shape: [batch, n(n+1)/2]
        L = torch.zeros(batch_size, n, n, device=out.device)
        
        idx = 0
        for i in range(n):
            for j in range(i+1):
                if i == j:
                    # Diagonal entries: exponentiate to ensure they are positive.
                    L[:, i, j] = torch.exp(cov_params[:, idx])
                else:
                    L[:, i, j] = cov_params[:, idx]
                idx += 1
        
        # Construct covariance matrix: Sigma = L Lᵀ (always positive definite)
        cov = torch.bmm(L, L.transpose(1, 2))

        if(self.verbose):
            print("[check] pnn_gmm mean: ", mean[0,:])
            print("[check] pnn_gmm cov:  ", cov[0,:,:])

        # Create a multivariate normal distribution with these parameters.
        # By construction, this density is normalized over R^4.
        dist = D.MultivariateNormal(mean, covariance_matrix=cov)

        # Evaluate the density at x_input.
        log_prob = dist.log_prob(x_input)  # shape: [batch]
        pdf = torch.exp(log_prob).unsqueeze(1)  # shape: [batch, 1]
        return pdf
                

def train_p_net_gmm(p_net_gmm, p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constants
    loss_history = []
    min_loss = np.inf
    iterations_per_decay = 1000
    normalize = p_net.scale

    N0_samples = 200; Nr_samples = 200
    # samples of initial condition
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    # samples of residual
    x, t = constants.sample_res_points(Nr_samples)

    # RAR
    S = 30000; RAR_eps = 5e-3; FLAG = False
    
    path_model = "output/p_net_gmm.pth"
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss 
        target_i = p_net(x_bc, t_bc)
        output_i = p_net_gmm(x_bc, t_bc)
        mse_u = mse_cost_function(output_i/normalize, target_i/normalize)

        target_r = p_net(x, t)
        output_r = p_net_gmm(x, t)
        mse_r = mse_cost_function(output_r/normalize, target_r/normalize)

        loss = mse_u + mse_r
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",l_i:", mse_u.data, 
                  ",l_r:", mse_r.data,
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net_gmm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'train_time': train_time,
                    }, path_model)
            min_loss = loss.data
            FLAG = True

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def load_trained_model(net, path, method="old"):
    print("[load pnet model from: "+ path)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if(method == "new"):
        loss_history = np.array(checkpoint['loss_history'])
        print("pnet best epoch: ", epoch, ", min loss:", np.min(loss_history), ", train time:", checkpoint['train_time'])
    else:
        print("pnet best epoch: ", epoch, ", min loss:", checkpoint['loss'], ", train time:", checkpoint['train_time'])
    
    # keys = p_net.state_dict().keys()
    # for k in keys:
    #     l2_norm = torch.norm(p_net.state_dict()[k], p=2)
    #     print(f"L2 norm of {k} : {l2_norm.item()}")
    # plot loss history
    # plt.figure()
    # plt.plot(np.arange(len(loss_history)), loss_history, "black", linewidth=1)
    # plt.ylim([min_loss, 10*min_loss])
    # plt.xlabel("epoch")
    # plt.ylabel("pnet loss")
    # plt.tight_layout()
    # plt.savefig("figs/pnet_loss_history.pdf", format='pdf', dpi=300)
    # plt.close()
    return net


def check_pdfnn_marginalize(p_net, t=0.0):
    """
    marginalize the joint pdf to a single coordinate, and compare it to the true analytical pdf at init time
    the true analytical pdf is obtained by 2 ways (but equivalent)
    1. using the univariate normal
    2. first compute the joint pdf (on the meshgrid) using multivariate normal, then marginalize
    """
    global constants
    print("[result] pdf_nn vs monte (or analytical at t=TI)")

    # load p(monte)
    x1s = np.load(DATA_FOLDER+"x1s.npy")
    x2s = np.load(DATA_FOLDER+"x2s.npy")
    x3s = np.load(DATA_FOLDER+"x3s.npy")
    x4s = np.load(DATA_FOLDER+"x4s.npy")
    pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t))
    print("[check] monte joint pdf shape, type: ", pdf_monte.shape, pdf_monte.dtype)

    # prepare grid points 
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    print("[check] grid points shape type: ", grid_points.shape, grid_points.dtype)
    
    # if t == 0.0, obtain analytical p(true)
    if(t == 0.0):
        pdf_true = p_init(grid_points).reshape(x1_grid.shape)
        print("[check] x1 ranges, true joint pdf shape, type: ", x1s.dtype, pdf_true.shape, pdf_true.dtype)

    # obtain pdf(nn)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
    t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t).to(device)
    print("[check] grid points tensor shape type: ", grid_points_tensor.shape, grid_points_tensor.dtype)
    pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)
    print("[check] nn joint pdf shape, type: ", pdf_nn.shape, pdf_nn.dtype)
    
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]

    # create figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for j in range(4):
        if(j == 0):
            ax = axs[0,0]
            x_axis = x1s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(1,2,3)) * dx2 * dx3 * dx4
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(1,2,3)) * dx2 * dx3 * dx4
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX1)
            ax.set_ylabel(r"$r$")
            if(t == 0.0):
                # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                marginalize_pdf_true = np.sum(pdf_true, axis=(1,2,3)) * dx2 * dx3 * dx4
                ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        elif(j == 1):
            ax = axs[0,1]
            x_axis = x2s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,2,3)) * dx1 * dx3 * dx4
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,2,3)) * dx1 * dx3 * dx4
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX2)
            ax.set_ylabel(r"$\phi$")
            if(t == 0.0):
                    # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                    marginalize_pdf_true = np.sum(pdf_true, axis=(0,2,3)) * dx1 * dx3 * dx4
                    ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        elif(j == 2):
            ax = axs[1,0]
            x_axis = x3s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,1,3)) * dx1 * dx2 * dx4
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,1,3)) * dx1 * dx2 * dx4
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX3)
            ax.set_ylabel(r"$v_r$")
            if(t == 0.0):
                    # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                    marginalize_pdf_true = np.sum(pdf_true, axis=(0,1,3)) * dx1 * dx2 * dx4
                    ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        else:
            ax = axs[1,1]
            x_axis = x4s
            marginalize_pdf_monte = np.sum(pdf_monte, axis=(0,1,2)) * dx1 * dx2 * dx3
            marginalize_pdf_nn = np.sum(pdf_nn, axis=(0,1,2)) * dx1 * dx2 * dx3
            ax.plot(x_axis, marginalize_pdf_monte, "blue", label="monte")
            ax.plot(x_axis, marginalize_pdf_nn, "r--", label="nn")
            ax.set_ylim(0.0, constants.MAX_PX4)
            ax.set_ylabel(r"$v_\phi$")
            if(t == 0.0):
                    # marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
                    marginalize_pdf_true = np.sum(pdf_true, axis=(0,1,2)) * dx1 * dx2 * dx3
                    ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical")
        ax.legend()
    plt.show()


def test_nn_cartesian_pdf_xy(p_net, model_name):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    the contour plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    global constants
    # Create the contour plot
    plt.figure(figsize=(8, 6))
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime).to(device)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4

        # convert pdf
        pdf_nn_rphi_data = np.empty((0,4))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1])  
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r**2)  
        # Define the grid where you want to plot the contours
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),  # Adjust 100 to get finer resolution
                                     np.linspace(y.min(), y.max(), 100))
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        cp = plt.contourf(grid_x, grid_y, grid_z_nn, levels=15, cmap="viridis", alpha=1.0)  # Filled contour plot
        # plt.colorbar(cp)  # Add a colorbar to indicate the values
    # Labels and title
    plt.axis('equal')
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.title(model_name+'(x,y) over T='+str(constants.TF)+' sec.')
    plt.show()


def check_pdfnn_cartesian_wrt_monte(p_net, model_name):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    global constants
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(nn) marginalized to xy at t'=", t_prime)
        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")
        # load pdf_monte on the domain
        pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))

        # obtain pdf_nn on the domain
        x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        t_tensor = (torch.ones(len(grid_points_tensor), 1, dtype=torch.float32) * t_prime).to(device)
        pdf_nn = p_net(grid_points_tensor, t_tensor).detach().numpy().reshape(x1_grid.shape)

        dx1 = x1s[1] - x1s[0] # dr'
        dx2 = x2s[1] - x2s[0] # dphi'
        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]
        dV = dx1*dx2*dx3*dx4
        # mean_x1 = np.sum(x1_grid * pdf_monte) * dV
        # mean_x2 = np.sum(x2_grid * pdf_monte) * dV
        # mean_x3 = np.sum(x3_grid * pdf_monte) * dV
        # mean_x4 = np.sum(x4_grid * pdf_monte) * dV
        # print("[check] pdf_monte mean, sum p: ", mean_x1, mean_x2, mean_x3, mean_x4, np.sum(pdf_monte)*dV)

        # marginalize to spherical position (r, phi)
        pdf_nn_Nrphi =  np.sum(pdf_nn, axis=(2,3)) * dx3 * dx4
        pdf_mo_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4

        sum_p_nn = np.sum(pdf_nn_Nrphi) * dx1 * dx2
        print("[check] sum p_nn (N-sphere): ", sum_p_nn)

        # convert pdf(r, phi)
        # NOTE: I store the area = "dr*(r*dphi)" associated to each pdf
        pdf_nn_rphi_data = np.empty((0,5))
        pdf_mo_rphi_data = np.empty((0,4))
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                # extract [r', phi', p(r',phi')]
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                t = constants.T*t_prime
                pdf_nn_rphi_data = np.vstack((pdf_nn_rphi_data, np.array([r, phi, t, pdf_nn_Nrphi[i,j]/(constants.R*constants.PHI), (dx1*constants.R)*(r*dx2*constants.PHI)])))
                pdf_mo_rphi_data = np.vstack((pdf_mo_rphi_data, np.array([r, phi, t, pdf_mo_Nrphi[i,j]/(constants.R*constants.PHI)])))

        # convert pdf(r,phi) to pdf(x,y)
        x = pdf_nn_rphi_data[:, 0] * np.cos(pdf_nn_rphi_data[:, 1]) 
        y = pdf_nn_rphi_data[:, 0] * np.sin(pdf_nn_rphi_data[:, 1])
        r = pdf_nn_rphi_data[:, 0]
        z_nn = pdf_nn_rphi_data[:, 3]/(r) # see derivation of the factor (1/r) in Nov 7 notes
        z_mo = pdf_mo_rphi_data[:, 3]/(r) 

        _p_test = (np.sum(pdf_nn_Nrphi)/(constants.R*constants.PHI)) * (dx1*constants.R * dx2*constants.PHI)
        __p_test = np.sum(z_nn * pdf_nn_rphi_data[:, 4])
        ___p_test = np.sum(z_mo * pdf_nn_rphi_data[:, 4])
        print("[check] p_monte in (x,y) {:.2f} & p_nn sum in (r,phi) {:.2f} and (x,y) {:.2f}".format(___p_test, _p_test, __p_test))

        # visualize p(x,y) using interpolation
        _grid_resolution = 70
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), _grid_resolution), np.linspace(y.min(), y.max(), _grid_resolution), indexing="ij")
        # Interpolate scattered data onto the grid
        grid_z_nn = griddata((x, y), z_nn, (grid_x, grid_y), method='cubic')
        grid_z_mo = griddata((x, y), z_mo, (grid_x, grid_y), method='cubic')

        if(t_prime == 0.0):
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--", label=model_name)
            surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, color="none", rstride=3, cstride=3, edgecolor='blue', linewidth=0.5, label="Monte")
        else:
            surf1 = ax.plot_surface(grid_x, grid_y, grid_z_nn, color="none", rstride=3, cstride=3, edgecolor='red',  linewidth=0.5, linestyle="--")
            surf2 = ax.plot_surface(grid_x, grid_y, grid_z_mo, color="none", rstride=3, cstride=3, edgecolor='blue', linewidth=0.5)
        # Optional: Add a color bar
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # Labels and title
    ax.legend()
    ax.set_xlabel('X, m')
    ax.set_ylabel('Y, m')
    ax.set_zlabel('PDF Value')
    ax.set_title(f'3D Surface Plot of PDF over t='+str(constants.TF))
    ax.view_init(elev=30, azim=-120)
    # Show the plot
    plt.show()


def main():
    global constants
    p_net = PNet().to(device)
    p_net_gmm = PNet_GMM().to(device)
    p_net.apply(init_weights_He)
    p_net_gmm.apply(init_weights_He)
    p_net.scale = get_p_init_max()
    p_net = load_trained_model(p_net, path="output/p_net.pth"); p_net.eval()

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net_gmm.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net_gmm(p_net_gmm, p_net, optimizer, scheduler, mse_cost_function, iterations=20000); print("p_net_gmm train complete")
    p_net_gmm = load_trained_model(p_net_gmm, path="output/p_net_gmm.pth", method="new"); p_net_gmm.eval(); p_net_gmm.verbose=False

    ### Post-process ###
    # for t_prime in constants.T_PRIME_SPAN:
    #    check_pdfnn_marginalize(p_net, t=t_prime)

    # test_nn_cartesian_pdf_xy(p_net, model_name="p_net")
    check_pdfnn_cartesian_wrt_monte(p_net, model_name="p_net")
    # test_nn_cartesian_pdf_xy(p_net_gmm, model_name="p_net_gmm")
    check_pdfnn_cartesian_wrt_monte(p_net_gmm, model_name="p_net_gmm")
    
    ### Finish printout ###
    if(TRAIN_FLAG == False):
        print("[complete] 4d perfect keplerian orbit Case2 (another pnn_gmm)")
    else:
        print("[complete] 4d perfect keplerian orbit Case2 with pre-trained models (another pnn_gmm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    # Set the TRAIN_FLAG
    TRAIN_FLAG = args.train
    main()
