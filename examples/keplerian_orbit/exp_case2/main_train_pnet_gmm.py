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
