"""
The first trial of keplerian orbit using rotating spherical coordiante and dynamics.
It is similar to the Case 2 of the paper: Uncertainty propagation in orbital mechanics via tensor decompostion.
Difference: do not have process noise, and J2 perturbation.
This case is a circular orbit on plannar motion. Hence, we reduce the dynamics to 4D.
Compared to exp2_sphere_4d (baseline):
    1) the final time is increased to 0.2*T.
    2) the solution domain is increased to ensure sum(p) ~= 1.0
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
import argparse
from monte import p_init, get_p_init_max
from pnet_models import PNet
# add ../utilities package
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utilities.post_exp_cas2 import *
from utilities.constants import Case2_4D_Constants

PNET_PATH = "output/v2/p_net.pth"
PNET_INTER_PATH = "output/v2/p_net_"
DATA_FOLDER = "data/1e+6/"
device = "cpu"
TRAIN_FLAG = False
constants = Case2_4D_Constants()

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def dyn_f1(x):
    return x[:,2]

def dyn_f2(x):
    return x[:,3]

def dyn_f3(x):
    global constants
    return constants.T**2 *x[:,0] *(constants.W +constants.PHI *x[:,3]/constants.T)**2 -constants.T**2 *constants.MU_EARTH/(constants.R**3 * x[:,0]**2)

def dyn_f4(x):
    return -2*constants.T*x[:,2]*(constants.W + constants.PHI * x[:,3]/constants.T)/(x[:,0]*constants.PHI)

def diff_opt_p(x, t, p_net, beta=1.0, verbose=False):
    output = p_net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x1 = output_x[:,0].view(-1,1)
    output_x2 = output_x[:,1].view(-1,1)
    output_x3 = output_x[:,2].view(-1,1)
    output_x4 = output_x[:,3].view(-1,1)
    f1 = dyn_f1(x).view(-1,1)
    f2 = dyn_f2(x).view(-1,1)
    f3 = dyn_f3(x).view(-1,1)
    f4 = dyn_f4(x).view(-1,1)
    f4_x = torch.autograd.grad(f4, x, grad_outputs=torch.ones_like(f4), create_graph=True)[0]
    f4_x4 = f4_x[:,3].view(-1,1)
    residual = output_t + beta*(output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + f4_x4*output)
    # print(residual.dtype)
    return residual

def init_weights_He(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        # init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
                
def train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=40000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    N0_samples = 2000
    Nr_samples = 2000
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    x, t = constants.sample_res_points(Nr_samples)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    # beta = np.float32(0.0)
    beta = np.float32(1.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc)
        mse_u = mse_cost_function(phat_i/normalize, p_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt_p(x, t, p_net, beta=beta)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_p/normalize, all_zeros)
        
        # TV_loss by random perturbation (epsilon)
        # res_x = torch.autograd.grad(res_p, x, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
        res_t = torch.autograd.grad(res_p/normalize, t, grad_outputs=torch.ones_like(res_p/normalize), create_graph=True)[0]
        # tv_x = (res_x ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        tv_t = (res_t ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        tv_loss = torch.mean(tv_t)

        # Loss Function
        loss = mse_u + mse_res + 0.0 * tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, ",res:", mse_res.data,
                  ",tv:", tv_loss.data,
                  ",beta: ", beta
                   )
            torch.save({
                    'epoch': epoch, 'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history, 'train_time': train_time,
                    }, PNET_PATH)
            min_loss = loss.data
            FLAG = True
            
            beta = beta + np.float32(0.02)
            if(beta > 1.0):
                beta = np.float32(1.0)

            FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            if(FLAG_SAVE_INTER >= 10):
                INTER_COUNT = INTER_COUNT + 1
                torch.save({
                    'epoch': epoch, 'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history, 'train_time': train_time,
                    }, PNET_INTER_PATH+str(INTER_COUNT)+".pth")
                FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG):
            x_bc_rar, t_bc_rar = constants.sample_init_points(S)
            x_rar, t_rar = constants.sample_res_points(S)
            # add initial points
            p_i = p_init(x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(p_i - phat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(p_i.squeeze() - phat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # add residual points
            res_p = diff_opt_p(x_rar, t_rar, p_net)/normalize
            max_error= torch.max(torch.abs(res_p))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res_p.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()

def main():
    global constants
    p_net = PNet().to(device)
    p_net.apply(init_weights_He)
    p_net.scale = get_p_init_max(DATA_FOLDER)
    print("p net scale: ", p_net.scale)

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_p_net(p_net, optimizer, scheduler, mse_cost_function, iterations=50000)
        print("p_net_reg train complete")
    p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()

    ### Post-process ###
    # 1. total variation 
    # for t_prime in constants.T_PRIME_SPAN:
    #     tv, tv_nn = compute_total_variation(t_prime, "data/", "data/1e+7/", p_net)
    #     print(tv, tv_nn)
    # 2. distribution plots
    check_pdf_Nrphi(p_net, constants, DATA_FOLDER)
    # check_pdfnn_cartesian_wrt_monte(p_net, constants, "data/")
    # (obsolete)
    # # check_pdfnn_marginalize(p_net, t=t_prime)
    # # test_nn_cartesian_pdf_xy(p_net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
