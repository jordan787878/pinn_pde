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
from monte import p_init, get_p_init_max, p_init_perturb
from main_train_pnetreg_cur import diff_opt_p
from pnet_models import PNet
from e1net_models import E1Net
import sys
import os
# Get the parent directory of the current directory (exp1)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from post.post_exp_cas2 import check_train_results, load_trained_model, get_max_e1_init


# curriculum training
E1NET_PATH = "output/v2/e1_net_seq2.pth"
E1NET_INTER_PATH = "output/v2/e1_net_seq2_"

PNET_PATH = "output/v2/p_net.pth"
E1NET_PATH_SEQ1 = "output/v2/e1_net_seq1.pth" # sequence 1
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
        # init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(e1_net, p_net, e1_net_seq1, optimizer, scheduler, mse_cost_function, iterations=50000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)
    t_seq = np.float32(np.array([constants.TF*0.5, constants.TF]))

    N0_samples = 2000
    Nr_samples = 2000
    # x_bc, t_bc = constants.sample_init_points(N0_samples)
    # x, t = constants.sample_res_points(Nr_samples)
    x_bc, t_bc = constants.sample_init_points_seq(N0_samples, t_seq)
    x, t = constants.sample_res_points_seq(Nr_samples, t_seq)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        e_i = e1_net_seq1(x_bc, t_bc).to(device)
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt_p(x, t, p_net, beta=beta)
        res_e = diff_opt_p(x, t, e1_net, beta=beta)
        res = res_e + res_p
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)

        # # tv
        tau = t + torch.randn_like(t) * np.float32(0.05)
        res_tau = diff_opt_p(x, tau, e1_net, beta=beta) + diff_opt_p(x, tau, p_net, beta=beta)
        tv_loss = mse_cost_function(res_tau/normalize, res/normalize)

        loss = mse_u + mse_res + 1e-2*tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,   
                  ",res:", mse_res.data,
                  ",tv:",tv_loss.data,
                  ",beta:", beta
                   )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'train_time': train_time,
                    }, E1NET_PATH)
            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(0.005)
            if(beta > 1.0):
                beta = np.float32(1.0)

            FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            if(FLAG_SAVE_INTER >= 10):
                INTER_COUNT = INTER_COUNT + 1
                torch.save({
                    'epoch': epoch, 'model_state_dict': e1_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history, 'train_time': train_time,
                    }, E1NET_INTER_PATH+str(INTER_COUNT)+".pth")
                FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG):
            x_bc_rar, t_bc_rar = constants.sample_init_points_seq(S, t_seq)
            x_rar, t_rar = constants.sample_res_points_seq(S, t_seq)

            # add initial points
            e_i = e1_net_seq1(x_bc_rar, t_bc_rar).to(device)
            ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # # add residual points
            res_p = diff_opt_p(x_rar, t_rar, p_net, beta=beta)
            res_e = diff_opt_p(x_rar, t_rar, e1_net, beta=beta)
            res = (res_e + res_p)/normalize
            max_error= torch.max(torch.abs(res))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def main():
    global constants
    p_net = PNet(scale=get_p_init_max()).to(device)
    p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()
    e1_net_seq1 = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants)).to(device)
    e1_net_seq1 = load_trained_model(e1_net_seq1, path=E1NET_PATH_SEQ1, method="new"); e1_net_seq1.eval()

    e1_net = E1Net(scale=get_max_e1_init(p_net, DATA_FOLDER, constants)).to(device)
    e1_net.apply(init_weights_He)

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if(TRAIN_FLAG):
        train_model(e1_net, p_net, e1_net_seq1, optimizer, scheduler, mse_cost_function, iterations=50000); print("e1_net train complete")
    e1_net = load_trained_model(e1_net, path=E1NET_PATH, method="new"); e1_net.eval()

    ### Post-process ###
    for t_prime in constants.T_PRIME_SPAN:
        if(t_prime >= 0.5*constants.TF/constants.T):
            check_train_results(e1_net, p_net, t_prime, DATA_FOLDER, constants)

    # for t_prime in constants.T_PRIME_SPAN:
    #    check_pdfnn_marginalize(p_net, t=t_prime)
    # test_nn_cartesian_pdf_xy(p_net, model_name="p_net")
    # check_pdfnn_cartesian_wrt_monte(p_net, model_name="p_net")
    # test_nn_cartesian_pdf_xy(p_net_gmm, model_name="p_net_gmm")
    # check_pdfnn_cartesian_wrt_monte(p_net_gmm, model_name="p_net_gmm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()