"""
Train PINN e1
"""
import numpy as np
import torch
import time
import argparse
from monte import p_init, get_p_init_max, get_max_e1_init
from train_p import diff_opt_p
from exp_utilities.plot_utilites import check_error_flatten
from exp_utilities.constants import Case2_4D_Constants
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, E1Net, load_trained_model, init_weights_He
import _General.train_pinn as PINN


DATA_FOLDER = "data/1e+6/"
TRAIN_FLAG = False
constants = Case2_4D_Constants()


def train_model(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=50000):
    global constants
    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)
    t_seq = np.float32(np.array([constants.TI, constants.TF*0.5]))

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
    # beta = np.float32(1.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(constants, x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc).to(device)
        e_i = p_i - phat_i
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt_p(x, t, p_net, beta=beta)
        res_e = diff_opt_p(x, t, e1_net, beta=beta)
        res = res_e + res_p
        # all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
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
            p_i = p_init(constants, x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            e_i = p_i - phat_i
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
    PNET_PATH = "output/v0/p_net.pth"

    # --- Init e1 pinn ---
    torch.manual_seed(0); np.random.seed(0) # set a fixed seed for reproducibility
    e1_net = E1Net(constants)
    e1_net.apply(init_weights_He)

    # --- Load pinn p_net and compute normalize scale ---
    p_net = PNet(constants, scale=get_p_init_max(constants))
    p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()
    e1_net.scale = get_max_e1_init(constants, p_net)

    configurations = {
        "ic_fcn": p_init,
        "diff_opt_fcn": diff_opt_p,
        "save_path": "output/base/e1_net_seq1.pth",
        "save_path_inter": "output/base/e1_net_seq1_",
    }
    if(TRAIN_FLAG):
        networks = (p_net, e1_net)
        # train_model(e1_net, p_net, optimizer, scheduler, mse_cost_function, iterations=50000); print("e1_net train complete")
        PINN.train_pinn_e1seq1_base(constants, networks, configurations, 
                                    iterations=20000, save_model=True)

    # --- Load best model after training ---
    e1_net = load_trained_model(e1_net, path=configurations["save_path"], method="new"); e1_net.eval()

    # --- Post-process ---
    for t_prime in constants.T_PRIME_SPAN:
        check_error_flatten(constants, p_init, e1_net, p_net, t_prime, DATA_FOLDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()