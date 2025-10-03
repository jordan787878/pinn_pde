"""
Train PINN e1
"""
import numpy as np
import torch
import argparse
from monte import p_init, get_p_init_max, get_max_e1_init
from train_p import diff_opt, MC_FOLDER
from exp_utilities.plot_utilites import check_error_flatten, visual_e1hat_training
from exp_utilities.constants import Case2_4D_Constants
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, E1Net, load_trained_model, init_weights_He
import _General.train_pinn as PINN


TRAIN_FLAG = False
constants = Case2_4D_Constants()
            

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
        "diff_opt_fcn": diff_opt,
        # "save_path": "output/v0/e1_net_seq1.pth",
        # "save_path_inter": "output/v0/e1_net_seq1_",
        "save_path": "output/base/e1_net.pth",
        "save_path_inter": "output/base/e1_net_",
    }

    # --- Train e1 pinn over first time seq ---
    if(TRAIN_FLAG):
        networks = (p_net, e1_net)
        # PINN.train_pinn_e1seq1_v0(constants, networks, configurations, 
        #                           iterations=120, save_model=False)
        PINN.train_pinn_e1seq1_base(constants, networks, configurations, 
                                  iterations=20000, save_model=True)
    
    # --- Load best model after training ---
    e1_net = load_trained_model(e1_net, path=configurations["save_path"]); e1_net.eval()

    # --- Post-process ---
    for t_prime in constants.T_PRIME_SPAN:
        check_error_flatten(constants, p_init, e1_net, p_net, t_prime, MC_FOLDER)
    # visual_e1hat_training(constants, (p_net, e1_net, e1_net), MC_FOLDER,
    #                       save_plot_path="figs/case2_e1net.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()