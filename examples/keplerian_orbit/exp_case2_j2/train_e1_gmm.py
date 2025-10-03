"""
Train PINN e1
"""
import numpy as np
import torch
import argparse
from monte import p_init, get_p_init_max
from train_p import diff_opt, MC_FOLDER
from exp_utilities.plot_utilites import check_error_flatten, check_error_batched_normsup, visual_e1hat_training
from exp_utilities.constants import Case2_4D_Constants
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import TimeToGMM6D_V0, E1Net_XL, load_trained_model
import _General.train_pinn as PINN


TRAIN_FLAG = False
constants = Case2_4D_Constants()
            

def main():
    global constants
    OUTPUT_PATH = "output/pinn-gmm(V0)"
    
    # --- Load pinn p_net and compute normalize scale ---
    scale = get_p_init_max(constants)
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    print(scale_torch)
    
    p_net = TimeToGMM6D_V0(constants, K=11, D=4)
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    # --- Init e1 pinn ---
    torch.manual_seed(0); np.random.seed(0) # set a fixed seed for reproducibility
    e1_net = E1Net_XL(constants, scale=0.02*scale_torch, normalize=scale_torch*0.02, input_feature=5)
    print("[check] e1_net scale: {:.5f}, normalize: {:.5f}".format(
        e1_net.scale, e1_net.normalize))

    configurations = {
        "ic_fcn":p_init,
        "diff_opt_fcn": diff_opt,
        "save_path": OUTPUT_PATH+"/e1_net.pth",
        "save_path_inter": None, #OUTPUT_PATH+"/e1_net_",
        "sample_res_uniform_fcn": constants.sample_res_points_uniform
    }

    # --- Train e1 pinn over first time seq ---
    if(TRAIN_FLAG):
        networks = (p_net, e1_net)
        PINN.train_pinn_e1gmm_v0_scaled_improved(constants, networks, configurations, 
                              iterations=750000, save_model=True)#, beta_incre=0.05)
    
    # --- Load best model after training ---
    e1_net = load_trained_model(e1_net, path=configurations["save_path"]); e1_net.eval()

    # --- Post-process ---
    t_check = np.round(constants.T_PRIME_SPAN, 2)
    t_check = [t_check[0], t_check[-1]] # [tmp]
    for t_prime in t_check:
        # check_error_flatten(constants, p_init, e1_net, p_net, t_prime, MC_FOLDER)
        res = check_error_batched_normsup(
            constants=constants,
            p_init_func=p_init,   # analytical p_true at t=0
            e1_net=e1_net,        # can be None
            p_net=p_net,
            t=t_prime,
            data_folder=MC_FOLDER,
            max_points_tile=250_000,
            batch_size_torch=64_000
        )
        print(res)

    # # visual_e1hat_training(constants, (p_net, e1_net, e1_net), MC_FOLDER,
    # #                       save_plot_path="figs/case2_e1net.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()