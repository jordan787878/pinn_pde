"""
Train PINN e1
"""
import numpy as np
import torch
import argparse
from monte import p_init
from train_p import diff_opt, MC_FOLDER
from exp_utilities.plot_util import plot_time_curves_3d, precompute_e1hat_streaming, compute_marginals_over_time
from exp_utilities.constants import Case1_6D_Constants
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, E1Net, load_trained_model, init_weights_He
import _General.train_pinn as PINN


TRAIN_FLAG = False
constants = Case1_6D_Constants()
            

def main():
    global constants
    OUTPUT_PATH = "output/v0"
    
    # --- Init e1 pinn ---
    torch.manual_seed(0); np.random.seed(0) # set a fixed seed for reproducibility
    e1_net = E1Net(constants)
    e1_net.apply(init_weights_He)

    # --- Load pinn p_net and compute normalize scale ---
    p_net = PNet(constants, input_feature=7)
    scale = np.load(MC_FOLDER+"pre_compute/p_init_max.npz")["value"]
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()
    e1_net.scale = p_net.scale*0.1 # assuming 10% percent error
    print("[check] e1_net scale: ", e1_net.scale)

    configurations = {
        "ic_fcn": p_init,
        "diff_opt_fcn": diff_opt,
        "save_path": OUTPUT_PATH+"/e1_net.pth",
        "save_path_inter": OUTPUT_PATH+"/e1_net_",
    }

    # --- Train e1 pinn over first time seq ---
    if(TRAIN_FLAG):
        networks = (p_net, e1_net)
        PINN.train_pinn_e1_v0(constants, networks, configurations, 
                              iterations=100000, save_model=True, beta_incre=0.01)
    
    # --- Load best model after training ---
    e1_net = load_trained_model(e1_net, path=configurations["save_path"]); e1_net.eval()

    # --- Pre-computation ---

    # 1) compute e1_init from analytical p_init
    # p_init_analy = np.load("data/1e+6/pdf_analy_t0.000.npy")
    # p_init_pinn = np.load(OUTPUT_PATH+"/pdf_t0.000.npy")
    # e1_init_analy = p_init_analy - p_init_pinn
    # print(np.max(np.abs(e1_init_analy)))
    # np.save(OUTPUT_PATH+"/e1_analy_t0.000.npy", e1_init_analy)

    # 2) compute e1 over time using MC data
    # precompute_error(constants, p_net, OUTPUT_PATH, MC_FOLDER)

    # 3) marginalize e1 
    # for i in range(1, 2): # pre-compute marginal pdf over time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir=OUTPUT_PATH,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="e1_t{:.3f}.npy",
    #     )
    #     np.savez(OUTPUT_PATH+"/pre_compute/marginal_e1_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)
    
    # 4) compute e1hat over time using pinn
    precompute_e1hat_streaming(constants, e1_net, OUTPUT_PATH)

    # 5) marginalize e1hat
    # for i in range(1,2):
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir=OUTPUT_PATH,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="e1hat_t{:.3f}.npy",
    #     )
    #     np.savez(OUTPUT_PATH+"/pre_compute/marginal_e1hat_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    # --- Plot ---
    # for i in range(1, 7):
    #     data_mc = np.load(OUTPUT_PATH+"/pre_compute/marginal_e1_x"+str(i)+"_t_M.npz")
    #     # data_pinn = np.load(PNET_PATH+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(data_mc, title="Marginal e1(x"+str(i)+"|t): curves")

    # for i in range(1, 2):
    #     data_mc = np.load(OUTPUT_PATH+"/pre_compute/marginal_e1_x"+str(i)+"_t_M.npz")
    #     data_pinn = np.load(OUTPUT_PATH+"/pre_compute/marginal_e1hat_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(data_pinn, data2=data_mc, title="Marginal error(x"+str(i)+"|t): curves")

    # --- Print ---
    # check MC accuracy by comparing e1_analy_t0.000.npy vs e1_t0.000.npy
    # e1_init_analy = np.load(OUTPUT_PATH+"/e1_analy_t0.000.npy")
    # e1_init_mc = np.load(OUTPUT_PATH+"/e1_t0.000.npy")
    # rel_acc = np.max(np.abs(e1_init_mc - e1_init_analy))/np.max(np.abs(e1_init_analy))
    # del e1_init_analy; del e1_init_mc
    # print("[check] rel. accuracy of MC at t0: ", rel_acc.item())

    # check how e1hat approximate e1
    for t in constants.T_PRIME_SPAN:
        e1_mc = np.load(OUTPUT_PATH+"/e1_t{:.3f}".format(t)+".npy")
        e1_pinn = np.load(OUTPUT_PATH+"/e1hat_t{:.3f}".format(t)+".npy")
        alpha = np.max(np.abs(e1_mc - e1_pinn))/np.max(np.abs(e1_pinn))
        print("[check] t: {:.2f}, alpha: {:.3f}, e1hat*: {:.3f}, e1*: {:.4f}".format(
            t, alpha, np.max(np.abs(e1_pinn)), np.max(np.abs(e1_mc))))
        if(np.abs(t) < 1e-5):
            e1_init_analy = np.load(OUTPUT_PATH+"/e1_analy_t0.000.npy")
            alpha = np.max(np.abs(e1_init_analy - e1_pinn))/np.max(np.abs(e1_pinn))
            print("[check] analy. t: {:.2f}, alpha: {:.3f}, e1hat*: {:.3f}, e1*: {:.4f}".format(
            t, alpha, np.max(np.abs(e1_pinn)), np.max(np.abs(e1_init_analy))))
            del e1_init_analy
        del e1_mc; del e1_pinn

    # --- Post-process ---
    # for t_prime in constants.T_PRIME_SPAN:
    #     check_error_flatten(constants, p_init, e1_net, p_net, t_prime, MC_FOLDER)
    # visual_e1hat_training(constants, (p_net, e1_net, e1_net), MC_FOLDER,
    #                       save_plot_path="figs/case2_e1net.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()