"""
Train PINN e1
"""
import numpy as np
import torch
from tqdm import tqdm
import argparse
import copy
from monte import p_init_scaled, p_sol
from train_p import diff_opt_scaled, MC_FOLDER
# from exp_utilities.plot_util import precompute_e1hat_streaming
from exp_utilities.plot_util import plot_e1_pinn_validation
from exp_utilities.constants import Case1_6D_Constants_Equin
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, PNet_XL, E1Net_Equin, E1Net_Scaled, E1Net_XL, load_trained_model, init_weights_He
import _General.train_pinn as PINN


TRAIN_FLAG = False
constants = Case1_6D_Constants_Equin()


def pre_compute_validation_data(p_net, e1_net, OUTPUT_PATH, PRE_COMPUTE_FLAG=False):
    if(PRE_COMPUTE_FLAG == False):
        return
    global constatns
    N_samples = 1000000
    N_trials = 50
    t_check = np.linspace(constants.T_PRIME_SPAN[0], constants.T_PRIME_SPAN[-1], 41, endpoint=True)
    data_e1_max = np.zeros((len(t_check), N_trials))
    data_e1_pinn_max = np.zeros((len(t_check), N_trials))
    for i in range(len(t_check)):
        t = t_check[i]
        e1_max_t = []
        e1_pinn_max_t = []
        pdf_sol_max = 0.0
        for j in tqdm(range(N_trials), desc="Generating checking samples for time: {:.3f}".format(t)):
            _x = np.column_stack([
                np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], N_samples),
                np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], N_samples),
                np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], N_samples),
                np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], N_samples),
                np.random.uniform(constants.X5_RANGE[0], constants.X5_RANGE[1], N_samples),
                np.random.uniform(constants.X6_RANGE[0], constants.X6_RANGE[1], N_samples),
            ])
            _x_tensor = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
            _t = np.ones((len(_x_tensor), 1)) * t
            _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=True).view(-1,1)
            pdf_pinn = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            pdf_sol = p_sol(constants, _x, t).reshape(-1,)
            e1 = pdf_sol - pdf_pinn
            e1_pinn = e1_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
            e1_max = np.max(np.abs(e1)).item()
            e1_pinn_max = np.max(np.abs(e1_pinn)).item()
            pdf_sol_max = max(pdf_sol_max, np.max(pdf_sol).item())
            if(j == 0):
                e1_max_t.append(e1_max)
                e1_pinn_max_t.append(e1_pinn_max)
            else:
                _max1 = np.max(np.array(e1_max_t)).item()
                _max2 = np.max(np.array(e1_pinn_max_t)).item()
                e1_max_t.append(max(e1_max, _max1))
                e1_pinn_max_t.append(max(e1_pinn_max, _max2))
        data_e1_max[i, :] = np.array(e1_max_t)/pdf_sol_max
        data_e1_pinn_max[i, :] = np.array(e1_pinn_max_t)/pdf_sol_max
        del _x, _x_tensor, _t, _t_tensor, pdf_pinn, pdf_sol, e1, e1_pinn
    print(data_e1_max)
    print(data_e1_pinn_max)
    np.savez(OUTPUT_PATH+"/data_e1_max.npz", times=t_check, N_trials=N_trials, values=data_e1_max)
    np.savez(OUTPUT_PATH+"/data_e1_pinn_max.npz", times=t_check, N_trials=N_trials, values=data_e1_pinn_max)
            

def main():
    global constants
    OUTPUT_PATH = "output/v0_scaled_T0.3"
    
    # --- Load pinn p_net and compute normalize scale ---
    p_net = PNet_XL(constants, input_feature=7)
    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale = p_max
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    print("p net scale: ", p_net.scale)
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    # --- Init e1 pinn ---
    torch.manual_seed(0); np.random.seed(0) # set a fixed seed for reproducibility
    
    # e1_net = E1Net_Scaled(constants)
    # e1_net.apply(init_weights_He)
    # e1_net.scale = scale_torch*0.05 # assuming 10% percent error
    # e1_net.set_p_net(copy.deepcopy(p_net))
    # print("[check] e1_net scale: ", e1_net.scale)
    e1_net = E1Net_XL(constants, copy.deepcopy(p_net), scale=scale_torch, input_feature=7)
    e1_net.normalize = scale_torch*0.02
    print("[check] e1_net scale: {:.5f}, normalize: {:.5f}".format(
        e1_net.scale, e1_net.normalize))

    configurations = {
        "ic_fcn": p_init_scaled,
        "diff_opt_fcn": diff_opt_scaled,
        "save_path": OUTPUT_PATH+"/e1_net.pth",
        "save_path_inter": OUTPUT_PATH+"/e1_net_",
    }

    # --- Train e1 pinn over first time seq ---
    if(TRAIN_FLAG):
        networks = (p_net, e1_net)
        PINN.train_pinn_e1_v0_scaled_improved(constants, networks, configurations, 
                              iterations=50000, save_model=True, beta_incre=0.05)
    
    # --- Load best model after training ---
    e1_net = load_trained_model(e1_net, path=configurations["save_path"]); e1_net.eval()

    # # --- Pre-computation ---
    PRE_COMPUTE_FLAG = False
    # pre_compute_validation_data(p_net, e1_net, OUTPUT_PATH, PRE_COMPUTE_FLAG)

    # data_e1_max = np.load(OUTPUT_PATH+"/data_e1_max.npz")
    # data_e1_pinn_max = np.load(OUTPUT_PATH+"/data_e1_pinn_max.npz")
    # plot_e1_pinn_validation(data_e1_max, data_e1_pinn_max)

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
    # precompute_e1hat_streaming(constants, e1_net, OUTPUT_PATH)

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

    # --- Check how e1hat approximate e1 ---
    # for t in constants.T_PRIME_SPAN:
    #     e1_mc = np.load(OUTPUT_PATH+"/e1_t{:.3f}".format(t)+".npy")
    #     e1_pinn = np.load(OUTPUT_PATH+"/e1hat_t{:.3f}".format(t)+".npy")
    #     alpha = np.max(np.abs(e1_mc - e1_pinn))/np.max(np.abs(e1_pinn))
    #     print("[check] t: {:.2f}, alpha: {:.3f}, e1hat*: {:.3f}, e1*: {:.4f}".format(
    #         t, alpha, np.max(np.abs(e1_pinn)), np.max(np.abs(e1_mc))))
    #     if(np.abs(t) < 1e-5):
    #         e1_init_analy = np.load(OUTPUT_PATH+"/e1_analy_t0.000.npy")
    #         alpha = np.max(np.abs(e1_init_analy - e1_pinn))/np.max(np.abs(e1_pinn))
    #         print("[check] analy. t: {:.2f}, alpha: {:.3f}, e1hat*: {:.3f}, e1*: {:.4f}".format(
    #         t, alpha, np.max(np.abs(e1_pinn)), np.max(np.abs(e1_init_analy))))
    #         del e1_init_analy
    #     del e1_mc; del e1_pinn

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