"""
Train PINN e1
"""
import numpy as np
import torch
import argparse
from tqdm import tqdm
from functools import partial
from monte import p_init_scaled, p_sol
from train_p import diff_opt_scaled
# from exp_utilities.plot_util import plot_e1_pinn_validation
from exp_utilities.constants import Case1_6D_Constants_Equin

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.neuralnetworks import PNet_XL, ENet_XL, TimeToGMM6D_V0, load_trained_model
import utilities._General.train_pinn as PINN
from utilities._General.util import RunLogger, save_config_human


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
            

def quick_training_check(e1_net, p_net, N_samples=100000):
    """
    checking joint PDF against p_sol
    """
    global constants

    # x points
    _x = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], N_samples),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], N_samples),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], N_samples),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], N_samples),
        np.random.uniform(constants.X5_RANGE[0], constants.X5_RANGE[1], N_samples),
        np.random.uniform(constants.X6_RANGE[0], constants.X6_RANGE[1], N_samples),
    ])
    # scaled x points
    _x_scaled = constants.scaled_x(_x)

    # # checking in scaled x
    # for t in constants.T_PRIME_SPAN:
    #     pdf_sol_scaled = p_sol_scaled(constants, _x_scaled, t).reshape(-1,)

    #     _x_tensor = torch.tensor(_x_scaled, dtype=torch.float32, requires_grad=True)
    #     _t = np.ones((len(_x_tensor), 1)) * t
    #     _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=True).view(-1,1)
    #     pdf_pinn = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
    #     rel_error = np.max(np.abs(pdf_sol_scaled-pdf_pinn)) / np.max(pdf_sol_scaled) # NOTE this metric needs sufficient large samples.
    #     print(100. *rel_error.item())
    
    # checking in x
    for t in constants.T_PRIME_SPAN:
        pdf_sol = p_sol(constants, _x, t).reshape(-1,)
        p_max = np.max(pdf_sol).item()

        _x_tensor = torch.tensor(_x_scaled, dtype=torch.float32, requires_grad=True)
        _t = np.ones((len(_x_tensor), 1)) * t
        _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=True).view(-1,1)
        pdf_pinn_scaled = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        pdf_pinn = pdf_pinn_scaled.copy() * constants.SCALING_PDF
        del pdf_pinn_scaled
        error_pinn_scaled = e1_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        error_pinn = error_pinn_scaled.copy() * constants.SCALING_PDF
        del error_pinn_scaled

        norm_worst_error = 100. * np.max(np.abs(pdf_sol-pdf_pinn)).item() / p_max # NOTE this metric needs sufficient large samples.
        norm_pinn_worst_error = 100. * np.max(np.abs(error_pinn)).item() / p_max
        print("[check] norm. worst error (%): ", norm_worst_error, norm_pinn_worst_error)


def config_training_ENet_XL(constants, fac=0.02, option=""):
    # Scale of p0 max
    _x_at_mean = constants.N_MEAN_I.copy()
    scale = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32)

    configuration = {
        "constants": constants,
        "iterations": 40000,
        "sample_ic": constants.sample_init_points_scaled,
        "sample_res": constants.sample_res_points_scaled_uniform,
        "p_ic": p_init_scaled,
        "res_func": diff_opt_scaled,
        "res_weight": 1.0,
        "save_path": "output/" + option,
        "beta_incre": 0.02,
        "fac": fac,
        "loss_normalize": scale_torch*fac,
        "reg_tv": None,
        "training_fcn": PINN.train_pinn_error_expcase1equin,
        "N0_samples_initial": 4000,
        "Nr_samples_initial": 4000,
        "iterations_per_decay" : 1000,
        "N_RAR": 30000,
        "N_RAR_TO_ADD": 32,
        "N_RAR_CAP": 4000,
        "iterations_per_rar": 100,
        "RAR_eps": 0.01,
        "bias_fac": 0.
    }

    if option == "pinn-xl":
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=fac*scale_torch, input_feature=7)
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        p_net = load_trained_model(p_net, path=configuration["save_path"]+"/p_net.pth"); p_net.eval()

    if option == "pinn-gmm":
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=fac*scale_torch, input_feature=7)
        p_net = TimeToGMM6D_V0(constants, K=11)
        p_net = load_trained_model(p_net, path=configuration["save_path"]+"/p_net.pth"); p_net.eval()

    if option == "pinn-gmm_bias-test":
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=fac*scale_torch, input_feature=7)
        p_net = TimeToGMM6D_V0(constants, K=11)
        p_net = load_trained_model(p_net, path=configuration["save_path"]+"/p_net.pth"); p_net.eval()
        configuration["bias_fac"] = 0.5

    return configuration, p_net, e1_net


def main():
    global constants #; constants.test_printout()

    # Set a fixed seed for reproducibility
    torch.manual_seed(0); np.random.seed(0)

    # Setup config
    fac = 0.02
    config, p_net, e1_net = config_training_ENet_XL(constants, fac=fac, option="pinn-gmm")
    save_config_human(config, model_name="e1_net")

    # Train & log
    if(TRAIN_FLAG):
        log_file = Path(config["save_path"]) / "e1_net-train.log"
        with RunLogger(log_file, tee=True):      # tee=False if you want file-only
            networks = (p_net, e1_net)
            config["training_fcn"](networks, config)
    
    # Load the best network after training
    e1_net = load_trained_model(e1_net, path=config["save_path"]+"/e1_net.pth"); e1_net.eval()
    # quick_training_check(e1_net, p_net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, default=False, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()