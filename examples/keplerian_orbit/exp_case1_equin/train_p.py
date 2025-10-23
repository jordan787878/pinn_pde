"""
Case 1 of the paper: Uncertainty propagation in orbital mechanics via tensor decomposition,
without J2 and process noise
x = [r, th, phi, vr, vth, vphi]
"""
import numpy as np
import torch
import argparse
from monte import p_init, p_init_scaled, p_sol, p_sol_scaled, print_mc_time
from exp_utilities.constants import Case1_6D_Constants_Equin

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.neuralnetworks import PNet_XL, TimeToGMM_V0, load_trained_model
import utilities._General.train_pinn as PINN
from utilities._General.util import RunLogger, save_config_human


TRAIN_FLAG = False
constants = Case1_6D_Constants_Equin()


def dyn_f1(x):
    return 0.0*x[:,0]

def dyn_f2(x):
    return 0.0*x[:,1]

def dyn_f3(x):
    return 0.0*x[:,2]

def dyn_f4(x):
    return 0.0*x[:,3]

def dyn_f5(x):
    return 0.0*x[:,4]

def dyn_f6(x):
    global constants
    return torch.sqrt(constants.MU_EARTH/ x[:,0]**3) * constants.T

def dyn_f6_scaled(x):
    global constants
    x1 = x[:,0]*(constants.COV_I[0,0]**0.5) + constants.MEAN_I[0]
    return torch.sqrt(constants.MU_EARTH/ x1**3) * constants.T / (constants.COV_I[5, 5]**0.5)

def diff_opt(x, t, p_net, beta=1.0, verbose=False):
    global constants
    output = p_net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_x1 = output_x[:,0].view(-1,1)
    output_x2 = output_x[:,1].view(-1,1)
    output_x3 = output_x[:,2].view(-1,1)
    output_x4 = output_x[:,3].view(-1,1)
    output_x5 = output_x[:,4].view(-1,1)
    output_x6 = output_x[:,5].view(-1,1)

    f1 = dyn_f1(x).view(-1,1)
    f2 = dyn_f2(x).view(-1,1)
    f3 = dyn_f3(x).view(-1,1)
    f4 = dyn_f4(x).view(-1,1)
    f5 = dyn_f5(x).view(-1,1)
    f6 = dyn_f6(x).view(-1,1)

    f5_x = torch.autograd.grad(f5, x, grad_outputs=torch.ones_like(f5), create_graph=True)[0]
    f5_x5 = f5_x[:,4].view(-1,1)

    f6_x = torch.autograd.grad(f6, x, grad_outputs=torch.ones_like(f6), create_graph=True)[0]
    f6_x6 = f6_x[:,5].view(-1,1)

    residual = output_t + beta*(output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + \
                                output_x5*f5 + f5_x5*output + \
                                output_x6*f6 + f6_x6*output)
    if(verbose):
        print(residual.dtype, residual.shape, residual[0:3, :])
    return residual

def diff_opt_scaled(x, t, p_net, beta=1.0, verbose=False):
    global constants
    output = p_net(x,t)
    output_x = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    output_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    # output_x1 = output_x[:,0].view(-1,1)
    # output_x2 = output_x[:,1].view(-1,1)
    # output_x3 = output_x[:,2].view(-1,1)
    # output_x4 = output_x[:,3].view(-1,1)
    # output_x5 = output_x[:,4].view(-1,1)
    output_x6 = output_x[:,5].view(-1,1)

    # f1 = dyn_f1(x).view(-1,1)
    # f2 = dyn_f2(x).view(-1,1)
    # f3 = dyn_f3(x).view(-1,1)
    # f4 = dyn_f4(x).view(-1,1)
    # f5 = dyn_f5(x).view(-1,1)
    f6 = dyn_f6_scaled(x).view(-1,1)

    # f5_x = torch.autograd.grad(f5, x, grad_outputs=torch.ones_like(f5), create_graph=True)[0]
    # f5_x5 = f5_x[:,4].view(-1,1)

    # f6_x = torch.autograd.grad(f6, x, grad_outputs=torch.ones_like(f6), create_graph=True)[0]
    # f6_x6 = f6_x[:,5].view(-1,1)

    # residual = output_t + beta*(output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + \
    #                             output_x5*f5 + f5_x5*output + \
    #                             output_x6*f6 + f6_x6*output)
    
    # Reduced to this since f1=f2=f3=f4=f5=0, and f6_x6 = 0
    residual = output_t + beta*(output_x6*f6)

    if(verbose):
        print(residual.dtype, residual.shape, residual[0:3, :])
    return residual

def quick_training_check(p_net, N_samples=100000):
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

        _x_tensor = torch.tensor(_x_scaled, dtype=torch.float32, requires_grad=True)
        _t = np.ones((len(_x_tensor), 1)) * t
        _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=True).view(-1,1)
        pdf_pinn_scaled = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)
        pdf_pinn = pdf_pinn_scaled.copy() * constants.SCALING_PDF

        rel_error = np.max(np.abs(pdf_sol-pdf_pinn)) / np.max(pdf_sol) # NOTE this metric needs sufficient large samples.
        print("[check] norm. worst error (%): ", 100. *rel_error.item())

def config_training(constants, option=""):
    # Scale of p0 max
    _x_at_mean = constants.N_MEAN_I.copy()
    scale = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32)

    configuration = {
        "constants": constants,
        "iterations": 60000,
        "sample_ic": constants.sample_init_points_scaled,
        "sample_res": constants.sample_res_points_scaled_uniform,
        "p_ic": constants.p_init_torch,
        "res_func": diff_opt_scaled,
        "res_weight": 1.0,
        "save_path": "output/" + option,
        "beta_incre": 0.02,
        "loss_normalize": scale_torch,
        "reg_tv": None,
        "N0_samples_initial": 4000,
        "Nr_samples_initial": 4000,
        "iterations_per_decay" : 1000,
        "N_RAR": 30000,
        "N_RAR_TO_ADD": 32,
        "N_RAR_CAP": 4000,
        "iterations_per_rar": 100,
        "RAR_eps": 0.01,
        "bias_fac": 0.,
    }

    # if option == "pinn-xl":
    #     configuration["training_fcn"] = PINN.train_pinn
    #     p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)

    if option == "pinn-xl_bias":
        configuration["training_fcn"] = PINN.train_pinn
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        configuration["sample_res"] = constants.sample_res_points_scaled_bias

    if option == "pinn-gmm":
        configuration["training_fcn"] = PINN.train_pinngmm
        p_net = TimeToGMM_V0(constants, K=11)

    # if option == "pinn-gmm_double-samples":
    #     configuration["training_fcn"] = PINN.train_pinngmm
    #     p_net = TimeToGMM_V0(constants, K=11)
    #     configuration["N0_samples_initial"] = 8000
    #     configuration["Nr_samples_initial"] = 8000
    #     configuration["N_RAR_TO_ADD"] = 64
    #     configuration["N_RAR"] = 60000
    #     configuration["N_RAR_CAP"] = 8000

    if option == "pinn-gmm_bias":
        configuration["training_fcn"] = PINN.train_pinngmm
        configuration["bias_fac"] = 0.5
        # configuration["reg_tv"] = torch.tensor(0.0)
        p_net = TimeToGMM_V0(constants, K=5, alpha_floor=0.01)

    return configuration, p_net

def main():
    global constants #; constants.test_printout()

    # Set a fixed seed for reproducibility
    torch.manual_seed(0); np.random.seed(0)

    # Setup config
    config, p_net = config_training(constants, option="pinn-gmm_bias")
    save_config_human(config, model_name="p_net")

    # Train & log
    if(TRAIN_FLAG):
        log_file = Path(config["save_path"]) / "p_net-train.log"
        with RunLogger(log_file, tee=True):      # tee=False if you want file-only
            config["training_fcn"](p_net, config)
    
    # Load the best network after training
    p_net = load_trained_model(p_net, path=config["save_path"]+"/p_net.pth"); p_net.eval()
    quick_training_check(p_net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, default=0, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
