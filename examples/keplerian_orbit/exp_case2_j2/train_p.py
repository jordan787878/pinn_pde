import numpy as np
import torch
import argparse
from monte import p_init
from exp_utilities.constants import Case2_4D_Constants

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.neuralnetworks import PNet_XL, TimeToGMM_V0, load_trained_model
import utilities._General.train_pinn as PINN
from utilities._General.util import RunLogger, save_config_human


TRAIN_FLAG = False
constants = Case2_4D_Constants()


def dyn_f1(x):
    return x[:,2]


def dyn_f2(x):
    return x[:,3]


def dyn_f3(x):
    global constants
    return constants.T**2 *x[:,0] *(constants.W +constants.PHI *x[:,3]/constants.T)**2 - \
           constants.T**2 *constants.MU_EARTH/(constants.R**3 * x[:,0]**2) + \
           constants.J2_VR/(x[:,0]**4)


def dyn_f4(x):
    global constants
    return -2*constants.T*x[:,2]*(constants.W + constants.PHI * x[:,3]/constants.T)/(x[:,0]*constants.PHI)


def diff_opt(x, t, net, beta=1.0, verbose=False):
    """
    differential operator with J2 and Brownian noise in velocity state:
    dw in R^2 with noise intensity diag([constants.N_Q_NOISE])
    As such the differential term becomes:
    0.5 * (Q[0] * P_x3x3 + Q[1] * P_x4x4)
    """
    global constants
    output = net(x,t)
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

    # only need ∂f4/∂x4
    ones_f4 = torch.ones_like(f4)
    f4_x = torch.autograd.grad(
        outputs=f4, inputs=x, grad_outputs=ones_f4,
        create_graph=True, retain_graph=True
    )[0]
    f4_x4 = f4_x[:, 3:4]

    # second-derivative diagonals we actually use
    output_x3x3 = torch.autograd.grad(
        outputs=output_x3, inputs=x, grad_outputs=torch.ones_like(output_x3),
        create_graph=True, retain_graph=True
    )[0][:, 2:3]                                         # (N,1)

    output_x4x4 = torch.autograd.grad(
        outputs=output_x4, inputs=x, grad_outputs=torch.ones_like(output_x4),
        create_graph=True, retain_graph=True
    )[0][:, 3:4]                                         # (N,1)

    Q = constants.N_Q_NOISE_TENSOR

    residual = output_t + beta*(output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + f4_x4*output \
                                - 0.5*(Q[0]*output_x3x3 + Q[1]*output_x4x4))
    return residual
                

def config_training(constants, option=""):
    # determine the scale of p_net
    scale = p_init(constants, constants.N_MEAN_I).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32); print(scale_torch)

    configuration = {
        "constants": constants,
        "iterations": 40000,
        "sample_ic": constants.sample_init_points,
        "sample_res": constants.sample_res_points_uniform,
        "p_ic": constants.p_init_torch,
        "res_func": diff_opt,
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

    if option == "pinn-xl":
        configuration["training_fcn"] = PINN.train_pinn
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=5)

    if option == "pinn-xl_bias":
        configuration["training_fcn"] = PINN.train_pinn
        configuration["sample_res"] = constants.sample_res_points_bias
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=5)
        
    # if option == "pinn-gmm":
    #     configuration["training_fcn"] = PINN.train_pinngmm_expcase1equin
    #     p_net = TimeToGMM6D_V0(constants, K=11, alpha_floor=0.01)

    if option == "pinn-gmm_bias":
        configuration["training_fcn"] = PINN.train_pinngmm
        configuration["bias_fac"] = 0.5
        p_net = TimeToGMM_V0(constants, D=4, K=11, alpha_floor=0.01)

    return configuration, p_net


def main():
    global constants
    torch.manual_seed(0); np.random.seed(0)  # Set a fixed seed for reproducibility

    # Setup config
    config, p_net = config_training(constants, option="pinn-xl")
    save_config_human(config, model_name="p_net")

    # Train & log
    if(TRAIN_FLAG):
        log_file = Path(config["save_path"]) / "p_net-train.log"
        with RunLogger(log_file, tee=True):      # tee=False if you want file-only
            config["training_fcn"](p_net, config)
    
    # Load the best network after training
    p_net = load_trained_model(p_net, path=config["save_path"]+"/p_net.pth"); p_net.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
