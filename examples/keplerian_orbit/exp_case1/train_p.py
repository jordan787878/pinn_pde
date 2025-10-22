import numpy as np
import torch
import argparse
from functools import partial
from monte import p_init
from exp_utilities.constants import Case1_6D_Constants

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.neuralnetworks import PNet_XL, TimeToGMM_V0, load_trained_model
import utilities._General.train_pinn as PINN
from utilities._General.util import RunLogger, save_config_human


TRAIN_FLAG = False
constants = Case1_6D_Constants()


def dyn_f1(x):
    r = x[:, 0]; th = x[:, 1]; ph = x[:, 2]
    vr = x[:, 3]; vth = x[:, 4]; vph = x[:, 5]
    return vr


def dyn_f2(x):
    r = x[:, 0]; th = x[:, 1]; ph = x[:, 2]
    vr = x[:, 3]; vth = x[:, 4]; vph = x[:, 5]
    return vth


def dyn_f3(x):
    r = x[:, 0]; th = x[:, 1]; ph = x[:, 2]
    vr = x[:, 3]; vth = x[:, 4]; vph = x[:, 5]
    return vph


def dyn_f4(x):
    global constants
    r = x[:, 0]; th = x[:, 1]; ph = x[:, 2]
    vr = x[:, 3]; vth = x[:, 4]; vph = x[:, 5]
    return r *constants.THETA**2 *vth**2 \
           + constants.T**2 *r *(torch.sin(constants.THETA *th))**2 *(constants.W +constants.PHI *vph/constants.T)**2 \
           - constants.T**2 *constants.MU_EARTH/(constants.R**3 * r**2)


def dyn_f5(x):
    global constants
    r = x[:, 0]; th = x[:, 1]; ph = x[:, 2]
    vr = x[:, 3]; vth = x[:, 4]; vph = x[:, 5]
    return -2. *vth *vr/r \
           + constants.T**2 * torch.sin(2.*constants.THETA *th) *(constants.W +constants.PHI * vph/constants.T)**2 /(2. *constants.THETA)


def torch_cot(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tan(x)


def dyn_f6(x):
    global constants
    r = x[:, 0]; th = x[:, 1]; ph = x[:, 2]
    vr = x[:, 3]; vth = x[:, 4]; vph = x[:, 5]
    return -2. *vth *constants.THETA*constants.T *(constants.W +constants.PHI *vph/constants.T) *torch_cot(constants.THETA*th)/constants.PHI \
           -2. *constants.T *vr *(constants.W + constants.PHI * vph/constants.T)/(r*constants.PHI)


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
                

def config_training(constants, option=""):
    # determine the scale of p_net
    scale = p_init(constants, [constants.N_MEAN_I]).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32); print(scale_torch)

    configuration = {
        "constants": constants,
        "iterations": 60000,
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
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)

    if option == "pinn-xl_bias":
        configuration["training_fcn"] = PINN.train_pinn
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
        configuration["sample_res"] = constants.sample_res_points_bias
        
    # NOTE: after testing, 0.5 bias is better
    # if option == "pinn-xl_bias-test":
    #     configuration["training_fcn"] = PINN.train_pinn
    #     p_net = PNet_XL(constants, scale=scale_torch, input_feature=7)
    #     configuration["bias_fac"] = 0.1
    #     configuration["sample_res"] = partial(constants.sample_res_points_bias, 
    #         bias_fac=configuration["bias_fac"])
        
    # if option == "pinn-gmm":
    #     configuration["training_fcn"] = PINN.train_pinngmm
    #     p_net = TimeToGMM_V0(constants, K=11, alpha_floor=0.01)

    if option == "pinn-gmm_bias":
        configuration["training_fcn"] = PINN.train_pinngmm
        configuration["bias_fac"] = 0.5
        p_net = TimeToGMM_V0(constants, K=11, alpha_floor=0.01)

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
    parser.add_argument("--train", type=int, default=0, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
