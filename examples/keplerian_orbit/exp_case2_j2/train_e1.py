import numpy as np
import torch
import argparse
from functools import partial
from monte import p_init
from train_p import diff_opt
from exp_utilities.constants import Case2_4D_Constants

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.neuralnetworks import PNet_XL, ENet_XL, TimeToGMM_V0, load_trained_model
import utilities._General.train_pinn as PINN
from utilities._General.util import RunLogger, save_config_human


TRAIN_FLAG = False
constants = Case2_4D_Constants()


def config_training_ENet_XL(constants, fac=0.02, option=""):
    # determine the scale of p_net
    scale = p_init(constants, constants.N_MEAN_I).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32); print(scale_torch)

    config = {
        "constants": constants,
        "iterations": 20000,
        "sample_ic": constants.sample_init_points,
        "sample_res": constants.sample_res_points_uniform,
        "p_ic": constants.p_init_torch,
        "res_func": diff_opt,
        "res_weight": 1.0,
        "save_path": "output/" + option,
        "beta_incre": 0.02,
        "loss_normalize": fac*scale_torch,
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
        "x_range": constants.NX_RANGE,
    }

    if option == "pinn-xl":
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=config["loss_normalize"], input_feature=5)
        config["training_fcn"] = PINN.train_pinn_error
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=5)
        p_net = load_trained_model(p_net, path=config["save_path"]+"/p_net.pth"); p_net.eval()

    if option == "pinn-xl_bias":
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=config["loss_normalize"], input_feature=5)
        config["training_fcn"] = PINN.train_pinn_error
        config["sample_res"] = constants.sample_res_points_bias
        p_net = PNet_XL(constants, scale=scale_torch, input_feature=5)
        p_net = load_trained_model(p_net, path=config["save_path"]+"/p_net.pth"); p_net.eval()

    if option == "pinn-gmm_bias":
        e1_net = ENet_XL(constants, scale=scale_torch, normalize=config["loss_normalize"], input_feature=5)
        config["training_fcn"] = PINN.train_pinn_error
        config["bias_fac"] = 0.5
        p_net = TimeToGMM_V0(constants, D=4, K=11, alpha_floor=0.01)
        p_net = load_trained_model(p_net, path=config["save_path"]+"/p_net.pth"); p_net.eval()

    return config, p_net, e1_net


def main():
    global constants #; constants.test_printout()

    # Set a fixed seed for reproducibility
    torch.manual_seed(0); np.random.seed(0)

    # Setup config
    fac = 0.02
    config, p_net, e1_net = config_training_ENet_XL(constants, fac=fac, option="pinn-xl")
    save_config_human(config, model_name="e1_net")

    # Train & log
    if(TRAIN_FLAG):
        log_file = Path(config["save_path"]) / "e1_net-train.log"
        with RunLogger(log_file, tee=True):      # tee=False if you want file-only
            networks = (p_net, e1_net)
            config["training_fcn"](networks, config)
    
    # Load the best network after training
    e1_net = load_trained_model(e1_net, path=config["save_path"]+"/e1_net.pth"); e1_net.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, default=False, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()