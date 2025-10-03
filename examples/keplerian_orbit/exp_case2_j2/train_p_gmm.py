"""
The first trial of keplerian orbit using rotating spherical coordiante and dynamics.
It is similar to the Case 2 of the paper: Uncertainty propagation in orbital mechanics via tensor decompostion.
Difference: do not have process noise, and J2 perturbation.
This case is a circular orbit on plannar motion. Hence, we reduce the dynamics to 4D.
Compared to exp2_sphere_4d (baseline):
    1) the final time is increased to 0.2*T.
    2) the solution domain is increased to ensure sum(p) ~= 1.0
"""
import numpy as np
import torch
import argparse
from monte import p_init, get_p_init_max, print_mc_time
from exp_utilities.plot_utilites import check_pinngmm_Nrphi, check_pinngmm_cartesian_wrt_monte, check_pdf_cartesian_wrt_samples, check_error_flatten
from exp_utilities.plot_utilites import check_error_flatten_new
from exp_utilities.constants import Case2_4D_Constants
from exp_utilities.classic_gmm import GMMWhitenedModel
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import TimeToGMM6D_V0, load_trained_model
import _General.train_pinn as PINN


MC_FOLDER = "data/1e+6/"
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

def diff_opt(x, t, p_net, beta=1.0, verbose=False):
    """
    differential operator with J2 and Brownian noise in velocity state:
    g*dw, where g = [0, 0, 0;
                     0, 0, 0;
                     0, 0, 0;
                     1, 0, 0;
                     0, 1, 0;
                     0, 0, 1]
    dw in R^3 with noise intensity diag([constants.N_Q_NOISE])
    As such the differential term becomes (in full 6D):
    0.5 * (Q[0] * P_x4x4 + Q[1] * P_x5x5 + Q[2] * P_x6x6)
    Note that this is a reduced 4D system: (x1,x2,x3,x4)_reduced = (x1,x3,x4,x6)_6D
    Hence: we have 0.5 * (Q[0] * P_x3x3 + Q[2] * P_x4x4) in the reduced 4d orbit
    """
    global constants
    output = p_net(x,t)
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
    f4_x = torch.autograd.grad(f4, x, grad_outputs=torch.ones_like(f4), create_graph=True)[0]
    f4_x4 = f4_x[:,3].view(-1,1)
    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(output_x.size(1)):
        grad2 = torch.autograd.grad(output_x[:, i], x, grad_outputs=torch.ones_like(output_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    output_xx = torch.stack(hessian, dim=-1)
    output_x4x4 = output_xx[:, 2, 2].view(-1,1)
    output_x6x6 = output_xx[:, 3, 3].view(-1,1)
    Q = torch.tensor(constants.N_Q_NOISE, dtype=torch.float32, requires_grad=False)
    residual = output_t + beta*(output_x1*f1 + output_x2*f2 + output_x3*f3 + output_x4*f4 + f4_x4*output + \
                                0.5*(Q[0]*output_x4x4 + Q[2]*output_x6x6))
    # print(residual.dtype)
    return residual


def config_training_TimeToGMM6D(constants, scale_torch, option=""):
    configuration = {
        "constants": constants,
        "iterations": 60000,
        "sample_ic": constants.sample_init_points,
        "sample_res": constants.sample_res_points,
        "sample_res_uniform": constants.sample_res_points_uniform,
        "p_ic": p_init,
        "res_func": diff_opt,
        "res_weight": 1.0,
        "save_path": "output/pinn-gmm",
        "beta_incre": 0.02,
        "loss_normalize": scale_torch,
        "reg_tv": None,
        "training_fcn": PINN.train_pinngmm_sol_v0
    }

    if(option == "V0"):
        configuration["save_path"] = "output/pinn-gmm(V0)"
    
    print(configuration["save_path"])

    return configuration


def main():
    global constants
    # Set a fixed seed for reproducibility
    torch.manual_seed(0); np.random.seed(0)

    scale = get_p_init_max(constants)
    scale_torch = torch.tensor(scale, dtype=torch.float32)

    p_net = TimeToGMM6D_V0(constants, K=11, D=4)
    configuration = config_training_TimeToGMM6D(constants, scale_torch, option="V0")


    if(TRAIN_FLAG):
        configuration["training_fcn"](p_net, configuration)

    # --- Load the best network after training ---   
    p_net = load_trained_model(p_net, path=configuration["save_path"]+"/p_net.pth"); p_net.eval()

    # --- Post-process ---
    # print_mc_time(MC_FOLDER)

    # check_pinngmm_Nrphi(constants, p_net_gmm=p_net)

    # check_pinngmm_cartesian_wrt_monte(constants, p_net, MC_FOLDER)
    
    # for t_prime in t_check:
    #     check_error_flatten(constants, p_init, None, p_net, t_prime, MC_FOLDER)

    # NOTE: use a fitted GMM PDF as 'true'
    t_check = [constants.T_PRIME_SPAN[-2]]
    for t_prime in t_check:
        gmm = GMMWhitenedModel.load("data/classic_gmm/gmm_whitened_t{:.2f}.npz".format(t_prime))
        check_error_flatten_new(constants, p_init, p_net, t_prime, MC_FOLDER, gmm)

    # check_pdf_cartesian_wrt_samples(constants, p_net=p_net)
    # --- (obsolete) ---
    # # check_pdfnn_marginalize(p_net, t=t_prime)
    # # test_nn_cartesian_pdf_xy(p_net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
