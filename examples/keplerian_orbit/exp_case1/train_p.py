"""
Case 1 of the paper: Uncertainty propagation in orbital mechanics via tensor decomposition,
without J2 and process noise
x = [r, th, phi, vr, vth, vphi]
"""
import numpy as np
import torch
import argparse
from monte import p_init
from exp_utilities.constants import Case1_6D_Constants
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, PNet_XL_Sphere, load_trained_model
from _General.neuralnetworks import TimeToGMM6D
import _General.train_pinn as PINN


TRAIN_FLAG = False
constants = Case1_6D_Constants()


# --- differential operator ---
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
                

# --- training configuration ---
def config_training_PNet_XL_Sphere(constants, scale_torch):
    configuration = {
        "constants": constants,
        "iterations": 50000,
        "sample_ic": constants.sample_init_points,
        "sample_res": constants.sample_res_points,
        "p_ic": p_init,
        "res_func": diff_opt,
        "res_weight": 1.0,
        "save_path": "output/v0",
        "beta_incre": 0.02,
        "loss_normalize": scale_torch,
        "reg_tv": None,
    }
    return configuration


def config_training_TimeToGMM6D(constants, scale_torch, option=""):
    configuration = {
        "constants": constants,
        "iterations": 50000,
        "sample_ic": constants.sample_init_points,
        "sample_res": constants.sample_res_points,
        "sample_ic_uniform": constants.sample_init_points_uniform,
        "sample_res_uniform": constants.sample_res_points_uniform,
        "p_ic": p_init,
        "res_func": diff_opt,
        "res_weight": 1.0,
        "save_path": "output/pinn-gmm",
        "beta_incre": 0.02,
        "loss_normalize": scale_torch,
        "reg_tv": None,
        "training_fcn": PINN.train_pinngmm_sol_v0
        # "increased_icsamples_factor": 7, # increase the number of IC sample size
        # "increased_ressamples_factor": 7,
        # "beta_0": 1.0, # initialized to 1.0 if pre-trained pinn-gmm model is used
    }

    if(option == "no-importance-sampling"):
        configuration["save_path"] = "output/pinn-gmm(no-imp)"
        configuration["training_fcn"] = PINN.train_pinn_sol_v0

    if(option == "uniform-sampling"):
        configuration["save_path"] = "output/pinn-gmm(uniform)"
        configuration["training_fcn"] = PINN.train_pinngmm_sol_uniform

    return configuration


# --- post-training checking ---
def check_pnet_against_pinit(p_net):
    # --- checking joint PDF against p_init ---
    N_samples = 1000000

    # x points
    _x = np.column_stack([
        np.random.uniform(constants.X1_RANGE[0], constants.X1_RANGE[1], N_samples),
        np.random.uniform(constants.X2_RANGE[0], constants.X2_RANGE[1], N_samples),
        np.random.uniform(constants.X3_RANGE[0], constants.X3_RANGE[1], N_samples),
        np.random.uniform(constants.X4_RANGE[0], constants.X4_RANGE[1], N_samples),
        np.random.uniform(constants.X5_RANGE[0], constants.X5_RANGE[1], N_samples),
        np.random.uniform(constants.X6_RANGE[0], constants.X6_RANGE[1], N_samples),
    ])

    t = constants.T_PRIME_SPAN[0]

    pdf_sol = p_init(constants, _x).reshape(-1,)

    _x_tensor = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
    _t = np.ones((len(_x_tensor), 1)) * t
    _t_tensor = torch.tensor(_t, dtype=torch.float32, requires_grad=True).view(-1,1)
    pdf_pinn = p_net(_x_tensor, _t_tensor).detach().cpu().numpy().reshape(-1,)

    rel_error = np.max(np.abs(pdf_sol-pdf_pinn)) / np.max(pdf_sol) # NOTE this metric needs sufficient large samples.
    print("[check]: rel error at t0: {:.4f} %".format(100. *rel_error.item()))


# --- main ---
def main():
    global constants
    torch.manual_seed(0); np.random.seed(0)  # Set a fixed seed for reproducibility

    # determine the scale of p_net
    scale = p_init(constants, [constants.N_MEAN_I]).item()
    scale_torch = torch.tensor(scale, dtype=torch.float32); print(scale_torch)

    # mlp
    # p_net = PNet_XL_Sphere(constants, scale=scale_torch)
    # configuration = config_training_PNet_XL_Sphere(constants, scale_torch)

    # pinn-gmm
    p_net = TimeToGMM6D(constants, K=11)
    # configuration = config_training_TimeToGMM6D(constants, scale_torch)
    # configuration = config_training_TimeToGMM6D(constants, scale_torch, option="no-importance-sampling")
    configuration = config_training_TimeToGMM6D(constants, scale_torch, option="uniform-sampling")

    if(TRAIN_FLAG):
        # mlp training
        # PINN.train_pinn_sol_v0(p_net, configuration)
        
        # pinn-gmm training
        configuration["training_fcn"](p_net, configuration)

    # --- Load the best network after training ---   
    p_net = load_trained_model(p_net, path=configuration["save_path"]+"/p_net.pth"); p_net.eval()
    
    # # --- Check PNet against p_init
    # check_pnet_against_pinit(p_net)


def obsolete_fcn():
    pass
    # --- Pre-computation for plotting data ---
    # precompute_pdf_streaming(constants, p_net, PNET_PATH) # pre-compute pdf on grid
    # for i in range(1,7): # pre-compute marginal pdf over time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir=PNET_PATH,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdf_t{:.3f}.npy",
    #     )
    #     np.savez(PNET_PATH+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    # --- Plots and Print-out ---
    # print_mc_time(MC_FOLDER) 

    # This print-out justifies why it is difficult to validate the error bound that defines on joint PDF
    # NOTE this illustrates the bottlneck of validating PINN method if an accurate joint PDF is not available (also mentioned in Sun and Kumar Pg. 19)
    # This is also validated in terms of e1(t0) using p(t0) analy v.s. e1(t0) using p(t0) MC ins train_e1.py code
    # p_init_analy = np.load(MC_FOLDER+"pdf_analy_t0.000.npy").reshape(-1,)
    # p_init_mc = np.load(MC_FOLDER+"pdf_t0.000.npy").reshape(-1,)
    # rel_acc = np.max(np.abs(p_init_mc - p_init_analy)) / np.max(np.abs(p_init_analy))
    # del p_init_mc
    # print("[check] rel. accuracy of MC   --> the deviation between p(t0) MC   and p(t0): {:.2f} %".format(
    #     100.0 * rel_acc.item()))
    # p_init_pinn = np.load(PNET_PATH+"/pdf_t0.000.npy").reshape(-1,)
    # rel_acc_pinn = np.max(np.abs(p_init_pinn - p_init_analy))/np.max(np.abs(p_init_analy))
    # print("[check] rel. accuracy of PINN --> the deviation between p(t0) PINN and p(t0): {:.2f} %".format(
    #     100.0 * rel_acc_pinn.item()))
    # del p_init_pinn; del p_init_analy

    # This print-out justifies why we visualize the marginal PDF of MC vs PINN even if the joint PDF of MC is very inaccurate
    # NOTE: the fact that two joint PDFs can be very different but have similar marginal PDF is illustrated by: 
    # exp_utilities/test_joint_vs_marginal.py
    # NOTE: this is consistent with how RMS error is computed (on 2D marginal) in Sun and Kumar
    # for i in range(1, 7):
    #     data_mc = np.load(MC_FOLDER+"pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     data_pinn = np.load(PNET_PATH+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(i, constants, data_pinn, data_mc, p_init=p_init, title="Marginal p(x"+str(i)+",t)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
