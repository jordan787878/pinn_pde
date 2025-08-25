"""
Case 1 of the paper: Uncertainty propagation in orbital mechanics via tensor decomposition,
without J2 and process noise
x = [r, th, phi, vr, vth, vphi]
"""
import numpy as np
import torch
import argparse
from monte import p_init, p_init_scaled, p_sol, p_sol_scaled, print_mc_time
# from exp_utilities.plot_utilites import check_pdf_Nrphi, check_pdfnn_cartesian_wrt_monte, check_pdf_cartesian_wrt_samples, check_error_flatten
from exp_utilities.constants import Case1_6D_Constants_Equin
# from exp_utilities.plot_util import precompute_pdf_streaming, precompute_error, compute_marginals_over_time, plot_time_curves_3d
from baseline_methods import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, PropagationData
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, PNet_Scaled, PNet_XL, load_trained_model, init_weights_He
import _General.train_pinn as PINN


MC_FOLDER = "data/1e+6/"
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
  
def main():
    global constants
    # constants.test_printout()

    # Set a fixed seed for reproducibility
    torch.manual_seed(0); np.random.seed(0)
    
    # p_net = PNet_Scaled(constants, input_feature=7)
    # p_net.apply(init_weights_He)

    p_net = PNet_XL(constants, input_feature=7)

    _x_at_mean = constants.N_MEAN_I.copy()
    p_max = p_init_scaled(constants, _x_at_mean.reshape(-1, 6)).item()
    scale = p_max
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    print("p net scale: ", p_net.scale)

    # --- v0 --- with regularization and curriculum training to enable subsequent training of e1_net
    OUTPUT_PATH = "output/v0_scaled_T0.3"
    
    # --- base --- the most basic PINN training attempted to compare with standard MC
    # OUTPUT_PATH = "output/base"

    if(TRAIN_FLAG):
        configurations = {
            "ic_fcn": p_init_scaled,
            "diff_opt_fcn": diff_opt_scaled,
            "pnet_path": OUTPUT_PATH+"/p_net.pth",
            "pnet_path_inter": OUTPUT_PATH+"/p_net_"
        }
        # --- v0 ---
        # PINN.train_pinn_sol_v0(constants, p_net, configurations, iterations=10000, save_model=True)
        PINN.train_pinn_sol_v0_scaled_improved(constants, p_net, configurations, iterations=100000, save_model=True)
    
    # --- Load the best network after training ---   
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()

    # --- checking joint PDF against p_sol ---
    N_samples = 10000000

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
        print(100. *rel_error.item())

    # --- Pre-computation for plotting data ---
    # precompute_pdf_streaming(constants, p_net, OUTPUT_PATH) # pre-compute pdf on grid

    # for i in range(1,7): # pre-compute marginal pdf over time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         data_dir=OUTPUT_PATH,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdf_t{:.3f}.npy",
    #     )
    #     np.savez(OUTPUT_PATH+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    # --- Plots and Print-out ---
    # print_mc_time(MC_FOLDER) 

    # # This print-out justifies why it is difficult to validate the error bound that defines on joint PDF
    # # NOTE this illustrates the bottlneck of validating PINN method if an accurate joint PDF is not available (also mentioned in Sun and Kumar Pg. 19)
    # # This is also validated in terms of e1(t0) using p(t0) analy v.s. e1(t0) using p(t0) MC ins train_e1.py code
    # p_init_analy = np.load(MC_FOLDER+"pdf_analy_t0.000.npy")
    # p_init_mc = np.load(MC_FOLDER+"pdf_t0.000.npy")
    # rel_acc = np.max(np.abs(p_init_mc - p_init_analy))/np.max(np.abs(p_init_analy))
    # del p_init_mc
    # print("[check] rel. accuracy of MC   --> the deviation between p(t0) MC   and p(t0): {:.2f} %".format(
    #     100.0 * rel_acc.item()))
    # p_init_pinn = np.load(OUTPUT_PATH+"/pdf_t0.000.npy")
    # rel_acc_pinn = np.max(np.abs(p_init_pinn - p_init_analy))/np.max(np.abs(p_init_analy))
    # print("[check] rel. accuracy of PINN --> the deviation between p(t0) PINN and p(t0): {:.2f} %".format(
    #     100.0 * rel_acc_pinn.item()))
    # del p_init_pinn; del p_init_analy

    # This print-out justifies why we visualize the marginal PDF of MC vs PINN even if the joint PDF of MC is very inaccurate
    # NOTE: the fact that two joint PDFs can be very different but have similar marginal PDF is illustrated by: 
    # exp_utilities/test_joint_vs_marginal.py
    # NOTE: this is consistent with how RMS error is computed (on 2D marginal) in Sun and Kumar
    # data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    # data_us = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    # for i in range(1, 7):
    #     data_mc = np.load(MC_FOLDER+"pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     data_sol = np.load("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz")
    #     data_pinn = np.load(OUTPUT_PATH+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
    #     plot_time_curves_3d(i, constants, data_pinn, data_sol, 
    #                         data_lp=data_us, p_init=p_init, leg_txt=["p PINN", "p sol."], title="Marginal p(x"+str(i)+",t)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass 1 to train, 0 to use the pre-trained models")
    parser.add_argument("--train", type=int, required=True, help="train bool")
    args = parser.parse_args()
    TRAIN_FLAG = args.train
    main()
