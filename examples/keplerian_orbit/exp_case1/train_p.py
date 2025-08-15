"""
Case 1 of the paper: Uncertainty propagation in orbital mechanics via tensor decomposition,
without J2 and process noise
x = [r, th, phi, vr, vth, vphi]
"""
import numpy as np
import torch
import argparse
from monte import p_init, print_mc_time
# from exp_utilities.plot_utilites import check_pdf_Nrphi, check_pdfnn_cartesian_wrt_monte, check_pdf_cartesian_wrt_samples, check_error_flatten
from exp_utilities.constants import Case1_6D_Constants
from exp_utilities.plot_util import precompute_pdf_streaming, compute_marginals_over_time, plot_time_curves_3d
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, load_trained_model, init_weights_He
import _General.train_pinn as PINN


MC_FOLDER = "data/1e+5/"
TRAIN_FLAG = False
constants = Case1_6D_Constants()


def dyn_f1(x):
    return x[:,3]

def dyn_f2(x):
    return x[:,4]

def dyn_f3(x):
    return x[:,5]

def dyn_f4(x):
    global constants
    return x[:,0] *constants.THETA**2 *x[:,4]**2 \
           + constants.T**2 *x[:,0] *torch.sin(constants.THETA*x[:,1])**2 *(constants.W +constants.PHI *x[:,5]/constants.T)**2 \
           - constants.T**2 *constants.MU_EARTH/(constants.R**3 * x[:,0]**2)

def dyn_f5(x):
    global constants
    return -2.*x[:,4]*x[:,5]/x[:,0] \
           + constants.T**2 * torch.sin(2.*constants.THETA*x[:,1]) *(constants.W +constants.PHI *x[:,5]/constants.T)**2 /(2. *constants.THETA)

# helper for cot()
def torch_cot(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tan(x)

def dyn_f6(x):
    global constants
    return -2. *x[:,4] *constants.THETA*constants.T *(constants.W +constants.PHI *x[:,5]/constants.T) *torch_cot(constants.THETA*x[:,1])/constants.PHI \
           -2. *constants.T*x[:,3]*(constants.W + constants.PHI * x[:,5]/constants.T)/(x[:,0]*constants.PHI)

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
                

def main():
    global constants
    # Set a fixed seed for reproducibility
    torch.manual_seed(0); np.random.seed(0)
    p_net = PNet(constants, input_feature=7)
    p_net.apply(init_weights_He)
    scale = np.load(MC_FOLDER+"pre_compute/p_init_max.npz")["value"]
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    print("p net scale: ", p_net.scale)

    # --- v0 --- with regularization and curriculum training to enable subsequent training of e1_net
    # PNET_PATH = "output/v0/p_net.pth"
    # PNET_INTER_PATH = "output/v0/p_net_"

    # --- base --- the most basic PINN training attempted to compare with standard MC
    PNET_PATH = "output/base"

    if(TRAIN_FLAG):
        configurations = {
            "ic_fcn": p_init,
            "diff_opt_fcn": diff_opt,
            "pnet_path": PNET_PATH+"/p_net.pth",
            "pnet_path_inter": PNET_PATH+"/p_net_"
        }
        # --- v0 ---
        # PINN.train_pinn_sol_v0(constants, p_net, configurations, iterations=50000, 
                            #    save_model=True,
                            #    beta_incre=0.005)
        
        # --- base ---
        PINN.train_pinn_sol_base(constants, p_net, configurations, iterations=50000, save_model=True, beta_incre=0.005)
    
    # --- Load the best network after training ---   
    p_net = load_trained_model(p_net, path=PNET_PATH+"/p_net.pth"); p_net.eval()

    # --- Pre-computation for plotting data ---
    # precompute_pdf_streaming(constants, p_net, PNET_PATH) # pre-compute pdf on grid
    # for i in range(1,7): # pre-compute marginal pdf over time
    #     x, t_kept, M = compute_marginals_over_time(
    #         times=constants.T_PRIME_SPAN,   # time array you mentioned
    #         pdf_dir=PNET_PATH,
    #         grid_dir="data/grids",
    #         keep_axis=(i-1),                # first dimension
    #         filename_fmt="pdf_t{:.3f}.npy",
    #         clip_negatives=True,
    #     )
    #     np.savez(PNET_PATH+"/marginal_data_x"+str(i)+"_t_M.npz", x=x, times=t_kept, M=M)

    # --- Plots and Print-out ---
    # print_mc_time(MC_FOLDER) 
    for i in range(1, 7):
        data_mc = np.load(MC_FOLDER+"pre_compute/marginal_data_x"+str(i)+"_t_M.npz")
        data_pinn = np.load(PNET_PATH+"/marginal_data_x"+str(i)+"_t_M.npz")
        plot_time_curves_3d(data_pinn, data_mc, title="Marginal p(x"+str(i)+"|t): curves (no interpolation)")

    # NOTE: not yet implemented 
    # check_pdf_Nrphi(constants, p_net=p_net)
    # check_pdfnn_cartesian_wrt_monte(constants, p_net, MC_FOLDER)
    # for t_prime in constants.T_PRIME_SPAN:
    #     check_error_flatten(constants, p_init, None, p_net, t_prime, MC_FOLDER)
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
