import numpy as np
import torch
from scipy.stats import multivariate_normal
from monte import p_init, p_sol, print_mc_time
from exp_utilities.constants import Case1_6D_Constants_Equin
from exp_utilities.plot_util import plot_time_curves_3d, plot_pdf_metrics
from baseline_methods import SAVE_PATH_LINEAR_PROPAGATE, SAVE_PATH_UNSCENT_PROPAGATE, PropagationData
# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, load_trained_model
from _General.util import compute_volume, save_metrics_npz, load_metrics_npz
constants = Case1_6D_Constants_Equin()


def p_normal(x, mean, cov):
    """
    x is numpy array of shape (N x X_dim), N is the sample size
    """
    # print(constants.COV_I)
    # pdf_func = multivariate_normal(mean=constants.MEAN_I, cov=constants.COV_I)
    # pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    # return pdf_eval
    # (the following is more stable due to decoupling)
    pdf_eval = np.float32(1.0)
    for i in range(6):
        pdf_func_i = multivariate_normal(mean=mean[i], cov=cov[i,i])
        x_i = x[:,i].astype(np.float32)
        pdf_eval = pdf_eval * pdf_func_i.pdf(x_i).reshape(-1,).astype(x_i.dtype)
    return pdf_eval


def p_total_variation(constants, p1, p2):
    bounds = np.array([
        constants.X1_RANGE,
        constants.X2_RANGE,
        constants.X3_RANGE,
        constants.X4_RANGE,
        constants.X5_RANGE,
        constants.X6_RANGE,
    ])
    vol_est = compute_volume(bounds)
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    tv = 0.5 * np.mean(np.abs(p1-p2)) * vol_est
    return 100.*np.clip(tv, 0., 1.)


def p_rel_worst_error(p1, p2):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    # print(np.max(p1), np.max(p2))
    rel_error = np.max(np.abs(p1-p2)) / np.max(p2)
    return 100.*rel_error.item()


def compute_pdf_variations(constants, p_net, data_lp=None, data_ut=None, save_path=None):
    metrics = {
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_pinn": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_pinn": [],
        "t": data_lp.data["times"]
    } 
    N_samples = 10000000
    for t in metrics["t"]:
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
        _, _mu_lp, _cov_lp = data_lp.get(t)
        pdf_lp = p_normal(_x, _mu_lp, _cov_lp)
        _, _mu_ut, _cov_ut = data_ut.get(t)
        pdf_ut = p_normal(_x, _mu_ut, _cov_ut)

        rel_error_pinn = p_rel_worst_error(pdf_pinn, pdf_sol)
        rel_error_lp = p_rel_worst_error(pdf_lp, pdf_sol)
        rel_error_ut = p_rel_worst_error(pdf_ut, pdf_sol)
        tv_pinn = p_total_variation(constants, pdf_pinn, pdf_sol)
        tv_lp = p_total_variation(constants, pdf_lp, pdf_sol)
        tv_ut = p_total_variation(constants, pdf_ut, pdf_sol)
        print("time {:.2f} total variation,  PINN: {:.3f} %,  LP: {:.3f} %,  UT: {:.3f} %".format(
            t, tv_pinn, tv_lp, tv_ut
            ))
        print("time {:.2f} worst rel. error, PINN: {:.3f} %,  LP: {:.3f} %,  UT: {:.3f} %".format(
            t, rel_error_pinn, rel_error_lp, rel_error_ut
        ))
        metrics["rel_error_lp"].append(rel_error_lp)
        metrics["rel_error_ut"].append(rel_error_ut)
        metrics["rel_error_pinn"].append(rel_error_pinn)
        metrics["tv_lp"].append(tv_lp)
        metrics["tv_ut"].append(tv_ut)
        metrics["tv_pinn"].append(tv_pinn)
    save_metrics_npz(metrics, save_path)


def compare_methods():
    global constants
    p_net = PNet(constants, input_feature=7)
    scale = np.load("data/pre_compute/p_init_max.npz")["value"]
    scale_torch = torch.tensor(scale, dtype=torch.float32)
    p_net.scale = scale_torch
    OUTPUT_PATH = "output/v0"
    p_net = load_trained_model(p_net, path=OUTPUT_PATH+"/p_net.pth"); p_net.eval()
    data_lp = PropagationData(SAVE_PATH_LINEAR_PROPAGATE)
    data_ut = PropagationData(SAVE_PATH_UNSCENT_PROPAGATE)
    
    metrics_path = "output/baseline_methods/metrics.npz"
    # compute_pdf_variations(constants, p_net, 
    #                        data_lp=data_lp, data_ut=data_ut, save_path=metrics_path)
    metrics = load_metrics_npz(metrics_path)
    
    # --- Visualize ---
    plot_pdf_metrics(metrics)

    for i in range(1, 7):
        # data_mc = np.load("data/1e+6/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
        data_sol = np.load("data/pre_compute/marginal_pdfsol_x"+str(i)+"_t_M.npz")
        data_pinn = np.load(OUTPUT_PATH+"/pre_compute/marginal_p_x"+str(i)+"_t_M.npz")
        plot_time_curves_3d(i, constants, data_sol, data_pinn, 
                            data_lp=data_lp, data_ut=data_ut, 
                            title="Marginal p(x"+str(i)+",t)")


def main():
    compare_methods()


if __name__ == "__main__":
    main()