import numpy as np
import torch
from monte import get_p_init_max, get_max_e1_init
from exp_utilities.constants import Case2_4D_Constants
import exp_utilities.plot_utilites as exp_plot
import exp_utilities.app1_utilities as app1_util
# import utilities
import sys
sys.path.insert(0, '../utilities/')
import FunctionalOpt.src as funcOpt
from _General.neuralnetworks import PNet, E1Net, load_trained_model
constants = Case2_4D_Constants()


def show_pdf(constants, options, p_net, target_r, target_phi):
    t_span = np.array([0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12])
    p_gmm = funcOpt.models.TorchGMM(constants)
    p_gmm_path = options["path_pdf_models"]+options["label"]
    for t in t_span:
        p_gmm.load_state_dict(torch.load(p_gmm_path+"_t{:.3f}.pth".format(t)))
        exp_plot.plot_pdf_gmm_wrt_pinn(constants, p_net, p_gmm, target_r, target_phi, t)


def app1(constants, p_net, e1_net_seq1, e1_net_seq2, options):
    # target region in (r, phi)
    target_r, target_phi = app1_util.set_target(constants)

    if(options['show_target']): 
        exp_plot.plot_target(constants, p_net, target_r, target_phi, options["mc_folder"])
    if(options['show_pdf']):
        show_pdf(constants, options, p_net, target_r, target_phi); return

    # Prob. (Event) when pdf are obtained by MC
    if(options["compute_prob_by_mc"]):
        app1_util.get_prob_MC(constants, options['t_span'], target_r, target_phi)

    # Prob. (Event) when pdf are obtained using PINN + B1(possibly continuous)
    networks = (p_net, e1_net_seq1, e1_net_seq2)
    app1_util.get_prob_PINN(constants, options['t_span'], target_r, target_phi, networks, options)
    

def main():
    global constants
    # --- Application ---
    options = {
        # solver
               "path_pdf_models": "data/app1/tar1/pdf_models/",
               "path_prob": "data/app1/tar1/prob/",
               "label": "pinnv0seq_diaggmmx64",
               "solver": "funcOpt",
        # others
               "show_app1": True,
               "show_pdf": False,
               "show_target": False,
               "mc_folder": "data/1e+6",
               "compute_prob_by_mc": False,
               "save_result": False,
        # pinn path
               "pnet_path" : "output/v0/p_net.pth",
               "e1net_path" : "output/v0/e1_net.pth",
               "e1net_path_seq1" : "output/v0/e1_net_seq1.pth",
               "e1net_path_seq2" : "output/v0/e1_net_seq2.pth"
               }

    # --- Visualization ---
    if(options["show_app1"]):
        exp_plot.plot_app1(save_plot_path="figs/pinnv0_diaggmmx64"); return
    p_net = PNet(constants, scale=get_p_init_max(constants))
    p_net = load_trained_model(p_net, path=options["pnet_path"], method="new"); p_net.eval()
    e1_net_seq1 = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    e1_net_seq1 = load_trained_model(e1_net_seq1, path=options["e1net_path_seq1"], method="new"); e1_net_seq1.eval()
    e1_net_seq2 = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    e1_net_seq2 = load_trained_model(e1_net_seq2, path=options["e1net_path_seq2"], method="new"); e1_net_seq2.eval()


    # --- Define a t_span to evaluate Pr(Event) ---
    dt = 0.01
    options['t_span'] = np.arange(0.08, 0.08+dt, dt)

    # --- Solve ---
    app1(constants, p_net, e1_net_seq1, e1_net_seq2, options)

    # --- Other Visualization ---
    # exp_plot.visual_phat_trainings(constants, p_net, DATA_FOLDER, save_plots=False,
    #                                save_plot_path=None)
    # exp_plot.visual_e1hat_training(constants, (p_net, e1_net_seq1, e1_net_seq2), DATA_FOLDER)
    # exp_plot.plot_app1_onlymc(constants, "data/app1/tar1/pr_mcs.npy")
    # # plot_train_loss(E1NET_PATH_SEQ2)


if __name__ == "__main__":
    main()