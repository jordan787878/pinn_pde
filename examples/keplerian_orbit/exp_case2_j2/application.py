import numpy as np
import torch
import os
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
    tspan = options["t_span"]
    for t in tspan:
        if(options["show_pdf_gmm"]):
            s = options['label']
            num_components = int(s.split("gmmx",1)[1].split("_",1)[0])
            p_gmm = funcOpt.models.TorchGMM(constants, num_components=num_components)
            p_gmm_path = options["path_pdf_models"]+options["label"]
            file_path = p_gmm_path + "_t{:.3f}.pth".format(t)
            if os.path.exists(file_path):
                p_gmm.load_state_dict(torch.load(file_path))
                # with torch.no_grad():
                    # ws, mus, _ = p_gmm.get_gmm_paramters()
                    # print(np.sum(ws))
                    # x_at_mus = torch.tensor(mus)
                    # t_tensor = torch.full((x_at_mus.shape[0], 1), t)
                    # pdf_at_mus = p_gmm(x_at_mus)
                    # pinn_at_mus = p_net(x_at_mus, t_tensor).view(-1,)
                    # print("[check] max deviation of pinn and gmm: ", torch.abs(pdf_at_mus - pinn_at_mus).max())
                exp_plot.plot_pdf_gmm_wrt_pinn(constants, p_net, p_gmm, target_r, target_phi, t,
                                        #    save_plot_path="figs/case2_target_and_fo"
                                           )
            else:
                continue
        else:
            exp_plot.plot_pdf_gmm_wrt_pinn(constants, p_net, None, target_r, target_phi, t)


def set_target(constants, options):
    """
    convert the normalize spherical pdf_nn to pdf_nn(x,y)
    and compare it with respect to pdf_monte(x,y)
    the surface plot is not exact, since we use interpolation to create x,y grid and pdf_nn(x,y) on this grid
    """
    # specify a fixedtarget region in spherical coordinate
    if(options["target"] == "tar1"):
        # --- tar1 ---
        target_r = np.array([21.3, 21.8])*constants.R
        target_ph = np.array([-3.5, 1.5])*constants.PHI + constants.W*constants.T*(0.1)
    elif(options["target"] == "tar2"):
        # --- tar2 ---
        target_r = np.array([20.6, 20.8])*constants.R
        target_ph = np.array([-0.1, 0.1])*constants.PHI + constants.W*constants.T*(0.05)
    else:
        raise("targets in option is not specified")
    # print("[check] target region of Radius {:.2f} km and Translational Distance {:.2f} km".format(
        # (target_r[1]-target_r[0])/1000.0, 0.5*(target_r[1]+target_r[0])*(target_ph[1]-target_ph[0])/1000.0 ))
    return target_r, target_ph


def app1(constants, p_net, e1_net_seq1, e1_net_seq2, options):
    # target region in (r, phi)
    target_r, target_phi = set_target(constants, options)

    if(options['show_target']): 
        exp_plot.plot_target(constants, p_net, target_r, target_phi, options["mc_folder"]); return
    if(options['show_pdf']):
        show_pdf(constants, options, p_net, target_r, target_phi); return

    # Prob. (Event) when pdf are obtained by MC
    if(options["compute_prob_by_mc"]):
        result = app1_util.get_prob_MC(constants, options['t_span'], target_r, target_phi)
        np.save(options["path_prob"]+"mc.npy", result)

    # Prob. (Event) when pdf are obtained using PINN + B1(possibly continuous)
    if(options["run_solver"]):
        networks = (p_net, e1_net_seq1, e1_net_seq2)
        app1_util.get_prob_PINN(constants, options['t_span'], target_r, target_phi, networks, options)
    

def show_plot(show_meta=True, show_pdf=False, gmm_basis=16):
    # funcOpt.helpers.scenario_base_guarantee(4, 50000, 1e-7); return 
    # NOTE: although this approach gives good result for probabilistically satisfying the constraint, 
    # it does not provide any additional 'confidence' to our problem for computing probability of an event ...

    global constants
    # --- Application ---
    target = "tar2"
    options = {
        # solver 
               "target": target,
               "path_pdf_models": "data/app1/"+target+"/pdf_models/",
               "path_prob": "data/app1/"+target+"/prob/",
               "label": "fo_gmmx"+str(gmm_basis)+"_nres50_100k", #fo_gmmx16_nres50_100k
               "solver": "fo",
               "run_solver": False,
               "save_result": False,
               "compute_prob_by_mc": False, 
        # special options for FO solver
               "num_iterations": 100000,
               "label_parent_gmm": None,
               "use_trained_gmm": False,
               "show_loss_landscape": False,
        # others
               "show_app1": show_meta,
               "show_pdf": show_pdf,
               "show_pdf_gmm": show_pdf,
               "show_target": False,
        # mc_data and pinn path
               "mc_folder": "data/1e+6/",
               "pnet_path" : "output/v0/p_net.pth",
               "e1net_path" : "output/base/e1_net.pth",
               "e1net_path_seq1" : "output/v0/e1_net_seq1.pth",
               "e1net_path_seq2" : "output/v0/e1_net_seq2.pth"
               }

    # --- Visualization ---
    if(options["show_app1"]):
        # exp_plot.plot_app1(target); return
        exp_plot.plot_app1(target, 
                        #    save_plot_path="figs/casestudy2_app1_"+target
                           )
        return
    
    p_net = PNet(constants, scale=get_p_init_max(constants))
    p_net = load_trained_model(p_net, path=options["pnet_path"], method="new"); p_net.eval()
    e1_net = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    e1_net = load_trained_model(e1_net, path=options["e1net_path"], method="new"); e1_net.eval()
    # --- Define a t_span to evaluate Pr(Event) ---
    dt = 0.005
    options['t_span'] = np.arange(0.00, 0.08+dt, dt)

    # --- Run application ---
    app1(constants, p_net, e1_net, e1_net, options)
    return

    # --- Other Visualization ---
    # exp_plot.visual_phat_trainings(constants, p_net, options["mc_folder"], 
    #                                save_plots=True, save_plot_path="figs/visual_mc_cartesian.pdf"
    #                                )
    # exp_plot.visual_e1hat_training(constants, (p_net, e1_net_seq1, e1_net_seq2), DATA_FOLDER)
    # exp_plot.plot_app1_onlymc(constants, "data/app1/tar1/pr_mcs.npy")
    # # plot_train_loss(E1NET_PATH_SEQ2)


def meta_app():
    global constants
    target = "tar2"
    options = {
    # solver 
        "target": target,
        "path_pdf_models": "data/app1/"+target+"/pdf_models/",
        "path_prob": "data/app1/"+target+"/prob/",
        "label": None,
        "solver": "fo",
        "run_solver": True,
        "save_result": True,
        "compute_prob_by_mc": False, 
    # special options for FO solver
        "num_iterations": 50000,
        "label_parent_gmm": None,
        "use_trained_gmm": False,
        "show_loss_landscape": False,
    # others
        "show_app1": False,
        "show_pdf": False,
        "show_pdf_gmm": False,
        "show_target": False,
    # mc_data and pinn path
        "mc_folder": "data/1e+6/",
        "pnet_path" : "output/v0/p_net.pth",
        "e1net_path" : "output/base/e1_net.pth",
        "e1net_path_seq1" : "output/v0/e1_net_seq1.pth",
        "e1net_path_seq2" : "output/v0/e1_net_seq2.pth"
        }
    
    p_net = PNet(constants, scale=get_p_init_max(constants))
    p_net = load_trained_model(p_net, path=options["pnet_path"], method="new"); p_net.eval()
    e1_net = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    e1_net = load_trained_model(e1_net, path=options["e1net_path"], method="new"); e1_net.eval()

    dt = 0.005
    options['t_span'] = np.arange(0.00, 0.08+dt, dt)
    
    gmm_basis = [16, 32, 48, 64, 80]
    for i in range(len(gmm_basis)):
        options["label"] = "fo_gmmx"+str(gmm_basis[i])+"_nres50_100k"
        if(i > 0):
            options["label_parent_gmm"] = "fo_gmmx"+str(gmm_basis[i-1])+"_nres50_100k"
        app1(constants, p_net, e1_net, e1_net, options)


if __name__ == "__main__":
    meta_app()
    # show_plot(show_meta=False, show_pdf=True, gmm_basis=80)