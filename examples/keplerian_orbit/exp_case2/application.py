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
"""
[check] Prob. by MC (time, Prob., Prob.)
[[0.     0.     0.    ]
 [0.01   0.     0.    ]
 [0.02   0.     0.    ]
 [0.03   0.     0.    ]
 [0.04   0.     0.    ]
 [0.05   0.     0.    ]
 [0.06   0.     0.    ]
 [0.07   0.     0.    ]
 [0.08   0.2188 0.2202]
 [0.09   0.2802 0.2792]
 [0.1    0.2939 0.2939]
 [0.11   0.1289 0.1292]
 [0.12   0.     0.    ]
 [0.13   0.     0.    ]
 [0.14   0.     0.    ]
 [0.15   0.     0.    ]
 [0.16   0.     0.    ]
 [0.17   0.     0.    ]
 [0.18   0.     0.    ]
 [0.19   0.     0.    ]
 [0.2    0.     0.    ]]
 """


def show_pdf(constants, options, p_net, target_r, target_phi):
    s = options['label']
    num_components = int(s.split("gmmx",1)[1].split("_",1)[0])
    p_gmm = funcOpt.models.TorchGMM(constants, num_components=num_components)
    p_gmm_path = options["path_pdf_models"]+options["label"]
    dt = 0.01
    tspan = np.arange(0.06, 0.11+dt, dt)
    for t in tspan:
        if(options["show_pdf_gmm"]):
            p_gmm.load_state_dict(torch.load(p_gmm_path+"_t{:.3f}.pth".format(t)))
            # print(p_gmm.logits)
            _, mus, _ = p_gmm.get_gmm_paramters()
            x_at_mus = torch.tensor(mus)
            pdf_at_mus = p_gmm(x_at_mus)
            print(pdf_at_mus)

            # print("[debug] gmm parameters: ", theta_gmm)
            exp_plot.plot_pdf_gmm_wrt_pinn(constants, p_net, p_gmm, target_r, target_phi, t)
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
        target_r = np.array([21.6, 21.8])*constants.R
        target_ph = np.array([-4.0, -3.5])*constants.PHI + constants.W*constants.T*(0.195)
    else:
        raise("targets in option is not specified")
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
    

def main():
    # funcOpt.helpers.scenario_base_guarantee(4, 50000, 1e-7); return 
    # NOTE: although this approach gives good result for probabilistically satisfying the constraint, 
    # it does not provide any additional 'confidence' to our problem for computing probability of an event ...

    global constants
    # --- Application ---
    target = "tar1"
    options = {
        # solver 
               "target": target,
               "path_pdf_models": "data/app1/"+target+"/pdf_models/",
               "path_prob": "data/app1/"+target+"/prob/",
               "label": "est_nres50",
               "solver": "est",
               "run_solver": True,
               "save_result": True,
        # special options for FO solver
               "num_iterations": 100000,
               "label_parent_gmm": None,
               "use_trained_gmm": False,
               "show_loss_landscape": False,
        # others
               "show_app1": True,
               "show_pdf": False,
               "show_pdf_gmm": False,
               "show_target": False,
               "compute_prob_by_mc": False, 
        # mc_data and pinn path
               "mc_folder": "data/1e+6/",
               "pnet_path" : "output/v0/p_net.pth",
               "e1net_path" : "output/base/e1_net.pth",
               "e1net_path_seq1" : "output/v0/e1_net_seq1.pth",
               "e1net_path_seq2" : "output/v0/e1_net_seq2.pth"
               }

    # --- Visualization ---
    if(options["show_app1"]):
        exp_plot.plot_app1(target, save_plot_path="figs/app1_"+target+"_new"); return
    
    p_net = PNet(constants, scale=get_p_init_max(constants))
    p_net = load_trained_model(p_net, path=options["pnet_path"], method="new"); p_net.eval()
    e1_net = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    e1_net = load_trained_model(e1_net, path=options["e1net_path"], method="new"); e1_net.eval()
    # e1_net_seq1 = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    # e1_net_seq1 = load_trained_model(e1_net_seq1, path=options["e1net_path_seq1"], method="new"); e1_net_seq1.eval()
    # e1_net_seq2 = E1Net(constants, scale=get_max_e1_init(constants, p_net))
    # e1_net_seq2 = load_trained_model(e1_net_seq2, path=options["e1net_path_seq2"], method="new"); e1_net_seq2.eval()


    # --- Define a t_span to evaluate Pr(Event) ---
    dt = 0.01
    options['t_span'] = np.arange(0.00, 0.20+dt, dt)

    # --- Run application ---
    app1(constants, p_net, e1_net, e1_net, options)

    # --- Other Visualization ---
    # exp_plot.visual_phat_trainings(constants, p_net, DATA_FOLDER, save_plots=False,
    #                                save_plot_path=None)
    # exp_plot.visual_e1hat_training(constants, (p_net, e1_net_seq1, e1_net_seq2), DATA_FOLDER)
    # exp_plot.plot_app1_onlymc(constants, "data/app1/tar1/pr_mcs.npy")
    # # plot_train_loss(E1NET_PATH_SEQ2)


if __name__ == "__main__":
    main()