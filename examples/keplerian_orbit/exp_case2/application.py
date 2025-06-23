import numpy as np

from utilities.constants import Case2_4D_Constants

from monte import get_p_init_max
from utilities.post_exp_cas2 import get_max_e1_init

import exp_utilities.plot_utilites as exp_plot
import exp_utilities.app1_utilities as app1_util

# import utilities
import sys
sys.path.insert(0, '../utilities/')
from _General.neuralnetworks import PNet, E1Net, load_trained_model


"""
select the NN models in output/
"""
PNET_PATH = "output/v2/p_net.pth"
E1NET_PATH_SEQ1 = "output/v2/e1_net_seq1.pth" # seq 1
E1NET_PATH_SEQ2 = "output/v2/e1_net_seq2.pth" # seq 2
DATA_FOLDER = "data/1e+6/"
constants = Case2_4D_Constants()


def app1(p_net, e1_net_seq1, e1_net_seq2):
    global constants
    
    # target region in (r, phi)
    target_r, target_phi = app1_util.set_target(constants, p_net, DATA_FOLDER, show_plot=False)

    # Define a t_span to evaluate Pr(Event)
    dt = 0.01
    t_span = np.arange(0.00, 0.20+dt, dt)
    # t_span = simple_interpolate(constants.T_PRIME_SPAN); t_span = simple_interpolate(t_span)

    # Prob. (Event) when pdf are obtained by MC
    # app1_util.get_prob_MC(constants, t_span, target_r, target_phi)

    # Prob. (Event) when pdf are obtained using PINN + B1(possibly continuous)
    networks = (p_net, e1_net_seq1, e1_net_seq2)
    app1_util.get_prob_PINN(constants, t_span, target_r, target_phi, networks)
    


def main():
    global constants

    p_net = PNet(constants, scale=get_p_init_max(DATA_FOLDER))
    p_net = load_trained_model(p_net, path=PNET_PATH, method="new"); p_net.eval()
    e1_net_seq1 = E1Net(constants, scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
    e1_net_seq1 = load_trained_model(e1_net_seq1, path=E1NET_PATH_SEQ1, method="new"); e1_net_seq1.eval()
    e1_net_seq2 = E1Net(constants, scale=get_max_e1_init(p_net, DATA_FOLDER, constants))
    e1_net_seq2 = load_trained_model(e1_net_seq2, path=E1NET_PATH_SEQ2, method="new"); e1_net_seq2.eval()

    ### Visualization ###
    # exp_plot.visual_phat_trainings(constants, p_net, DATA_FOLDER, save_plots=False,
    #                                save_plot_path=None)
    # exp_plot.visual_e1hat_training(constants, (p_net, e1_net_seq1, e1_net_seq2), DATA_FOLDER)
    # plot_train_loss(E1NET_PATH_SEQ2)
    # exp_plot.plot_app1_onlymc(constants, "data/app1/tar1/pr_mcs.npy")
    # exp_plot.plot_app1()
    # return

    ### Application ###
    app1(p_net, e1_net_seq1, e1_net_seq2)


if __name__ == "__main__":
    main()