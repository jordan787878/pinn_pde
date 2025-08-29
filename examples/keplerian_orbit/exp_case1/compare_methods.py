import os
import numpy as np
from exp_utilities.constants import Case1_6D_Constants
from exp_utilities.plot_util import plt, plot_full_corner
constants = Case1_6D_Constants()


def compare_corner_plots(data_mc=None, data_lp=None, data_ut=None, save_path=None, 
                         OUTPUT_PATH=None):
    global constants
    t_show = [constants.T_PRIME_SPAN[-1]]

    # N_samples = 1000000
    # X = get_uniform_Xsamples_numpy(N_samples=N_samples) # samples from random uniform
    # X_scaled = constants.scaled_x(X)
    # _x_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False)

    for idx, t in enumerate(t_show):
        print("\ntime {:.4f}".format(t))
        if(data_mc is None):
            continue

        print("[info] ref")
        # load samples from MC
        filename_mc = data_mc+"xsamples_t{:.3f}.npy".format(t)
        if(os.path.exists(filename_mc)):
            X_ref = np.load(filename_mc)
        else:
            continue

        # Full corner plot
        plot_full_corner(constants, t, X_ref, OUTPUT_PATH=OUTPUT_PATH)

        plt.show()


def main():
    global constants
    data_mc = "data/1e+6/"
    OUTPUT_PATH = "output/v0"
    compare_corner_plots(data_mc, OUTPUT_PATH=OUTPUT_PATH)


if __name__ == "__main__":
    main()
