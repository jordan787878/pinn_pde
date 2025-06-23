import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata

from utilities.util import *
from utilities.constants import Case2_4D_Constants

# Compute Time 
# 1e+5/
# MC time (sec):  [18.44 19.04 18.87 18.73 18.7  19.14 18.99 19.04 19.07 19.11 18.91]

# 1e+8/
# [10985.15 10279.26 10203.24 10229.59 10386.13 17258.86*]

constants = Case2_4D_Constants()
np.random.seed(0)


def p_sol_monte(t=0.0, linespace_num=51, stat_sample=10000000):
    global constants
    X_four_dim = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)

    # append constant theta' = 0.5pi/THETA, theta'_dot = 0.0
    X = np.zeros((stat_sample, 6))
    X[:,0] = X_four_dim[:,0]
    X[:,1] = np.ones((stat_sample,)) * 0.5 * np.float32(np.pi) / constants.THETA
    X[:,2] = X_four_dim[:,1]
    X[:,3] = X_four_dim[:,2]
    X[:,5] = X_four_dim[:,3]
    
    # convert rns to cartesian
    X_cart = sphere_to_cartesian(rnsphere_to_sphere(X, constants.TI, constants), reduced_to_four_dim=True)
    for i in tqdm(range(stat_sample), desc="Processing samples"):
        _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(X_cart[i,:], constants.MU_EARTH)
        _m = true_to_mean_anomaly(_e, _nu)
        _m = _m + np.sqrt(constants.MU_EARTH/_a**3) * constants.T * t
        _nu = solve_kepler(_e, _m)
        x_cart_t = COE2RV(_a, _e, _i, _RAAN, _w, _nu, _lonper, constants.MU_EARTH)
        x_t = sphere_to_rnsphere(cartesian_to_sphere(x_cart_t.reshape(1,-1)), t, constants)
        # update state
        X[i,:] = x_t
    # extract 4D data (0,2,3,5 columns)
    X = X[:, [0, 2, 3, 5]]
    
    # Define bins for each dimension
    bins_x1 = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x2 = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x3 = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x4 = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)

    # Digitize to find bin indices for each dimension
    bin_indices_x1 = np.digitize(X[:, 0], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[:, 1], bins_x2) - 1
    bin_indices_x3 = np.digitize(X[:, 2], bins_x3) - 1
    bin_indices_x4 = np.digitize(X[:, 3], bins_x4) - 1  

    # Initialize frequency array for 4D
    frequency_4d = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1, len(bins_x3) - 1, len(bins_x4) - 1)).astype(np.float32)

    # Count occurrences in each bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        idx_x1 = bin_indices_x1[i]
        idx_x2 = bin_indices_x2[i]
        idx_x3 = bin_indices_x3[i]
        idx_x4 = bin_indices_x4[i]

        # Check if the indices are valid
        if (0 <= idx_x1 < frequency_4d.shape[0] and
            0 <= idx_x2 < frequency_4d.shape[1] and
            0 <= idx_x3 < frequency_4d.shape[2] and
            0 <= idx_x4 < frequency_4d.shape[3]):
            frequency_4d[idx_x1, idx_x2, idx_x3, idx_x4] += 1
            # print("bin into: ", idx_x1, idx_x2, idx_x3, idx_x4)

    # Normalize the frequency to get the probability density
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    dx3 = bins_x3[1] - bins_x3[0]
    dx4 = bins_x4[1] - bins_x4[0]
    frequency_4d /= (dx1 * dx2 * dx3 * dx4 * stat_sample) 
    # NOTE: I do not normalize the p with "r"*dr*dphi*dr_dot*dphi_dot. Instead, a simple version is used.

    # Check the sum of the probability density function
    print("[check] sum pdf(monte) = 1.0", np.sum(frequency_4d) * dx1 * dx2 * dx3 * dx4)

    # Calculate the midpoints for bins (optional, depending on your needs)
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
    midpoints_x4 = (bins_x4[:-1] + bins_x4[1:]) / 2
    return midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4, frequency_4d


def p_init(x):
    """
    x is numpy array of shape (N x 4), N is the sample size
    """
    global constants
    pdf_func = multivariate_normal(mean=constants.N_MEAN_I, cov=constants. N_COV_I)
    pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
    return pdf_eval


def get_p_init_max(data_folder):
    x1s = np.load(data_folder+"x1s.npy")
    x2s = np.load(data_folder+"x2s.npy")
    x3s = np.load(data_folder+"x3s.npy")
    x4s = np.load(data_folder+"x4s.npy")
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    joint_pdf_true = p_init(grid_points)
    p_init_max = np.max(joint_pdf_true)
    return p_init_max


def test_monte_accuracy():
    """
    marginalize the joint pdf to a single coordinate, and compare it to the true analytical pdf at init time
    the true analytical pdf is obtained by 2 ways (but equivalent)
    1. using the univariate normal
    2. first compute the joint pdf (on the meshgrid) using multivariate normal, then marginalize
    """
    set_publication_style()
    global constants
    print("[test] pdf(monte) accuracy at t=0")

    x1s = np.load(DATA_FOLDER+"x1s.npy")
    x2s = np.load(DATA_FOLDER+"x2s.npy")
    x3s = np.load(DATA_FOLDER+"x3s.npy")
    x4s = np.load(DATA_FOLDER+"x4s.npy")
    pdf = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(0.0))
    print("x1s, pdf(monte) data type: ", x1s.dtype, pdf.dtype)

    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    joint_pdf_true = p_init(grid_points).reshape(x1_grid.shape)
    print("x1 ranges type: ", x1s.dtype)
    print("true joint pdf shape, type: ", joint_pdf_true.shape, joint_pdf_true.dtype)

    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]

    fig, axs = plt.subplots(2, 2, figsize=(8,6))
    for j in range(4):
        if(j == 0):
            marginalize_pdf = np.sum(pdf, axis=(1,2,3)) * dx2 * dx3 * dx4
            marginalize_pdf_true = norm.pdf(x1s, loc=constants.N_MEAN_I[0], scale=constants._N_COV_I[0,0]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(1,2,3)) * dx2 * dx3 * dx4
            ax = axs[0,0]
            x_axis = x1s
            ax.set_xlabel(r"$r'$")
            ax.set_ylabel(r"$p(r',t_0)$")
        elif(j == 1):
            marginalize_pdf = np.sum(pdf, axis=(0,2,3)) * dx1 * dx3 * dx4
            marginalize_pdf_true = norm.pdf(x2s, loc=constants.N_MEAN_I[1], scale=constants._N_COV_I[1,1]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,2,3)) * dx1 * dx3 * dx4
            ax = axs[1,0]
            x_axis = x2s
            ax.set_xlabel(r"$\phi'$")
            ax.set_ylabel(r"$p(\phi',t_0)$")
        elif(j == 2):
            marginalize_pdf = np.sum(pdf, axis=(0,1,3)) * dx1 * dx2 * dx4
            marginalize_pdf_true = norm.pdf(x3s, loc=constants.N_MEAN_I[2], scale=constants._N_COV_I[2,2]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,1,3)) * dx1 * dx2 * dx4
            ax = axs[0,1]
            x_axis = x3s
            ax.set_xlabel(r"$v'_r$")
            ax.set_ylabel(r"$p(v'_r,t_0)$")
        else:
            marginalize_pdf = np.sum(pdf, axis=(0,1,2)) * dx1 * dx2 * dx3
            marginalize_pdf_true = norm.pdf(x4s, loc=constants.N_MEAN_I[3], scale=constants._N_COV_I[3,3]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,1,2)) * dx1 * dx2 * dx3
            ax = axs[1,1]
            x_axis = x4s
            ax.set_xlabel(r"$v_{\phi}'$")
            ax.set_xlabel(r"$p(v_{\phi}', t_0)$")
        # ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical (marginal)")
        ax.plot(x_axis, pdf_test, "k--", label="analytical", linewidth=2, alpha=0.7)
        ax.plot(x_axis, marginalize_pdf, "b:", label="monte", marker='o', markersize=4, linewidth=1, alpha=0.8)
        ax.legend()
    # plt.show()
    # Save the figure as a high-quality PDF file
    plt.savefig("figs/figure.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def test_monte_spherical_pdf_Nrphi(data_folder):
    """
    marginalize the pdf of normalize spherical to [r,phi]
    """
    set_publication_style()
    global constants
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(monte) marginalized to rphi at t=", t_prime)

        x1s = np.load(data_folder+"x1s.npy")
        x2s = np.load(data_folder+"x2s.npy")
        x3s = np.load(data_folder+"x3s.npy")
        x4s = np.load(data_folder+"x4s.npy")
        pdf_monte = np.load(data_folder+"pdf_t{:.3f}.npy".format(t_prime))
        print("[check] x1s, pdf(monte) data type: ", x1s.dtype, pdf_monte.dtype)

        samples = np.load(data_folder+"samples_t{:.3f}.npy".format(t_prime))
        r_samples = samples[:,0]
        phi_samples = samples[:,2]

        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        pdf_monte_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        # for i in range(len(x1_grid)):
        #     for j in range(len(x2_grid)):
        #         print(x1_grid[i,j], x2_grid[i,j], pdf_monte_Nrphi[i,j])

        # Plotting the contour plot
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(x1_grid, x2_grid, pdf_monte_Nrphi, levels=30, cmap="viridis", alpha=0.8)
        # Adding color bar
        plt.colorbar(cp)
        # scatter samples of (r, phi) on to the plot
        plt.scatter(r_samples, phi_samples, s=8, c='white', edgecolor='black', alpha=1.0, label='Samples')
        # Adding labels and title
        plt.xlabel(r"$r'$")
        plt.ylabel(r"$\phi'$")
        plt.title(r"$p(r',\phi')$ from MC and 200 Samples at t="+str(np.round(t_prime,2))+"T")
        plt.legend()
        plt.show()


def test_monte_cartesian_pdf_xy():
    """
    convert the normalize spherical pdf to pdf(x,y)
    the contour plot is not exact, since we use interpolation to create x,y grid and p(x,y) on this grid
    """
    set_publication_style()
    global constants
    # Create the contour plot
    plt.figure(figsize=(8, 6))
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(monte) marginalized to xy at t=", t_prime)

        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")
        pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))
        print("[check] x1s, pdf(monte) data type: ", x1s.dtype, pdf_monte.dtype)

        dx3 = x3s[1] - x3s[0]
        dx4 = x4s[1] - x4s[0]

        pdf_monte_Nrphi =  np.sum(pdf_monte, axis=(2,3)) * dx3 * dx4
        x1_grid, x2_grid =  np.meshgrid(x1s, x2s, indexing="ij") # the indexing is very important
        pdf_rphi_data = np.empty((0,4))
        for i in range(len(x1_grid)):
            for j in range(len(x2_grid)):
                Nr = x1_grid[i,j]
                Nphi = x2_grid[i,j]
                pdf_Nrphi = pdf_monte_Nrphi[i,j]
                r = Nr*constants.R
                phi = Nphi*constants.PHI + constants.W*constants.T*t_prime
                pdf_rphi = (pdf_Nrphi/(constants.R*constants.PHI))/r**2
                t = constants.T*t_prime
                pdf_rphi_data = np.vstack((pdf_rphi_data, np.array([r, phi, t, pdf_rphi])))

        # Coordinates (x and y) and values (z)
        x = pdf_rphi_data[:, 0] * np.cos(pdf_rphi_data[:, 1])  # x-coordinates (1st column)
        y = pdf_rphi_data[:, 0] * np.sin(pdf_rphi_data[:, 1])  # y-coordinates (2nd column)
        z = pdf_rphi_data[:, 3]  # values to plot (4th column)
        print(np.max(z))
        # Define the grid where you want to plot the contours
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),  # Adjust 100 to get finer resolution
                                     np.linspace(y.min(), y.max(), 100))
        # Interpolate scattered data onto the grid
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        cp = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap="viridis", alpha=0.9)  # Filled contour plot
        # plt.colorbar(cp)  # Add a colorbar to indicate the values

        samples = np.load(DATA_FOLDER+"X_t{:.3f}.npy".format(t_prime))
        r_samples = samples[1:30,0]*constants.R
        phi_samples = samples[1:30,2]*constants.PHI + constants.W*constants.T*t_prime
        x_samples = r_samples * np.cos(phi_samples)
        y_samples = r_samples * np.sin(phi_samples)
        if(t_prime == 0.0):
            plt.scatter(x_samples, y_samples, s=3, c='white', linewidths=0.2, edgecolor='red', alpha=0.9, label='Samples')
        else:
            plt.scatter(x_samples, y_samples, s=3, c='white', linewidths=0.2, edgecolor='red', alpha=0.9)

    # Labels and title
    plt.axis('equal')
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.title('p(x,y) from MC and (30) Samples over 0.2T')
    plt.legend()
    plt.savefig("figs/figure1.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
      

def generate_data(data_folder, N_samples):
    mc_time = []
    t_span = constants.T_PRIME_SPAN
    print(t_span)

    for t_prime in t_span:
        start_time = time.time()
        x1s, x2s, x3s, x4s, pdf = p_sol_monte(t=t_prime, linespace_num=51, stat_sample=N_samples)   
        mc_time.append(time.time() - start_time)
        np.save(data_folder+"pdf_t{:.3f}.npy".format(t_prime), pdf)
        if t_prime == 0.0:
            np.save(data_folder+"x1s.npy", x1s)
            np.save(data_folder+"x2s.npy", x2s)
            np.save(data_folder+"x3s.npy", x3s)
            np.save(data_folder+"x4s.npy", x4s)
    print("MC time (sec): ", np.round(np.array(mc_time),2) )

    
def main():
    ######################
    ## Generate data
    ######################
    data_folder = "data/1e+6/"
    # generate_data(data_folder, 1000000)
    # exp_case2_generate_samples(data_folder, constants, use_j2=False, N_samples=300, dtt=1e-4)

    ######################
    ## Test MC results
    ######################
    # test_monte_accuracy()
    test_monte_spherical_pdf_Nrphi(data_folder)
    # test_monte_cartesian_pdf_xy()
    

if __name__ == "__main__":
    main()