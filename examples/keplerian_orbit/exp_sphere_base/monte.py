import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from constants import Case2_4D_Constants
from util import rnsphere_to_sphere, sphere_to_rnsphere, sphere_to_cartesian, cartesian_to_sphere, RV2COE, true_to_mean_anomaly, solve_kepler, COE2RV

DATA_FOLDER = "data/"
constants = Case2_4D_Constants()
np.random.seed(0)


def p_sol_monte(t=0.0, linespace_num=100, stat_sample=100000):
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
    for i in range(stat_sample):
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


def get_p_init_max():
    x1s = np.load(DATA_FOLDER+"x1s.npy")
    x2s = np.load(DATA_FOLDER+"x2s.npy")
    x3s = np.load(DATA_FOLDER+"x3s.npy")
    x4s = np.load(DATA_FOLDER+"x4s.npy")
    x1_grid, x2_grid, x3_grid, x4_grid = np.meshgrid(x1s, x2s, x3s, x4s, indexing="ij") # the indexing is very important
    grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel()]).T
    joint_pdf_true = p_init(grid_points)
    p_init_max = np.max(joint_pdf_true)
    # print(p_init_max)
    return p_init_max


def test_monte_accuracy():
    """
    marginalize the joint pdf to a single coordinate, and compare it to the true analytical pdf at init time
    the true analytical pdf is obtained by 2 ways (but equivalent)
    1. using the univariate normal
    2. first compute the joint pdf (on the meshgrid) using multivariate normal, then marginalize
    """
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
        elif(j == 1):
            marginalize_pdf = np.sum(pdf, axis=(0,2,3)) * dx1 * dx3 * dx4
            marginalize_pdf_true = norm.pdf(x2s, loc=constants.N_MEAN_I[1], scale=constants._N_COV_I[1,1]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,2,3)) * dx1 * dx3 * dx4
            ax = axs[1,0]
            x_axis = x2s
        elif(j == 2):
            marginalize_pdf = np.sum(pdf, axis=(0,1,3)) * dx1 * dx2 * dx4
            marginalize_pdf_true = norm.pdf(x3s, loc=constants.N_MEAN_I[2], scale=constants._N_COV_I[2,2]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,1,3)) * dx1 * dx2 * dx4
            ax = axs[0,1]
            x_axis = x3s
        else:
            marginalize_pdf = np.sum(pdf, axis=(0,1,2)) * dx1 * dx2 * dx3
            marginalize_pdf_true = norm.pdf(x4s, loc=constants.N_MEAN_I[3], scale=constants._N_COV_I[3,3]**0.5)
            pdf_test = np.sum(joint_pdf_true, axis=(0,1,2)) * dx1 * dx2 * dx3
            ax = axs[1,1]
            x_axis = x4s
        ax.plot(x_axis, marginalize_pdf_true, "black", label="analytical (marginal)")
        ax.plot(x_axis, pdf_test, "g:", label="analytical (from joint)")
        ax.plot(x_axis, marginalize_pdf, "b--", label="monte")
        ax.legend()
    plt.show()


# def test_propagate_using_oe(t_prime=0.0):
#     global constants
#     X = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=1).astype(np.float32)
#     # normalize spherical 4d to cartesian 6d
#     X_cart = normspherical4d_to_cartesian(X, 0.0, constants)
#     for i in range(1):
#         _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(X_cart[i,:], constants.MU_EARTH)
#         _m = true_to_mean_anomaly(_e, _nu)
#         print(X[i,:])
#         print(X_cart[i,:])
#         print(_a, _e, _i, _RAAN, _w, _nu, _lonper)
#         print(_m)
#         _m = _m + np.sqrt(constants.MU_EARTH/_a**3) * constants.T * t_prime
#         _nu = solve_kepler(_e, _m)
#         x_cart_t = COE2RV(_a, _e, _i, _RAAN, _w, _nu, _lonper, constants.MU_EARTH)
#         print(_m, _nu)
#         print(x_cart_t)
#         x_t = cartesian_to_normspherical4d(x_cart_t.reshape(1,-1), t_prime, constants)
#         print(x_t)


def test_monte_spherical_pdf_Nrphi():
    """
    marginalize the pdf of normalize spherical to [r,phi]
    """
    global constants
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(monte) marginalized to rphi at t=", t_prime)

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
        # for i in range(len(x1_grid)):
        #     for j in range(len(x2_grid)):
        #         print(x1_grid[i,j], x2_grid[i,j], pdf_monte_Nrphi[i,j])

        # Plotting the contour plot
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(x1_grid, x2_grid, pdf_monte_Nrphi, levels=30, cmap="viridis")
        # Adding color bar
        plt.colorbar(cp)
        # Adding labels and title
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Contour plot of pdf_monte_rphi_prime')
        plt.show()


def test_monte_cartesian_pdf_xy():
    """
    convert the normalize spherical pdf to pdf(x,y)
    the contour plot is not exact, since we use interpolation to create x,y grid and p(x,y) on this grid
    """
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
        cp = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap="viridis")  # Filled contour plot
        # plt.colorbar(cp)  # Add a colorbar to indicate the values
    # Labels and title
    plt.axis('equal')
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.title('pdf_monte(x,y) over T='+str(constants.TF)+' sec.')
    # Show the plot
    plt.show()

      

def main():
    # Generate data
    # for t_prime in constants.T_PRIME_SPAN:
    #     x1s, x2s, x3s, x4s, pdf = p_sol_monte(t=t_prime, linespace_num=51, stat_sample=100000)   
    #     np.save(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime), pdf)
    #     if t_prime == 0.0:
    #         np.save(DATA_FOLDER+"x1s.npy", x1s)
    #         np.save(DATA_FOLDER+"x2s.npy", x2s)
    #         np.save(DATA_FOLDER+"x3s.npy", x3s)
    #         np.save(DATA_FOLDER+"x4s.npy", x4s)

    # # Testing functions
    test_monte_accuracy()

    # # [obsolete] test_propagate_using_oe(t_prime=constants.TF/constants.T)

    test_monte_spherical_pdf_Nrphi()

    test_monte_cartesian_pdf_xy()
    


if __name__ == "__main__":
    main()