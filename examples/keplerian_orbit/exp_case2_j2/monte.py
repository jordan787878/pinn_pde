import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from constants import Case2_4D_Constants
from util import rnsphere_to_sphere, sphere_to_rnsphere, sphere_to_cartesian, cartesian_to_sphere, RV2COE, true_to_mean_anomaly, solve_kepler, COE2RV, set_publication_style

DATA_FOLDER = "data/"
constants = Case2_4D_Constants()
np.random.seed(0)


def p_sol_monte(t=0.0, linespace_num=51, stat_sample=10000):
    """
    Propagate stat_sample samples from the initial 4D Gaussian distribution
    (augmented to a 6D state) using the dynamics function dyn_normsph, and directly
    bin the final 4D states (columns 0, 2, 3, 5) without storing all samples.
    
    Parameters:
        t            : time at which to propagate the samples.
        linespace_num: number of bins along each dimension.
        stat_sample  : number of samples to generate.
        
    Returns:
        midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4: midpoints for the bins.
        frequency_4d : the 4D probability density function (normalized).
    """
    global constants

    # Define bins for each dimension.
    bins_x1 = np.linspace(constants.X1_RANGE[0], constants.X1_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x2 = np.linspace(constants.X2_RANGE[0], constants.X2_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x3 = np.linspace(constants.X3_RANGE[0], constants.X3_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)
    bins_x4 = np.linspace(constants.X4_RANGE[0], constants.X4_RANGE[1], num=linespace_num, endpoint=True).astype(np.float32)

    # Initialize the frequency array for the 4D pdf.
    frequency_4d = np.zeros((len(bins_x1)-1, len(bins_x2)-1, len(bins_x3)-1, len(bins_x4)-1), dtype=np.float32)

    # Time-stepping parameters for propagation.
    dtt = 1e-4
    kf = int(t / dtt)

    # Process each sample individually.
    for i in tqdm(range(stat_sample), desc="Processing samples"):
        # Generate one sample from the 4D Gaussian distribution.
        sample_4d = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I).astype(np.float32)
        
        # Build the 6D state from the 4D sample.
        # X[0] = sample_4d[0], X[1] = constant, X[2] = sample_4d[1], X[3] = sample_4d[2], X[5] = sample_4d[3]
        x = np.zeros(6, dtype=np.float32)
        x[0] = sample_4d[0]
        x[1] = 0.5 * np.float32(np.pi) / constants.THETA
        x[2] = sample_4d[1]
        x[3] = sample_4d[2]
        x[5] = sample_4d[3]
        
        # Propagate the sample using the dynamics (e.g., RK4 with J2 dynamics).
        for k in range(kf):
            x = x + dyn_normsph(x) * dtt  # dyn_normsph(x) is assumed to be defined elsewhere.
        
        # Extract the 4D state (columns 0, 2, 3, 5).
        x_final = np.array([x[0], x[2], x[3], x[5]], dtype=np.float32)
        
        # Determine the bin indices for each dimension.
        idx1 = np.digitize(x_final[0], bins_x1) - 1
        idx2 = np.digitize(x_final[1], bins_x2) - 1
        idx3 = np.digitize(x_final[2], bins_x3) - 1
        idx4 = np.digitize(x_final[3], bins_x4) - 1
        
        # Only update the frequency if the indices are within the valid range.
        if (0 <= idx1 < frequency_4d.shape[0] and
            0 <= idx2 < frequency_4d.shape[1] and
            0 <= idx3 < frequency_4d.shape[2] and
            0 <= idx4 < frequency_4d.shape[3]):
            frequency_4d[idx1, idx2, idx3, idx4] += 1

    # Normalize the frequency array to get the probability density.
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    dx3 = bins_x3[1] - bins_x3[0]
    dx4 = bins_x4[1] - bins_x4[0]
    frequency_4d /= (dx1 * dx2 * dx3 * dx4 * stat_sample)

    # Check that the integrated pdf is approximately 1.
    print("[check] sum pdf(monte) =", np.sum(frequency_4d) * dx1 * dx2 * dx3 * dx4)

    # Calculate midpoints for each bin.
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


def propagate_samples(t=0.2, dtt=1e-4, stat_sample=1):
    global constants
    X_four_dim = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)
    # append constant theta' = 0.5pi/THETA, theta'_dot = 0.0
    X = np.zeros((stat_sample, 6))
    X[:,0] = X_four_dim[:,0]
    X[:,1] = np.ones((stat_sample,)) * 0.5 * np.float32(np.pi) / constants.THETA
    X[:,2] = X_four_dim[:,1]
    X[:,3] = X_four_dim[:,2]
    X[:,5] = X_four_dim[:,3]
    # convert rns to sphere
    kf = int(t/dtt)
    for i in range(stat_sample):
        x = X[i, :]
        # ode45 propogate the X_sph(t)
        for k in range(kf):
            x = x + dyn_normsph(x) * dtt # + g*dw
        X[i, :] = x
    return X


def dyn_normsph(x):
    # normalized spherical coordinate dynamics
    global constants
    r = x[0]
    th = x[1]
    phi = x[2]
    vr = x[3]
    vth = x[4]
    vphi = x[5]
    f1 = vr
    f2 = 0.0 # (constants)
    f3 = vphi
    aux = constants.W + constants.PHI/constants.T * vphi
    f4 = constants.T**2 * r * aux**2 - constants.T**2 * constants.MU_EARTH /(constants.R**3 * r**2) + 2.0*(3*constants.T**2 * constants.J2 * constants.MU_EARTH * constants.R_EARTH**2)/(2*constants.R**5 * r**4)
    f5 = 0.0 # (constants)
    f6 = -2*constants.T/(r * constants.PHI) * vr * aux
    return np.array([f1, f2, f3, f4, f5, f6])


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


def test_monte_spherical_pdf_Nrphi():
    """
    marginalize the pdf of normalize spherical to [r,phi]
    """
    set_publication_style()
    global constants
    for t_prime in constants.T_PRIME_SPAN:
        print("[test] pdf(monte) marginalized to rphi at t=", t_prime)

        x1s = np.load(DATA_FOLDER+"x1s.npy")
        x2s = np.load(DATA_FOLDER+"x2s.npy")
        x3s = np.load(DATA_FOLDER+"x3s.npy")
        x4s = np.load(DATA_FOLDER+"x4s.npy")
        pdf_monte = np.load(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime))
        print("[check] x1s, pdf(monte) data type: ", x1s.dtype, pdf_monte.dtype)

        samples = np.load(DATA_FOLDER+"X_t{:.3f}.npy".format(t_prime))
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
        plt.title(r"$p(r',\phi')$"+ "from MC and 200 Samples at t="+str(np.round(t_prime,2))+"T")
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
      

def generate_data():
    mc_time = []
    for t_prime in constants.T_PRIME_SPAN:
        start_time = time.time()
        x1s, x2s, x3s, x4s, pdf = p_sol_monte(t=t_prime, linespace_num=51, stat_sample=10000)   
        mc_time.append(time.time() - start_time)
        np.save(DATA_FOLDER+"pdf_t{:.3f}.npy".format(t_prime), pdf)
        if t_prime == 0.0:
            np.save(DATA_FOLDER+"x1s.npy", x1s)
            np.save(DATA_FOLDER+"x2s.npy", x2s)
            np.save(DATA_FOLDER+"x3s.npy", x3s)
            np.save(DATA_FOLDER+"x4s.npy", x4s)
    print("MC time (sec): ", np.round(np.array(mc_time),2) )


def generate_samples():
    for t_prime in constants.T_PRIME_SPAN:
        X = propagate_samples(t_prime, stat_sample=200)
        np.save(DATA_FOLDER+"X_t{:.3f}.npy".format(t_prime), X)
    

def main():
    # # [Generate data] # #
    # generate_data()
    # generate_samples()

    # # [Testing functions] # #
    # test_monte_accuracy()

    test_monte_spherical_pdf_Nrphi()
    # test_monte_cartesian_pdf_xy()
    


if __name__ == "__main__":
    main()