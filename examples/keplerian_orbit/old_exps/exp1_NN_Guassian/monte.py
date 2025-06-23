import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
from scipy.stats import skew, kurtosis

DATA_FOLDER = "data/"

# Constants
pi = np.float32(np.pi)
n_d = 2
a_0 = np.float32(8000.0)
lam_0 = np.float32(0.0)
mu_0 = np.float32(np.array([a_0, lam_0]).reshape(2,))
cov_0 = np.float32(np.array([[250.0**2, 0.0], [0.0, (pi/9)**2]]))
x1_low = np.float32(7000.0)
x1_hig = np.float32(9000.0)
x2_low = np.float32(-2.0 * pi)
x2_hig = np.float32(28.0)
ti = np.float32(0.0)
tf = np.float32(6.0 * 3600)
t1s = np.float32([0.0, 1.*3600.0, 2.*3600.0, 3.*3600.0, 4.*3600.0, 5.*3600.0, 6.*3600.0])
mu_gravity = np.float32(398600.4418)
t_orbit = np.float32(2 * pi * np.sqrt(a_0**3 / mu_gravity))
print("Nominal orbit period: ", t_orbit)


# other fixed orbit elements
ecc = np.float32(0.15)
inc = np.float32(pi/3.0)
RAAN = np.float32(0.0)
w = np.float32(0.0)


# [testing] initial beta distribution
def draw_scaled_beta_samples(alpha, beta, x1_low, x1_hig, num_samples=1000):
    # Create a random generator
    rng = np.random.default_rng()
    # Draw samples from the Beta distribution
    samples = rng.beta(alpha, beta, size=num_samples)
    # Scale the samples to the desired range
    scaled_samples = samples * (x1_hig - x1_low) + x1_low
    # Plot the histogram of the scaled samples
    # plt.figure(figsize=(10, 6))
    # plt.hist(scaled_samples, bins=30, density=True, alpha=0.6, color='b')
    # # Plot the theoretical PDF of the scaled Beta distribution
    # x = np.linspace(x1_low, x1_hig, 100)
    # pdf = ( (x - x1_low) / (x1_hig - x1_low) )**(alpha - 1) * (1 - (x - x1_low) / (x1_hig - x1_low))**(beta - 1)
    # pdf /= np.math.gamma(alpha) * np.math.gamma(beta) / np.math.gamma(alpha + beta) * (x1_hig - x1_low)
    # plt.plot(x, pdf, 'r-', lw=2, label='Scaled Beta PDF')
    # plt.title(f'Scaled Beta Distribution (α={alpha}, β={beta})')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # _mean = np.mean(scaled_samples)
    # _var  = np.var(scaled_samples)
    # print(_mean, _var)
    return scaled_samples


def p_sol_monte(t1=ti, linespace_num=50, stat_sample=10000000):

    # initial distribution (normal)
    # X = np.float32(np.zeros((2, stat_sample)))
    # X[0,:] = np.random.normal(a_0,   cov_0[0,0]**0.5, stat_sample)
    # X[1,:] = np.random.normal(lam_0, cov_0[1,1]**0.5, stat_sample)
    X = np.float32( np.random.multivariate_normal(mu_0, cov_0, size=stat_sample).T )

    # initial distribution (beta)
    # X[0,:] = draw_scaled_beta_samples(7, 2, x1_low+200, x1_hig-300, stat_sample)
    # X[1,:] = draw_scaled_beta_samples(7, 2, x2_low, x2_hig-4*pi, stat_sample)
    
    # propagate each sample over time
    X[1,:] = X[1,:] + t1*np.sqrt(mu_gravity/X[0,:]**3)

    print("x1 sample var, skew, excess kurt: ", np.var(X[0,:]), skew(X[0,:]), kurtosis(X[0,:]))
    print("x2 sample var, skew, excess kurt: ", np.var(X[1,:]), skew(X[1,:]), kurtosis(X[1,:]))
    
    # Define bins as the edges of x1 and x2
    bins_x1 = np.linspace(x1_low, x1_hig, num=linespace_num, endpoint=True, dtype=np.float32)
    bins_x2 = np.linspace(x2_low, x2_hig, num=linespace_num, endpoint=True, dtype=np.float32)

    # Digitize v to find which bin each value falls into for both dimensions
    bin_indices_x1 = np.digitize(X[0, :], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[1, :], bins_x2) - 1  

    # Initialize frequency array
    frequency_2d = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1), dtype=np.float32)

    # Count occurrences in each bin
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        idx_x1 = bin_indices_x1[i]
        idx_x2 = bin_indices_x2[i]
        # Check if the indices are valid
        if 0 <= idx_x1 < frequency_2d.shape[0] and 0 <= idx_x2 < frequency_2d.shape[1]:
            frequency_2d[idx_x2, idx_x1] += 1

    # Normalize the frequency to get the probability density
    frequency_2d /= stat_sample
    dx1 = bins_x1[1] - bins_x1[0]
    dx2 = bins_x2[1] - bins_x2[0]
    frequency_2d /= (dx1 * dx2)

    # Check the sum of the probability density function
    print("[check sum pdf=1.0]", np.sum(frequency_2d) * dx1 * dx2)

    # Calculate the midpoints for bins
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    X_grid, Y_grid = np.meshgrid(midpoints_x1, midpoints_x2)

    return X_grid, Y_grid, frequency_2d


def visual_pdf(x1_grid, x2_grid, pdf):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, pdf, cmap='viridis', edgecolor='none')
    # Set labels and title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Probability Density')
    ax.set_title('Surface Plot of 2D Normal Distribution Density')
    plt.show()


# def expected_value_y(pdf, x1_grid, x2_grid):
#     """
#     Compute the expected value of y given the joint PDF p(x, y) on the provided grids.
#     Parameters:
#     p_xy (2D array): Joint probability density function values at (x, y) points.
#     x_grid (1D array): Array of x values.
#     y_grid (1D array): Array of y values.
#     Returns:
#     float: The expected value of y.
#     """
#     x1s = x1_grid[0,:]
#     x2s = x2_grid[:,0]
#     dx1 = x1s[1] - x1s[0]
#     dx2 = x2s[1] - x2s[0]
#     sum = 0
#     for i in range(len(x1s)):
#         for j in range(len(x2s)):
#             pdf_value = pdf[j, i]  # pdf is accessed with (y, x) due to meshgrid order
#             integrand = pdf_value * x2s[j] * dx1 * dx2
#             sum = sum + integrand
#     return sum
        

def main():
    # [testing] beta initial distribution
    # draw_scaled_beta_samples(7, 2, x1_low, x1_hig)

    for t1 in t1s:
        print("t: ", t1)
        x1_grid, x2_grid, pdf = p_sol_monte(t1, linespace_num=50, stat_sample=100000000)
        np.save(DATA_FOLDER+"pdf_t"+str(t1)+".npy", pdf)
        if t1 == 0.0:
            np.save(DATA_FOLDER+"x1_grid.npy", x1_grid)
            np.save(DATA_FOLDER+"x2_grid.npy", x2_grid)
    time.sleep(5)
    
    for t1 in t1s:
        # load data
        x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
        x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
        pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
        # expected_lam_t = lam_0 + t1 * np.sqrt(mu_gravity/a_0**3)
        # print("[debug exp. lam(t)] ", expected_lam_t)
        # expected_lam_t_monte = expected_value_y(pdf, x1_grid, x2_grid)
        # print("[debug exp. lam(t) monte] ", expected_lam_t_monte)
        visual_pdf(x1_grid, x2_grid, pdf)
    


if __name__ == "__main__":
    main()