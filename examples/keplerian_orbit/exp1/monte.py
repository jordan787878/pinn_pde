import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
from scipy.stats import skew, kurtosis

DATA_FOLDER = "meta_data/exp1/"

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


# # [testing] initial beta distribution
# def draw_scaled_beta_samples(alpha, beta, x1_low, x1_hig, num_samples=1000):
#     # Create a random generator
#     rng = np.random.default_rng()
#     # Draw samples from the Beta distribution
#     samples = rng.beta(alpha, beta, size=num_samples)
#     # Scale the samples to the desired range
#     scaled_samples = samples * (x1_hig - x1_low) + x1_low
#     # Plot the histogram of the scaled samples
#     # plt.figure(figsize=(10, 6))
#     # plt.hist(scaled_samples, bins=30, density=True, alpha=0.6, color='b')
#     # # Plot the theoretical PDF of the scaled Beta distribution
#     # x = np.linspace(x1_low, x1_hig, 100)
#     # pdf = ( (x - x1_low) / (x1_hig - x1_low) )**(alpha - 1) * (1 - (x - x1_low) / (x1_hig - x1_low))**(beta - 1)
#     # pdf /= np.math.gamma(alpha) * np.math.gamma(beta) / np.math.gamma(alpha + beta) * (x1_hig - x1_low)
#     # plt.plot(x, pdf, 'r-', lw=2, label='Scaled Beta PDF')
#     # plt.title(f'Scaled Beta Distribution (α={alpha}, β={beta})')
#     # plt.xlabel('Value')
#     # plt.ylabel('Density')
#     # plt.legend()
#     # plt.grid()
#     # plt.show()
#     # _mean = np.mean(scaled_samples)
#     # _var  = np.var(scaled_samples)
#     # print(_mean, _var)
#     return scaled_samples


def p_sol_monte(t1=ti, linespace_num=50, stat_sample=10000000):
    # initial samples
    X = np.float32( np.random.multivariate_normal(mu_0, cov_0, size=stat_sample).T )
    # propagate each sample over time
    X[1,:] = X[1,:] + t1*np.sqrt(mu_gravity/X[0,:]**3)

    print("x1 sample mean, var, skew, excess kurt: ", np.mean(X[0,:]), np.var(X[0,:]), skew(X[0,:]), kurtosis(X[0,:]))
    print("x2 sample mean, var, skew, excess kurt: ", np.mean(X[1,:]), np.var(X[1,:]), skew(X[1,:]), kurtosis(X[1,:]))

    # [test] update domain.
    # NOTE: If this is used, then the training should also consider moving domain ...
    # x2_low = np.min(X[1,:])-pi
    # x2_hig = np.max(X[1,:])+pi
    
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
        

def main():
    # [Run this to create folders] for i in {1..100}; do mkdir data_$i; done
    N_samples = int(1e+8)
    N_monteruns = 100

    for j in range(1, N_monteruns+1):
        # run monte-carlo to specific time and save data
        for t1 in t1s:
            print("run "+str(j)+"-th monte-carlo for t: ", t1)
            x1_grid, x2_grid, pdf = p_sol_monte(t1, linespace_num=50, stat_sample=N_samples)
            if(t1 == 0.0 and j == 1):
                # save to meta data folder
                np.save(DATA_FOLDER+"x1_grid.npy", x1_grid)
                np.save(DATA_FOLDER+"x2_grid.npy", x2_grid)
            np.save(DATA_FOLDER+"data_"+str(j)+"/pdf_t"+str(t1)+".npy", pdf)

    # time.sleep(5)

    # load and quick visualize data
    # for t1 in t1s:
    #     x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
    #     x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
    #     print("[debug] x1_grid shape: ", x1_grid.shape)
    #     pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
    #     visual_pdf(x1_grid, x2_grid, pdf)
    


if __name__ == "__main__":
    main()