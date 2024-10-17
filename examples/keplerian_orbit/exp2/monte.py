import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

DATA_FOLDER = "data/"

pi = np.pi
n_d = 2
a_0 = 8000
lam_0 = 0.0
mu_0 = np.array([a_0, lam_0]).reshape(2,)
cov_0 = np.array([[100.0**2, 0.0], [0.0, (pi/10)**2]])
x1_low = 7000
x1_hig = 9000
x2_low = -pi
x2_hig = pi
ti = 0.0
tf = 60.0
mu_gravity = 398600.4418


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def p_sol_monte(t1=ti, linespace_num=50, stat_sample=1000000):
    mean = mu_0
    cov = cov_0
    X = np.random.multivariate_normal(mean, cov, size=stat_sample).T
    # X[1,:] = wrap_to_pi(X[1,:])
    
    X[1,:] = X[1,:] + t1*np.sqrt(mu_gravity/X[0,:]**3)
    X[1,:] = wrap_to_pi(X[1,:])
    
    # Define bins as the edges of x1 and x2
    bins_x1 = np.linspace(x1_low, x1_hig, num=linespace_num, endpoint=True)
    bins_x2 = np.linspace(x2_low, x2_hig, num=linespace_num, endpoint=True)

    # Digitize v to find which bin each value falls into for both dimensions
    bin_indices_x1 = np.digitize(X[0, :], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[1, :], bins_x2) - 1  

    # Initialize frequency array
    frequency_2d = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1))

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


def expected_value_y(pdf, x1_grid, x2_grid):
    """
    Compute the expected value of y given the joint PDF p(x, y) on the provided grids.
    Parameters:
    p_xy (2D array): Joint probability density function values at (x, y) points.
    x_grid (1D array): Array of x values.
    y_grid (1D array): Array of y values.

    Returns:
    float: The expected value of y.
    """
    x1s = x1_grid[0,:]
    x2s = x2_grid[:,0]
    dx1 = x1s[1] - x1s[0]
    dx2 = x2s[1] - x2s[0]
    sum = 0
    for i in range(len(x1s)):
        for j in range(len(x2s)):
            pdf_value = pdf[j, i]  # pdf is accessed with (y, x) due to meshgrid order
            integrand = pdf_value * x2s[j] * dx1 * dx2
            sum = sum + integrand
    return sum
        
    


def main():
    t1s = [0.0, 3600.0, 7200.0, 86400.0, 2.0*86400.0]

    for t1 in t1s:
        x1_grid, x2_grid, pdf = p_sol_monte(t1)
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
        expected_lam_t = lam_0 + t1 * np.sqrt(mu_gravity/a_0**3)
        print("[debug expected lam(t) analytical] ", expected_lam_t)
        # expected_lam_t_monte = expected_value_y(pdf, x1_grid, x2_grid)
        # print("[debug exp. lam(t) monte] ", expected_lam_t_monte)
        visual_pdf(x1_grid, x2_grid, pdf)
    


if __name__ == "__main__":
    main()