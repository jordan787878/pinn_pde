import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal

DATA_FOLDER = "data/"

# from util import kepler_to_cartesian, solve_kepler, orbital_elements_2d, mu_gravity
from util import COE2RV, RV2COE, true_to_mean_anomaly, solve_kepler, mu_gravity

ti = 0.0
tf = 1.0*3600.0

# mean pos-vel
# X = [x (km),y (km), vx (km/s), vy (km/s) ]
a_axis = 8000.0
r_IJK, v_IJK = COE2RV(a_axis, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0)
print(r_IJK, v_IJK)

mean_0 = np.array([r_IJK[0], r_IJK[1], v_IJK[0], v_IJK[1]]).reshape(-1,)
cov_0 = np.diag([(1.0)**2, (1.0)**2, (0.1)**2, (0.1)**2]) # unit test

# X' = non dimensional X: select t_scaled and xy_scaled such that mu(non-d)=1
# t_scaled = 2*np.pi*np.sqrt(a_axis**3/mu_gravity)
# xy_scaled = a_axis*(4*np.pi**2)**(1/3)
# v_scaled = xy_scaled/t_scaled

# [test] try to scale the domain such that the scaled cov is standard ...
t_scaled = 1.0; xy_scaled = 1.0; v_scaled = 0.1
print(t_scaled, xy_scaled, v_scaled)

# Specify domain
# [test] the domain is 10 times the original standard deviation
# [test] could we have different domain size (xy is 100 times)?
x1_low = (mean_0[0]-100.0)/xy_scaled  
x1_hig = (mean_0[0]+100.0)/xy_scaled  
x2_low = (mean_0[1]-100.0)/xy_scaled  
x2_hig = (mean_0[1]+100.0)/xy_scaled
x3_low = (mean_0[2]-1.0)/v_scaled
x3_hig = (mean_0[2]+1.0)/v_scaled
x4_low = (mean_0[3]-1.0)/v_scaled
x4_hig = (mean_0[3]+1.0)/v_scaled
r_low = 6378.0/xy_scaled
r_hig = 12000.0/xy_scaled

mu_gravity_scaled = mu_gravity/xy_scaled
A_scaled = np.diag([1/xy_scaled, 1/xy_scaled, 1/v_scaled, 1/v_scaled])
mean_scaled = np.matmul(A_scaled, mean_0).astype(np.float32)
cov_scaled  = np.matmul(A_scaled, np.matmul(cov_0, A_scaled.T)).astype(np.float32)
print("scaled mean: ", mean_scaled)
print("scaled cov:  ", cov_scaled)

pdf_func = multivariate_normal(mean_scaled, cov_scaled, allow_singular=True)
max_p0 = pdf_func.pdf(mean_scaled).reshape(-1,)[0]
print("max p0: ",max_p0)


def p_sol_monte(t1=ti, linespace_num=100, stat_sample=100000):
    X = np.random.multivariate_normal(mean_0, cov_0, size=stat_sample).T
    for i in range(stat_sample):
        # cartesian to orbit element
        r = np.array([X[0,i], X[1,i], 0.0]).reshape(3,)
        v = np.array([X[2,i], X[3,i], 0.0]).reshape(3,)
        _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(r, v)
        
        # checking eccentricity < 1
        # while(_e > 1):
        #     # if not, resampling
        #     print("[debug] resample initial state at sample: ", i)
        #     X[:,i] = np.random.multivariate_normal(mean_0, cov_0).T
        #     r = np.array([X[0,i], X[1,i], 0.0]).reshape(3,)
        #     v = np.array([X[2,i], X[3,i], 0.0]).reshape(3,)
        #     _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(r, v)

        _M = true_to_mean_anomaly(_e, _nu)
        _M = _M + np.sqrt(mu_gravity/_a**3)*t1
        _nu = solve_kepler(_e, _M)
        r_t, v_t = COE2RV(_a, _e, _i, _RAAN, _w, _nu, _lonper)
        X[0,i] = r_t[0]/xy_scaled
        X[1,i] = r_t[1]/xy_scaled
        X[2,i] = v_t[0]/v_scaled
        X[3,i] = v_t[1]/v_scaled
    
    # Define bins for each dimension
    bins_x1 = np.linspace(x1_low, x1_hig, num=linespace_num, endpoint=True)
    bins_x2 = np.linspace(x2_low, x2_hig, num=linespace_num, endpoint=True)
    bins_x3 = np.linspace(x3_low, x3_hig, num=linespace_num, endpoint=True)
    bins_x4 = np.linspace(x4_low, x4_hig, num=linespace_num, endpoint=True)

    # Digitize to find bin indices for each dimension
    bin_indices_x1 = np.digitize(X[0, :], bins_x1) - 1
    bin_indices_x2 = np.digitize(X[1, :], bins_x2) - 1
    bin_indices_x3 = np.digitize(X[2, :] , bins_x3) - 1
    bin_indices_x4 = np.digitize(X[3, :] , bins_x4) - 1  

    # Initialize frequency array for 4D
    frequency_4d = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1, len(bins_x3) - 1, len(bins_x4) - 1))

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

    # Check the sum of the probability density function
    print("[check sum pdf=1.0]", np.sum(frequency_4d) * dx1 * dx2 * dx3 * dx4)

    # Calculate the midpoints for bins (optional, depending on your needs)
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
    midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
    midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
    midpoints_x4 = (bins_x4[:-1] + bins_x4[1:]) / 2
    return midpoints_x1, midpoints_x2, midpoints_x3, midpoints_x4, frequency_4d


def pdf_xy(idx_x, idx_y, pdf, x3s, x4s):
    dx3 = x3s[1] - x3s[0]
    dx4 = x4s[1] - x4s[0]
    p_sum = 0
    for k in range(pdf.shape[2]):
        for l in range(pdf.shape[3]):
            p_sum += pdf[idx_x, idx_y, k, l] * dx3 * dx4
    return p_sum


# def visual_pdf(x1_grid, x2_grid, pdf):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x1_grid, x2_grid, pdf, cmap='viridis', edgecolor='none')
#     # Set labels and title
#     ax.set_xlabel('X1')
#     ax.set_ylabel('X2')
#     ax.set_zlabel('Probability Density')
#     ax.set_title('Surface Plot of 2D Normal Distribution Density')
#     plt.show()
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
    t1s = [ti]
    for t1 in t1s:
        x1s, x2s, x3s, x4s, pdf = p_sol_monte(t1, linespace_num=50, stat_sample=100000)

        # Print coordinates and pdf values greater than the threshold
        # threshold = 1e-15
        # for i in range(pdf.shape[0]):
        #     for j in range(pdf.shape[1]):
        #         for k in range(pdf.shape[2]):
        #             for l in range(pdf.shape[3]):
        #                 # print(pdf[i, j,k,l])
        #                 if pdf[i, j, k, l] > threshold:
        #                     print(f"x1: {x1s[i]:.4f}, x2: {x2s[j]:.4f}, "
        #                         f"x3: {x3s[k]:.4f}, x4: {x4s[l]:.4f}, "
        #                         f"pdf value: {pdf[i, j, k, l]:.6f}")

        # Marginalize to pdf(X,Y)
        X1, X2 = np.meshgrid(x1s, x2s)
        P_XY = np.zeros_like(X1)
        # print((x1s[1]-x1s[0])*(x2s[1]-x2s[0]))
        for i in range(pdf.shape[0]):
            for j in range(pdf.shape[1]):
                p_xy = pdf_xy(i, j, pdf, x3s, x4s)
                P_XY[i, j] = p_xy
                # print(x1s[i], x2s[j], p_xy)
                # if(p_xy > 0):
                #     print(x1s[i], x2s[j], p_xy)

        # Find the maximum P_XY value and its coordinates
        max_value = np.max(P_XY)
        max_index = np.unravel_index(np.argmax(P_XY, axis=None), P_XY.shape)
        # Get the corresponding X, Y coordinates
        X_max = X1[max_index]
        Y_max = X2[max_index]
        # Print the results
        print(f"Maximum P_XY value: {max_value}")
        print(f"Corresponding X coordinate (km): {X_max*xy_scaled}")
        print(f"Corresponding Y coordinate (km): {Y_max*xy_scaled}")
        
        # Plot and save pdf(X,Y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vmax = np.max(P_XY)
        ax.plot_surface(X1, X2, P_XY.T, cmap='viridis', vmin=0.0, vmax=vmax)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('P(X1, X2)')
        plt.title('3D Surface Plot of P(X1, X2)')
        fig.savefig('figs/p_xy(monte)_t'+str(t1)+".pdf", format='pdf', dpi=300)
        plt.close()
                            
        np.save(DATA_FOLDER+"p_xy(monte)_t"+str(t1)+".npy", P_XY)
        if t1 == 0.0:
            np.save(DATA_FOLDER+"x1s.npy", x1s)
            np.save(DATA_FOLDER+"x2s.npy", x2s)

    # time.sleep(5)
    # for t1 in t1s:
    #     # load data
    #     x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
    #     x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")
    #     pdf = np.load(DATA_FOLDER+"pdf_t"+str(t1)+".npy")
    #     # expected_lam_t = lam_0 + t1 * np.sqrt(mu_gravity/a_0**3)
    #     # print("[debug exp. lam(t)] ", expected_lam_t)
    #     # expected_lam_t_monte = expected_value_y(pdf, x1_grid, x2_grid)
    #     # print("[debug exp. lam(t) monte] ", expected_lam_t_monte)
    #     visual_pdf(x1_grid, x2_grid, pdf)
    


if __name__ == "__main__":
    main()