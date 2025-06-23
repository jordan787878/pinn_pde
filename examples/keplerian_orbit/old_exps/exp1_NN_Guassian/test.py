# purpose: unit test to decide the setting of monte carlo, 
# as well as quickly compare the monte-carlo vs linear propagated Guassian
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
from scipy.stats import skew, kurtosis
from scipy.stats import wasserstein_distance
from scipy.stats import wasserstein_distance_nd #(on probability mass)
import ot
import torch
from geomloss import SamplesLoss

DATA_FOLDER = "data/"

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

# [testing]
# mu_0 = np.float32(np.array([8000.0, 0.0]).reshape(2,))
# cov_0 = np.float32(np.array([[2000.0**2, 0.0], [0.0, 1.0**2]]))
# mu_gravity = np.float32(1.0)


def sample_skewness(X, stat_sample):
    mu = np.mean(X)
    std = np.std(X)
    skew = np.mean(((X-mu)/std)**3)
    # skew = ((stat_sample*(stat_sample-1))**0.5/(stat_sample-2)) * np.mean((X - mu)**3)/ ( np.mean((X - mu)**2) )**(3/2)  
    return skew


def sample_kurtosis(X, stat_sample):
    mu = np.mean(X)
    std = np.std(X)
    kurt = np.mean(((X-mu)/std)**4)
    # skew = ((stat_sample*(stat_sample-1))**0.5/(stat_sample-2)) * np.mean((X - mu)**3)/ ( np.mean((X - mu)**2) )**(3/2)  
    return kurt-3.0


def draw_scaled_beta_samples(alpha, beta, x1_low, x1_hig, num_samples=1000):
    rng = np.random.default_rng()
    samples = rng.beta(alpha, beta, size=num_samples)
    scaled_samples = samples * (x1_hig - x1_low) + x1_low
    return scaled_samples


def draw_samples_from_pdf(pdf_func, num_samples, z_low=0.01, z_high=5.0):
    """Draw samples from the given PDF using rejection sampling."""
    samples = []
    # Estimate the maximum value of the PDF for normalization
    z_values = np.linspace(z_low, z_high, 1000)
    pdf_values = pdf_func(z_values)
    M = np.max(pdf_values)  # Find maximum value of the PDF
    while len(samples) < num_samples:
        # Generate a candidate sample from a uniform distribution
        z_sample = np.random.uniform(z_low, z_high)
        u = np.random.uniform(0, M)  # Uniform sample for acceptance
        # Check acceptance condition
        if u < pdf_func(z_sample):
            samples.append(z_sample)
    return np.array(samples)


# test to show that the analytical form of the RV Y matches the samples
def test_1(t1=3600.0, stat_sample=100):
    # initial distribution (normal)
    X = np.float32( np.random.multivariate_normal(mu_0, cov_0, size=stat_sample).T )
    print("X2 stats at t=0: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(X[1, :]),
        (np.var(X[1, :]))**0.5,
        sample_skewness(X[1, :], stat_sample),
        sample_kurtosis(X[1, :], stat_sample)
    ))

    C = t1*np.sqrt(mu_gravity)
    Z = np.sqrt(1/(X[0,:]**3))
    G = C*Z
    g_min = np.min(G)
    g_max = np.max(G)
    gs = np.linspace(g_min, g_max, num=100)
    pdf_g = (1/C)*pdf_Z(gs/C)
    print("Y stats: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(G),
        (np.var(G))**0.5,
        sample_skewness(G, stat_sample),
        sample_kurtosis(G, stat_sample)
    ))

    # [test] sampling arbitrary pdf
    # g_samples = draw_samples_from_pdf(pdf_Z, stat_sample, z_low=g_min, z_high=g_max)
    # print(g_samples)

    X2t = X[1,:] + G
    print("X2 stats at t="+str(t1)+": {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(X2t),
        (np.var(X2t))**0.5,
        sample_skewness(X2t, stat_sample),
        sample_kurtosis(X2t, stat_sample)
    ))

    plt.figure(figsize=(10, 6))
    plt.hist(G, bins=100, density=True, alpha=0.3, color='b')
    plt.plot(gs, pdf_g, "blue")
    # plt.hist(g_samples, bins=100, density=True, alpha=0.3, color='r')
    # plt.plot(gs, fit_g, "r--")
    plt.xlabel('Value')
    plt.ylabel('p_Y Density')
    plt.grid()
    plt.show()


# Z = sqrt(1/X^3)
def pdf_Z(z):
    pdf_value = (1/np.sqrt(2*pi*cov_0[0,0])) * np.exp(-(z**(-2/3)-mu_0[0])**2/(2*cov_0[0,0])) * (2/3) * z**(-5/3)
    return pdf_value


def pdf_normal(x, mean, cov):
    pdf_value = (1/np.sqrt(2*pi*cov))*np.exp(-0.5*(x-mean)**2/cov)
    return pdf_value


def estimate_pdf(X, x_low, x_high, dx):
    # Define the bin edges
    bins = np.arange(x_low, x_high + dx, dx)
    # Compute the histogram (counts)
    counts, _ = np.histogram(X, bins=bins)
    # Normalize to get the probability density
    bin_area = dx * len(X)  # Area of each bin
    pdf = counts / bin_area  # Normalize by the total number of samples
    return bins[:-1], pdf  # Return the bin centers and the PDF


# test to show the mean, std, skew, kurtosis of (true samples) vs (lineaur Guassian)
# The skew is different (the most)
def test_2(t1=3600.0, stat_sample=100):
    # initial distribution (normal)
    X = np.float32( np.random.multivariate_normal(mu_0, cov_0, size=stat_sample).T )

    print("X2 stats at t=0: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(X[1, :]),
        (np.var(X[1, :]))**0.5,
        sample_skewness(X[1, :], stat_sample),
        sample_kurtosis(X[1, :], stat_sample)
    ))
    C = t1*np.sqrt(mu_gravity)
    Z = np.sqrt(1/(X[0,:]**3))
    G = C*Z
    print("Y stats: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(G),
        (np.var(G))**0.5,
        sample_skewness(G, stat_sample),
        sample_kurtosis(G, stat_sample)
    ))
    X[1,:] = X[1,:] + G
    # update x2 domain at time t
    # x2_low = np.min(X[1,:]); x2_hig = np.max(X[1,:])
    # estimate pdf(x2) at time t
    dx2 = (x2_hig - x2_low)/100.0
    x2_centers, pdf_x2_monte = estimate_pdf(X[1,:], x2_low, x2_hig, dx2)
    print("X2 stats at t="+str(t1)+": {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(X[1,:]),
        (np.var(X[1,:]))**0.5,
        sample_skewness(X[1,:], stat_sample),
        sample_kurtosis(X[1,:], stat_sample)
    ))

    # Gaussian approximation
    mu_G, cov_G = propagate_guassian(t1, mu_0, cov_0)
    # generate samples using mu_G, cov_G
    X_G = np.float32(np.random.multivariate_normal(mu_G, cov_G, size=stat_sample).T)
    print("Linear Gaussian Propagation Stats (mean, std): {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
        np.mean(X_G[1,:]),
        np.var(X_G[1,:])**0.5,
        sample_skewness(X_G[1,:], stat_sample),
        sample_kurtosis(X_G[1,:], stat_sample)
    ))
    pdf_x2_gaussian = pdf_normal(x2_centers, mu_G[1], cov_G[1,1])
    
    # Compute W-distance (using samples)
    # [scipy-1d]: requires O(sample^2) memory, this is not possible when samples are large
    # w_distance_x1 = wasserstein_distance(X[0,:], X_G[0,:])
    # w_distance_x2 = wasserstein_distance(X[1,:], X_G[1,:])
    # print("W distance using scipy on each dimension (x1,x2): ", w_distance_x1, w_distance_x2)

    # [scipy-nd]
    # POV: this is not decent, because the w_distance changes a lot with different samples...
    # w_distance = wasserstein_distance_nd(X, X_G)/stat_sample
    # print("w-distance [samples scipy-nd]: ", w_distance)

    # [pot] (using pdfs)
    # NOTE this method can also be done for pdf on higher dimension
    # However, the cost matrix becomes R^n x R^n, and in the same time, what should be the weight...?
    # NOTE x1 is the semi-major axis (km), and x2 is the mean motion (rad)...
    # So this measure depends on how we "weight" different states...
    # 1. check if the constraint sum(pX) = sum(pY). if not, normalize the approx to true
    pdf_x2_gaussian = pdf_x2_gaussian/np.sum(pdf_x2_gaussian*dx2) * np.sum(pdf_x2_monte*dx2)
    # print(np.sum(pdf_x2_gaussian*dx2), np.sum(pdf_x2_monte*dx2))
    # 2. Create a grid of positions and compute cost matrix cost(xi, xj) = ||xi - xj||^2
    M = ot.dist(x2_centers.reshape((-1,1)), x2_centers.reshape((-1,1)))
    # general case
    w_distance = (ot.emd2(pdf_x2_gaussian, pdf_x2_monte, M))**0.5
    print("w-distance [pdf, emd2, Square Euclidean Distance Cost]: ", w_distance)
    # (special case for 1D)
    w_distance = (ot.wasserstein_1d(x2_centers, x2_centers, pdf_x2_gaussian, pdf_x2_monte, p=2))**0.5
    print("w-distance [pdf, wasserstein_1d, Square Euclidean Distance Cost]: ", w_distance)
    # (entropy regularized OT) NOT GOOD
    # w_distance = (ot.sinkhorn2(pdf_x2_gaussian, pdf_x2_monte, M, 1))**0.5
    # print("w-distance [pdf, entropy regularized, Square Euclidean Distance Cost]: ", w_distance)
    # (Sinkhorn Divergence geomLoss)
    
    # [geomLoss] Even official code does not work ...
    # X_torch = torch.from_numpy(X).reshape(-1,2)
    # X_G_torch = torch.from_numpy(X_G).reshape(-1,2)
    # print(X_torch.shape, X_G_torch.shape)
    # w_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    # L = w_loss(X_torch, X_G_torch).item()
    # print("W distnace [samples, geomLoss]: ", L)
    # (official code)
    # x = torch.randn(100000, 3)
    # y = torch.randn(200000, 3)
    # # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    # loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    # L = loss(x, y)  # By default, use constant weights = 1/number of samples
    
    # [pot] run out of memeory or divide by zero error
    # b = np.zeros((stat_sample,1))/stat_sample
    # xx = X[1,:].reshape(-1,1)
    # print(xx.shape, b.shape)
    # loss = ot.bregman.empirical_sinkhorn2(xx, xx, b=b, reg=0.1, verbose=False)

    # it seems that L is extremely large even if X_G_torch is just slightly different from X_G

    plt.figure(figsize=(10, 6))
    # plt.hist(X[1,:], bins=100, density=True, alpha=0.3, color='g')
    # plt.hist(X_G[1,:], bins=100, density=True, alpha=0.3, color='b')
    plt.plot(x2_centers, pdf_x2_monte, "b", label="Monte")
    plt.plot(x2_centers, pdf_x2_gaussian, "r--", label="Linear Guassian")
    plt.legend(loc="upper right")
    plt.xlabel('Value')
    plt.ylabel('p_X2 Density')
    plt.grid()
    plt.show()


def nonlinear_mean_dynamics(x):
    xdot = np.float32(np.zeros(2))
    xdot[1] = np.sqrt(mu_gravity/x[0]**3)
    return xdot


def linear_cov_dynamics(p, x):
    A = np.float32(np.zeros((2,2)))
    A[1,0] = mu_gravity**0.5*(-1.5)*(x[0]**(-2.5))
    p_dot = np.matmul(p, A.T) + np.matmul(A, p)
    return p_dot


def propagate_guassian(t, _mu_0, _cov_0):
    dt = np.float32(0.1)
    T_end = t
    K = int(T_end/dt)
    x = _mu_0
    p = _cov_0
    for k in range(K):
        xnew = x + nonlinear_mean_dynamics(x) * dt
        pnew = p + linear_cov_dynamics(p, x) * dt
        # runge-kutta
        # h1 = linear_dynamics(x)
        # h2 = linear_dynamics(x+0.5*h1*dt)
        # h3 = linear_dynamics(x+0.5*h2*dt)
        # h4 = linear_dynamics(x+1.0*h3*dt)
        # xnew = x + dt*(h1 + 2.0*h2 + 2.0*h3 + h4)/6.0
        x = xnew
        p = pnew
    return x, p


# test to show that two pdfs can have same mean and covariance, but different shapes
# Hence, need a metric to quantify the difference
def test_3():
    # domain
    N = 100
    x = np.linspace(-5, 5, N)
    
    # Bimodal distribution parameters
    weight1 = 0.5
    weight2 = 0.5
    mean1 = -2
    mean2 = 2
    std_dev = 0.5

    # Bimodal distribution PDF
    pY = (weight1 * (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean1) / std_dev) ** 2) +
          weight2 * (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean2) / std_dev) ** 2))
        
    # Mean and covariance of pY can be calculated and matched with pX
    dx = x[1] - x[0]
    mean_Y = np.sum(x*pY*dx)
    var_Y =  np.sum(x*x*pY*dx) - mean_Y**2

    mu = mean_Y
    sigma = var_Y**0.5
    pX = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    print("mean of pX, pY: ", mu, mean_Y)
    print("cov  of pX, pY: ", sigma**2, var_Y)

    # Compute Wasserstein distance (using pdfs)
    # check if the constraint sum(pX) = sum(pY). if not, normalize the approx to true
    pY = pY/np.sum(pY*dx) * np.sum(pX*dx)
    print(np.sum(pX*dx), np.sum(pY*dx))

    # Create a grid of positions and compute cost matrix cost(xi, xj) = ||xi - xj||^2 (manually)
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i,j] = (x[i]-x[j])**2
    
    # Compute the Wasserstein distance
    W_distance = ot.emd2(pX, pY, M)**0.5
    print("w-distance [pdf emd2, Euclidean Distance Cost]: ", W_distance)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, pX, label='Normal Distribution (pX)', color='blue')
    plt.plot(x, pY, label='Bimodal Distribution (pY)', color='orange')
    plt.title('Comparison of Distributions with Same Mean and Variance')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()


# n_samples_a = 2
# n_samples_b = 2
# X_s = np.reshape(np.arange(n_samples_a, dtype=np.float64), (n_samples_a, 1))
# X_t = np.reshape(np.arange(0, n_samples_b, dtype=np.float64), (n_samples_b, 1))
# b = np.full((n_samples_b, 1), 1/n_samples_b)
# print(X_s.shape, X_t.shape, b.shape)
# print(b)
# loss = ot.bregman.empirical_sinkhorn2(X_s, X_t, b=b, reg=0.1, verbose=False)


def main():
    # p_sol_monte(t1=864000.0)

    # test_1(t1=12.0*3600.0, stat_sample=1000000)

    test_2(t1=6.0*3600.0, stat_sample=10000000)
    # NOTE: the resolution of space will affect how "similar" the monte and guassian pdfs are, 
    # In general, the finer the resolution, the more "similar" the two pdfs will be.
    # Hence, it changes the w-distance 
    # but it does not "significantly change" the SAMPLE statistics: skew/kurtosis
    # So, using samples and higher order moment as metric is invariant w.r.t. the domain and its discretization

    # test_3()
    

if __name__ == "__main__":
    main()