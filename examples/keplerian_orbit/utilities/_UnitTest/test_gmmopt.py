import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

from util_test_gmmopt import *


def single_gmm_1d():
    # Domain and discretization.
    x_l, x_u, dx = -5.0, 5.0, 0.01
    x = np.arange(x_l, x_u + dx, dx)
    
    # Define baseline pdf p0 over the domain.
    base_mu = 0.0
    base_sigma = 1.0
    p0 = norm.pdf(x, loc=base_mu, scale=base_sigma)
    p0 = p0 / (np.sum(p0) * dx)
    
    # Error bound
    B = 0.05
    
    # Target subset (where we want to maximize probability mass).
    x_subset = np.array([0.0, 2.0])
    
    # Initial guess for parameters (mu, sigma).
    initial_guess = [0.0, 1.0]
    # Parameters constraints: (mean \in R, sigma > 0)
    bounds = [(-np.inf, np.inf), (1e-6, np.inf)]

    # Define the inequality constraint: constraint_fun(params, x, p0, B) >= 0.
    cons = {'type': 'ineq',
            'fun': lambda params: constraint_fun(params, x, p0, B)}
    
    # Optimize parameters.
    runtime_start = time.time()
    method = "analy"
    result = minimize(objective, 
                      initial_guess, 
                      args=(x, x_subset, method),
                      constraints=cons, 
                      bounds=bounds)
    runtime_end = time.time() - runtime_start
    
    if result.success:
        print("Optimization successful! Run Time: {:.4f}".format(runtime_end))
    else:
        print("Optimization failed!")
    
    mu_opt, sigma_opt = result.x
    print("Optimized parameters: mu = {:.4f}, sigma = {:.4f}".format(mu_opt, sigma_opt))

    Pr_opt_subset = gmm_integral_1dsubset(result.x, x_subset)
    print("Optimized Probability: {:.4f}".format(Pr_opt_subset))

    plot_gamm_1dsubset(x, x_subset, dx, p0, B, result.x, Pr_opt_subset)


if __name__ == '__main__':
    single_gmm_1d()