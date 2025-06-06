import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

from util_test_gmmopt import *


def optimize_single_gmm_1d(x, x_subset, p0, B, Pr_subset, method):
    """
    method = {"baseline", "iterative"}
    NOTE: this "iterative" method is a intermediate testing method because it uses the true Prob.
          Next step, an interative method with guaranteed bound without using the true Prob.
    """
    # Parameters constraints: (mean \in R, sigma > 0)
    bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
    # Define the inequality constraint: constraint_fun(params, x, p0, B) >= 0.
    cons = {'type': 'ineq',
            'fun': lambda params: constraint_fun(params, x, p0, B)}
    # Initial guess for parameters (mu, sigma).
    initial_guess = [0.0, 1.0]
    
    # Optimize parameters.
    objective_fcn = "analy"

    # Ensure the optimization satisfy the constraints
    opt_sol = False
    while(opt_sol == False):
        result = minimize(objective, 
                        initial_guess, 
                        args=(x, x_subset, objective_fcn),
                        constraints=cons, 
                        bounds=bounds)
        opt_sol = result.success

        # iterative method: justifying that initial guess does affect the optimized GMM
        if(opt_sol):
            Pr_opt_subset = gmm_integral_1dsubset(result.x, x_subset)
            if(Pr_opt_subset < Pr_subset and method == "iterative(fake)"):
                opt_sol = False
        initial_guess[0] = np.random.uniform(np.min(x_subset), np.max(x_subset)) # re-initialize gmm to optimize

    return result


def single_gmm_1d(show_plots, method):
    # Domain and discretization.
    x_l, x_u, dx = -5.0, 5.0, 0.05
    x = np.arange(x_l, x_u + dx, dx)
    
    # Define baseline pdf p0 over the domain.
    p0, p0_params = generate_base_pdf(x, num=1)
    # print("=== Base PDF params ==="); print(p0_params)

    # Error bound
    B = np.max(p0)*0.1
    
    # Target subset (where we want to maximize probability mass).
    x_subset = np.array([-1.0, 1.0])
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    Pr_subset = np.sum(p0[mask]) * dx

    runtime_start = time.time()
    result = optimize_single_gmm_1d(x, x_subset, p0, B, Pr_subset, method)
    runtime_end = time.time() - runtime_start

    if not result.success:
        raise RuntimeError("Optimization FAIL")
    
    print("Optimization Run Time: {:.4f}".format(runtime_end))
    Pr_opt_subset = gmm_integral_1dsubset(result.x, x_subset)
    Pr_tv_bound = 0.0
    if(method == "tv_bound"):
        Pr_tv_bound = gmm_tv_bound_1d(x_subset, p0, B, iter=1)
    print("Opt Prob: {:.4f} (+ {:.4f}), True Prob {:.4f}".format(Pr_opt_subset, Pr_tv_bound, Pr_subset))

    plot_gamm_1dsubset(show_plots, x, x_subset, p0, B, result.x, Pr_opt_subset, Pr_subset)

    np.testing.assert_array_less(Pr_subset, Pr_opt_subset + Pr_tv_bound, err_msg="Pr_opt <= Pr")    
    # mu_opt, sigma_opt = result.x
    # print("Optimized parameters: mu = {:.4f}, sigma = {:.4f}".format(mu_opt, sigma_opt))


def test_single_gmm_1d():
    """
    This test shows that Pr_opt <= Pr if the initial guess of the GMM is not appropriate
    """
    np.random.seed(13)
    single_gmm_1d(show_plots=True, method="iterative(fake)")

    np.random.seed(13)
    single_gmm_1d(show_plots=True, method="baseline")


def test_single_gmm_1d_iterative():
    for i in range(100):
        np.random.seed(i)
        print(i)
        single_gmm_1d(show_plots=False, method="iterative(fake)")


def test_single_gmm_1d_tvbound():
    for i in range(100):
        np.random.seed(i)
        print(i)
        single_gmm_1d(show_plots=False, method="tv_bound")


if __name__ == '__main__':
    test_single_gmm_1d()

    # test_single_gmm_1d_iterative()

    # test_single_gmm_1d_tvbound()