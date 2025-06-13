import numpy as np
import time
import warnings
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.special import erfc
from util_test_gmmopt import generate_base_pdf, plot_Kmode_gmm_1d


def mixture_pdf(K, theta, x):
    """
    Compute the mixture PDF for a K-component GMM.
    theta: length 3*K array [mu_1..mu_K, sigma_1..sigma_K, w_1..w_K]
    x: array of evaluation points
    """
    mus    = theta[0:K]
    sigmas = theta[K:2*K]
    weights= theta[2*K:3*K]
    pdfs = np.array([
        w * norm.pdf(x, loc=mu, scale=sigma)
        for mu, sigma, w in zip(mus, sigmas, weights)
    ])
    return pdfs.sum(axis=0)


def gmm_constraint(K, theta, x, p0, B):
    """
    Ensure mixture PDF lies within [p0-B, p0+B] for all x.
    Returns a vector of >=0 values when satisfied.
    """
    slack = B*0.01
    p_mix = mixture_pdf(K, theta, x)
    return np.hstack([
        p_mix - (p0 - B) - slack,    # >= 0
        (p0 + B) - p_mix - slack     # >= 0
    ])


def gmm_integral_1dsubset(K, theta, x_subset):
    """
    Analytical integral of the GMM over the interval [a, b].
    Uses the error function for Gaussian CDF.
    """
    a = x_subset[0]; b = x_subset[1]
    mus    = theta[0:K]
    sigmas = theta[K:2*K]
    weights= theta[2*K:3*K]
    integral = 0.0
    for mu, sigma, w in zip(mus, sigmas, weights):
        integral += w*0.5*(1-erfc((b-mu)/(np.sqrt(2)*sigma)) - (1-erfc((a-mu)/(np.sqrt(2)*sigma))) ) # erfc = 1 - erf (is numerically more stable)
    return integral


def optimize_gmm_k(K, x, x_subset, p0, B, Pr_iter, method="analytical"):
    """
    Optimize a K-component GMM to maximize probability mass over x_subset
    subject to PDF bounds [p0 - B, p0 + B] over the full domain x.

    Returns the SciPy OptimizeResult with .x = optimal theta.
    """
    # re-seed from the current time (in microseconds)
    seed = int(time.time() * 1e6) & 0xFFFFFFFF
    np.random.seed(seed)
    # Initial guess: evenly spaced means, unit sigmas, uniform weights
    theta0 = np.zeros(3 * K + 1)
    theta0[:K]     = np.random.uniform(np.min(x_subset), np.max(x_subset), K)
    theta0[K:2*K]  = 1.0
    theta0[2*K:]   = 1.0 / K
    theta0[-1]     = 1.0

    # Bounds for (mu, sigma, weights, convex)
    if(Pr_iter == 0.0):
        bounds = [(-np.inf, np.inf)] * K + \
            [(1e-2, np.inf)] * K + \
            [(0.0, 1.0)] * K + \
            [(1, 1)]
    else:
        bounds = [(-np.inf, np.inf)] * K + \
            [(1e-2, np.inf)] * K + \
            [(0.0, 1.0)] * K + \
            [(1.0, 1.0)]

    # Linear constraint: sum(weights) = 1
    A = np.zeros((1, 3 * K + 1))
    A[0, 2*K:3*K] = 1
    weight_sum = LinearConstraint(A, lb=[1], ub=[1])

    # Nonlinear constraint: PDF within [p0 - B, p0 + B]
    pdf_bounds = NonlinearConstraint(
        lambda th: gmm_constraint(K, th, x, p0, B),
        lb=0, ub=np.inf
    )

    # Objective: negative mass over x_subset
    def objective(th):
        if method == "analytical":
            p_new = gmm_integral_1dsubset(K, th, x_subset)
            return -(p_new * th[-1] + Pr_iter * (1-th[-1]))
        else:
            dx = x[1]-x[0]
            p_vals = mixture_pdf(K, th, x)
            mask = (x >= x_subset[0]) & (x <= x_subset[1])
            return -(np.sum(p_vals[mask])*dx * th[-1] + Pr_iter * (1-th[-1]))
        
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)

        result = minimize(
                    objective, theta0,
                    bounds=bounds,
                    constraints=[weight_sum, pdf_bounds],
                    method="SLSQP"
                )
        
        saw_outside_bounds = any(
            isinstance(w.message, Warning) and 
            "outside bounds" in str(w.message)
            for w in caught
        )
    # ensure incremental construction
    if(-result.fun < Pr_iter):
        result.success = False
    return result


def test_optimize_gmm_k(show_plots=True):
    # Example
    x_l, x_u, dx = -5.0, 5.0, 0.01
    x = np.arange(x_l, x_u + dx, dx)
    x_subset = np.array([-2.0, 3.0])

    np.random.seed(5)
    p0, _ = generate_base_pdf(x, num=2)
    B  = np.max(p0)*0.1

    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    Pr_subset = np.sum(p0[mask]) * dx

    K=12
    max_iter = 200; break_iter = 10
    success_iter = 0
    Pr_iter = 0.0
    p_gmm = 0.0*x
    for iter in range(max_iter):
        result = optimize_gmm_k(K, x, x_subset, p0, B, Pr_iter, method="analytical")
        # print(iter, result.success)
        if(result.success):
            success_iter += 1
            theta_gmm = result.x
            Pr_new = gmm_integral_1dsubset(K, theta_gmm, x_subset)
            p_gmm_new = mixture_pdf(K, theta_gmm, x)
  
            Pr_new_samples = np.sum(p_gmm_new[mask])*dx
            np.testing.assert_allclose(Pr_new_samples, Pr_new, atol=1e-1, rtol=1e-1)

            print("iter {:5d}, conv: {:.3f}, Pr_old {:.4f}, Pr_new {:.10f}".format(
                iter, theta_gmm[-1], Pr_iter, Pr_new))
            # print(theta_gmm)
            
            Pr_iter = Pr_new*(theta_gmm[-1]) + Pr_iter*(1-theta_gmm[-1])
            p_gmm = p_gmm_new*(theta_gmm[-1]) + p_gmm*(1-theta_gmm[-1])
        if(success_iter >= break_iter):
            break

    Pr_samples = np.sum(p_gmm[mask])*dx

    Pr_gmm_subset = Pr_iter
    print("Opt Prob: {:.4f} ({:.4f}), True Prob {:.4f}".format(Pr_gmm_subset, Pr_samples, Pr_subset))

    plot_Kmode_gmm_1d(show_plots, x, x_subset, p0, B, Pr_subset, p_gmm, Pr_gmm_subset) 

    # check if added slack in the constraint works
    np.testing.assert_array_less(np.abs(p0-p_gmm),
                                 B,
                                 err_msg="p_gmm cannot satisfy constraints of (p0 +/- B1)")
    
    np.testing.assert_array_less(Pr_subset, Pr_gmm_subset, err_msg="Pr_opt does not upper bound Pr")    


if __name__ == '__main__':
    test_optimize_gmm_k()
