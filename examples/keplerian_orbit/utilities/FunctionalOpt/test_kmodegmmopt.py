import numpy as np
import pickle
import warnings
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from util_test_gmmopt import *


def optimize_gmm_k(K, x, x_subset, p0, B, strategy, info):
    """
    Optimize a K-component GMM to maximize probability mass over x_subset
    subject to PDF bounds [p0 - B, p0 + B] over the full domain x.

    Returns the SciPy OptimizeResult with .x = optimal theta.

    Inputs:
        strategy: {'baseline', 'inverse_square'}
        info: { 
                'Pr_old': optimized Probabililty of previous iteration
                'theta_old': optimized gmm paramters of previous iteration
                'success_iter': success_iter,
                'c': Pr_iter at first success_iter
               }
    """
    # Initial guess: uniformly spaced means, unit sigmas, uniform weights
    min_sigma = 1e-1
    theta0 = np.zeros(3 * K + 1)
    theta0[:K]     = np.random.uniform(np.min(x), np.max(x), K)
    theta0[K:2*K]  = 1.0 # NOTE: plot the distribution of thetas
    theta0[2*K:3*K]   = 1.0 / K
    theta0[-1]     = 1.0

    # Bounds for (mu, sigma, weights, convex)
    if(info['success_iter'] == 0):
        bounds = [(-np.inf, np.inf)] * K + \
            [(min_sigma, np.inf)] * K + \
            [(0.0, 1.0)] * K + \
            [(1.0, 1.0)]
    else:
        bounds = [(-np.inf, np.inf)] * K + \
            [(min_sigma, np.inf)] * K + \
            [(0.0, 1.0)] * K + \
            [(1.0, 1.0)]
    
    # using the theta gmm of previous iteration
    if(info['theta_old'] is not None):
        theta_old = info['theta_old']
        K_old = theta_old.shape[0] // 3
        mus    = theta_old[0:K_old]
        sigmas = theta_old[K_old:2*K_old]
        ws = theta_old[2*K_old:3*K_old]
        for i in range(K_old):
            theta0[i] = mus[i] + np.random.normal(loc=0, scale=0.1)
            theta0[K+i] = np.clip(sigmas[i]+ np.random.normal(loc=0, scale=0.1), a_min=min_sigma, a_max=np.inf)
            theta0[2*K+i] = ws[i]
        # ensure weights sum to 1
        w_sum = np.sum(theta0[2*K:3*K])
        theta0[2*K:3*K] /= w_sum
        # print(theta_old); print(theta0)

    # Linear constraint: sum(weights) = 1
    A = np.zeros((1, 3 * K + 1))
    A[0, 2*K:3*K] = 1
    weight_sum = LinearConstraint(A, lb=[1], ub=[1])

    # Nonlinear constraint: PDF within [p0 - B, p0 + B]
    pdf_bounds = NonlinearConstraint(
        lambda th: gmm_constraint(K, th, x, p0, B),
        lb=0, ub=np.inf
    )

    constraints=[weight_sum, pdf_bounds]

    # Objective: negative mass over x_subset
    def objective(th):
        if strategy['integral'] == "analytical":
            p_new = gmm_integral_1dsubset(K, th, x_subset)
            return -(p_new* th[-1] + info['Pr_old'] * (1-th[-1]))
        else:
            dx = x[1]-x[0]
            p_vals = mixture_pdf(K, th, x)
            mask = (x >= x_subset[0]) & (x <= x_subset[1])
            return -(np.sum(p_vals[mask])*dx * th[-1] + info['Pr_old'] * (1-th[-1]))
        
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = minimize(
                    objective, theta0,
                    bounds=bounds,
                    constraints=constraints,
                    method="SLSQP",
                    options={'maxiter': 50, 'disp': False},
                    tol=1e-3
                )
    # non-decreasing incremental construction
    Pr_new = -result.fun
    if(Pr_new <= info['Pr_old']):
        result.success = False

    if(strategy['increment'] == "inverse_square"):
        if(info['success_iter'] > 0):
            fac = info['success_iter']
            Pr_thres = info['Pr_old'] + \
                info['rho']*(1+info['rho'])*(1/(fac*(fac+info['rho'])) - 1/((fac+1)*(fac+1+info['rho'])))
            # print(Pr_new, Pr_thres)
            if(Pr_new <= Pr_thres):
                result.success = False
                info['rho'] *= 0.99

    return result, info


def test_convergence(data_folder, load_data=False):
    if(load_data):
        with open(data_folder +'problem.pkl', 'rb') as f:
            problem = pickle.load(f)
        with open(data_folder+'all_histories.pkl', 'rb') as f:
            all_histories = pickle.load(f)
        plot_convergence_test(True, problem, all_histories, show_extrapolation=True)
        return

    # --- problem setup ---
    # np.random.seed(11)
    # domain
    x_l, x_u, dx = -6.0, 6.0, 0.02
    x = np.arange(x_l, x_u + dx, dx)
    # region of interest
    # x_subset = np.array([-4.0, 4.0]) # deterministic
    x_subset = np.random.uniform(x_l, x_u, size=2); x_subset.sort() # random
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    
    # p_hat and error bound
    K_p0 = 3
    p0, _, theta_p0 = generate_base_pdf(x, num=K_p0)
    # B  = np.max(p0)*np.random.uniform(0.1, 0.2)
    B  = 0.04
    
    # Pr_subset = np.sum(p0[mask]) * dx
    Pr_subset = gmm_integral_1dsubset(K_p0, theta_p0, x_subset)
    Pr_max = B*(x_subset[1]-x_subset[0])+Pr_subset
    problem = {
        'x': x, 'x_subset': x_subset,
        'p0': p0, 'B': B,
        'Pr_subset': Pr_subset, 'Pr_max': Pr_max,
        'mask': mask, 'dV': dx
    }

    # --- linear program comparision ---
    Pr_lp, pdf_lp, x_lp = optimize_linearprogram(problem)
    Pr_lp_new, _ = solve_linearprogram(problem)
    print("[debug]: ", Pr_lp, Pr_lp_new)
    problem['Pr_lp'] = Pr_lp
    problem['Pr_neg'] = optimize_negation(problem)
    problem['pdf_lp'] = pdf_lp
    problem['x_lp'] = x_lp

    # --- gmm optimization setup ---
    num_test_trials = 1
    K_gmm = 3
    max_iter = 100
    break_iter = 10
    strategy = {
        'integral' : 'analytical', # this specify how to compute gmm integral over a region
        'increment' : 'inverse_square' # this specify the incremental construction approach 
    }
    all_histories = []
    k_incret = K_gmm

    for iter_test in range(num_test_trials):
        # --- initialization ---
        history = {
            'iter':[],
            'Pr_iter': [],
            'tv_bound': [],
            'theta_iter': [],
            'delta':[],
        }
        K = K_gmm
        info = { 
                'Pr_old': 0.0,
                'theta_old': None,
                'success_iter': 0,
                'rho': 1.0,
               }
        p_gmm = 0.0*x
        # --- solving ---
        for iter in range(max_iter):
            result, info = optimize_gmm_k(K, x, x_subset, p0, B, strategy, info)
            if(result.success):
                info['success_iter'] += 1
                theta_new = result.x
                Pr_new = gmm_integral_1dsubset(K, theta_new, x_subset)
                p_gmm_new = mixture_pdf(K, theta_new, x)
    
                Pr_new_samples = np.sum(p_gmm_new[mask])*dx
                np.testing.assert_allclose(Pr_new_samples, Pr_new, atol=1e-1, rtol=1e-1)
                delta = 1-Pr_new # distance between max Prob.(which is 1) - Pr_new
                print("i {:3d}, increment {:3d}, Pr_new {:.8f}, delta {:.4f}".format(
                      iter, info['success_iter'], Pr_new, delta))
                
                # -- local line search along gradient direction ---
                #   NOTE: there seems little difference when line search is enabled
                # Pr_new, theta_new, p_gmm_new, _ = update_if_sat(K, Pr_new, theta_new, p_gmm_new, problem)

                # --- update the old result --
                info['Pr_old'] = Pr_new
                info['theta_old'] = theta_new
                # info['rho'] = 1.0 # reset
                p_gmm = p_gmm_new
                K = K + k_incret # direct incremental construction with additional gaussian mixtures

                history['iter'].append(info['success_iter'])
                history['Pr_iter'].append(Pr_new)
                history['tv_bound'].append(gmm_tv_bound_1d(x_subset, p0, B, iter=info['success_iter']))
                history['theta_iter'].append(theta_new)
                history['delta'].append(delta)
            # else:
            #     print("i {:3d}, rho {:.8f}".format(iter, info["rho"]))

            if(info['success_iter'] >= break_iter):
                break
        # --- storing solver history & testing ---
        all_histories.append(history)
        tv_bound = gmm_tv_bound_1d(x_subset, p0, B, iter=info['success_iter'])
        Pr_gmm_subset = Pr_new
        Pr_gmm_with_tv = Pr_gmm_subset+tv_bound
        print("(Opt Prob: {:.4f} + TV: {:.4f}) = {:.4f}, True Prob. {:.4f}, Direct Inegral Prob. {:.4f}".format(
            Pr_gmm_subset, tv_bound, Pr_gmm_with_tv, Pr_subset, Pr_max))
        np.testing.assert_array_less(np.max(np.abs(p0-p_gmm)),
                                     B,
                                     err_msg="p_gmm cannot satisfy constraints of (p0 +/- B1)")
        np.testing.assert_array_less(Pr_subset, 
                                     Pr_gmm_with_tv, 
                                     err_msg="(Pr_gmm + TV) does not upper bound Pr_subset")
        np.testing.assert_array_less(Pr_gmm_subset, 
                                     Pr_max, 
                                     err_msg="Pr_gmm does not lower bound Pr_max")
    # summarize test
    # --- save to file ---
    with open(data_folder +'problem.pkl', 'wb') as f:
        pickle.dump(problem, f)
    with open(data_folder+'all_histories.pkl', 'wb') as f:
        pickle.dump(all_histories, f)
    plot_convergence_test(True, problem, all_histories)


if __name__ == '__main__':
    data_folder = 'data/case2/'
    test_convergence(data_folder, load_data=True)
