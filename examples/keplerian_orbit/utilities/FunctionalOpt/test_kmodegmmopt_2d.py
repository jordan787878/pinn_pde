import numpy as np
import pickle
import warnings
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from util_test_gmmopt import generate_base_pdf_2d, plot_pdf_2d, solve_linearprogram, optimize_linearprogram_2d
from util_test_gmmopt import gmm_constraint_2d, gmm_integral_2dsubset, mixture_pdf_2d


def optimize_gmm_2d(K, info, problem, inner_iterations=10):
    """
    Analogous to your 1D version but in 2D.
    - x:    (N_pts, 2) full‐grid points
    - x_subset: [[x1_l,x1_u],[x2_l,x2_u]]
    - p0:   target pdf on the full grid, shape (N_pts,)
    - B:    bound tolerance
    - info: dict carrying previous Pr_old, theta_old, etc.
    """
    x = problem['x']
    x_subset = problem['x_subset']
    p0 = problem['p0']
    B = problem['B']

    # 1) initial guess
    min_sigma = 1e-2
    theta0 = np.zeros(5*K)
    # means random in bounding box:
    theta0[     :   K] = np.random.uniform(x[:,0].min(), x[:,0].max(), K)
    theta0[K   : 2*K] = np.random.uniform(x[:,1].min(), x[:,1].max(), K)
    # sigmas = 1
    theta0[2*K : 4*K] = 1.0
    # uniform weights
    theta0[4*K : 5*K] = 0.0

    theta_p0 = info['theta_p0']
    K_old = 2
    for i in range(5):
        theta0[i*K : i*K+K_old] = theta_p0[i*K_old : i*K_old + K_old]

    # 2) bounds on each block of theta
    b_mu   = [(-np.inf, np.inf)] * (2*K)
    b_sig  = [(min_sigma, np.inf)] * (2*K)
    b_w    = [(0.0, 1.0)] * K
    bounds = b_mu + b_sig + b_w

    # 3) constraint: sum(weights)=1
    A = np.zeros((1, 5*K))
    A[0, 4*K:5*K] = 1
    weight_sum = LinearConstraint(A, lb=1.0, ub=1.0)

    # 4) constraint: pdf within [p0-B, p0+B]
    pdf_bounds = NonlinearConstraint(
        lambda th: gmm_constraint_2d(K, th, x, p0, B),
        lb=0, ub=np.inf
    )

    # 5) objective: maximize mass in subset
    # Objective with penalty
    def objective(theta):
        # primary objective: maximize probability in subset
        Pr = gmm_integral_2dsubset(K, theta, x_subset)
        obj = -Pr
        # penalty for bound violations
        penalty_coef = 1e+10
        p_mix = mixture_pdf_2d(K, theta, x)
        lower_violation = np.clip((p0 - B) - p_mix, 0, None)
        upper_violation = np.clip(p_mix - (p0 + B), 0, None)
        penalty = np.sum(lower_violation**2 + upper_violation**2)
        return obj + penalty_coef * penalty

    cons = [weight_sum]

    # --- solving ---
    best_res = None
    Pr_max = 0.0
    for jj in range(inner_iterations):
        print(jj, Pr_max)
        res = minimize(
            objective, theta0,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={'maxiter':100, 'disp':False},
            tol=1e-3,
        )

        # 6) feasibility check
        p_mix_opt = mixture_pdf_2d(K, res.x, x)
        lower_res = p_mix_opt - (p0 - B)
        upper_res = (p0 + B) - p_mix_opt
        if np.any(lower_res < 0) or np.any(upper_res < 0):
            # violation detected → discard solution
            print("⚠️ Constraint violated. No feasible solution returned.")
            res = None

        if(res is not None):
            Pr = gmm_integral_2dsubset(K, res.x, x_subset)
            if(Pr > Pr_max):
                Pr_max = Pr
                best_res = res
                theta0 = res.x

        # perturb the initialization
        noise_scale_mu = 1e-2
        noise_scale_std = 1e-3
        theta0[    :  K] += np.random.randn(K) * noise_scale_mu
        theta0[K  :2*K] += np.random.randn(K) * noise_scale_mu 
        theta0[2*K  :4*K] += np.random.randn(2*K) * noise_scale_std 
        theta0[2*K : 4*K] = np.clip(theta0[2*K : 4*K], a_min=min_sigma, a_max=None)

    return best_res


def test_convergence(data_folder, load_data=False):
    if(load_data):
        with open(data_folder +'problem.pkl', 'rb') as f:
            problem = pickle.load(f)
        plot_pdf_2d(problem)
        return

    # --- problem setup ---
    np.random.seed(0)

    # domain
    x1_l, x1_u, dx1 = -6.0, 6.0, 0.1
    x2_l, x2_u, dx2 = -6.0, 6.0, 0.1
    dV = dx1*dx2
    x1 = np.arange(x1_l, x1_u + dx1, dx1)
    x2 = np.arange(x2_l, x2_u + dx2, dx2)
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing="ij")
    x = np.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
    # region of interest
    x_subset = np.array([[-3.0, 3.0],
                         [-6.0, 6.0]]) # deterministic
    # x_subset = np.random.uniform(x_l, x_u, size=2); x_subset.sort() # random

    mask = np.all((x >= x_subset[:, 0]) & (x <= x_subset[:, 1]), axis=1)
    
    # p_hat and error bound
    K_p0 = 2
    p0, _, theta_p0 = generate_base_pdf_2d(x1_grid, x2_grid, num=K_p0)
    B  = np.max(p0)*np.random.uniform(0.1, 0.2)
    
    Pr_subset = np.sum(p0.flatten()[mask]) * dV
    # Pr_subset = gmm_integral_1dsubset(K_p0, theta_p0, x_subset)
    # Pr_max = B*(x_subset[1]-x_subset[0])+Pr_subset
    problem = {
        'x1_grid': x1_grid, 'x2_grid': x2_grid,
        'x': x, 'x_subset': x_subset,
        'p0_grid': p0,
        'p0': p0.flatten(), 'B': B,
        'Pr_subset': Pr_subset, 
        'mask': mask,
        'dV': dV
    }

    # --- linear program comparision ---
    Pr_lp_new, pdf_lp, x_lp = optimize_linearprogram_2d(problem)
    pdf_lp = pdf_lp.reshape(x1_grid.shape)
    Pr_lp, _ = solve_linearprogram(problem)
    problem["Pr_LP"] = Pr_lp_new
    problem["pdf_LP"] = pdf_lp

    # --- FO solving ---
    info = {
        'Pr_old': 0.0,
        'theta_p0': theta_p0,
    }
    K_gmm = 4
    result = optimize_gmm_2d(K_gmm, info, problem, inner_iterations=200)
    theta_fo = result.x
    Pr_fo_analy = gmm_integral_2dsubset(K_gmm, theta_fo, x_subset)
    pdf_fo = mixture_pdf_2d(K_gmm, theta_fo, x)
    pdf_fo = pdf_fo.reshape(x1_grid.shape)
    problem["Pr_FO"] = Pr_fo_analy
    problem["pdf_FO"] = pdf_fo

    # summarize test
    print("Pr true: {:.4f}, LP: {:.4f}, FO: {:.4f}".format(problem['Pr_subset'], Pr_lp_new, Pr_fo_analy))
    plot_pdf_2d(problem)
    # --- save to file ---
    with open(data_folder +'problem.pkl', 'wb') as f:
        pickle.dump(problem, f)


if __name__ == '__main__':
    data_folder = 'data/case5/'
    test_convergence(data_folder, load_data=True)
