import numpy as np
from scipy.stats import norm
from scipy.special import erf
import matplotlib.pyplot as plt
import cvxpy as cp


def generate_base_pdf(x, num, special=False):
    w = np.random.rand(num) + 1e-2
    sum_w = np.sum(w)
    w = w/sum_w
    mu = np.random.uniform(low=np.min(x), high=np.max(x), size=num)
    sigma = np.random.uniform(low=1e-1, high=6.0, size=num)

    if(special):
        w = np.array([0.5, 0.5])
        mu = np.array([-3.0, 3.0])
        sigma = np.array([0.7, 0.7])

    # initialize
    p0 = x*0.0
    for i in range(num):
        p0 = p0 + w[i]*norm.pdf(x, loc=mu[i], scale=sigma[i])
    params = (w, mu, sigma)
    theta = np.zeros(3 * num)
    theta[0:num] = mu
    theta[num:2*num]  = sigma
    theta[2*num:]   = w
    return p0, params, theta


def constraint_fun(params, x, p0, B):
    """
    Constraint function: For each x in our domain, ensure that the Gaussian PDF
    (parameterized by params) lies within [p0 - B, p0 + B].
    
    We define g(params) >= 0 when the constraint is satisfied, e.g.,
      (p0 + B) - pdf(x; params) >= 0  and  pdf(x; params) - (p0 - B) >= 0  for all x.
    
    To combine these, we can require that the minimum over x of the gap is nonnegative.
    """
    mu, sigma = params
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    # For all x, we need:
    #   pdf <= p0 + B   =>   (p0 + B) - pdf >= 0, and
    #   pdf >= p0 - B   =>   pdf - (p0 - B) >= 0.
    gap_upper = (p0 + B) - pdf   # should be >= 0 for all x
    gap_lower = pdf - (p0 - B)   # should be >= 0 for all x
    # Our constraint will be that both gaps are nonnegative; we return the minimal gap.
    return min(np.min(gap_upper), np.min(gap_lower))


def objective(params, x, x_subset, method):
    """
    Objective function to minimize.
    We wish to maximize the probability mass of the Gaussian (parameterized by params)
    over the subset x_subset. Since minimize minimizes, we return the negative of that mass.
    
    params: [mu, sigma] for our Gaussian pdf.
    x: discretized state domain.
    x_subset: [x_min, x_max] of the target subset.
    """
    if(method == "sample"):
        mu, sigma = params
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        # Compute the approximate integral (mass) over the target subset.
        mask = (x >= x_subset[0]) & (x <= x_subset[1])
        total_mass = np.sum(pdf[mask])
        return -total_mass  # negative because we maximize
    
    else: # analytical
        return -gmm_integral_1dsubset(params, x_subset)
    

def constraint_fun_iter(params, x, p0, B):
    """
    Constraint function: For each x in our domain, ensure that the Gaussian PDF
    (parameterized by params) lies within [p0 - B, p0 + B].
    
    We define g(params) >= 0 when the constraint is satisfied, e.g.,
      (p0 + B) - pdf(x; params) >= 0  and  pdf(x; params) - (p0 - B) >= 0  for all x.
    
    To combine these, we can require that the minimum over x of the gap is nonnegative.
    """
    mu, sigma, alpha = params
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    # For all x, we need:
    #   pdf <= p0 + B   =>   (p0 + B) - pdf >= 0, and
    #   pdf >= p0 - B   =>   pdf - (p0 - B) >= 0.
    gap_upper = (p0 + B) - pdf   # should be >= 0 for all x
    gap_lower = pdf - (p0 - B)   # should be >= 0 for all x
    # Our constraint will be that both gaps are nonnegative; we return the minimal gap.
    return min(np.min(gap_upper), np.min(gap_lower))


def objective_iter(params, x_subset, Pr_iter):
    """
    Objective function to minimize.
    """
    mu, sigma, alpha = params
    gmm_integral = gmm_integral_1dsubset((mu, sigma), x_subset)
    Pr = Pr_iter*(1-alpha) + gmm_integral*(alpha)
    return -Pr


def gmm_integral_1dsubset(params, x_subset):
    mu_opt, sigma_opt = params
    a = x_subset[0]
    b = x_subset[1]
    p_opt_integral = 0.5*(erf((b-mu_opt)/(np.sqrt(2)*sigma_opt)) - erf((a-mu_opt)/(np.sqrt(2)*sigma_opt)))
    return p_opt_integral


def gmm_tv_bound_1d(x_subset, p0, B, iter=1):
    max_p_value = np.max(p0) + B
    subset_vol_1d = np.max(x_subset)-np.min(x_subset)
    if(max_p_value <= 1):
        return 0.5*np.sqrt(subset_vol_1d)*1*np.sqrt(1/iter)
    else:
        return 0.5*np.sqrt(subset_vol_1d)* (subset_vol_1d*max_p_value**2) *np.sqrt(1/iter)


def plot_singlegmm_1dsubset(show_plots, x, x_subset, p0, B, params, Pr_opt_subset, Pr_subset):
    if(show_plots == False):
       return
    mu_opt, sigma_opt = params
    # Evaluate the optimized Gaussian pdf.
    optimized_pdf = norm.pdf(x, loc=mu_opt, scale=sigma_opt)
    
    # Compute the bounds based on the baseline pdf.
    lower_bound = p0 - B
    upper_bound = p0 + B

    # set_publication_plot_style()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, p0, label=r"$\hat{p}$", linestyle="--", color="gray")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.3,
                    label=r"$\hat{p} \pm B_1$")
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    plt.fill_between(x[mask], 0, upper_bound[mask], color="green", alpha=0.3, label=r"$X^'_{tar}$")  
    plt.plot(x, optimized_pdf, label=r"$p_{FO}$", color="blue")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(r"True $\mathbb{P}(X^'_{tar})$= "+str(np.round(Pr_subset,3))+
              r", Max $\mathbb{P}_{FO}(X^'_{tar})$="+str(np.round(Pr_opt_subset, 3))
              )
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()


def plot_gmm_1dsubset(show_plots, x, x_subset, p0, B, gmm_results, Pr_opt_subset, Pr_subset):
    if(show_plots == False):
       return

    optimized_pdf = x*0.0
    for i in range(len(gmm_results)):
        mu_i, sigma_i, alpha_i = gmm_results[i]
        pdf_i = norm.pdf(x, loc=mu_i, scale=sigma_i)
        optimized_pdf = optimized_pdf*(1-alpha_i) + pdf_i*alpha_i

    # Compute the bounds based on the baseline pdf.
    lower_bound = p0 - B
    upper_bound = p0 + B

    # set_publication_plot_style()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, p0, label=r"$\hat{p}$", linestyle="--", color="gray")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.3,
                    label=r"$\hat{p} \pm B_1$")
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    plt.fill_between(x[mask], 0, upper_bound[mask], color="green", alpha=0.3, label=r"$X^'_{tar}$")  
    plt.plot(x, optimized_pdf, label=r"$p_{FO}$", color="blue")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(r"True $\mathbb{P}(X^'_{tar})$= "+str(np.round(Pr_subset,3))+
              r", Max $\mathbb{P}_{FO}(X^'_{tar})$="+str(np.round(Pr_opt_subset, 3))
              )
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()


def plot_gmm_1dsubset_increment(show_plots, x, x_subset, p0, B, gmm_results, Pr_opt_subset, Pr_subset):
    if(show_plots == False):
       return
    
    # Compute the bounds based on the baseline pdf.
    lower_bound = p0 - B
    upper_bound = p0 + B

    # set_publication_plot_style()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, p0, label=r"$\hat{p}$", linestyle="--", color="gray")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.3,
                    label=r"$\hat{p} \pm B_1$")
    mask = (x >= x_subset[0]) & (x <= x_subset[1])

    optimized_pdf = x*0.0
    colors = plt.cm.rainbow(np.linspace(0, 1, len(gmm_results))); colors = colors[::-1]
    for i in range(len(gmm_results)):
        mu_i, sigma_i, alpha_i = gmm_results[i]
        pdf_i = norm.pdf(x, loc=mu_i, scale=sigma_i)
        optimized_pdf = optimized_pdf*(1-alpha_i) + pdf_i*alpha_i
        plt.plot(x, optimized_pdf, color=colors[i,:], linewidth=1.0)
    plt.fill_between(x[mask], 0, upper_bound[mask], color="green", alpha=0.3, label=r"$X^'_{tar}$")  

    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(r"True $\mathbb{P}(X^'_{tar})$= "+str(np.round(Pr_subset,3))+
              r", Max $\mathbb{P}_{FO}(X^'_{tar})$="+str(np.round(Pr_opt_subset, 3))
              )
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()


def plot_gmm_1dresults_increment(show_plots, gmm_results):
    if(show_plots == False):
        return
    fig = plt.figure(figsize=(8,6))
    iter_alphas = np.array(gmm_results)[:, 2]
    plt.plot(iter_alphas, marker='o', linestyle='-')
    plt.xlabel("iterations")
    plt.ylabel("convex weights")
    plt.show()


### Kmode gmm 1D ###
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


def gmm_constraint(K, theta, x, p0, B, info=None):
    """
    Ensure mixture PDF lies within [p0-B, p0+B] for all x.
    Returns a vector of >=0 values when satisfied.
    """
    slack = B*0.01
    # if(info is not None):
    #     slack = slack*(0.9**info)
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
        integral += w*0.5*( erf( (b-mu)/(np.sqrt(2)*sigma) ) - erf( (a-mu)/(np.sqrt(2)*sigma) ) ) # erfc = 1 - erf (is numerically more stable)
    return integral


# testing analytical jacobian
def gmm_integral_jacobian(K, theta, x_subset):
    """
    Gradient of the GMM integral over a 1D subset [a, b] using erf.
    theta = [mu_1,...,mu_K, sigma_1,...,sigma_K, w_1,...,w_K]
    Returns grad: shape (3K,)
    """
    a, b = x_subset
    mus     = theta[0:K]
    sigmas  = theta[K:2*K]
    weights = theta[2*K:3*K]

    grad = np.zeros_like(theta)
    for i in range(K):
        mu = mus[i]
        sigma = sigmas[i]
        w = weights[i]

        u_b = (b - mu) / (np.sqrt(2) * sigma)
        u_a = (a - mu) / (np.sqrt(2)* sigma)
        exp_b = np.exp(-u_b**2)
        exp_a = np.exp(-u_a**2)

        # ∂/∂mu_i
        grad[i] = w/np.sqrt(np.pi)*(exp_b*(-1/(np.sqrt(2)*sigma)) - exp_a*(-1/(np.sqrt(2)*sigma)))

        # ∂/∂sigma_i
        grad[K + i] = w/np.sqrt(np.pi)*(exp_b*(-1)*u_b/sigma + exp_a*(-1)*u_a/sigma)

        # ∂/∂w_i
        grad[2*K + i] = 0.5 * (erf(u_b) - erf(u_a))

    return grad


def plot_Kmode_gmm_1d(show_plots, x, x_subset, p0, B, Pr_subset, p_gmm, Pr_gmm_subset):
    if(show_plots == False):
       return

    # Compute the bounds based on the baseline pdf.
    lower_bound = p0 - B
    upper_bound = p0 + B

    # set_publication_plot_style()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, p0, label=r"$\hat{p}$", linestyle="--", color="gray")
    plt.plot(x, p_gmm, label=r"$p_{GMM}$", linestyle="-", color="blue")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.3,
                    label=r"$\hat{p} \pm B_1$")
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    plt.fill_between(x[mask], 0, upper_bound[mask], color="green", alpha=0.3, label=r"$X^'_{tar}$")  
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(r"True $\mathbb{P}(X^'_{tar})$= "+str(np.round(Pr_subset,3))+
              r", Max $\mathbb{P}_{FO}(X^'_{tar})$="+str(np.round(Pr_gmm_subset, 3))
              )
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()


def plot_convergence_test(show_plots, problem, all_histories, show_extrapolation=False):
    if(show_plots == False):
        return
    
    Pr_subset = problem['Pr_subset']
    Pr_max = problem['Pr_max']
    x = problem['x']
    x_subset = problem['x_subset']
    p0 = problem['p0']
    B  = problem['B']
    
    print("Prob. (Approximate Integral) Direct: {:.4f}, Negation: {:.4f}".format(problem['Pr_max'], problem['Pr_neg']))
    print("Prob. LP: {:.4f}".format(problem['Pr_lp']))

    fig = plt.figure(figsize=(8, 6))
    colors = plt.cm.spring(np.linspace(0, 1, len(all_histories)))
    for i in range(len(all_histories)):
        lower_bound = np.array(all_histories[i]['Pr_iter'])
        upper_bound = np.array(all_histories[i]['Pr_iter'])+ np.array(all_histories[i]['tv_bound'])
        plt.fill_between(all_histories[i]['iter'], lower_bound, upper_bound, color=colors[i,:], alpha=0.3)
        plt.plot(all_histories[i]['iter'], lower_bound, marker='*', color=colors[i,:], alpha=0.3, label=str(i)+r"-trial: $P_{GMM}$")
        plt.plot(all_histories[i]['iter'], upper_bound, marker='o', color=colors[i,:], alpha=0.3, label=str(i)+r"-trial: $P_{GMM}+TV$")
        print("trial: {:2d} Prob. Pr_gmm: {:.4f}".format(i, lower_bound[-1]))
        
        if(show_extrapolation):
            factor_tv_bound = np.array(all_histories[i]['tv_bound'])[0]
            Pr_iter_asymp = all_histories[i]['Pr_iter'][-1]
            N_extr = 5000
            x_extr = np.arange(all_histories[i]['iter'][-1], N_extr, step=N_extr//100)
            ub_extr = factor_tv_bound/(np.sqrt(x_extr))
            plt.plot(x_extr, ub_extr + Pr_iter_asymp, color=colors[i, :], linestyle=":", label="extrapolation")
            plt.plot(x_extr, ub_extr*0 + Pr_iter_asymp, color=colors[i, :], linestyle=":")
            plt.fill_between(x_extr, ub_extr*0 + Pr_iter_asymp, ub_extr + Pr_iter_asymp, color=colors[i,:], alpha=0.1)

    plt.axhline(y=Pr_subset, color="black", linestyle="--", label=r"True $\mathbb{P}$")
    colors = plt.cm.winter(np.linspace(0, 1, 3)); colors = colors[::-1]
    plt.axhline(y=Pr_max, color=colors[0,:], linestyle="-", label=r"$\mathbb{P}_{Integral}$")
    plt.axhline(y=problem['Pr_neg'], color=colors[1,:], linestyle="-", label=r"$\mathbb{P}_{Neg. Integral}$")
    plt.axhline(y=problem['Pr_lp'], color=colors[2,:], linestyle="-", label=r"$\mathbb{P}_{LP}$")
    plt.legend()
    plt.ylabel("Prob.")
    plt.xlabel("increment iteration")
    plt.ylim([0, 2.0*np.min(upper_bound)])
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    upper_bound = p0+B
    lower_bound = p0-B
    plt.fill_between(x, upper_bound, lower_bound, color="gray", alpha=0.4,
                label=r"$\hat{p} \pm B_1$")
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    plt.fill_between(x[mask], lower_bound[mask], upper_bound[mask], color="green", alpha=0.15, label=r"$X^'_{tar}$")  
    plt.plot(problem['x_lp'], problem['pdf_lp'], color="black", linestyle="--", label=r"$f_{LP}$")
    for i in range(len(all_histories)):
    # for i in range(0, 1):
        theta_history = all_histories[i]['theta_iter']
        colors = plt.cm.spring(np.linspace(0, 1, len(theta_history)))
        for j in range(len(theta_history)):
            theta_j = theta_history[j]
            K_j = theta_j.shape[0] // 3
            p_gmm = mixture_pdf(K_j, theta_j, x)
            # plt.plot(x, p_gmm, linewidth=1.0, color=colors[j,:], alpha=0.5)
            if(j == len(theta_history)-1):
                plt.plot(x, p_gmm, linewidth=1.5, color=colors[j,:], alpha=1.0)
                plt.scatter(x[::10], p_gmm[::10], s=16, marker="*", color=colors[j,:], 
                            # label=r"$f_{GM}$"+"[trial="+str(i) +", N=" + str(K_j) + "]")
                            label=r"$f_{GM}$"+"[trial="+str(i) +", final]", alpha=1.0)
            # if(j == 0):
            #     plt.plot(x, p_gmm, linewidth=1.0, color=colors[j,:], alpha=1.0)
            #     plt.scatter(x[::10], p_gmm[::10], s=8, marker="o", color=colors[j,:], 
            #                 # label=r"$f_{GM}$"+"[trial="+str(i) +", N=" + str(K_j) + "]", alpha=alpha_j)
            #                 label=r"$f_{GM}$"+"[trial="+str(i) +", init]", alpha=1.0)
                
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    for i in range(len(all_histories)):
    # for i in range(0, 1):
        theta_history = all_histories[i]['theta_iter']
        theta_end = theta_history[-1]
        K = theta_end.shape[0] // 3
        mus = theta_end[0:K]
        sigmas = theta_end[K:2*K]
        ws = theta_end[2*K:3*K]
        plt.hist(sigmas, bins=30)
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()


def optimize_negation(problem):
    x = problem['x']
    x_subset = problem['x_subset']
    dx = x[1] - x[0]
    p0 = problem['p0']
    B = problem['B']

    # Define the subset x_sub (e.g., points between -1 and 1).
    sub_idx = np.where((x < x_subset[0]) | (x > x_subset[1]))[0]
    p0_subset = p0[sub_idx]
    Pr = 0.0
    for p in p0_subset:
        if((p-B) >= 0):
            Pr += (p-B)*dx
    return 1 - Pr
    

def optimize_linearprogram(problem):
    """
    linear program 
    # NOTE: we assume that the discretization over state space is small enough such that
    #   p_ub and p_lb can be directly calculated by the vertices of the cell
    """
    x = problem['x']
    x_subset = problem['x_subset']
    n = len(x)-1
    dx = x[1] - x[0]
    p0 = problem['p0']
    B = problem['B']

    # Define the subset x_sub (e.g., points between -1 and 1).
    sub_idx = np.where((x >= x_subset[0]) & (x <= x_subset[1]))[0]
    
    # Decision variable: p, representing the probability at each discretized point.
    p = cp.Variable(n)
    p_ub = np.zeros(n)
    p_lb = np.zeros(n)
    x_center = np.zeros(n)
    for i in range(n):
        p_ub[i] = max(p0[i], p0[i+1])
        p_lb[i] = min(p0[i], p0[i+1])
        x_center[i] = 0.5*(x[i]+x[i+1])
    
    # Define constraints.
    constraints = [
        cp.sum(p) * dx <= 1,  # Total probability integrates to 1.
        p >= 0,               # Non-negativity.
    ]
    
    # Each p(x_j) must lie in [p0(x_j)-B, p0(x_j)+B].
    for i in range(n):
        constraints += [p[i] >= p_ub[i] - B,
                        p[i] <= p_lb[i] + B]
    
    # Objective: maximize the total probability mass over the subset x_sub.
    objective = cp.Maximize(cp.sum(p[sub_idx]) * dx)
    
    # Set up and solve the problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result, np.array(p.value), x_center


def update_if_sat(K, Pr_new, theta_new, p_gmm_new, problem):
    """
    Performs a local line search to update GMM parameters if PDF constraint is satisfied.

    Returns:
        Pr_new (float)
        theta_new (np.ndarray)
        p_gmm_new (np.ndarray)
        sat (bool) – whether update was successful
    """
    x = problem['x']
    x_subset = problem['x_subset']
    p0 = problem['p0']
    B = problem['B']

    slack = B * 0.01
    line_search = 1.0
    sat = False
    while not sat:
        jac = gmm_integral_jacobian(K, theta_new, x_subset)
        theta_try = theta_new + line_search * jac
        # Normalize weights
        w_try = theta_try[2*K:3*K]
        w_sum = np.sum(w_try)
        w_try = w_try / w_sum
        theta_try[2*K:3*K] = w_try
        # Compute new integral and PDF
        Pr_try = gmm_integral_1dsubset(K, theta_try, x_subset)
        pdf_try = mixture_pdf(K, theta_try, x)
        # Check constraint satisfaction
        sat = np.max(np.abs(pdf_try - p0)) + slack <= B
        line_search *= 0.99
        if line_search < 1e-4:
            break
    if sat:
        np.testing.assert_array_less(0, Pr_try-Pr_new, err_msg="gradient direction is not increasing Prob.")
        print("SAT {:.6f}".format(Pr_try - Pr_new))
        return Pr_try, theta_try, pdf_try, True
    else:
        return Pr_new, theta_new, p_gmm_new, False
