import numpy as np
from scipy.stats import norm
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import cvxpy as cp


def set_publication_plot_style(font_family='Times New Roman', font_size=18):
    """
    Update Matplotlib settings to use publication-ready fonts.

    Parameters:
        font_family (str): Font family to be used for all texts.
        font_size (int): Base font size for labels, titles, legends, and ticks.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.titlesize'] = font_size
    plt.rcParams['lines.linewidth'] = 2


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


# --- Kmode gmm 1D ---
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
    
    set_publication_plot_style()

    Pr_subset = problem['Pr_subset']
    Pr_max = problem['Pr_max']
    x = problem['x']
    x_subset = problem['x_subset']
    p0 = problem['p0']
    B  = problem['B']
    Pr_ni = min(problem['Pr_max'], problem['Pr_neg'])
    Pr_lp = problem['Pr_lp']
    Pr_fo = []
    
    print("Prob. (Approximate Integral) Direct: {:.4f}, Negation: {:.4f}".format(problem['Pr_max'], problem['Pr_neg']))
    print("Prob. LP: {:.4f}".format(Pr_lp))

    fig = plt.figure(figsize=(8, 6))
    colors = plt.cm.spring(np.linspace(0, 1, len(all_histories)))
    for i in range(len(all_histories)):
        lower_bound = np.array(all_histories[i]['Pr_iter'])
        upper_bound = np.array(all_histories[i]['Pr_iter'])+ np.array(all_histories[i]['tv_bound'])
        plt.fill_between(all_histories[i]['iter'], lower_bound, upper_bound, color=colors[i,:], alpha=0.3)
        plt.plot(all_histories[i]['iter'], lower_bound, marker='*', color=colors[i,:], alpha=0.3, label=str(i)+r"-trial: $P_{GMM}$")
        plt.plot(all_histories[i]['iter'], upper_bound, marker='o', color=colors[i,:], alpha=0.3, label=str(i)+r"-trial: $P_{GMM}+TV$")
        print("trial: {:2d} Prob. Pr_gmm: {:.4f}".format(i, lower_bound[-1]))
        Pr_fo.append(lower_bound[-1])
        
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
    plt.plot(x, p0, linewidth=1.5, color="black", linestyle="--", label=r"$p$") 
    plt.plot(problem['x_lp'], problem['pdf_lp'], linewidth=1.5, color="blue", linestyle="-", label=r"$p_i$ (LP)")
    plt.title(r"$\mathbb{P}$"+" = {:.4f},   ".format(Pr_subset) +\
              r"$\mathbb{P}^+_{NI}$"+" = {:.4f}, ".format(Pr_ni) +\
              r"$\mathbb{P}^+_{LP}$"+" = {:.4f}, ".format(Pr_lp) +\
              r"$\mathbb{P}^+_{FO}$"+" = {:.4f}".format(Pr_fo[0]))
    # for i in range(len(all_histories)):
    for i in range(0, 1):
        theta_history = all_histories[i]['theta_iter']
        colors = plt.cm.Reds(np.linspace(0, 1, len(theta_history)))
        for j in range(len(theta_history)):
        # for j in range(1):
            theta_j = theta_history[j]
            K_j = theta_j.shape[0] // 3
            p_gmm = mixture_pdf(K_j, theta_j, x)
            plt.plot(x, p_gmm, linewidth=0.5, color=colors[j,:], alpha=0.5)
            if(j == len(theta_history)-1):
                plt.plot(x, p_gmm, linewidth=1.5, color=colors[j,:], alpha=1.0)
                plt.scatter(x[::10], p_gmm[::10], s=16, marker="*", color=colors[j,:], 
                            label=r"$\phi$ (FO)"+", [N=" + str(K_j*3) + "]",
                            # label=r"$f_{GM}$"+"[trial="+str(i) +", final]", 
                            alpha=1.0)
            # if(j == 0):
            #     plt.plot(x, p_gmm, linewidth=1.0, color=colors[j,:], alpha=1.0)
            #     plt.scatter(x[::10], p_gmm[::10], s=8, marker="o", color=colors[j,:], 
            #                 # label=r"$f_{GM}$"+"[trial="+str(i) +", N=" + str(K_j) + "]", alpha=alpha_j)
            #                 label=r"$f_{GM}$"+", [N=" + str(K_j*3) + "]",
            #                 # label=r"$f_{GM}$"+"[trial="+str(i) +", init]"
            #                 alpha=1.0)
                
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    # fig.savefig("figs/1dexample_case3.pdf", format='pdf'); plt.close()
    plt.show()

    # fig = plt.figure(figsize=(8, 6))
    # for i in range(len(all_histories)):
    # # for i in range(0, 1):
    #     theta_history = all_histories[i]['theta_iter']
    #     theta_end = theta_history[-1]
    #     K = theta_end.shape[0] // 3
    #     mus = theta_end[0:K]
    #     sigmas = theta_end[K:2*K]
    #     ws = theta_end[2*K:3*K]
    #     plt.hist(sigmas, bins=30)
    # plt.grid(True)
    # plt.tight_layout(pad=0.2)
    # plt.show()


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


def solve_linearprogram(problem):
    """
    linear program 
    # NOTE: we assume that the discretization over state space is small enough such that
    #   p_ub and p_lb can be directly calculated by the vertices of the cell
    """
    B = problem['B']
    p0 = problem['p0']
    mask = problem['mask']
    dV = problem['dV']

    # --- Solving via specialized ----
    delta = 1e-6
    # sorting
    front = p0[mask]    # all the True’s
    front.sort()
    front[:] = front[::-1]
    front_count = front.shape[0]
    back  = p0[~mask]   # all the False’s
    p0 = np.concatenate([front, back])
    # init
    Pr = 0.0
    p_ub = p0+B
    p_lb = np.clip(p0-B, 0.0, None)
    p_result = p_lb
    for i in range(p0.shape[0]):
        p_ub_i = (p_ub[i])*dV
        sum_p_lb_rest = (p_lb[i:].sum())*dV
        if(Pr + p_ub_i + sum_p_lb_rest <= 1):
            p_result[i] = p_ub_i/dV
        else:
            p_result[i] = (1.0 - Pr - sum_p_lb_rest)/dV
        Pr += p_result[i]*dV
        # print(i, p_result[i]*dV, Pr)
        if(i >= front_count-1):
            break
        if(p_result[i]*dV < delta):
            break
    p_total = p_result.sum()*dV
    print(" [Solved] Pr_tar: {:.4f}, Pr_total: {:.4f}".format(Pr, p_total))
    deviation = np.abs(p_result - p0)
    np.testing.assert_array_less(deviation, B+delta)
    np.testing.assert_array_less(p_total, 1.0+delta)
    return Pr, None


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


# --- Kmode gmm 2D ---
def generate_base_pdf_2d(X, Y, num):
    """
    X, Y : 2D arrays of the same shape (e.g. from np.meshgrid)
    num   : number of Gaussian components
    special: if True, use a fixed 2-component toy example
    """
    # 1) initialize weights, means and stds
    w = np.random.rand(num) + 1e-2
    w /= w.sum()
    # for 2D, mu_i = [mu_x, mu_y]
    mu = np.column_stack([
        np.random.uniform(X.min(), X.max(), size=num),
        np.random.uniform(Y.min(), Y.max(), size=num)
    ])
    # diagonal std devs
    sigma = np.column_stack([
        np.random.uniform(0.1, 6.0, size=num),
        np.random.uniform(0.1, 6.0, size=num)
    ])

    # 2) build the density on the grid
    # flatten for easy summation
    xs = X.ravel()
    ys = Y.ravel()
    p0_flat = np.zeros_like(xs)

    for i in range(num):
        # product of independent normals
        p0_flat += (
            w[i]
            * norm.pdf(xs, loc=mu[i,0], scale=sigma[i,0])
            * norm.pdf(ys, loc=mu[i,1], scale=sigma[i,1])
        )
    P0 = p0_flat.reshape(X.shape)
    # 3) pack parameters into theta
    # order: [mu_x, mu_y, sigma_x, sigma_y, w]
    theta = np.zeros(5 * num)
    theta[     :  num] = mu[:, 0]
    theta[num   :2*num] = mu[:, 1]
    theta[2*num:3*num] = sigma[:, 0]
    theta[3*num:4*num] = sigma[:, 1]
    theta[4*num:5*num] = w

    params = (w, mu, sigma)
    return P0, params, theta



    set_publication_plot_style()

    X, Y = problem["x1_grid"], problem["x2_grid"]
    p0, B     = problem['p0_grid'], problem['B']
    pdf_lp    = problem['pdf_LP']
    pdf_fo    = problem['pdf_FO']
    Pr, Pr_LP, Pr_FO = problem['Pr_subset'], problem['Pr_LP'], problem['Pr_FO']
    
    ub = p0 + B
    lb = p0 - B
    lb[lb < 0] = 0
    
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    
    # # bounds as light gray wireframe
    # ax.plot_wireframe(X, Y, ub,  rstride=16, cstride=16, color='gray', linewidth=0.8, alpha=0.5)
    # ax.plot_wireframe(X, Y, lb,  rstride=16, cstride=16, color='gray', linewidth=0.8, alpha=0.5)
    
    # true pdf as coarse wireframe
    # ax.plot_wireframe(X, Y, p0, rstride=16, cstride=16, color='black', linewidth=1.2)
    # surf_true = ax.plot_surface(
    #     X, Y, p0,
    #     cmap=cm.Greys,          # red colormap
    #     alpha=0.8,
    #     edgecolor='grey',
    #     antialiased=True
    # )
    
    # LP and FO surfaces
    surf_lp = ax.plot_surface(
        X, Y, pdf_lp,
        cmap=cm.Blues,         # blue colormap
        alpha=0.7,
        edgecolor='blue',
        antialiased=True
    )
    surf_fo = ax.plot_surface(
        X, Y, pdf_fo,
        cmap=cm.Reds,          # red colormap
        alpha=0.7,
        edgecolor='red',
        antialiased=True
    )

    # get normalized target bound
    target_bounds = problem['x_subset']
    x1_tar_bounds = target_bounds[0,:]
    x2_tar_bounds = target_bounds[1,:]
    patch_vertices = [
        [x1_tar_bounds[0], x2_tar_bounds[0], 0],
        [x1_tar_bounds[1], x2_tar_bounds[0], 0],
        [x1_tar_bounds[1], x2_tar_bounds[1], 0],
        [x1_tar_bounds[0], x2_tar_bounds[1], 0],
    ]
    # Create a Poly3DCollection and add it to the 3D axis.
    patch = Poly3DCollection([patch_vertices], facecolor='green', 
                              alpha=1.0, edgecolor='k', 
                              linewidths=2,
                              zorder=10,
                            #   label=r"$X^'_{tar}$"
                              )
    ax.add_collection3d(patch)
    
    # manual legend
    legend_elems = [
        # Line2D([0], [0], color='black', lw=1.2, label='True PDF'),
        # Patch(facecolor=cm.Greys(0.6), label=r'$p_{true}$'),
        Patch(facecolor=cm.Blues(0.6), label=r'$f_{LP}$'),
        Patch(facecolor=cm.Reds(0.6),  label=r'$\phi_{FO}$'),
        Patch(facecolor="green",  label=r"$X^'_{tar}$"),
        # Line2D([0], [0], color='gray', lw=0.8, alpha=0.5, label='Bounds')
    ]
    ax.legend(handles=legend_elems, loc='upper left')
    
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel('\n Probability Distribution')
    ax.set_zlim(0, ub.max())
    ax.view_init(elev=14, azim=130)
    plt.title(f'2D Test, Pr true: {Pr:.4f},   LP: {Pr_LP:.4f},   FO: {Pr_FO:.4f}')
    plt.tight_layout()
    plt.show()


def plot_pdf_2d_overlay(problem):
    set_publication_plot_style()

    X, Y      = problem["x1_grid"], problem["x2_grid"]
    p0        = problem['p0_grid']
    pdf_lp    = problem['pdf_LP']
    pdf_fo    = problem['pdf_FO']
    Pr, Pr_LP, Pr_FO = problem['Pr_subset'], problem['Pr_LP'], problem['Pr_FO']
    x1b, x2b  = problem['x_subset']

    # precompute the verts for the green patch
    verts = [
        [x1b[0], x2b[0], 0],
        [x1b[1], x2b[0], 0],
        [x1b[1], x2b[1], 0],
        [x1b[0], x2b[1], 0],
    ]

    fig = plt.figure(figsize=(14, 6))
    ax_lp = fig.add_subplot(1, 2, 1, projection='3d')
    ax_fo = fig.add_subplot(1, 2, 2, projection='3d')

    # draw the gray "true" surface and green patch on both axes
    for ax in (ax_lp, ax_fo):
        ax.plot_surface(
            X, Y, p0,
            color='gray', alpha=0.5,
            edgecolor='none', antialiased=True
        )
        patch = Poly3DCollection([verts],
                                 facecolor='green',
                                 edgecolor='k',
                                 linewidths=2,
                                 alpha=1.0,
                                 zorder=10)
        ax.add_collection3d(patch)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.view_init(elev=22, azim=110)

    # LEFT: LP
    ax_lp.plot_surface(
        X, Y, pdf_lp,
        cmap=cm.Blues, alpha=0.7,
        edgecolor='blue', antialiased=True
    )
    zmax_lp = max(p0.max(), pdf_lp.max())
    ax_lp.set_zlim(0, zmax_lp)
    ax_lp.set_zlabel("\n Prob. Dist.")

    # info + legend‐like text
    # ax_lp.text2D(
    #     0.02, 0.98,
    #     f"Pr true: {Pr:.4f}\nPr LP:   {Pr_LP:.4f}",
    #     transform=ax_lp.transAxes,
    #     va='top', fontsize='small'
    # )
    ax_lp.text2D(0.02, 0.80, u"\u25A0 True p, "+r"$\mathbb{P}=$"+" {:.4f}".format(Pr), 
                 color='gray', transform=ax_lp.transAxes)
    ax_lp.text2D(0.02, 0.75, u"\u25A0 $p_i$ (LP), "+r"$\mathbb{P}^+=$"+" {:.4f}".format(Pr_LP),    
                 color='blue', transform=ax_lp.transAxes)
    ax_lp.text2D(0.02, 0.70, u"\u25A0 $X^'_{tar}$", color='green', transform=ax_lp.transAxes)

    # RIGHT: FO
    ax_fo.plot_surface(
        X, Y, pdf_fo,
        cmap=cm.Reds, alpha=0.7,
        edgecolor='red', antialiased=True
    )
    zmax_fo = max(p0.max(), pdf_fo.max())
    zmax_fo = max(zmax_fo, zmax_lp)
    ax_fo.set_zlim(0, zmax_fo)
    ax_fo.set_zlabel("\n Prob. Dist.")

    # ax_fo.text2D(
    #     0.02, 0.98,
    #     f"Pr true: {Pr:.4f}\nPr FO:   {Pr_FO:.4f}",
    #     transform=ax_fo.transAxes,
    #     va='top', fontsize='small'
    # )
    ax_fo.text2D(0.02, 0.80, u"\u25A0 True p, "+r"$\mathbb{P}=$"+" {:.4f}".format(Pr), 
                 color='gray', transform=ax_fo.transAxes)
    ax_fo.text2D(0.02, 0.75, u"\u25A0 $\phi$ (FO), "+r"$\mathbb{P}^+=$"+" {:.4f}".format(Pr_FO),    
                 color='red',  transform=ax_fo.transAxes)
    ax_fo.text2D(0.02, 0.70, u"\u25A0 $X^'_{tar}$", color='green', transform=ax_fo.transAxes)

    # aggressively trim the white margins between & around panels
    plt.tight_layout(pad=0.1)
    plt.savefig("figs/test_2d_case6.pdf",format="pdf"); plt.close()
    # plt.show()



def plot_pdf_2d(problem):
    """Plot a 2D GMM density as a 3D surface with wireframe overlay."""
    set_publication_plot_style()
    X = problem["x1_grid"]
    Y = problem["x2_grid"]
    p0 = problem['p0_grid']
    B = problem['B']
    pdf_lp = problem['pdf_LP']
    pdf_fo = problem['pdf_FO']
    Pr = problem['Pr_subset']
    Pr_LP = problem['Pr_LP']
    Pr_FO = problem['Pr_FO']
    ub = p0 + B
    lb = p0 - B
    lb[lb < 0] = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_strides = 8
    lw = 1.0
    ax.plot_surface(X, Y, ub, color="gray", alpha=0.3)
    ax.plot_surface(X, Y, lb, color="gray", alpha=0.3)
    ax.plot_wireframe(X, Y, p0, linewidth=lw,
                      color="black", label="p")
    ax.plot_surface(X, Y, pdf_lp, linewidth=lw,
                      color="blue", alpha=0.5, label=r"$f_{LP}$")
    ax.plot_surface(X, Y, pdf_fo, linewidth=lw,
                      color="yellow", alpha=0.5, label=r"$f_{FO}$")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Density')
    ax.set_zlim([0, ub.max()])
    ax.legend()
    plt.title('2D Test, Pr true: {:.4f},   LP: {:.4f},   FO: {:.4f}'.format(Pr, Pr_LP, Pr_FO))
    plt.show()


def optimize_linearprogram_2d(problem):
    """
    linear program 
    # NOTE: we assume that the discretization over state space is small enough such that
    #   p_ub and p_lb can be directly calculated by the vertices of the cell
    """
    x = problem['x']
    n = x.shape[0]
    p0 = problem['p0']
    B = problem['B']

    # Define the subset x_sub (e.g., points between -1 and 1).
    sub_idx = problem['mask']
    dV = problem['dV']
    
    # Decision variable: p, representing the probability at each discretized point.
    p = cp.Variable(n)
    p_ub = np.zeros(n)
    p_lb = np.zeros(n)
    
    # Define constraints.
    constraints = [
        cp.sum(p) * dV <= 1,  # Total probability integrates to 1.
        p >= 0,               # Non-negativity.
    ]
    
    # Each p(x_j) must lie in [p0(x_j)-B, p0(x_j)+B].
    for i in range(n):
        constraints += [p[i] >= p0[i] - B,
                        p[i] <= p0[i] + B]
    
    # Objective: maximize the total probability mass over the subset x_sub.
    objective = cp.Maximize(cp.sum(p[sub_idx]) * dV)
    
    # Set up and solve the problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result, np.array(p.value), None


def mixture_pdf_2d(K, theta, x):
    """
    x: array of shape (N_pts, 2)
    theta: length 5*K array [mu_x, mu_y, sigma_x, sigma_y, w]
    returns: array of shape (N_pts,) giving pdf at each x[i]
    """
    mu_x    = theta[     :   K]
    mu_y    = theta[K   : 2*K]
    sig_x   = theta[2*K : 3*K]
    sig_y   = theta[3*K : 4*K]
    weights = theta[4*K : 5*K]
    
    pdfs = np.zeros((K, x.shape[0]))
    for i in range(K):
        pdfs[i] = (
            weights[i]
            * norm.pdf(x[:,0], loc=mu_x[i], scale=sig_x[i])
            * norm.pdf(x[:,1], loc=mu_y[i], scale=sig_y[i])
        )
    return pdfs.sum(axis=0)


def gmm_integral_2dsubset(K, theta, subset):
    """
    Analytical integral of a diagonal‐GMM over a rectangle:
      subset = [[x1_l, x1_u], [x2_l, x2_u]]
    Uses the fact that for each component:
      ∫N(x|μ,σ) dx = ½[erf((u-μ)/(√2σ)) - erf((l-μ)/(√2σ))]
    """
    x1_l, x1_u = subset[0]
    x2_l, x2_u = subset[1]
    
    mu_x    = theta[     :   K]
    mu_y    = theta[K   : 2*K]
    sig_x   = theta[2*K : 3*K]
    sig_y   = theta[3*K : 4*K]
    weights = theta[4*K : 5*K]
    
    integral = 0.0
    for i in range(K):
        Fx = 0.5*(erf((x1_u-mu_x[i])/(np.sqrt(2)*sig_x[i]))
                 - erf((x1_l-mu_x[i])/(np.sqrt(2)*sig_x[i])))
        Fy = 0.5*(erf((x2_u-mu_y[i])/(np.sqrt(2)*sig_y[i]))
                 - erf((x2_l-mu_y[i])/(np.sqrt(2)*sig_y[i])))
        integral += weights[i] * Fx * Fy
    return integral


def gmm_constraint_2d(K, theta, x, p0, B):
    """
    Returns residuals >=0 for p0 - B <= p_mix <= p0 + B.
    """
    p_mix = mixture_pdf_2d(K, theta, x)
    lower = p_mix - (p0 - B)   # >= 0 → p_mix >= p0 - B
    upper = (p0 + B) - p_mix   # >= 0 → p_mix <= p0 + B
    return np.hstack([lower, upper])
