import numpy as np
from scipy.stats import norm
from scipy.special import erf
import matplotlib.pyplot as plt


def generate_base_pdf(x, num):
    w = np.random.rand(num) + 1e-2
    sum_w = np.sum(w)
    w = w/sum_w
    mu = np.random.uniform(low=np.min(x), high=np.max(x), size=num)
    sigma = np.ones(num)
    # initialize
    p0 = x*0.0
    for i in range(num):
        p0 = p0 + w[i]*norm.pdf(x, loc=mu[i], scale=sigma[i])
    params = (w, mu, sigma)
    return p0, params


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


def gmm_integral_1dsubset(params, x_subset):
    mu_opt, sigma_opt = params
    a = x_subset[0]
    b = x_subset[1]
    p_opt_integral = 0.5*(erf((b-mu_opt)/(np.sqrt(2)*sigma_opt)) - erf((a-mu_opt)/(np.sqrt(2)*sigma_opt)))
    return p_opt_integral


def plot_gamm_1dsubset(show_plots, x, x_subset, p0, B, params, Pr_opt_subset, Pr_subset):
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
    plt.plot(x, optimized_pdf, label=r"$p_{FO}$", marker="o", markersize=4, color="blue")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(r"True $\mathbb{P}(X^'_{tar})$= "+str(np.round(Pr_subset,3))+
              r", Max $\mathbb{P}_{FO}(X^'_{tar})$="+str(np.round(Pr_opt_subset, 3))
              )
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()