import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def solve_numer_integral(problem):
    """
    compute upper bound of P* by numerical integral
    """
    B = problem['B']
    p0 = problem['p0']
    mask = problem['mask']
    dV = problem['dV']

    # --- Solving ---
    inside = mask
    outside = ~inside
    p0_inside = p0[inside]
    p0_outside = p0[outside]
    # direct integral inside target
    Pr = 0.0
    for p in p0_inside:
        Pr += (p+B)*dV

    # 1 - integral outside target
    Pr_neg = 0.0
    for p in p0_outside:
        if((p-B) >= 0):
            Pr_neg += (p-B)*dV
    Pr_neg = 1 - Pr_neg
    print(Pr, Pr_neg)
    Pr_opt = min(Pr, Pr_neg)
    print(" [Solved] time:{:.3f}, Pr_tar: {:.4f}".format(problem["time"], Pr_opt))
    return Pr_opt, None


def solve_estimate(problem):
    """
    compute upper bound of P* by numerical integral
    """
    # B = problem['B']
    p0 = problem['p0']
    mask = problem['mask']
    dV = problem['dV']

    # --- Solving ---
    inside = mask
    outside = ~inside
    p0_inside = p0[inside]
    # direct integral inside target
    Pr = 0.0
    for p in p0_inside:
        Pr += (p)*dV
    print(" [Solved] time:{:.3f}, Pr_tar: {:.4f}".format(problem["time"], Pr))
    return Pr, None


def solve_linearprogram(problem, show_plots=False):
    """
    linear program 
    # NOTE: we assume that the discretization over state space is small enough such that
    #   p_ub and p_lb can be directly calculated by the vertices of the cell
    """
    B = problem['B']
    p0 = problem['p0']
    mask = problem['mask']
    dV = problem['dV']
    t = problem['time']

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
    print(" [Solved] t: {:.3f}, Pr_tar: {:.4f}, Pr_total: {:.4f}".format(t, Pr, p_total))
    deviation = np.abs(p_result - p0)
    np.testing.assert_array_less(deviation, B+delta)
    np.testing.assert_array_less(p_total, 1.0+delta)
    if(show_plots):
        plt.figure()
        plt.plot(p0[:front_count]*dV, color="black", linestyle="--", label="PINN p")
        plt.plot(p_ub[:front_count]*dV, color="red", linestyle="--", label="Error Bounds")
        plt.plot(p_lb[:front_count]*dV, color="red", linestyle="--")
        plt.plot(p_result[:front_count]*dV, color="blue", label="LP sol.")
        plt.legend()
        plt.xlabel("4D cell index")
        plt.ylabel("p_i * dV")
        plt.title(r"$\mathbb{P}^+ =$"+"{:.4f}".format(Pr))
        plt.show()
    return Pr, None

    # --- Solving via cvxpy (Very slow) ---
    n = p0.shape[0]
    p = cp.Variable(n)
    p_ub = np.zeros(n)
    p_lb = np.zeros(n)
    for i in range(n):
        p_ub[i] = p0[i]
        p_lb[i] = p0[i]
    # Define constraints.
    constraints = [
        cp.sum(p) * dV <= 1,  # Total probability integrates to 1.
        p >= 0,               # Non-negativity.
    ]
    # Each p(x_j) must lie in [p0(x_j)-B, p0(x_j)+B].
    for i in range(n):
        constraints += [p[i] >= p_ub[i] - B,
                        p[i] <= p_lb[i] + B]
    # Objective: maximize the total probability mass over the subset x_sub.
    objective = cp.Maximize(cp.sum(p[mask_flat]) * dV)
    # Set up and solve the problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return result, np.array(p.value)