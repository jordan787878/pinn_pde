import cvxpy as cp
import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.sparse as sp
from scipy.optimize import minimize
from matplotlib.lines import Line2D



def test_optimize_probability_mass_normal_p0():
    # Define the domain and discretization parameters.
    x_l = -5.0       # lower bound of x
    x_u = 5.0        # upper bound of x
    dx = 0.1         # discretization step
    x = np.arange(x_l, x_u + dx, dx)
    n = len(x)
    
    # Define parameters for the normal distribution.
    mu = 0.0         # mean of the normal distribution
    sigma = 1.0      # standard deviation of the normal distribution
    # Compute the normal pdf over the discretized domain.
    p0 = norm.pdf(x, loc=mu, scale=sigma)
    # Normalize p0 so that ∑ p0(x_j)*dx = 1.
    p0 = p0 / (np.sum(p0) * dx)
    print("[check] sum of phat:", np.sum(p0) * dx)
    
    # Allowed deviation.
    B = 0.1  # Adjust as needed (ensure feasibility with the chosen p0)

    # Target of interest
    x_subset = np.array([0.0, 2.0])

    # Smoothness regularization parameter.
    lambda_reg = 5.0  # Increase lambda_reg for stronger smoothness penalization.

    # Compute the bounds for p0.
    lower_bound = p0 - B
    upper_bound = p0 + B
    
    # Define the subset x_sub (e.g., points between -1 and 1).
    sub_idx = np.where((x >= x_subset[0]) & (x <= x_subset[1]))[0]
    
    # Decision variable: p, representing the probability at each discretized point.
    p = cp.Variable(n)
    
    # Define constraints.
    constraints = [
        cp.sum(p) * dx == 1,  # Total probability integrates to 1.
        p >= 0                # Non-negativity.
    ]
    
    # Each p(x_j) must lie in [p0(x_j)-B, p0(x_j)+B].
    for i in range(n):
        constraints += [p[i] >= p0[i] - B,
                        p[i] <= p0[i] + B]
        
    # Add linear (piecewise) smoothness constraints:
    c1 = 0.5
    for i in range(1, n-1):
        # Enforce: |p[i]-p[i-1]| <= c1 * dx
        constraints += [  (p[i+1] - 2*p[i] + p[i-1])/(dx*dx) <= c1,
                         -(p[i+1] - 2*p[i] + p[i-1])/(dx*dx) <= c1]
    # Add smoothness constraints (second difference) only within the target domain.
    # For interior indices of the subdomain only:
    # if len(sub_idx) >= 3:  # Ensure there are enough points to impose a second difference.
    #     for j in range(1, len(sub_idx) - 1):
    #         i = sub_idx[j]
    #         # Use the neighbor indices from the subdomain.
    #         i_prev = sub_idx[j - 1]
    #         i_next = sub_idx[j + 1]
    #         constraints += [
    #             (p[i_next] - 2 * p[i] + p[i_prev]) / (dx**2) <= c1,
    #             -(p[i_next] - 2 * p[i] + p[i_prev]) / (dx**2) <= c1
    #         ]
    
    # Objective: maximize the total probability mass over the subset x_sub.
    # (We removed the penalization term previously added for smoothness.)
    objective = cp.Maximize(cp.sum(p[sub_idx]) * dx)
    
    # Set up and solve the problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    # --- Verification ---
    # Compute maximum possible mass on x_sub given the upper bound at each point.
    true_Pr = np.sum(p0[sub_idx])*dx
    max_Pr =  np.sum(p.value[sub_idx]) * dx
    # Assert that the total probability mass equals 1.
    np.testing.assert_allclose(
        np.sum(p.value) * dx, 
        1, 
        atol=1e-6, 
        err_msg="Total probability mass does not sum to 1."
    )
    # Check that each probability is within the bounds.
    for i in range(n):
        assert p.value[i] >= p0[i] - B - 1e-6, f"p[{i}] is below its lower bound."
        assert p.value[i] <= p0[i] + B + 1e-6, f"p[{i}] is above its upper bound."
    
    # Post
    print("Test passed: Normal baseline distribution p0 produces an optimal probability distribution satisfying all constraints.")
    plt.figure(figsize=(10, 6))
    plt.plot(x, p0, label="Baseline $p_0$", linestyle="--", color="gray")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.3,
                    label="$p_0 \\pm B$")
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    plt.fill_between(x[mask], 0, upper_bound[mask], color="green", alpha=0.3, label="$X$")  
    plt.plot(x, p.value, label="Optimal $p$", marker="o", markersize=4, color="blue")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Optimal Probability Distribution $p$, Pr True= "+str(np.round(true_Pr,3))+", Pr(S)="+str(np.round(max_Pr, 3)))
    plt.legend()
    plt.grid(True)
    plt.show()


def test_optimize_probability_mass_multivariate_normal_p0():
    # Domain parameters.
    x_l, x_u = -5.0, 5.0
    y_l, y_u = -5.0, 5.0
    dx = 0.1   # grid spacing in x
    dy = 0.1   # grid spacing in y
    
    # Create grid.
    x_vals = np.arange(x_l, x_u + dx, dx)
    y_vals = np.arange(y_l, y_u + dy, dy)
    X, Y = np.meshgrid(x_vals, y_vals)  # X, Y shape: (n_y, n_x)
    n_y, n_x = X.shape

    # Define parameters for the bivariate normal.
    mu = np.array([0.0, 0.0])
    sigma = np.eye(2)  # Identity covariance
    rv = multivariate_normal(mean=mu, cov=sigma)
    
    # Compute the density over the grid.
    pos = np.dstack((X, Y))
    p0 = rv.pdf(pos)
    # Normalize p0 so that the total mass is 1.
    p0 = p0 / (np.sum(p0) * dx * dy)
    print("[check] Sum of p0:", np.sum(p0) * dx * dy)
    
    # Allowed deviation.
    B = 0.01  # Adjust as needed.

    lambda_reg = 1.0

    # Compute bounds.
    lower_bound = p0 - B
    upper_bound = p0 + B

    # Define target region.
    # For example, the target region is [0,2] in x and [0,2] in y.
    target_region = [1.0, 2.0, -5.0, 5.0]  # [x_min, x_max, y_min, y_max]
    mask = (X >= target_region[0]) & (X <= target_region[1]) & \
           (Y >= target_region[2]) & (Y <= target_region[3])
    
    # Decision variable: p (a 2D array over the grid).
    p = cp.Variable((n_y, n_x))

    # Define constraints.
    constraints = [
        cp.sum(p) * dx * dy == 1,  # Total probability mass is 1.
        p >= 0                     # Non-negativity.
    ]
    
    # Bound constraints: each p[i,j] must lie in [p0[i,j]-B, p0[i,j]+B].
    for i in range(n_y):
        for j in range(n_x):
            constraints += [
                p[i, j] >= p0[i, j] - B,
                p[i, j] <= p0[i, j] + B
            ]

    # Vectorized smoothness penalty:
    smoothness_penalty = cp.sum_squares(p[1:, :] - p[:-1, :]) + cp.sum_squares(p[:, 1:] - p[:, :-1])
    
    # Objective: maximize the total probability mass in the target region.
    objective = cp.Maximize(cp.sum(cp.multiply(p, mask)) * dx * dy - lambda_reg * smoothness_penalty)
    
    # Set up and solve the problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    print("True value (mass in target region):", np.sum(p0[mask]) * dx * dy)
    print("Optimal objective value (mass in target region):", result)
    print("Total mass (should be 1):", cp.sum(p).value * dx * dy)
    
    # Plot the baseline p0 and the optimal p.
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, p0, cmap='viridis')
    plt.colorbar()
    plt.title("Baseline Multivariate Normal $p_0$")
    
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, p.value, cmap='viridis')
    plt.colorbar()
    plt.title("Optimal $p$")
    
    plt.show()


def test_optimize_probability_mass_multivariate_normal_p0_4d():
    # Domain parameters for each dimension.
    # Using a coarser grid spacing to keep the total grid size manageable.
    x_l, x_u, dx = -5.0, 5.0, 1.0
    y_l, y_u, dy = -5.0, 5.0, 1.0
    z_l, z_u, dz = -5.0, 5.0, 1.0
    w_l, w_u, dw = -5.0, 5.0, 1.0

    # Create grid values for each dimension.
    x_vals = np.arange(x_l, x_u + dx, dx)
    y_vals = np.arange(y_l, y_u + dy, dy)
    z_vals = np.arange(z_l, z_u + dz, dz)
    w_vals = np.arange(w_l, w_u + dw, dw)

    # Create a 4D grid using meshgrid (using 'ij' indexing to preserve order).
    X, Y, Z, W = np.meshgrid(x_vals, y_vals, z_vals, w_vals, indexing='ij')
    grid_shape = X.shape  # (n_x, n_y, n_z, n_w)
    
    # Define parameters for the 4D multivariate normal.
    mu = np.zeros(4)
    sigma = np.eye(4)
    rv = multivariate_normal(mean=mu, cov=sigma)
    
    # Build a 4D array of positions with shape (n_x, n_y, n_z, n_w, 4).
    pos = np.stack((X, Y, Z, W), axis=-1)
    p0 = rv.pdf(pos)
    # Normalize p0 so that its total mass is 1.
    p0 = p0 / (np.sum(p0) * dx * dy * dz * dw)
    print("[check] Sum of p0:", np.sum(p0) * dx * dy * dz * dw)
    
    # Allowed deviation.
    B = 0.01  # Adjust as needed.
    lambda_reg = 10.0

    # Specify target region as a 4x2 numpy array.
    # Each row corresponds to a dimension: [lower_bound, upper_bound].
    # For example: x in [1, 2], y in [1, 2], and z, w use the full range [-5, 5].
    target_region = np.array([
        [1.0, 1.1],   # x bounds
        [1.0, 1.1],   # y bounds
        [-5.0, 5.0],  # z bounds
        [-5.0, 5.0]   # w bounds
    ])
    
    # Create a mask for grid points within the target region.
    mask = ((X >= target_region[0, 0]) & (X <= target_region[0, 1]) &
            (Y >= target_region[1, 0]) & (Y <= target_region[1, 1]) &
            (Z >= target_region[2, 0]) & (Z <= target_region[2, 1]) &
            (W >= target_region[3, 0]) & (W <= target_region[3, 1]))
    
    # Decision variable: p (a 4D array over the grid).
    p = cp.Variable(grid_shape)
    
    # Define constraints.
    constraints = [
        cp.sum(p) * dx * dy * dz * dw == 1,  # Total probability mass is 1.
        p >= 0,                              # Non-negativity.
        p >= p0 - B,                         # Lower bound.
        p <= p0 + B                          # Upper bound.
    ]
    
    # Define a smoothness penalty along each dimension.
    smoothness_penalty = (
        cp.sum_squares(p[1:, :, :, :] - p[:-1, :, :, :]) +
        cp.sum_squares(p[:, 1:, :, :] - p[:, :-1, :, :]) +
        cp.sum_squares(p[:, :, 1:, :] - p[:, :, :-1, :]) +
        cp.sum_squares(p[:, :, :, 1:] - p[:, :, :, :-1])
    )
    
    # Objective: maximize the total probability mass in the target region minus the smoothness penalty.
    objective = cp.Maximize(
        cp.sum(cp.multiply(p, mask)) * dx * dy * dz * dw - lambda_reg * smoothness_penalty
    )
    
    # Set up and solve the optimization problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    print("True value (mass in target region):", np.sum(p0[mask]) * dx * dy * dz * dw)
    print("Optimal objective value (mass in target region):", np.sum(p.value[mask])* dx * dy * dz * dw)
    print("Total mass (should be 1):", cp.sum(p).value * dx * dy * dz * dw)
    
    # For visualization, take a 2D slice of the 4D solution.
    # For example, fix z and w to their middle indices.
    z_mid = len(z_vals) // 2
    w_mid = len(w_vals) // 2
    p0_slice = p0[:, :, z_mid, w_mid]
    p_opt_slice = p.value[:, :, z_mid, w_mid]
    X_slice = X[:, :, z_mid, w_mid]
    Y_slice = Y[:, :, z_mid, w_mid]
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.contourf(X_slice, Y_slice, p0_slice, cmap='viridis')
    plt.colorbar()
    plt.title("Baseline 4D Multivariate Normal $p_0$ (slice at z, w midpoints)")
    
    plt.subplot(1, 2, 2)
    plt.contourf(X_slice, Y_slice, p_opt_slice, cmap='viridis')
    plt.colorbar()
    plt.title("Optimal $p$ (slice at z, w midpoints)")
    
    plt.show()


def build_difference_matrix(grid_shape, dim):
    """
    Build a sparse difference matrix for the specified dimension.
    
    For each grid point that has a neighbor along dimension `dim`,
    the matrix has a row that computes the difference between the
    neighbor and the point.
    
    Parameters
    ----------
    grid_shape : tuple
        The shape of the grid (n_x, n_y, n_z, n_w).
    dim : int
        The dimension along which to compute differences.
    
    Returns
    -------
    D : scipy.sparse.coo_matrix
        A sparse matrix such that (D @ p_vec)[i] = p[neighbor] - p[current].
    """
    dims = grid_shape
    N = np.prod(dims)
    rows = []
    cols = []
    data = []
    row_counter = 0
    # Iterate over all indices in the grid.
    for index in np.ndindex(dims):
        if index[dim] < dims[dim] - 1:  # has a neighbor in this dimension
            current_index = np.ravel_multi_index(index, dims)
            # Build the neighbor index by adding 1 in the dim-th coordinate.
            neighbor_index = np.ravel_multi_index(
                tuple(index[i] + (1 if i == dim else 0) for i in range(len(dims))), dims
            )
            # p[neighbor] - p[current]
            rows.append(row_counter)
            cols.append(neighbor_index)
            data.append(1.0)
            
            rows.append(row_counter)
            cols.append(current_index)
            data.append(-1.0)
            row_counter += 1
    D = sp.coo_matrix((data, (rows, cols)), shape=(row_counter, N))
    return D


def test_optimize_probability_mass_multivariate_normal_p0_4d_vectorized():
    # Domain parameters for each dimension.
    # Using a coarser grid spacing to keep the total grid size manageable.
    x_l, x_u, dx = -5.0, 5.0, 0.5
    y_l, y_u, dy = -5.0, 5.0, 0.5
    z_l, z_u, dz = -5.0, 5.0, 0.5
    w_l, w_u, dw = -5.0, 5.0, 0.5

    # Create grid values.
    x_vals = np.arange(x_l, x_u + dx, dx)
    y_vals = np.arange(y_l, y_u + dy, dy)
    z_vals = np.arange(z_l, z_u + dz, dz)
    w_vals = np.arange(w_l, w_u + dw, dw)

    # Create a 4D grid (using 'ij' indexing).
    X, Y, Z, W = np.meshgrid(x_vals, y_vals, z_vals, w_vals, indexing='ij')
    grid_shape = X.shape  # (n_x, n_y, n_z, n_w)
    N = np.prod(grid_shape)

    # Define parameters for the 4D multivariate normal.
    mu = np.zeros(4)
    sigma = np.eye(4)
    rv = multivariate_normal(mean=mu, cov=sigma)
    
    # Build positions: shape (n_x, n_y, n_z, n_w, 4)
    pos = np.stack((X, Y, Z, W), axis=-1)
    p0 = rv.pdf(pos)
    # Normalize p0.
    p0 = p0 / (np.sum(p0) * dx * dy * dz * dw)
    print("[check] Sum of p0:", np.sum(p0) * dx * dy * dz * dw)
    
    # Allowed deviation and regularization.
    B = 0.028*0.1  # Adjust as needed.
    lambda_reg = 1.0

    # Specify target region as a 4x2 array:
    # Each row corresponds to [lower_bound, upper_bound] in that dimension.
    target_region = np.array([
        [1.0, 1.1],   # x bounds
        [1.0, 1.1],   # y bounds
        [-5.0, 5.0],  # z bounds
        [-5.0, 5.0]   # w bounds
    ])
    
    # Create a mask for grid points in the target region.
    mask = ((X >= target_region[0, 0]) & (X <= target_region[0, 1]) &
            (Y >= target_region[1, 0]) & (Y <= target_region[1, 1]) &
            (Z >= target_region[2, 0]) & (Z <= target_region[2, 1]) &
            (W >= target_region[3, 0]) & (W <= target_region[3, 1]))
    
    # Flatten p0 and mask to vector form.
    p0_flat = p0.flatten()
    mask_flat = mask.flatten()
    
    # Decision variable: p_vec is now a vector of length N.
    p_vec = cp.Variable(N)
    
    # Constraints.
    constraints = [
        cp.sum(p_vec) * dx * dy * dz * dw == 1,  # total mass constraint.
        p_vec >= 0,                              # non-negativity.
        p_vec >= p0_flat - B,                      # lower bound.
        p_vec <= p0_flat + B                       # upper bound.
    ]
    
    # Build difference matrices for each of the 4 dimensions.
    # D_matrices = [build_difference_matrix(grid_shape, dim) for dim in range(4)]
    # Compute smoothness penalty as sum of squared differences along each dimension.
    # smoothness_penalty = sum(cp.sum_squares(D @ p_vec) for D in D_matrices)
    
    # Objective: maximize probability mass in target region minus smoothness penalty.
    objective = cp.Maximize(
        cp.sum(cp.multiply(p_vec, mask_flat)) * dx * dy * dz * dw #- lambda_reg * smoothness_penalty
    )
    
    # Set up and solve the problem.
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    print("True value (mass in target region):", np.sum(p0_flat[mask_flat]) * dx * dy * dz * dw)
    print("Optimal objective value (mass in target region):", 
          np.sum(p_vec.value[mask_flat]) * dx * dy * dz * dw)
    print("Total mass (should be 1):", np.sum(p_vec.value) * dx * dy * dz * dw)
    
    # Reshape the optimized p_vec back into the original grid shape.
    p_opt = p_vec.value.reshape(grid_shape)
    
    # For visualization, take a 2D slice: fix z and w to their middle indices.
    z_mid = len(z_vals) // 2
    w_mid = len(w_vals) // 2
    p0_slice = p0[:, :, z_mid, w_mid]
    p_opt_slice = p_opt[:, :, z_mid, w_mid]
    X_slice = X[:, :, z_mid, w_mid]
    Y_slice = Y[:, :, z_mid, w_mid]
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(X_slice, Y_slice, p0_slice, cmap='viridis')
    plt.colorbar()
    plt.title("Baseline 4D Multivariate Normal $p_0$ (slice at z, w midpoints)")
    plt.subplot(1, 2, 2)
    plt.contourf(X_slice, Y_slice, p_opt_slice, cmap='viridis')
    plt.colorbar()
    plt.title("Optimal $p$ (slice at z, w midpoints)")
    plt.show()


def objective(params, x, x_subset, dx):
    """
    Objective function to minimize.
    We wish to maximize the probability mass of the Gaussian (parameterized by params)
    over the subset x_subset. Since minimize minimizes, we return the negative of that mass.
    
    params: [mu, sigma] for our Gaussian pdf.
    x: discretized state domain.
    x_subset: [x_min, x_max] of the target subset.
    dx: spacing between x values.
    """
    mu, sigma = params
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    # Compute the approximate integral (mass) over the target subset.
    mask = (x >= x_subset[0]) & (x <= x_subset[1])
    total_mass = np.sum(pdf[mask]) * dx
    return -total_mass  # negative because we maximize

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

def test_function_opt_1D():
    # 1. Domain and discretization.
    x_l, x_u, dx = -5.0, 5.0, 0.1
    x = np.arange(x_l, x_u + dx, dx)
    
    # 2. Define baseline pdf p0 over the domain.
    base_mu = 0.0
    base_sigma = 1.0
    p0 = norm.pdf(x, loc=base_mu, scale=base_sigma)
    # (Optionally, we renormalize to the Riemann-sum integral on our grid.)
    p0 = p0 / (np.sum(p0) * dx)
    
    # 3. Allowed deviation.
    B = 0.1
    
    # 4. Target subset (where we want to maximize probability mass).
    x_subset = np.array([0.0, 2.0])
    
    # 5. Initial guess for parameters (mu, sigma).
    initial_guess = [0.0, 1.0]
    # Enforce sigma > 0
    bounds = [(-np.inf, np.inf), (1e-6, np.inf)]
    
    # 6. Define the inequality constraint.
    # Our constraint is: constraint_fun(params, x, p0, B) >= 0.
    cons = {'type': 'ineq',
            'fun': lambda params: constraint_fun(params, x, p0, B)}
    
    # 7. Optimize parameters.
    result = minimize(objective, initial_guess, args=(x, x_subset, dx),
                      constraints=cons, bounds=bounds)
    
    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed!")
    mu_opt, sigma_opt = result.x
    print("Optimized parameters: mu = {:.4f}, sigma = {:.4f}".format(mu_opt, sigma_opt))
    
    # Evaluate the optimized Gaussian pdf.
    optimized_pdf = norm.pdf(x, loc=mu_opt, scale=sigma_opt)
    
    # Compute the bounds based on the baseline pdf.
    lower_bound = p0 - B
    upper_bound = p0 + B

    # Compute the mask for the target subset.
    mask = (x >= x_subset[0]) & (x <= x_subset[1])

    # Compute the integrated probability masses.
    true_Pr = np.sum(p0[mask]) * dx
    max_Pr = np.sum(optimized_pdf[mask]) * dx

    plt.figure(figsize=(10, 6))
    plt.plot(x, p0, label="Baseline $p_0$", linestyle="--", color="gray")
    plt.fill_between(x, lower_bound, upper_bound, color="gray", alpha=0.3, label="$p_0 \\pm B$")
    plt.fill_between(x[mask], 0, upper_bound[mask], color="green", alpha=0.3, label="$X$")  
    plt.plot(x, optimized_pdf, label="Optimal $p$", marker="o", markersize=4, color="blue")

    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Optimal Probability Distribution $p$, Pr True= " + str(np.round(true_Pr, 3)) + ", Pr(S)= " + str(np.round(max_Pr, 3)))
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


###
def multivariate_gaussian_pdf_fullL(grid, params):
    """
    Evaluate the 2D Gaussian pdf on the grid with covariance parameterized by a full matrix L.

    Parameters:
      grid  : array of shape (N, 2) with N grid points.
      params: list or array with 6 elements: [mu1, mu2, L11, L12, L21, L22].
              The mean vector is [mu1, mu2], and the covariance is defined as Sigma = L L^T
              with L = [[L11, L12], [L21, L22]].
    Returns:
      1D array of pdf values at each grid point.
    """
    mu1, mu2, L11, L12, L21, L22 = params
    mu = np.array([mu1, mu2])
    L = np.array([[L11, L12],
                  [L21, L22]])
    Sigma = L @ L.T  # covariance is L * L^T
    return multivariate_normal.pdf(grid, mean=mu, cov=Sigma)

def objective_mv_fullL(params, grid, mask, dx, dy):
    """
    Objective function: maximize the probability mass over the target region.
    Since minimize() minimizes, we return the negative of the mass.
    
    Parameters:
      params: parameters of the 2D Gaussian [mu1, mu2, L11, L12, L21, L22].
      grid  : array of grid points (N x 2).
      mask  : boolean 2D array for target region.
      dx, dy: grid spacings.
    """
    pdf_vals = multivariate_gaussian_pdf_fullL(grid, params)
    pdf_2d = pdf_vals.reshape(mask.shape)
    total_mass = np.sum(pdf_2d[mask]) * dx * dy
    return -total_mass

def constraint_fun_mv_fullL(params, grid, p0, B):
    """
    Constraint: for every point, the optimized pdf must be within [p0 - B, p0 + B].
    
    Parameters:
      params: parameters [mu1, mu2, L11, L12, L21, L22].
      grid  : array of grid points (N x 2).
      p0    : baseline pdf, given as a 2D array.
      B     : allowed deviation.
      
    Returns:
      A scalar (minimum gap) that should be nonnegative if all constraints are satisfied.
    """
    pdf_vals = multivariate_gaussian_pdf_fullL(grid, params)  # shape (N,)
    p0_flat = p0.ravel()  # shape (N,)
    gap_upper = (p0_flat + B) - pdf_vals  # should be >= 0
    gap_lower = pdf_vals - (p0_flat - B)    # should be >= 0
    return min(np.min(gap_upper), np.min(gap_lower))

def test_function_opt_mv_2D():
    # Define the 2D domain and discretization parameters.
    x_l, x_u, dx = -5.0, 5.0, 0.05
    y_l, y_u, dy = -5.0, 5.0, 0.05
    x_vals = np.arange(x_l, x_u + dx, dx)
    y_vals = np.arange(y_l, y_u + dy, dy)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid = np.column_stack([X.ravel(), Y.ravel()])  # shape (N, 2)
    
    # Compute the baseline pdf p0 using a base Gaussian (e.g., mean=[0,0], covariance=I).
    base_mu = np.array([0.0, 0.0])
    base_Sigma = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
    p0 = multivariate_normal.pdf(grid, mean=base_mu, cov=base_Sigma)
    p0 = p0.reshape(X.shape)
    p0 = p0 / (np.sum(p0) * dx * dy)  # Normalize via Riemann sum
    
    # Allowed deviation.
    B = 0.01  # adjust as needed
    
    # Define the target subset: for example, x in [0, 2] and y in [0, 2].
    mask = (X >= 0.0) & (X <= 2.0) & (Y >= -5.0) & (Y <= 5.0)
    
    # Initial guess for parameters: [mu1, mu2, L11, L12, L21, L22].
    # Starting from the base Gaussian: mu = [0,0] and L = identity.
    initial_guess = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    
    # Bounds: mu are free; enforce L11 > 0 and L22 > 0.
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf),
              (-np.inf, np.inf), (-np.inf, np.inf),
              (-np.inf, np.inf), (-np.inf, np.inf)]
    
    # Define the inequality constraint.
    cons = {'type': 'ineq',
            'fun': lambda params: constraint_fun_mv_fullL(params, grid, p0, B)}
    
    # Optimize using SciPy's minimize.
    result = minimize(objective_mv_fullL, initial_guess, args=(grid, mask, dx, dy),
                      constraints=cons, bounds=bounds)
    
    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed!")
    print("Optimized parameters: [mu1, mu2, L11, L12, L21, L22] =", result.x)
    
    # Compute the optimized pdf.
    opt_pdf = multivariate_gaussian_pdf_fullL(grid, result.x).reshape(X.shape)
    
    # Compute integrated probability masses over the target region.
    true_Pr = np.sum(p0[mask]) * dx * dy
    max_Pr = np.sum(opt_pdf[mask]) * dx * dy
    
    plt.figure(figsize=(12, 6))
    # Contour of baseline pdf.
    CS0 = plt.contour(X, Y, p0, colors='gray', linestyles='--', levels=10)
    # Shade the target region.
    plt.contourf(X, Y, mask.astype(float), alpha=0.2, cmap='Greens')
    # Contours for the optimized pdf.
    CS_opt = plt.contour(X, Y, opt_pdf, levels=10, cmap='jet')
    plt.clabel(CS_opt, inline=True, fontsize=8)
    plt.colorbar(CS_opt)

    plt.xlabel("x")
    plt.ylabel("y")
    title_str = ("Optimized Multivariate Normal PDF\n"
                "Pr(True) = {:.3f},  Pr(Target) = {:.3f}"
                .format(true_Pr, max_Pr))
    plt.title(title_str)
    print(title_str)

    # Create custom legend handles instead of using CS0.collections directly.
    custom_lines = [
        Line2D([0], [0], color='gray', linestyle='--', label="Baseline $p_0$"),
        Line2D([0], [0], color='green', marker='s', markersize=8, linestyle='', alpha=0.3, label="$X$"),
        Line2D([0], [0], color='blue', marker='o', markersize=4, linestyle='-', label="Optimal $p$")
    ]
    plt.legend(handles=custom_lines, loc="upper right")

    plt.grid(True)
    plt.show()


# 4D 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

# Define the Gaussian model with learnable mean and lower-triangular L.
class GaussianModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize mean with zeros.
        self.mu = nn.Parameter(torch.zeros(dim))
        # Initialize raw L as an identity matrix. We will constrain it to be lower triangular.
        # Note: We treat the entire matrix as parameters and later extract the lower-triangular part.
        self.L_raw = nn.Parameter(torch.eye(dim))
    
    def forward(self, x):
        cov = self.L_raw @ self.L_raw.T
        mvn = D.MultivariateNormal(self.mu, covariance_matrix=cov)
        # Return the pdf values for each x sample.
        # (pdf = exp(log_prob))
        return mvn.log_prob(x).exp()

# Define a target density function p(x). In this example,
# we use a fixed multivariate normal with predetermined parameters.
def target_pdf(x):
    # For example, set a target Gaussian with mean [1, -1, 0.5, -0.5] and identity covariance.
    mu_target = torch.tensor([3.0, 2.0, -1.0, -0.5], device=x.device)
    cov_target = torch.eye(4, device=x.device)
    dist_target = D.MultivariateNormal(mu_target, covariance_matrix=cov_target)
    return dist_target.log_prob(x).exp()

# Training routine that uses full-domain and region bounds passed as arguments.
def train_model(full_domain_bounds, region_bounds, B=0.01, num_iterations=10000, full_batch_size=1024, region_batch_size=512,
                device=torch.device("cpu"), lambda_region=1.0):
    # Determine dimensionality from the length of full_domain_bounds.
    dim = len(full_domain_bounds)
    model = GaussianModel(dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # Build full-domain bounds as tensors.
    full_lower = torch.tensor([b[0] for b in full_domain_bounds], device=device, dtype=torch.float)
    full_upper = torch.tensor([b[1] for b in full_domain_bounds], device=device, dtype=torch.float)
    full_domain_volume = torch.prod(full_upper - full_lower).item()
    
    # Build region bounds as tensors.
    region_lower = torch.tensor([b[0] for b in region_bounds], device=device, dtype=torch.float)
    region_upper = torch.tensor([b[1] for b in region_bounds], device=device, dtype=torch.float)
    region_volume = torch.prod(region_upper - region_lower).item()
    
    for it in range(num_iterations):
        optimizer.zero_grad()
        
        # --- Full Domain Loss ---
        x_full = (torch.rand(full_batch_size, dim, device=device) * (full_upper - full_lower)) + full_lower
        print(x_full. shape)
        p_target_full = target_pdf(x_full)
        p_model_full = model(x_full)
        
        mask = (p_model_full >= (p_target_full - B)) & (p_model_full <= (p_target_full + B))
        if not mask.all():
            loss_full = 1e7 * torch.mean((p_target_full - p_model_full) ** 2)
        else:
            loss_full = torch.mean((p_target_full - p_model_full) ** 2)
        
        # --- Region Mass Maximization Term ---
        x_region = (torch.rand(region_batch_size, dim, device=device) * (region_upper - region_lower)) + region_lower
        p_model_region = model(x_region)
        region_mass_estimate = region_volume * torch.mean(p_model_region)
        loss_region = -region_mass_estimate
        
        total_loss = loss_full + lambda_region * loss_region
        total_loss.backward()
        optimizer.step()
        
        if it % 100 == 0:
            print(f"Iteration {it:4d}, full_loss: {loss_full.item():.6f}, "
                  f"region_mass: {region_mass_estimate.item():.6f}, total_loss: {total_loss.item():.6f}")
    
    return model


# Testing routine.
def test_4D():
    device = torch.device("cpu")  # or torch.device("cuda") if available.
    # Example: full domain and region bounds can be arbitrary per dimension.
    full_domain_bounds = [(-1, 6), (-5, 5), (-3, 4), (-4, 2)]
    region_bounds = [(0.0, 1.0), (0.0, 1.0), full_domain_bounds[2], full_domain_bounds[3]]
    B = 0.01
    trained_model = train_model(B=B, num_iterations=5000, full_batch_size=1024,
                                region_batch_size=1024, device=device, lambda_region=1e-3,
                                full_domain_bounds=full_domain_bounds,
                                region_bounds=region_bounds)
    
    # Check constraints over the full domain.
    dim = len(full_domain_bounds)
    N_full = 10000
    full_lower = torch.tensor([b[0] for b in full_domain_bounds], device=device, dtype=torch.float)
    full_upper = torch.tensor([b[1] for b in full_domain_bounds], device=device, dtype=torch.float)
    x_full = (torch.rand(N_full, dim, device=device) * (full_upper - full_lower)) + full_lower
    p_target_full = target_pdf(x_full)
    p_model_full = trained_model(x_full)
    B_check = B  # Bound to check (can be different than training bound if desired)
    constraint_mask = (p_model_full >= (p_target_full - B_check)) & (p_model_full <= (p_target_full + B_check))
    percentage_satisfied = (constraint_mask.sum().item() / N_full) * 100
    print(f"Constraint satisfied for {percentage_satisfied:.2f}% of samples over the full domain.")
    
    violation_indices = (~constraint_mask).nonzero(as_tuple=False)
    if violation_indices.numel() > 0:
        print(f"Number of constraint violations: {violation_indices.shape[0]}")
        print("Example violations (first 5):")
        for idx in violation_indices[:5]:
            i = idx.item()
            print(f"  Sample {i}: p_target = {p_target_full[i].item():.6f}, p_model = {p_model_full[i].item():.6f}")
    else:
        print("All samples satisfy the constraint.")
    
    # Evaluate integrated masses over the target region.
    region_lower_t = torch.tensor([b[0] for b in region_bounds], device=device, dtype=torch.float)
    region_upper_t = torch.tensor([b[1] for b in region_bounds], device=device, dtype=torch.float)
    region_volume = torch.prod(region_upper_t - region_lower_t).item()
    N_eval = 10000
    x_eval = (torch.rand(N_eval, dim, device=device) * (region_upper_t - region_lower_t)) + region_lower_t
    true_pdf_values = target_pdf(x_eval)
    learned_pdf_values = trained_model(x_eval)
    integrated_true_pdf = region_volume * torch.mean(true_pdf_values).item()
    integrated_learned_pdf = region_volume * torch.mean(learned_pdf_values).item()
    
    print("\nIntegrated True PDF over target region: {:.6f}".format(integrated_true_pdf))
    print("Integrated Learned PDF over target region: {:.6f}".format(integrated_learned_pdf))
    
    visualize_marginal_2d(trained_model, full_domain_bounds, region_bounds)


def compute_true_marginal(grid_points, integration_bounds, integration_samples=1000, device=torch.device("cpu")):
    """
    Numerically approximate the marginal of target_pdf for the first two dimensions.
    For each 2D grid point, we sample points (for the remaining dimensions)
    uniformly from integration_bounds and average target_pdf over those samples.
    
    Parameters:
      grid_points        : N x 2 numpy array for the first two dims.
      integration_bounds : list of tuples for the remaining dims (e.g. for a 4D problem, two tuples)
      integration_samples: number of Monte Carlo samples to use for integration.
      device             : torch.device to run computations on.
    
    Returns:
      marginal: 1D numpy array of length N containing the approximated marginal density.
    """
    # Determine the dimensionality to integrate over.
    num_integ_dims = len(integration_bounds)
    # Build lower and upper bounds for these dims.
    integ_lower = torch.tensor([b[0] for b in integration_bounds], device=device, dtype=torch.float)
    integ_upper = torch.tensor([b[1] for b in integration_bounds], device=device, dtype=torch.float)
    integ_volume = torch.prod(integ_upper - integ_lower).item()
    
    marginal = np.zeros(len(grid_points))
    # For each grid point, average target_pdf over integration_samples in remaining dims.
    # (A vectorized implementation may require more memory; here we use a simple loop.)
    for i, pt in enumerate(grid_points):
        # Create integration_samples x num_integ_dims samples uniformly from integration_bounds.
        samples = (torch.rand(integration_samples, num_integ_dims, device=device) * (integ_upper - integ_lower)) + integ_lower
        # Expand the grid point to match samples.
        pt_tensor = torch.tensor(pt, device=device, dtype=torch.float).unsqueeze(0).expand(integration_samples, -1)
        # Concatenate to form full-dimensional points.
        # Here we assume the full dimension equals 2 + num_integ_dims.
        x_full = torch.cat([pt_tensor, samples], dim=1)
        # Evaluate target_pdf (which is assumed to work on full-dimensional inputs).
        with torch.no_grad():
            vals = target_pdf(x_full)
        # Average and multiply by integration volume.
        marginal[i] = integ_volume * torch.mean(vals).item()
    return marginal

def visualize_marginal_2d(model, 
                          full_domain_bounds, 
                          region_bounds, 
                          integration_bounds_full=None,
                          integration_bounds_region=None,
                          num_points=100,
                          integration_samples=1000,
                          device=torch.device("cpu")):
    """
    Visualize the 2D marginal (first two dimensions) of the true PDF (computed using target_pdf)
    and of the learned PDF.
    
    For the true PDF, the marginal is computed numerically by integrating out 
    the remaining dimensions using Monte Carlo integration over the provided
    integration bounds.
    
    Two plots are produced:
      - Left: grid determined by the first two dimensions of full_domain_bounds.
      - Right: grid determined by the first two dimensions of region_bounds.
    
    Parameters:
      model                : the trained Gaussian model (instance of GaussianModel).
      full_domain_bounds   : list of bounds [(low, high), ...] for the full domain (for all dims).
      region_bounds        : list of bounds for the target region (for all dims).
      integration_bounds_full : list of bounds for the integration (dims 3..end) for full-domain marginal.
                                If None, defaults to full_domain_bounds[2:].
      integration_bounds_region: similar but for the region case. If None, defaults to region_bounds[2:].
      num_points           : number of grid points per axis for the 2D grid.
      integration_samples  : number of Monte Carlo samples for integrating out the remaining dimensions.
      device               : torch.device to run computations on.
    """
    # Use first two dimensions from the provided bounds for grid creation.
    # For the full domain plot:
    full_bounds_2d = full_domain_bounds[:2]
    x_full = np.linspace(full_bounds_2d[0][0], full_bounds_2d[0][1], num_points)
    y_full = np.linspace(full_bounds_2d[1][0], full_bounds_2d[1][1], num_points)
    X_full, Y_full = np.meshgrid(x_full, y_full)
    grid_full = np.column_stack([X_full.ravel(), Y_full.ravel()])
    
    # For the region plot:
    region_bounds_2d = region_bounds[:2]
    x_reg = np.linspace(region_bounds_2d[0][0], region_bounds_2d[0][1], num_points)
    y_reg = np.linspace(region_bounds_2d[1][0], region_bounds_2d[1][1], num_points)
    X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
    grid_reg = np.column_stack([X_reg.ravel(), Y_reg.ravel()])
    
    # Determine integration bounds for remaining dimensions.
    # For a 4D problem, these are bounds for dims 3 and 4.
    if integration_bounds_full is None:
        integration_bounds_full = full_domain_bounds[2:]
    if integration_bounds_region is None:
        integration_bounds_region = region_bounds[2:]
    
    # Compute the numerical marginal for the true PDF using target_pdf.
    print("Computing true marginal (full domain)...")
    true_marginal_full = compute_true_marginal(grid_full, integration_bounds_full,
                                               integration_samples=integration_samples, device=device)
    true_marginal_full = true_marginal_full.reshape(X_full.shape)
    
    print("Computing true marginal (region)...")
    true_marginal_reg = compute_true_marginal(grid_reg, integration_bounds_region,
                                              integration_samples=integration_samples, device=device)
    true_marginal_reg = true_marginal_reg.reshape(X_reg.shape)
    
    # For the learned PDF, we compute the marginal analytically:
    learned_mu = model.mu.detach().cpu().numpy()[:2]
    learned_L = model.L_raw.detach().cpu().numpy()
    learned_cov_full = learned_L @ learned_L.T
    learned_cov = learned_cov_full[:2, :2]
    learned_rv = multivariate_normal(mean=learned_mu, cov=learned_cov)
    learned_marginal_full = learned_rv.pdf(grid_full).reshape(X_full.shape)
    learned_marginal_reg = learned_rv.pdf(grid_reg).reshape(X_reg.shape)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    # Full domain plot
    plt.subplot(1, 2, 1)
    cp1 = plt.contourf(X_full, Y_full, true_marginal_full, levels=10, cmap='viridis')
    plt.colorbar(cp1)
    plt.contour(X_full, Y_full, learned_marginal_full, levels=10, colors='white', linewidths=1)
    plt.title("True vs Learned PDF Marginal (Full Domain)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    
    # Region plot
    plt.subplot(1, 2, 2)
    cp2 = plt.contourf(X_reg, Y_reg, true_marginal_reg, levels=10, cmap='viridis')
    plt.colorbar(cp2)
    plt.contour(X_reg, Y_reg, learned_marginal_reg, levels=10, colors='white', linewidths=1)
    plt.title("True vs Learned PDF Marginal (Region)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test_optimize_probability_mass_normal_p0()
    # test_function_opt_1D()
    # test_function_opt_mv_2D()
    test_4D()


    # test_optimize_probability_mass_multivariate_normal_p0()
    # test_optimize_probability_mass_multivariate_normal_p0_4d()
    # test_optimize_probability_mass_multivariate_normal_p0_4d_vectorized()
