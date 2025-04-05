import cvxpy as cp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.sparse as sp


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
        
    # Define smoothness penalty: we penalize squared differences between adjacent p values.
    smoothness_penalty = cp.sum_squares(p[1:] - p[:-1])
    
    # Objective: maximize the total probability mass over the subset x_sub.
    objective = cp.Maximize(cp.sum(p[sub_idx]) * dx - lambda_reg * smoothness_penalty)
    
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


if __name__ == '__main__':
    # test_optimize_probability_mass_normal_p0()
    # test_optimize_probability_mass_multivariate_normal_p0()
    # test_optimize_probability_mass_multivariate_normal_p0_4d()
    test_optimize_probability_mass_multivariate_normal_p0_4d_vectorized()
