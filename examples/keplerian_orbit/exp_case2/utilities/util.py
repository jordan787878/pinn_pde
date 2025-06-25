import numpy as np
from scipy.optimize import fsolve
import matplotlib as mpl


def set_publication_style():
    """
    Configure matplotlib's rcParams for publication-ready plots.
    Adjusts figure size, font sizes, line widths, tick styles, and grid properties.
    """
    mpl.rcParams.update({
        'figure.figsize': (8, 6),
        'savefig.dpi': 300,
        'axes.titlesize': 16,       # Slightly larger title size
        'axes.labelsize': 14,       # Increase axis label size
        'axes.linewidth': 1.2,      # Thinner axes lines
        'lines.linewidth': 2.5,     # Thicker plot lines for clarity
        'lines.markersize': 8,
        'xtick.labelsize': 12,      # Slightly smaller tick labels
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,      # Adjust tick length
        'ytick.major.size': 4,
        'legend.fontsize': 12,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.color': '0.8',        # Light gray grid lines
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'figure.autolayout': True   # Enable automatic layout adjustment
    })


def sphere_to_cartesian(input_array, reduced_to_four_dim=False):
    """
    Convert spherical coordinates (r, theta, phi) and their corresponding velocities
    (r_dot, theta_dot, phi_dot) to Cartesian coordinates and velocities for each row
    in the input array.
    input_array: A 2D array with shape (N, 4) where each row is of the form:
        [r, (theta=0.5pi), phi, r_dot, (theta_dot=0), phi_dot]
    Returns:
        result: A 2D array with shape (N, 6), where each row is of the form:
            [x, y, z, v_x, v_y, v_z]
    """
    # Unpack input columns for vectorized operations
    r = input_array[:, 0]  # Radial distance
    theta = input_array[:, 1]  # Polar angle (theta)
    phi = input_array[:, 2]  # Azimuthal angle (phi)
    r_dot = input_array[:, 3]  # Radial velocity (r_dot)
    theta_dot = input_array[:, 4]  # Polar velocity (theta_dot)
    phi_dot = input_array[:, 5]  # Azimuthal velocity (phi_dot)
    if(reduced_to_four_dim):
        sin_theta = np.float32(1.0)
        cos_theta = np.float32(0.0)
        theta_dot = np.float32(0.0)
    else:
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
    # Convert position from spherical to Cartesian
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    # Convert velocity from spherical to Cartesian
    v_x = r_dot * sin_theta * np.cos(phi) + r * theta_dot * cos_theta * np.cos(phi) - r * sin_theta * np.sin(phi) * phi_dot
    v_y = r_dot * sin_theta * np.sin(phi) + r * theta_dot * cos_theta * np.sin(phi) + r * sin_theta * np.cos(phi) * phi_dot
    v_z = r_dot * cos_theta - r * sin_theta * theta_dot
    # print("[debug] ", x, y, z, v_x, v_y, v_z)
    result = np.column_stack((x, y, z, v_x, v_y, v_z))
    return result


def rnsphere_to_sphere(input_array, t_prime, constants):
    """
    input_array: A 2D array with shape (N,6) where each row is of the form:
        [r', theta', phi', r'_dot, theta'_dot, phi'_dot]
    output_array: A 2D array with shape (N,6) where each row is of the form:
        [r, theta, phi, r_dot, theta_dot, phi_dot]
    """
    r = input_array[:, 0] * constants.R
    theta = input_array[:, 1] * constants.THETA
    phi = input_array[:, 2] * constants.PHI + constants.W*constants.T*t_prime
    r_dot = input_array[:, 3] * (constants.R/constants.T)
    theta_dot = input_array[:, 4] * (constants.THETA/constants.T)
    phi_dot = input_array[:,5] * (constants.PHI/constants.T) + constants.W
    result = np.column_stack((r, theta, phi, r_dot, theta_dot, phi_dot))
    return result


def sphere_to_rnsphere(input_array, t_prime, constants):
    """
    inverse of the fcn: rnsphere_to_sphere
    """
    r =         input_array[:, 0]/constants.R
    theta =     input_array[:, 1]/constants.THETA
    phi =      (input_array[:, 2]-constants.W*constants.T*t_prime)/constants.PHI
    r_dot =     input_array[:, 3]/(constants.R/constants.T)
    theta_dot = input_array[:, 4]/(constants.THETA/constants.T)
    phi_dot =  (input_array[:, 5]-constants.W)/(constants.PHI/constants.T)
    result = np.column_stack((r, theta, phi, r_dot, theta_dot, phi_dot))
    return result

    
def RV2COE(input_array, mu_gravity):
    r_IJK = input_array[0:3]
    v_IJK = input_array[3:6]
    K_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    h_vec = np.cross(r_IJK, v_IJK)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross(K_vec, h_vec) ###
    n = np.linalg.norm(n_vec)
    r = np.linalg.norm(r_IJK)
    v = np.linalg.norm(v_IJK)
    e_vec = ((v**2 - mu_gravity/r)*r_IJK - np.dot(r_IJK, v_IJK)*v_IJK)/mu_gravity
    e = np.linalg.norm(e_vec)
    xi = v**2/2 - mu_gravity/r
    a = -mu_gravity/(2*xi)
    i = np.arccos(h_vec[2]/h)
    if(abs(i) < 1e-10):
        RAAN = 0.0
        lonper = np.arccos(e_vec[0]/e)
        if(e_vec[1] < 0):
            lonper = 2.0*np.pi - lonper
        w = lonper
    else:
        RAAN = np.arccos(n_vec[0]/n)
        if(n_vec[1] < 0):
            RAAN = 2.0*np.pi - RAAN
        w = np.arccos(np.dot(n_vec, e_vec)/ (n*e))
        if(e_vec[2] < 0):
            w = 2.0*np.pi - w
        lonper = RAAN + w
    nu_cos = np.dot(e_vec, r_IJK)/ (e*r)
    nu_cos = np.clip(nu_cos, -1.0, 1.0)
    nu = np.arccos(nu_cos)
    if(np.dot(r_IJK, v_IJK) < 0):
        nu = 2.0*np.pi - nu
    return a, e, i, RAAN, w, nu, lonper


def true_to_mean_anomaly(e, nu):
    if(1-e**2 < 0):
        print("[debug] not valid keplerian eccentricity: ", e)
    E = np.arctan2(np.sin(nu)*np.sqrt(1-e**2), np.cos(nu)+e)
    M = E - e*np.sin(E)
    return M


def solve_kepler(e, M):
    # Define the function to solve
    def fun(psi):
        return psi - e * np.sin(psi) - M
    # Improved initial guess
    psi0 = M if M < np.pi else M - 2 * np.pi  # Adjust based on the mean anomaly
    # Solve the equation
    psi_solution = fsolve(fun, psi0)[0]
    # Calculate f
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(psi_solution / 2))
    # Adjust f to be in the range [0, 2*pi]
    if f < 0:
        f += 2 * np.pi
    return f


def COE2RV(a, e, i, RAAN, w, nu, lonper, mu_gravity):
    p = a*(1-e**2)
    if(abs(i) < 1e-10):
        RAAN = 0.0
        w    = lonper
    r_vec_PQW = np.array([p*np.cos(nu)/(1+e*np.cos(nu)),
                      p*np.sin(nu)/(1+e*np.cos(nu)),
                      0.0])
    v_vec_PQW = np.array([-np.sqrt(mu_gravity/p)*np.sin(nu),
                          np.sqrt(mu_gravity/p)*(e+np.cos(nu)),
                          0.0])
    Rot = np.array([[np.cos(RAAN)*np.cos(w) - np.sin(RAAN)*np.sin(w)*np.cos(i),
                     -np.cos(RAAN)*np.sin(w) - np.sin(RAAN)*np.cos(w)*np.cos(i),
                     np.sin(RAAN)*np.sin(i)],
                    [np.sin(RAAN)*np.cos(w) + np.cos(RAAN)*np.sin(w)*np.cos(i),
                     -np.sin(RAAN)*np.sin(w) + np.cos(RAAN)*np.cos(w)*np.cos(i),
                     -np.cos(RAAN)*np.sin(i)],
                    [np.sin(w)*np.sin(i),
                     np.cos(w)*np.sin(i),
                     np.cos(i)]])
    r_IJK = np.matmul(Rot, r_vec_PQW)
    v_IJK = np.matmul(Rot, v_vec_PQW)
    result = np.concatenate((r_IJK, v_IJK))
    return result


def cartesian_to_sphere(input_array):
    """
    input_array: A 2D array with shape (N, 6), where each row is of the form:
            [x, y, z, v_x, v_y, v_z]
    Returns:
        result: A 2D array with shape (N, 4) where each row is of the form:
        [r, (theta=0.5pi), phi, r_dot, (theta_dot=0), phi_dot]
    """
    # Unpack the position and velocity vectors
    x = input_array[:,0]
    y = input_array[:,1]
    z = input_array[:,2]
    v_x = input_array[:,3]
    v_y = input_array[:,4]
    v_z = input_array[:,5]

    # Compute the spherical position components
    r = np.sqrt(x**2 + y**2 + z**2) # Radial distance
    theta = np.arccos(z / r);       # Polar angle (inclination)
    phi = np.arctan2(y, x);         # Azimuthal angle (longitude)

    # Compute the spherical velocity components
    r_dot = (x * v_x + y * v_y + z * v_z) / r
    theta_dot = (-1/np.sqrt(1-(z/r)**2))*(v_z/r - z*r_dot/r**2)
    phi_dot = (1/(1+(y/x)**2))*(v_y/x - y*v_x/x**2)
    # print("[debug] ", x, y, z, v_x, v_y, v_z)
    result = np.column_stack((r, theta, phi, r_dot, theta_dot, phi_dot))
    return result


def simple_interpolate(arr):
    """
    Given a 1D numpy array, return a new array that inserts the average
    of each pair of adjacent elements between them.
    
    For example:
    If arr = [0.0, 0.4, 0.8]
    then the result will be [0.0, 0.2, 0.4, 0.6, 0.8]
    """
    # Number of original elements
    n = len(arr)
    # New array length will be (2*n - 1)
    new_arr = np.empty(2 * n - 1, dtype=arr.dtype)
    
    # Place the original values in the even indices of the new array
    new_arr[0::2] = arr
    
    # Calculate averages and place in the odd indices
    new_arr[1::2] = (arr[:-1] + arr[1:]) * 0.5
    
    return new_arr

##############
## Exp Case2
##############    

def exp_case2_generate_samples(constants, use_j2=False, N_samples=100, dtt=1e-4):
    for t_prime in constants.T_PRIME_SPAN:
        X = exp_case2_propagate_samples(constants, use_j2=use_j2, t=t_prime, stat_sample=N_samples, dtt=dtt)
        np.save("data/samples/samples_t{:.3f}.npy".format(t_prime), X)


def exp_case2_propagate_samples(constants, use_j2=False, t=0.2, stat_sample=1, dtt=1e-4):
    X_four_dim = np.random.multivariate_normal(constants.N_MEAN_I, constants.N_COV_I, size=stat_sample).astype(np.float32)
    # append constant theta' = 0.5pi/THETA, theta'_dot = 0.0
    X = np.zeros((stat_sample, 6))
    X[:,0] = X_four_dim[:,0]
    X[:,1] = np.ones((stat_sample,)) * 0.5 * np.float32(np.pi) / constants.THETA
    X[:,2] = X_four_dim[:,1]
    X[:,3] = X_four_dim[:,2]
    X[:,5] = X_four_dim[:,3]
    # convert rns to sphere
    kf = int(t/dtt)
    for i in range(stat_sample):
        x = X[i, :]
        # ode45 propogate the X_sph(t)
        for k in range(kf):
            x = x + exp_case2_dyn_normsph(x, constants, use_j2) * dtt # + g*dw
        X[i, :] = x
    return X


def exp_case2_dyn_normsph(x, constants, use_j2):
    # normalized spherical coordinate dynamics
    if(use_j2):
        J2 = constants.J2
    else:
        J2 = 0.0
    r = x[0]
    th = x[1]
    phi = x[2]
    vr = x[3]
    vth = x[4]
    vphi = x[5]
    f1 = vr
    f2 = 0.0 # (constants)
    f3 = vphi
    aux = constants.W + constants.PHI/constants.T * vphi
    f4 = constants.T**2 * r * aux**2 - constants.T**2 * constants.MU_EARTH /(constants.R**3 * r**2) + \
         2.0*(3*constants.T**2 * J2 * constants.MU_EARTH * constants.R_EARTH**2)/(2*constants.R**5 * r**4)
    f5 = 0.0 # (constants)
    f6 = -2*constants.T/(r * constants.PHI) * vr * aux
    return np.array([f1, f2, f3, f4, f5, f6])