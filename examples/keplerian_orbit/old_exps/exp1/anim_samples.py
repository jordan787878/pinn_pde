# purpose: animate the uncertainty propagation with monte-carlo setting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.optimize import fsolve
from monte import x1_low, x1_hig, x2_low, x2_hig, ti, tf, mu_0, cov_0, mu_gravity
from monte import ecc, inc, RAAN, w

N_samples = 100
dtt = 0.01
t_span = np.arange(ti, tf, dtt)


def generate_traj():
    X = np.zeros((2, len(t_span)))
    X[:,0] = np.random.multivariate_normal(mu_0, cov_0).T
    dX2 = t_span * np.sqrt(mu_gravity/(X[0,0]**3))
    X[0,:] = X[0,0]
    X[1,:] = X[1,0] + dX2
    return X


# Generate N_samples of trajectories
trajectories = [generate_traj() for _ in range(N_samples)]

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatters = []
# Initialize lines and scatter points
for traj in trajectories:
    scatter = ax.scatter([], [], [], color='blue', s=10, alpha=0.2)  # Size of scatter points
    scatters.append(scatter)


def solve_kepler(e, M):
    # Define the function to solve
    def fun(psi):
        return psi - e * np.sin(psi) - M
    # Initial guess for psi
    psi0 = 0
    # Solve the equation
    psi_solution = fsolve(fun, psi0)[0]
    # Calculate f
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(psi_solution / 2))
    # Adjust f to be in the range [0, 2*pi]
    if f < 0:
        f += 2 * np.pi
    return f


def kepler_to_cartesian(a, ecc, inc, w, RAAN, nu, mu_earth=3.986e5, re = 6378.1):
    """
    Convert Keplerian orbital elements to Cartesian position and velocity in a geocentric-equatorial reference system.

    Parameters:
    alt : float
        Altitude (km)
    ecc : float
        Eccentricity
    inc : float
        Inclination (radians)
    w : float
        Argument of perigee (radians)
    nu : float
        Satellite position (radians) 
    RAAN : float
        Right Ascension of Ascending Node (radians)

    Returns:
    X, Y, Z : float
        Position components (km)
    Vx, Vy, Vz : float
        Velocity components (km/s)
    """
    # Constants
    # mu_earth = 3.986e5  # Earth Gravitational Constant (km^3/s^2)
    # re = 6378.1         # Earth radius (km)
    p = a * (1 - ecc ** 2)
    r_0 = p / (1 + ecc * np.cos(nu))
    # Position vector in the perifocal reference system Oxyz
    x = r_0 * np.cos(nu)
    y = r_0 * np.sin(nu)
    # Velocity vector in the perifocal reference system Oxyz
    Vx_ = -np.sqrt(mu_earth / p) * np.sin(nu)
    Vy_ = np.sqrt(mu_earth / p) * (ecc + np.cos(nu))
    # Position vector components in the geocentric-equatorial reference system OXYZ
    X = (np.cos(RAAN) * np.cos(w) - np.sin(RAAN) * np.sin(w) * np.cos(inc)) * x + \
        (-np.cos(RAAN) * np.sin(w) - np.sin(RAAN) * np.cos(w) * np.cos(inc)) * y
    Y = (np.sin(RAAN) * np.cos(w) + np.cos(RAAN) * np.sin(w) * np.cos(inc)) * x + \
        (-np.sin(RAAN) * np.sin(w) + np.cos(RAAN) * np.cos(w) * np.cos(inc)) * y
    Z = (np.sin(w) * np.sin(inc)) * x + (np.cos(w) * np.sin(inc)) * y
    # Velocity vector components in the geocentric-equatorial reference system OXYZ
    Vx = (np.cos(RAAN) * np.cos(w) - np.sin(RAAN) * np.sin(w) * np.cos(inc)) * Vx_ + \
         (-np.cos(RAAN) * np.sin(w) - np.sin(RAAN) * np.cos(w) * np.cos(inc)) * Vy_
    Vy = (np.sin(RAAN) * np.cos(w) + np.cos(RAAN) * np.sin(w) * np.cos(inc)) * Vx_ + \
         (-np.sin(RAAN) * np.sin(w) + np.cos(RAAN) * np.cos(w) * np.cos(inc)) * Vy_
    Vz = (np.sin(w) * np.sin(inc)) * Vx_ + (np.cos(w) * np.sin(inc)) * Vy_
    return X, Y, Z, Vx, Vy, Vz


def update_sample(X_data, frame):
    a_k = X_data[0, frame]
    M_k = X_data[1, frame]
    # print(frame, a_k, M_k)
    # orbit elements to position
    true_anomaly = solve_kepler(ecc, M_k)
    x, y, z, vx, vy, vz = kepler_to_cartesian(a_k, ecc, inc, w, RAAN, true_anomaly)
    ax.scatter(x, y, z, color='blue', s=20, alpha=0.3)


# Animation update function
def update(frame):
    ax.clear()
    ax.scatter(0, 0, 0, color='black', s=50)
    for traj in trajectories:
        update_sample(traj, frame)
    ax.set_title(f'Orbit Element Animation, t = {frame * dtt:.0f}')  # Update the title
    ax.set_xlim(-2.0*x1_hig, 2.0*x1_hig)
    ax.set_ylim(-2.0*x1_hig, 2.0*x1_hig)
    ax.set_zlim(-2.0*x1_hig, 2.0*x1_hig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(t_span), 6000))

# Save the animation as an MP4 file
ani.save('figs/sample_animation.mp4', writer='ffmpeg', fps=60)

plt.show()