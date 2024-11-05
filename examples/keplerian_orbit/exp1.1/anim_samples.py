import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

from util import kepler_to_cartesian, solve_kepler, orbital_elements_2d, mu_gravity
from util import x1_low, x1_hig, x2_low, x2_hig


N_samples = 50
dtt = 10.0
t_span = np.arange(0.0, 86400.0, dtt)


# mean pos-vel
x, y, z, vx, vy, vz = kepler_to_cartesian(a=8000.0, ecc=0.15, inc=0.0, w=0.0, RAAN=0.0, nu=0.0)
mu_0  = np.array([x, y, vx, vy]).reshape(4,)
cov_0 = np.array([[0.1**2, 0.0, 0.0, 0.0],
                  [0.0   , 0.1**2, 0.0, 0.0],
                  [0.0   , 0.0, (1e-5)**2, 0.0],
                  [0.0   , 0.0, 0.0, (1e-5)**2]] )


# state def: X = [x, y, vx, vy]
def generate_traj():
    X = np.zeros((4, len(t_span)))
    X[:,0] = np.random.multivariate_normal(mu_0, cov_0).T
    # cartesian to orbit element
    r = np.array([X[0,0], X[1,0], 0.0]).reshape(3,)
    v = np.array([X[2,0], X[3,0], 0.0]).reshape(3,)
    S = np.zeros((5, len(t_span)))
    S[:,0] = orbital_elements_2d(r, v)
    # propagation over time
    dS5 = t_span * np.sqrt(mu_gravity/(S[0,0]**3))
    S[0,:] = S[0,0]
    S[1,:] = S[1,0]
    S[2,:] = S[2,0]
    S[3,:] = S[3,0]
    S[4,:] = S[4,0] + dS5
    # orbit element to cartesian
    for k in range(len(t_span)):
        M_k = S[4, k]
        nu = solve_kepler(e=0.15, M=M_k)
        _x, _y, _z, _vx, _vy, _vz = kepler_to_cartesian(S[0,k], S[1,k], S[2,k], S[3,k], 0.0, nu)
        X[0,k] = _x
        X[1,k] = _y
        X[2,k] = _vx
        X[3,k] = _vy
    return X


# Generate N_samples of trajectories
trajectories = [generate_traj() for _ in range(N_samples)]


# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatters = []
# Initialize lines and scatter points
for traj in trajectories:
    scatter = ax.scatter([], [], [], color='blue', s=20, alpha=0.3)  # Size of scatter points
    scatters.append(scatter)


def update_sample(X_data, frame):
    ax.scatter(X_data[0,frame], X_data[1,frame], 0.0, color='blue', s=20, alpha=0.3)


# Animation update function
def update(frame):
    ax.clear()
    ax.scatter(0, 0, 0, color='black', s=50)
    for traj in trajectories:
        update_sample(traj, frame)
    ax.set_title(f'Orbit Element Animation, t = {frame * dtt:.0f}')  # Update the title
    ax.set_xlim(x1_low, x1_hig)
    ax.set_ylim(x2_low, x2_hig)
    ax.set_zlim(x1_low, x1_hig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(t_span), 10))


# Save the animation as an MP4 file
ani.save('figs/sample_animation.mp4', writer='ffmpeg', fps=30)


plt.show()