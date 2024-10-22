import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from main import x_low, x_hig, ti, tf, pi, g, l, f_sde, mu_0, cov_0, n_d, B, FOLDER

N_samples = 10
dtt = 0.01
t_span = np.arange(ti, tf, dtt)


def generate_traj():
    X = np.zeros((n_d, len(t_span)+1))
    x = np.random.multivariate_normal(mu_0, cov_0).T
    for i in range(0,len(t_span)):
        X[:,i] = x
        w1 = np.random.normal(0, np.sqrt(dtt))
        w2 = np.random.normal(0, np.sqrt(dtt))
        w = np.array([w1, w2]).reshape(2,)
        x = x + f_sde(x)*dtt + np.matmul(B, w)
    X[:,-1] = x
    return X

# Generate N_samples of trajectories
trajectories = [generate_traj() for _ in range(N_samples)]

# Set up the figure and axis
fig, ax = plt.subplots()
lines = []
scatters = []

# Initialize lines and scatter points
for traj in trajectories:
    line, = ax.plot([0.0, np.cos(traj[0, 0] - 0.5 * pi)], 
                     [0, np.sin(traj[0, 0] - 0.5 * pi)], color="blue", alpha=0.3)
    lines.append(line)
    scatter = ax.scatter([], [], color='blue', s=50, alpha=0.3)  # Size of scatter points
    scatters.append(scatter)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

def update_sample(line_obj, scatter_obj, X_data, frame):
    tip_x = np.cos(X_data[0, frame] - 0.5 * pi)
    tip_y = np.sin(X_data[0, frame] - 0.5 * pi)
    line_obj.set_xdata([0, tip_x])
    line_obj.set_ydata([0, tip_y])  # Update the data for the line
    scatter_obj.set_offsets([tip_x, tip_y])  # Update the scatter point position

# Animation update function
def update(frame):
    for line, scatter, traj in zip(lines, scatters, trajectories):
        update_sample(line, scatter, traj, frame)
    ax.set_title(f'Inverted Pendulum Animation, t = {frame * dtt:.1f}')  # Update the title

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, len(t_span), 2))

# Save the animation as an MP4 file
ani.save(FOLDER+'figs/sample_animation.mp4', writer='ffmpeg', fps=50)

plt.show()


# X1 = generate_traj()
# X2 = generate_traj()
# X3 = generate_traj()


# # Set up the figure and axis
# fig, ax = plt.subplots()
# line1, = ax.plot([0.0, np.cos(X1[0,0]-0.5*pi)], [0, np.sin(X1[0,0]-0.5*pi)], color="blue", alpha=0.5)
# line2, = ax.plot([0.0, np.cos(X2[0,0]-0.5*pi)], [0, np.sin(X2[0,0]-0.5*pi)], color="blue", alpha=0.5)
# line3, = ax.plot([0.0, np.cos(X3[0,0]-0.5*pi)], [0, np.sin(X3[0,0]-0.5*pi)], color="blue", alpha=0.5)

# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)


# def update_sample(line_obj, X_data, frame):
#     line_obj.set_xdata([0, np.cos(X_data[0,frame]-0.5*pi)])
#     line_obj.set_ydata([0, np.sin(X_data[0,frame]-0.5*pi)])  # Update the data for the line


# # Animation update function
# def update(frame):
#     update_sample(line1, X1, frame)
#     update_sample(line2, X2, frame)
#     update_sample(line3, X3, frame)
#     ax.set_title(f'Inverted Pendulum Animation, t = {frame*dtt:.1f}')  # Update the title

# # Create the animation
# ani = FuncAnimation(fig, update, frames=np.arange(1, 501, 2))

# # Save the animation as an MP4 file
# ani.save(FOLDER+'figs/sample_animation.mp4', writer='ffmpeg', fps=50)

# plt.show()

