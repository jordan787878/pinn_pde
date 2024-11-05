import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np


from main import x1_low, x1_hig, x2_low, x2_hig, ti, tf, pi
from main import Net
from main import pos_p_net_train
from main import get_pnet_output

DATA_FOLDER = "data/"
device = "cpu"

# Fixed grids for x1 and x2
x1_grid = np.load(DATA_FOLDER+"x1_grid.npy")
x2_grid = np.load(DATA_FOLDER+"x2_grid.npy")

# Neural network
p_net = Net().to(device)
p_net = pos_p_net_train(p_net, PATH="output/p_net.pth", PATH_LOSS="output/p_net_train_loss.npy"); p_net.eval()

# Example function for generating probability density based on time
def generate_density(t):
    return get_pnet_output(p_net, x1_grid, x2_grid, t)

# Create a figure for the animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the limits for the axes
ax.set_xlim(x1_low, x1_hig)
ax.set_ylim(x2_low, x2_hig)
ax.set_zlim(0, 0.004)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('PDF')

# Initialize a surface plot
t = 0
z = generate_density(t)
surface = ax.plot_surface(x1_grid, x2_grid, z, cmap='viridis', edgecolor='none', vmin=0, vmax=0.004)

# Function to update the surface plot for each frame
def update(t):
    ax.cla()  # Clear the current axes
    ax.set_xlim(x1_low, x1_hig)
    ax.set_ylim(x2_low, x2_hig)
    ax.set_zlim(0, 0.004)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('PDF')
    ax.set_title(f'Surface Plot of p(x1, x2, t={t})')

    z = generate_density(t)
    surface = ax.plot_surface(x1_grid, x2_grid, z, cmap='viridis', edgecolor='none', vmin=0, vmax=0.004)
    return surface,

# Create an animation
anim = FuncAnimation(fig, update, frames=np.arange(0, 7201, 60), blit=False)

# Save the animation as an MP4 file
anim.save('figs/pdf_animation.mp4', writer='ffmpeg', fps=30)

plt.show()