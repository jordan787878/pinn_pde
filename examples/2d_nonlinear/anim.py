import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np


from main import x_low, x_hig, ti, tf, pi, FOLDER_DATA, FOLDER, Variable, torch
from main import Net
from main import pos_p_net_train

device = "cpu"

# Fixed grids for x1 and x2
x1_grid = np.load(FOLDER_DATA+"x_sim_grid.npy")
x2_grid = np.load(FOLDER_DATA+"y_sim_grid.npy")

# Neural network
p_net = Net().to(device)
p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()

# Example function for generating probability density based on time
def generate_density(t):
    x = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])#; print(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t1 = Variable(torch.from_numpy(x[:,0]*0+t).float(), requires_grad=True).view(-1,1).to(device)
    p_hat = p_net(pt_x, pt_t1)
    sample_size = len(x1_grid)
    p_hat_numpy = p_hat.data.cpu().numpy().reshape((sample_size, sample_size))
    return p_hat_numpy

# Create a figure for the animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the limits for the axes
ax.set_xlim(x_low, x_hig)
ax.set_ylim(x_low, x_hig)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r'$\omega$')
ax.set_zlabel('PDF')
ax.set_zlim(0, 0.2)
ax.view_init(elev=55, azim=-130)

# Initialize a surface plot
t = 0
z = generate_density(t)
surface = ax.plot_surface(x1_grid, x2_grid, z, cmap='viridis', edgecolor='none', vmin=0, vmax=0.2)

# Function to update the surface plot for each frame
def update(t):
    ax.cla()  # Clear the current axes
    ax.set_xlim(x_low, x_hig)
    ax.set_ylim(x_low, x_hig)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r'$\omega$')
    ax.set_zlabel('PDF')
    ax.set_zlim(0, 0.2)
    ax.set_title(f'PDF approximate, t = {t:.1f}')

    z = generate_density(t)
    surface = ax.plot_surface(x1_grid, x2_grid, z, cmap='viridis', edgecolor='none', vmin=0, vmax=0.2)
    return surface,

# Create an animation
anim = FuncAnimation(fig, update, frames=np.arange(ti, tf+0.1, 0.1), blit=False)

# Save the animation as an MP4 file
anim.save(FOLDER+'figs/pdf_animation.mp4', writer='ffmpeg', fps=10)

plt.show()