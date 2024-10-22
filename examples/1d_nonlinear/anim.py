import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.autograd import Variable
import torch

from main import x_low, x_hig, t0, T_end, PNet, load_trained_model, FOLDER, p_init, t1s, DATA_FOLDER, datas


device = "cpu"
p_net = PNet().to(device)
p_net = load_trained_model(p_net, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")


# Set up the figure and axis
fig, ax = plt.subplots()
x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
_x = x.reshape(-1,1)
pt_x = Variable(torch.from_numpy(_x.reshape(-1,1)).float()).to(device)


def generate_density(t):
    pt_t = pt_x*0.0 + t
    p_hat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(-1,1)
    return p_hat


t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
for t1 in t1s:
    p = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
    if t1 == 0:
        ax.plot(x, p, color="black", linewidth=0.5, alpha=0.5, label="sol.")
    ax.plot(x, p, color="black", linewidth=0.5, alpha=0.5)


p0_hat = generate_density(t0)
line2, = ax.plot(x, generate_density(t0), color='red', label="sol. approx.")
ax.set_xlabel('x')
ax.set_ylabel('p(x, t)')
ax.legend(loc="upper right")
title = ax.text(0.5, 1.05, 'Animation of p(x, t): t = 0', ha='center', va='center', transform=ax.transAxes)


# Animation update function
def update(frame):
    print(frame)
    line2.set_ydata(generate_density(frame))  # Update the data for the line
    # title.set_text(f'Animation of p(x, t): t = {frame:.2f}')  # Update the title
    title.set_text(f'Animation of p(x, t): t = {frame:.2f}')  # Update the title

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(t0, T_end+0.1, 0.1))

# Save the animation as an MP4 file
ani.save(FOLDER+'figs/pdf_animation.mp4', writer='ffmpeg', fps=10)

# Display the animation
plt.show()