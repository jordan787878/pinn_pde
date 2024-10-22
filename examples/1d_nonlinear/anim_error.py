import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.autograd import Variable
import torch

from main import x_low, x_hig, t0, T_end, PNet, load_trained_model, FOLDER, p_init, t1s, DATA_FOLDER, datas, get_e1_normalize, E1Net

device = "cpu"
p_net = PNet().to(device)
p_net = load_trained_model(p_net, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
e_net = E1Net(scale=get_e1_normalize(p_net)).to(device)
e_net = load_trained_model(e_net, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e_net.eval()


# Set up the figure and axis
fig, ax = plt.subplots()
x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
_x = x.reshape(-1,1)
pt_x = Variable(torch.from_numpy(_x.reshape(-1,1)).float()).to(device)


def generate_density(t):
    pt_t = pt_x*0.0 + t
    p_hat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(-1,1)
    return p_hat


def generate_error(t):
    pt_t = pt_x*0.0 + t
    e1_hat = e_net(pt_x, pt_t).data.cpu().numpy().reshape(-1,1)
    return e1_hat


t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
for t1 in t1s:
    p = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
    phat = generate_density(t1)
    e = p - phat
    if t1 == 0.0:
        ax.plot(x, e, color="black", linewidth=0.5, alpha=0.5, label="error")
    else:
        ax.plot(x, e, color="black", linewidth=0.5, alpha=0.5)


pt_t0 = pt_x*0.0 + t0
p0     = p_init(x).reshape(-1,1)
p0_hat = p_net(pt_x, pt_t0).data.cpu().numpy().reshape(-1,1)
e0 = p0-p0_hat
e0_hat = e_net(pt_x, pt_t0).data.cpu().numpy().reshape(-1,1)
line2, = ax.plot(x, e0_hat, color='red', label="error approx.")
ax.set_xlabel('x')
ax.set_ylabel('e(x, t)')
ax.set_ylim([-0.1, 0.1])
ax.legend(loc="upper right")
title = ax.text(0.5, 1.05, 'Animation of e(x, t): t = 0', ha='center', va='center', transform=ax.transAxes)


# Animation update function
def update(frame):
    # p = p_sol(x,frame).reshape(-1,1)
    # phat = generate_density(frame)
    e1_hat = generate_error(frame)
    # line1.set_ydata(p-phat)  # Update the data for the line
    line2.set_ydata(e1_hat)  # Update the data for the line
    title.set_text(f'Animation of e(x, t): t = {frame:.2f}')  # Update the title
    ax.set_ylim([-0.1, 0.1])

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(t0, T_end+0.1, 0.1))

# Save the animation as an MP4 file
ani.save(FOLDER+'figs/error_animation.mp4', writer='ffmpeg', fps=10)

# Display the animation
plt.show()