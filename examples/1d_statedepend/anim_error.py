import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.autograd import Variable
import torch

from main_epochs import x_low, x_hig, t0, T_end, Net, pos_p_net_train, FOLDER, p_init, t1s, p_sol, get_e1_scale, E1Net, pos_e1_net_train

device = "cpu"
p_net = Net().to(device)
p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()
e_net = E1Net(scale=get_e1_scale(p_net)).to(device)
e_net = pos_e1_net_train(e_net, PATH=FOLDER+"output/e1_net.pt", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy"); e_net.eval()


# Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(x_low, x_hig, 100)
_x = x.reshape(-1,1)
pt_x = Variable(torch.from_numpy(_x.reshape(-1,1)).float()).to(device)
pt_t0 = pt_x*0.0 + t0
p0     = p_init(x).reshape(-1,1)
p0_hat = p_net(pt_x, pt_t0).data.cpu().numpy().reshape(-1,1)
e0 = p0-p0_hat
e0_hat = e_net(pt_x, pt_t0).data.cpu().numpy().reshape(-1,1)
line1, = ax.plot(x, e0, color='black', label="error")
line2, = ax.plot(x, e0_hat, color='red', label="error approx.")
ax.set_xlabel('x')
ax.set_ylabel('e(x, t)')
ax.set_ylim([-0.02, 0.02])
ax.legend(loc="upper right")

# for t1 in t1s:
#     p = p_sol(x, t1)
#     ax.plot(x, p, color="black", linewidth=0.5, alpha=0.5)

title = ax.text(0.5, 1.05, 'Animation of p(x, t): t = 0', ha='center', va='center', transform=ax.transAxes)

def generate_density(t):
    pt_t = pt_x*0.0 + t
    p_hat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(-1,1)
    return p_hat


def generate_error(t):
    pt_t = pt_x*0.0 + t
    e1_hat = e_net(pt_x, pt_t).data.cpu().numpy().reshape(-1,1)
    return e1_hat


# Animation update function
def update(frame):
    p = p_sol(x,frame).reshape(-1,1)
    phat = generate_density(frame)
    e1_hat = generate_error(frame)
    line1.set_ydata(p-phat)  # Update the data for the line
    line2.set_ydata(e1_hat)  # Update the data for the line
    title.set_text(f'Animation of e(x, t): t = {frame:.2f}')  # Update the title
    ax.set_ylim([-0.02, 0.02])

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(t0, T_end, 0.1))

# Save the animation as an MP4 file
ani.save(FOLDER+'figs/error_animation.mp4', writer='ffmpeg', fps=30)

# Display the animation
plt.show()