import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.autograd import Variable
import torch

from main_epochs import x_low, x_hig, t0, T_end, Net, pos_p_net_train, FOLDER, p_init, t1s, p_sol

device = "cpu"
p_net = Net().to(device)
p_net = pos_p_net_train(p_net, PATH=FOLDER+"output/p_net.pt", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy"); p_net.eval()


# Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(x_low, x_hig, 100)
_x = x.reshape(-1,1)
pt_x = Variable(torch.from_numpy(_x.reshape(-1,1)).float()).to(device)


def generate_density(t):
    pt_t = pt_x*0.0 + t
    p_hat = p_net(pt_x, pt_t).data.cpu().numpy()
    return p_hat


p0     = p_init(x)
p0_hat = generate_density(t0)
line1, = ax.plot(x, p0, color='black', label="sol.")
line2, = ax.plot(x, p0, color='red', label="sol. approx.")
ax.set_xlabel('x')
ax.set_ylabel('p(x, t)')
ax.legend(loc="upper right")

# for t1 in t1s:
#     p = p_sol(x, t1)
#     ax.plot(x, p, color="black", linewidth=0.5, alpha=0.5)

title = ax.text(0.5, 1.05, 'Animation of p(x, t): t = 0', ha='center', va='center', transform=ax.transAxes)


# Animation update function
def update(frame):
    line1.set_ydata(p_sol(x, frame))
    line2.set_ydata(generate_density(frame))  # Update the data for the line
    # title.set_text(f'Animation of p(x, t): t = {frame:.2f}')  # Update the title
    title.set_text(f'Animation of p(x, t): t = {frame:.2f}')  # Update the title

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(t0, T_end, 0.1))

# Save the animation as an MP4 file
ani.save(FOLDER+'figs/pdf_animation.mp4', writer='ffmpeg', fps=30)

# Display the animation
plt.show()