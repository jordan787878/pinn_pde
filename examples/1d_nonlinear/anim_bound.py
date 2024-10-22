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
line_list = []
for t1 in t1s:
    p = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
    line_k, = ax.plot(x, p, color="black", linewidth=0.2, alpha=0.5, label="sol." if t1 == 0 else "")
    line_list.append(line_k)


pt_t0 = pt_x*0.0 + t0
p0     = p_init(x).reshape(-1,1)
p0_hat = p_net(pt_x, pt_t0).data.cpu().numpy().reshape(-1,1)
e0_hat = e_net(pt_x, pt_t0).data.cpu().numpy().reshape(-1,1)
line1, = ax.plot(x, p0, color='black', linewidth=0.7)
line2, = ax.plot(x, p0_hat, color='red', label="sol. approx.")
eS = 2*np.max(np.abs(e0_hat))
fill_between_area = ax.fill_between(x.reshape(-1), 
                                    p0_hat.reshape(-1) + eS, 
                                    p0_hat.reshape(-1) - eS, color="green", alpha=0.3, label=r"$e_S$")
ax.set_xlabel('x')
ax.set_ylabel('p(x, t)')
ax.legend(loc="upper right")
title = ax.text(0.5, 1.05, 'Animation of p(x, t): t = 0', ha='center', va='center', transform=ax.transAxes)


# Animation update function
ii = 0
def update(frame):
    global fill_between_area
    global ii
    # p = p_sol(x,frame).reshape(-1,1)
    phat = generate_density(frame)
    e1_hat = generate_error(frame)
    eS = 2*np.max(np.abs(e1_hat))
    # Remove the previous fill area
    fill_between_area.remove()
    fill_between_area = ax.fill_between(x.reshape(-1), 
                                    phat.reshape(-1) + eS, 
                                    phat.reshape(-1) - eS, color="green", alpha=0.2, label=r"$e_S$")
    line2.set_ydata(phat)  # Update the data for the line
    
    # hide a line in the line_list at certain frame
    if(ii < len(t1s)):
        if(abs(frame-t1s[ii]) < 1e-3):
            print(frame, ii)
            line1.set_ydata(np.load(DATA_FOLDER + datas[0] + "psim_t" + str(np.round(frame,1)) + ".npy").reshape(-1, 1))
            ii = ii + 1

    title.set_text(f'Animation of p(x, t): t = {frame:.2f} with Error Bound')  # Update the title

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(t0, T_end+0.1, 0.1))

# Save the animation as an MP4 file
ani.save(FOLDER+'figs/bound_animation.mp4', writer='ffmpeg', fps=10)

# Display the animation
plt.show()