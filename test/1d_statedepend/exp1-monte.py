import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from tqdm import tqdm
import warnings

# global variable
# Check if MPS (Apple's Metal Performance Shaders) is available
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# FOLDER = "exp4/run-8.1/"

DATA_FOLDER = "exp1/data/monte1/"
device = "cpu"; print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def np_float32(*args, **kwargs):
    return np.array(*args, dtype=np.float32, **kwargs)

# Constants
n_d = np_float32(1)
mu = np_float32(9.0)
std = np_float32(1.0)
a = np_float32(-0.01)
b = np_float32(0.1)
d = np_float32(0.0)
x_low = np_float32(3.0)
x_hig = np_float32(15.0)

t0 = np_float32(0)
T_end = np_float32(5)
t1s = np_float32([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

def f_sde(x):
    return a * x + d

def p_init(x):
    x = np_float32(x)
    return np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(np_float32(2 * np.pi)))

def p_sol_monte(linespace_num=200, stat_sample=100000000):
    dtt = np_float32(0.001)
    dt_save = np_float32(1.0)
    step_save = int(np.round(dt_save/dtt, 2))
    print(step_save)
    t_span = np_float32(np.arange(t0, T_end, dtt))
    num_steps = len(t_span)
    
    # Initialize arrays
    X_last = np_float32(np.random.normal(mu, std, stat_sample))
    bins_x1 = np_float32(np.linspace(x_low, x_hig, num=linespace_num))
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2

    # Digitize v to find which bin each value falls into
    bin_indices_x1 = np.digitize(X_last, bins_x1) - 1
    frequency = np_float32(np.zeros((len(bins_x1) - 1, 1)))
    
    for i in tqdm(range(stat_sample), desc="Counting samples"):
        if 0 <= bin_indices_x1[i] < frequency.shape[0]:
            frequency[bin_indices_x1[i], :] += 1
    
    frequency = frequency / np_float32(stat_sample)
    dx = bins_x1[1] - bins_x1[0]
    frequency = frequency / (dx ** n_d)
    
    np.save(DATA_FOLDER + "psim_t" + str(0.0) + ".npy", frequency)
    np.save(DATA_FOLDER + "xsim.npy", midpoints_x1)
    print("save data to: ", DATA_FOLDER + "psim_t" + str(0.0) + ".npy")

    for step in tqdm(range(1, num_steps + 1), desc="Simulating samples"):
        dW = np_float32(np.random.normal(0, 1, stat_sample)) * (dtt ** 0.5)
        X_new = X_last + f_sde(X_last) * dtt + b * X_last * dW
        X_last = X_new

        if step % step_save == 0:
            t_k = np.round(step * dtt, 5)

            bin_indices_x1 = np.digitize(X_last, bins_x1) - 1
            frequency = np_float32(np.zeros((len(bins_x1) - 1, 1)))
            
            for i in tqdm(range(stat_sample), desc="Counting samples"):
                if 0 <= bin_indices_x1[i] < frequency.shape[0]:
                    frequency[bin_indices_x1[i], :] += 1
            
            frequency = frequency / np_float32(stat_sample)
            dx = bins_x1[1] - bins_x1[0]
            frequency = frequency / (dx ** n_d)
            
            np.save(DATA_FOLDER + "psim_t" + str(t_k) + ".npy", frequency)
            print("step: ", step, ", save data to: ", DATA_FOLDER + "psim_t" + str(t_k) + ".npy")


def plot_p_monte():
    plt.figure()
    t_ks = np.arange(t0, T_end+1.0, 1.0)
    print(t_ks)
    for t1 in t_ks:
        x_sim = np.load(DATA_FOLDER+"xsim.npy")
        p_sim = np.load(DATA_FOLDER+"psim_t"+str(t1)+".npy")
        plt.plot(x_sim, p_sim, label="t="+str(t1))
    plt.grid()
    plt.legend()
    plt.savefig(DATA_FOLDER+"figs/p_sol_monte.png")
    print("save fig to: "+DATA_FOLDER+"figs/p_sol_monte.png")
    plt.close()



def main():
    FLAG_GENERATE_DATA = True
    if(FLAG_GENERATE_DATA):
        p_sol_monte(linespace_num=200, stat_sample=100000)

    # Plot generated data
    plot_p_monte()


if __name__ == "__main__":
    main()
