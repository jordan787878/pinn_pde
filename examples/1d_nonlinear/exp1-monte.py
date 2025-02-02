import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from tqdm import tqdm
import warnings


DATA_FOLDER = "exp1/data/10+6_samples/"
device = "cpu"; print(device)

# Set a fixed seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

n_d = 1
mu = -2
std = 0.5
a = -0.1
b = 0.1
c = 0.5
d = 0.5
e = 0.8
x_low = -6
x_hig = 6

t0 = 0
T_end = 5
t1s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def f_sde(x):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = a*np.power(x, 3) + b*np.power(x, 2) + c*x + d
        except RuntimeWarning as e:
            print(f"Warning occurred at x = {x}: {e}")
            result = np.nan  # or handle the error in another way
    return result


def p_init(x):
  return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def test_p_init():
    x = np.arange(x_low, x_hig, 0.01).reshape(-1,1)
    p = p_init(x)
    return max(abs(p))[0]


def p_sol_monte(linespace_num=200, stat_sample=100000000):
    dtt = 0.0005
    dt_save = 0.5
    step_save = int(dt_save/dtt)
    t_span = np.arange(t0, T_end, dtt)
    num_steps = len(t_span)
    
    # Initialize arrays
    X_last = np.random.normal(mu, std, stat_sample)
    bins_x1 = np.linspace(x_low, x_hig, num=linespace_num)
    midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2

    # Digitize v to find which bin each value falls into for both dimensions
    bin_indices_x1 = np.digitize(X_last, bins_x1) - 1
    # Initialize the frequency array
    frequency = np.zeros((len(bins_x1) - 1, 1))
    # Count the occurrences in each 2D bin
    for i in range(stat_sample):
        if 0 <= bin_indices_x1[i] < frequency.shape[0]:
            frequency[bin_indices_x1[i], :] += 1
    # Normalize the frequency to get the proportion
    frequency = frequency / stat_sample
    dx = bins_x1[1]-bins_x1[0]
    frequency = frequency/(dx**n_d)
    np.save(DATA_FOLDER+"psim_t"+str(0.0)+".npy", frequency)
    np.save(DATA_FOLDER+"xsim.npy", midpoints_x1)

    # Vectorized simulation of the SDE
    for step in tqdm(range(1, num_steps + 1), desc="Simulating samples"):
        dW = np.random.normal(0, np.sqrt(dtt), stat_sample)
        X_new = X_last + f_sde(X_last) * dtt + e * dW
        X_last = X_new

        if(step % step_save == 0):
            t_k = np.round(step*dtt,2)
            print(t_k)

            # Digitize v to find which bin each value falls into for both dimensions
            bin_indices_x1 = np.digitize(X_last, bins_x1) - 1

            # Initialize the frequency array
            frequency = np.zeros((len(bins_x1) - 1, 1))

            # Count the occurrences in each 2D bin
            for i in range(stat_sample):
                if 0 <= bin_indices_x1[i] < frequency.shape[0]:
                    frequency[bin_indices_x1[i], :] += 1

            # Normalize the frequency to get the proportion
            frequency = frequency / stat_sample
            dx = bins_x1[1]-bins_x1[0]
            frequency = frequency/(dx**n_d)
            np.save(DATA_FOLDER+"psim_t"+str(t_k)+".npy", frequency)


def main():
    # exp1/data
    p_sol_monte(linespace_num=200, stat_sample=1000000)


if __name__ == "__main__":
    main()
