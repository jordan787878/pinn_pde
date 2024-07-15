import numpy as np
import matplotlib.pyplot as plt

def gbm_transition_density(S_t, S_0, mu, sigma, t):
    """
    Calculate the transition density p(S_t | S_0) for Geometric Brownian Motion (GBM).

    Parameters:
    - S_t: The value of the process at time t (current value).
    - S_0: The initial value of the process (value at time t=0).
    - mu: The drift coefficient.
    - sigma: The volatility coefficient.
    - t: Time elapsed.

    Returns:
    - The probability density p(S_t | S_0).
    """
    # Calculate the mean and variance of the log-normal distribution
    mean = np.log(S_0) + (mu - 0.5 * sigma**2) * t
    variance = sigma**2 * t

    # Compute the probability density function
    coeff = 1 / (S_t * sigma * np.sqrt(2 * np.pi * t))
    exponent = - (np.log(S_t) - mean)**2 / (2 * variance)
    density = coeff * np.exp(exponent)

    return density

mu = 0.002   # Drift
sigma = 0.01 # Volatility

x = np.linspace(90, 110)
x0 = 100

plt.figure()
for t in range(1,6):
    p = gbm_transition_density(x, x0, mu, sigma, t=t)
    print(np.sum(p)*(x[1]-x[0]))
    plt.plot(x, p, label=str(t))
plt.show()