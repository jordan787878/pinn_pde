import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.animation import FuncAnimation

# -------------------------------
# Define the domain and nominal prior
# -------------------------------
# Use a full normal on x in [-10, 10] (covers most of the probability mass)
x = np.linspace(-10, 10, 400)
hat_p = norm.pdf(x, loc=1.0, scale=2)  # Full normal density

# Bound for the prior uncertainty (choose small enough so that hat_p - B is nonnegative for significant mass)
B = 0.02

# -------------------------------
# Likelihood function parameters and model
# -------------------------------
# Measurement model: y = x + w, with w ~ N(0, sigma^2)
sigma = 0.1

def likelihood(x, y, sigma):
    """Gaussian likelihood: p(y|x) = N(y; x, sigma^2)"""
    return norm.pdf(y, loc=x, scale=sigma)

# -------------------------------
# Compute posteriors given an observation y
# -------------------------------
def compute_posteriors(y):
    """
    For a given observation y, compute:
      - Standard posterior using nominal prior
      - p_min: using max(hat_p(x)-B, 0) in numerator and hat_p(x)+B in the evidence
      - p_max: using hat_p(x)+B in numerator and max(hat_p(x)-B, 0) in the evidence
    """
    L = likelihood(x, y, sigma)
    
    # Standard posterior (Bayes update with the nominal prior)
    numerator_std = L * hat_p
    norm_std = np.trapz(numerator_std, x)
    posterior_std = numerator_std / norm_std

    # Ensure that hat_p - B is nonnegative.
    lower_prior = np.maximum(hat_p - B, 0)
    
    # Lower extreme posterior:
    numerator_min = L * lower_prior
    norm_min = np.trapz(L * (hat_p + B), x)
    posterior_min = numerator_min / norm_min

    # Upper extreme posterior:
    numerator_max = L * (hat_p + B)
    norm_max = np.trapz(L * lower_prior, x)
    posterior_max = numerator_max / norm_max

    return posterior_std, posterior_min, posterior_max, L

# -------------------------------
# Static Plot for a fixed observation y
# -------------------------------
y_obs = 1.0  # Example observation
posterior_std, posterior_min, posterior_max, L = compute_posteriors(y_obs)

plt.figure(figsize=(10,6))
# Plot the nominal prior and its uncertainty band (ensuring nonnegative lower bound)
plt.plot(x, hat_p, label="Nominal Prior", color="black", linestyle=":")
plt.fill_between(x, np.maximum(hat_p - B, 0), hat_p + B, color="orange", alpha=0.3, label="Prior Uncertainty Band")
# Plot the posterior update and its envelope
plt.plot(x, posterior_std, label="Nominal Posterior (Nominal Prior)", color="blue")
plt.plot(x, posterior_min, label="Posterior Min (Lower Extreme Prior)", color="red", linestyle="--")
plt.plot(x, posterior_max, label="Posterior Max (Upper Extreme Prior)", color="green", linestyle="--")
plt.fill_between(x, posterior_min, posterior_max, color="gray", alpha=0.3, label="Posterior Band")
plt.xlabel("x")
plt.ylabel("Density")
plt.title(f"Prior and Posterior Update with Prior Uncertainty for y = {y_obs}, σ = {sigma}")
plt.legend()
plt.show()

# -------------------------------
# Animation: Varying the Observation y over user-specified range
# -------------------------------
# User-specified parameters: y will vary from -a to b.
a = 5.0  # y minimum is -a
b = 5.0  # y maximum is b
frames = 100  # total number of animation frames

fig, ax = plt.subplots(figsize=(10,6))
# Plot the nominal prior and its uncertainty band once.
line_prior, = ax.plot(x, hat_p, label="Nominal Prior", color="black", linestyle=":")
ax.fill_between(x, np.maximum(hat_p - B, 0), hat_p + B, color="orange", alpha=0.3, label="Prior Uncertainty Band")
line_std, = ax.plot([], [], label="Standard Posterior", color="blue")
line_min, = ax.plot([], [], label="Posterior Min", color="red", linestyle="--")
line_max, = ax.plot([], [], label="Posterior Max", color="green", linestyle="--")
ax.set_xlim(-10, 10)
ax.set_ylim(0, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("Density")
title = ax.set_title("Prior and Posterior Update with Prior Uncertainty")

# We'll store the current posterior band fill in a mutable container.
posterior_fill = [None]

def init():
    line_std.set_data([], [])
    line_min.set_data([], [])
    line_max.set_data([], [])
    return line_std, line_min, line_max, line_prior

def animate(i):
    # y varies smoothly from -a to b.
    y_val = -a + (a + b) * (i / frames)
    posterior_std, posterior_min, posterior_max, _ = compute_posteriors(y_val)
    
    # Update the posterior lines.
    line_std.set_data(x, posterior_std)
    line_min.set_data(x, posterior_min)
    line_max.set_data(x, posterior_max)
    
    # Remove the previous posterior band fill, if it exists.
    if posterior_fill[0] is not None:
        posterior_fill[0].remove()
    # Plot new posterior band.
    posterior_fill[0] = ax.fill_between(x, posterior_min, posterior_max, color="gray", alpha=0.3, label="Posterior Band")
    
    title.set_text(f"Prior and Posterior Update with Prior Uncertainty for y = {y_val:.2f}")
    return line_std, line_min, line_max, line_prior, posterior_fill[0]

anim = FuncAnimation(fig, animate, frames=frames, init_func=init, interval=100, blit=False)
plt.legend()
plt.show()
