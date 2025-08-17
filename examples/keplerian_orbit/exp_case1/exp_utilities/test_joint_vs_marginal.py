import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def sample_bivariate_normal(n, rho, rng):
    """Return samples (x, y) ~ N([0,0], [[1, rho], [rho, 1]])."""
    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, 2))
    return (z @ L.T)  # shape (n, 2)

# --- generate two *different* joints with the same N(0,1) marginals ---
n = 100_000
xy_indep = sample_bivariate_normal(n, rho=0.0, rng=rng)   # independent
xy_corr  = sample_bivariate_normal(n, rho=0.9, rng=rng)   # strongly correlated

x0, y0 = xy_indep[:,0], xy_indep[:,1]
x1, y1 = xy_corr[:,0],  xy_corr[:,1]

# --- print quick stats: marginals match, correlation differs ---
print("Means (indep):", np.mean(x0), np.mean(y0), "  Stds:", np.std(x0), np.std(y0), "  Corr:", np.corrcoef(x0,y0)[0,1])
print("Means (corr) :", np.mean(x1), np.mean(y1), "  Stds:", np.std(x1), np.std(y1), "  Corr:", np.corrcoef(x1,y1)[0,1])

# --- plot: marginals overlap; joints differ ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# X marginal: overlay histograms
axes[0,0].hist(x0, bins=60, density=True, alpha=0.5, label=r"$\rho=0$")
axes[0,0].hist(x1, bins=60, density=True, alpha=0.5, label=r"$\rho=0.9$")
axes[0,0].set_title("Marginal of X (both ~ N(0,1))")
axes[0,0].legend()

# Y marginal: overlay histograms
axes[0,1].hist(y0, bins=60, density=True, alpha=0.5, label=r"$\rho=0$")
axes[0,1].hist(y1, bins=60, density=True, alpha=0.5, label=r"$\rho=0.9$")
axes[0,1].set_title("Marginal of Y (both ~ N(0,1))")
axes[0,1].legend()

# Joint: scatter/hexbin (use hexbin for clarity)
hb0 = axes[1,0].hexbin(x0, y0, gridsize=60, cmap="Blues")
axes[1,0].set_title("Joint PDF (independent, ρ=0)")
axes[1,0].set_xlabel("X"); axes[1,0].set_ylabel("Y")

hb1 = axes[1,1].hexbin(x1, y1, gridsize=60, cmap="Reds")
axes[1,1].set_title("Joint PDF (correlated, ρ=0.9)")
axes[1,1].set_xlabel("X"); axes[1,1].set_ylabel("Y")

plt.tight_layout()
plt.show()
