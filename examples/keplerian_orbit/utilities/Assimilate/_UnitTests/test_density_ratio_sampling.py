import os
import sys
import numpy as np

# Adjust this import depending on where you put the function:
# e.g. if it's in Assimilate/density_ratio_sampling.py:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from density_ratio_sampling import sample_density_ratio_class_gmm_1d  # or your module name


def test_sample_density_ratio_class_gmm_1d(show_plot: bool = False):
    """
    Unit test + visualization for sample_density_ratio_class_gmm_1d.

    Strategy:
      1. Define a simple 1D bimodal GMM and a domain [x_min, x_max].
      2. Construct l(x) = max(GMM(x) - B1, 0) and u(x) = GMM(x) + B1.
      3. Call sample_density_ratio_class_gmm_1d to get:
            - x_u: samples from u(x) (normalized on the domain)
            - w:   importance weights for the lower-bound density f_low ∝ l(x)
      4. Check moment matching:
            - For u(x): E_u[x], E_u[x^2] using x_u
            - For l(x): E_l[x], E_l[x^2] using (x_u, w)
         against numerical quadrature over the grid.
      5. Optionally, plot histograms + analytic densities for visual inspection.
    """

    rng = np.random.default_rng(123)

    # --- 1. Define a simple 1D GMM ---
    K = 2
    weights = np.array([0.3, 0.7])
    means = np.array([[-1.0],
                      [2.0]])
    covs = np.array([[[0.5 ** 2]],
                     [[1.0 ** 2]]])

    B1 = 0.05
    x_min, x_max = -5.0, 6.0
    M = 20000  # number of samples from u

    # --- 2. Helper: GMM pdf, l(x), u(x), normalized densities ---
    def gmm_pdf_1d(x):
        x = np.asarray(x, dtype=float)
        x_col = x[None, :]                       # (1,N)
        vars_ = covs[:, 0, 0]                    # (K,)
        vars_col = vars_[:, None]                # (K,1)

        norm_consts = 1.0 / np.sqrt(2.0 * np.pi * vars_col)   # (K,1)
        exponents = -0.5 * ((x_col - means) ** 2 / vars_col)  # (K,N)
        comp_pdfs = norm_consts * np.exp(exponents)           # (K,N)
        return np.dot(weights, comp_pdfs)                     # (N,)

    grid = np.linspace(x_min, x_max, 4001)
    dx = grid[1] - grid[0]
    gmm_vals = gmm_pdf_1d(grid)

    l_vals = np.clip(gmm_vals - B1, 0.0, None)
    u_vals = gmm_vals + B1

    # Normalize on [x_min, x_max]
    int_u = np.trapz(u_vals, grid)
    int_l = np.trapz(l_vals, grid)
    if int_u <= 0.0:
        raise ValueError("u(x) integrates to zero or negative on the domain.")
    if int_l <= 0.0:
        raise ValueError("l(x) integrates to zero or negative on the domain.")

    u_norm = u_vals / int_u
    l_norm = l_vals / int_l

    # --- 3. Draw samples from u and weights for l ---
    x_u, w = sample_density_ratio_class_gmm_1d(
        weights, means, covs, B1, M, x_min, x_max, rng=rng
    )

    # --- 4. Moment matching checks ---

    def true_moments(density_vals, x_grid):
        """Return (E[x], E[x^2]) for a normalized density on the given grid."""
        ex = np.trapz(x_grid * density_vals, x_grid)
        ex2 = np.trapz((x_grid ** 2) * density_vals, x_grid)
        return ex, ex2

    # True moments for u_norm and l_norm
    Ex_u_true, Ex2_u_true = true_moments(u_norm, grid)
    Ex_l_true, Ex2_l_true = true_moments(l_norm, grid)

    # Monte Carlo estimates:
    #  - For u(x): unweighted moments using x_u
    #  - For l(x): weighted moments using (x_u, w)
    Ex_u_mc = np.mean(x_u)
    Ex2_u_mc = np.mean(x_u ** 2)

    Ex_l_mc = np.sum(w * x_u)
    Ex2_l_mc = np.sum(w * (x_u ** 2))

    # Relative errors (add small epsilon to avoid division by 0)
    eps = 1e-12
    rel_err_Ex_u = abs(Ex_u_mc - Ex_u_true) / max(abs(Ex_u_true), eps)
    rel_err_Ex2_u = abs(Ex2_u_mc - Ex2_u_true) / max(abs(Ex2_u_true), eps)

    rel_err_Ex_l = abs(Ex_l_mc - Ex_l_true) / max(abs(Ex_l_true), eps)
    rel_err_Ex2_l = abs(Ex2_l_mc - Ex2_l_true) / max(abs(Ex2_l_true), eps)

    print("True moments for u(x):     E[x]={:.4f}, E[x^2]={:.4f}"
          .format(Ex_u_true, Ex2_u_true))
    print("MC   moments for u(x):     E[x]={:.4f}, E[x^2]={:.4f}"
          .format(Ex_u_mc, Ex2_u_mc))
    print("Rel errors for u:          E[x]={:.3e}, E[x^2]={:.3e}"
          .format(rel_err_Ex_u, rel_err_Ex2_u))

    print("True moments for l(x):     E[x]={:.4f}, E[x^2]={:.4f}"
          .format(Ex_l_true, Ex2_l_true))
    print("MC   moments for l(x):     E[x]={:.4f}, E[x^2]={:.4f}"
          .format(Ex_l_mc, Ex2_l_mc))
    print("Rel errors for l:          E[x]={:.3e}, E[x^2]={:.3e}"
          .format(rel_err_Ex_l, rel_err_Ex2_l))

    # Tolerances – you can tighten/loosen as you like
    tol_u = 0.05
    tol_l = 0.05

    assert rel_err_Ex_u < tol_u and rel_err_Ex2_u < tol_u, (
        "Upper density u(x) moments not matched well enough. "
        f"rel_err_Ex={rel_err_Ex_u:.3e}, rel_err_Ex2={rel_err_Ex2_u:.3e}"
    )
    assert rel_err_Ex_l < tol_l and rel_err_Ex2_l < tol_l, (
        "Lower density l(x) moments not matched well enough via importance weights. "
        f"rel_err_Ex={rel_err_Ex_l:.3e}, rel_err_Ex2={rel_err_Ex2_l:.3e}"
    )

    print("Test passed: moment matching looks good for both u(x) and l(x).")

    # --- 5. Optional visualization ---
    if show_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ax0, ax1 = axes

        # PDF of u and histogram of x_u
        ax0.plot(grid, u_norm, label="u_norm(x)", linewidth=2)
        # scale histogram to integrate ~1 under u_norm
        ax0.hist(
            x_u,
            bins=80,
            range=(x_min, x_max),
            density=True,
            alpha=0.3,
            label="samples from u (hist)",
        )
        ax0.set_ylabel("density")
        ax0.set_title("Upper density u(x) and samples x_u")
        ax0.legend()

        # PDF of l and weighted histogram using w
        ax1.plot(grid, l_norm, label="l_norm(x)", linewidth=2)

        # For a weighted histogram, we can approximate via:
        # replicate weights across bins using np.histogram with weights.
        counts, bin_edges = np.histogram(
            x_u,
            bins=80,
            range=(x_min, x_max),
            weights=w,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # convert counts to a density: counts_k ≈ ∫_bin_k f(x) dx
        bin_width = bin_edges[1] - bin_edges[0]
        density_est = counts / bin_width
        ax1.plot(bin_centers, density_est, "o", alpha=0.7,
                 label="weighted hist ~ l_norm(x)")

        ax1.set_xlabel("x")
        ax1.set_ylabel("density")
        ax1.set_title("Lower density l(x) and importance-weighted samples")
        ax1.legend()

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    # run the test with plots if this file is executed directly
    test_sample_density_ratio_class_gmm_1d(show_plot=True)
