"""
test_credible_region.py

Single test function for credible_region.outer_credible_region_gmm_1d.

Layout (from repo root 'Assimilate/'):

    Assimilate/
        credible_region.py
        _UnitTests/
            test_credible_region.py
"""

import os
import sys
import numpy as np
from math import erf as _math_erf
from functools import partial
import matplotlib.pyplot as plt

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]   # repo_root
sys.path.insert(0, str(ROOT))

from utilities.Assimilate.credible_region import outer_credible_region_gmm_1d, outer_credible_region_gmm_1d_with_measurement, make_gmm_pdf_1d, outer_credible_region_1d_from_bounds
from utilities.Assimilate.bayes import data_assimilation_gmm_1d_prior, like_func
from utilities.Assimilate.density_ratio_sampling import sample_density_ratio_class_gmm_1d
from utilities._General.util import set_publication_plot_style, custom_save_plot


def _gmm_cdf_1d_for_test(x, weights, means, covs):
    """
    Test-only helper: evaluate the CDF of a 1D Gaussian mixture model on a grid x.
    Uses math.erf so it does not depend on scipy.
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1, 1)   # (K,1)
    covs = np.asarray(covs, dtype=float).reshape(-1, 1, 1)  # (K,1,1)

    vars_ = covs[:, 0, 0]
    if np.any(vars_ <= 0):
        raise ValueError("All covariance entries must be positive in 1D.")

    x_col = x[None, :]            # (1,N)
    vars_col = vars_[:, None]     # (K,1)

    z = (x_col - means) / np.sqrt(2.0 * vars_col)  # (K,N)
    erf_vec = np.vectorize(_math_erf)
    Phi = 0.5 * (1.0 + erf_vec(z))                 # (K,N)

    return np.dot(weights, Phi)                    # (N,)


def test_outer_credible_region_gmm_1d(show_plot: bool = False):
    """
    Simple unit test / sanity check for outer_credible_region_gmm_1d.

    - Define a 2-component 1D GMM.
    - Compute the outer 95% credible region for the density ratio class.
    - Compute the nominal GMM 95% credible interval from the true GMM CDF.
    - Check that the nominal GMM interval is contained inside the outer region.

    If show_plot is True, also plot:
      - GMM pdf and PDF band with both the outer credible region
        and the GMM nominal 95% interval highlighted.
      - CDF bounds and nominal GMM CDF with both sets of endpoints marked.
    """
    # --- example GMM ---
    K = 2
    weights = np.array([0.4, 0.6])
    means = np.array([[0.0],
                      [2.0]])
    covs = np.array([[[0.5 ** 2]],
                     [[0.8 ** 2]]])

    conf_level = 0.95
    B1 = 0.01

    # --- compute outer credible interval ---
    q_left, q_right, x_grid, F_upper, F_lower = outer_credible_region_gmm_1d(
        weights, means, covs, B1, conf_level=conf_level
    )

    # --- nominal GMM CDF and 95% interval on same grid ---
    gmm_cdf = _gmm_cdf_1d_for_test(x_grid, weights, means, covs)
    gmm_cdf = np.maximum.accumulate(gmm_cdf)
    gmm_cdf = np.clip(gmm_cdf, 0.0, 1.0)

    alpha = 1. - conf_level
    p_left = alpha
    p_right = 1.0 - alpha

    q_gmm_left = float(np.interp(p_left, gmm_cdf, x_grid))
    q_gmm_right = float(np.interp(p_right, gmm_cdf, x_grid))

    outer_width = q_right - q_left
    tol = 1e-3 * max(1.0, abs(outer_width))

    assert q_left - tol <= q_gmm_left <= q_gmm_right <= q_right + tol, (
        "Nominal GMM 95% interval is not contained in the outer credible region.\n"
        f"Outer:  [{q_left:.6f}, {q_right:.6f}]\n"
        f"GMM  :  [{q_gmm_left:.6f}, {q_gmm_right:.6f}]"
    )

    print("Outer credible region 95%: [{:.6f}, {:.6f}]".format(q_left, q_right))
    print("GMM nominal 95% interval: [{:.6f}, {:.6f}]".format(q_gmm_left, q_gmm_right))
    print("Test passed: GMM interval is contained in the outer region.")

    if show_plot:
        # reconstruct GMM pdf and band for plotting
        x = x_grid
        x_col = x[None, :]
        vars_ = covs[:, 0, 0]
        vars_col = vars_[:, None]

        norm_consts = 1.0 / np.sqrt(2.0 * np.pi * vars_col)
        exponents = -0.5 * ((x_col - means) ** 2 / vars_col)
        comp_pdfs = norm_consts * np.exp(exponents)
        gmm_pdf_values = np.dot(weights, comp_pdfs)

        l = np.clip(gmm_pdf_values - B1, 0.0, None)
        u = gmm_pdf_values + B1

        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ax0, ax1 = axes

        # --- PDF band plot ---
        ax0.plot(x, gmm_pdf_values, label="GMM pdf")
        ax0.plot(x, l, "--", label="lower bound l(x)")
        ax0.plot(x, u, "--", label="upper bound u(x)")

        # Outer credible region (band-based)
        ax0.axvspan(q_left, q_right, alpha=0.15,
                    label="outer 95% credible region")

        # Nominal GMM 95% interval
        ax0.axvspan(q_gmm_left, q_gmm_right, alpha=0.15,
                    hatch="///", edgecolor="k", fill=False,
                    label="GMM nominal 95% interval")

        ax0.set_ylabel("density")
        ax0.set_title("PDF band, outer credible region, and GMM nominal interval")
        ax0.legend()

        # --- CDF plot ---
        ax1.plot(x, F_lower, label="F_lower(x)")
        ax1.plot(x, F_upper, label="F_upper(x)")
        ax1.plot(x, gmm_cdf, label="GMM CDF", linewidth=1.5)

        ax1.axhline(p_left, color="gray", linestyle=":")
        ax1.axhline(p_right, color="gray", linestyle=":")

        # Outer interval endpoints
        ax1.axvline(q_left, color="k", linestyle=":", label="q_left (outer)")
        ax1.axvline(q_right, color="k", linestyle="--", label="q_right (outer)")

        # Nominal GMM interval endpoints
        ax1.axvline(q_gmm_left, color="tab:red", linestyle=":",
                    label="q_left (GMM)")
        ax1.axvline(q_gmm_right, color="tab:red", linestyle="--",
                    label="q_right (GMM)")

        ax1.set_xlabel("x")
        ax1.set_ylabel("CDF")
        ax1.set_title("CDF bounds vs nominal GMM CDF and intervals")
        ax1.legend()

        fig.tight_layout()
        plt.show()


def test_outer_credible_region_gmm_1d_with_measurement(show_plot: bool = False, seed=2):
    """
    Simple unit test / sanity check for outer_credible_region_gmm_1d.

    - Define a 2-component 1D GMM.
    - Compute the outer 95% credible region for the density ratio class.
    - Compute the nominal GMM 95% credible interval from the true GMM CDF.
    - Check that the nominal GMM interval is contained inside the outer region.

    If show_plot is True, also plot:
      - GMM pdf and PDF band with both the outer credible region
        and the GMM nominal 95% interval highlighted.
      - CDF bounds and nominal GMM CDF with both sets of endpoints marked.
    """
    # --- example GMM ---
    weights = np.array([0.7, 0.3])
    means = np.array([[0.0],
                      [2.0]])
    covs = np.array([[[0.5 ** 2]],
                     [[0.8 ** 2]]])

    conf_level = 0.95
    B1 = 0.03
    H = 1.0
    R = 0.5  # variance
    obs_model = (H, R)
    x_true = 4
    np.random.seed(seed=seed)
    y = x_true + np.random.normal(loc=0., scale=np.sqrt(R))
    
    # --- compute outer credible interval (without measurement) ---
    gmm_prior = make_gmm_pdf_1d(weights, means, covs)
    q_left, q_right, x_grid, F_upper, F_lower = outer_credible_region_gmm_1d(
        weights, means, covs, B1, conf_level=conf_level
    )
    x_min = x_grid[0]
    x_max = x_grid[-1]

    # --- compute outer credible interval (with measurement),
    # the prior is a single PDF defined by gmm(weights, covs, means) ---
    weights_pos_nom, means_pos_nom, covs_pos_nom = data_assimilation_gmm_1d_prior(y, obs_model, weights, means, covs)
    gmm_cdf = _gmm_cdf_1d_for_test(x_grid, weights_pos_nom, means_pos_nom, covs_pos_nom)
    gmm_cdf = np.maximum.accumulate(gmm_cdf)
    gmm_cdf = np.clip(gmm_cdf, 0.0, 1.0)
    alpha = 1. - conf_level
    p_left = alpha
    p_right = 1.0 - alpha
    q_left_pos_nom = float(np.interp(p_left, gmm_cdf, x_grid))
    q_right_pos_nom = float(np.interp(p_right, gmm_cdf, x_grid))
    gmm_pos_nom = make_gmm_pdf_1d(weights_pos_nom, means_pos_nom, covs_pos_nom)

    # --- compute outer credible interval (with measurement) using the single pos PDF gmm(weights_pos_nom, means_pos_nom, cov_pos_nom) above ---
    def l_func(x):
        x = np.asarray(x, dtype=float)
        g = gmm_prior(x)
        l_prior = np.clip(g - B1, 0.0, None)
        # like_l = like_func(y, x, obs_model)
        # return l_prior * like_l
        ratio = np.zeros_like(g)
        mask = g > 0
        ratio[mask] = l_prior[mask] / g[mask]
        return ratio * gmm_pos_nom(x)

    def u_func(x):
        x = np.asarray(x, dtype=float)
        g = gmm_prior(x)
        u_prior = g + B1
        # like_l = like_func(y, x, obs_model)
        # return u_prior * like_l
        # ratio = np.zeros_like(g)
        # mask = g > 0
        # ratio[mask] = u_prior[mask] / g[mask]
        ratio = u_prior / g
        return ratio * gmm_pos_nom(x)
    
    q_left_pos, q_right_pos, _, _, _ = outer_credible_region_1d_from_bounds(l_func, u_func, conf_level=conf_level, x_min=x_min, x_max=x_max)
    print(q_left_pos, q_right_pos)

    if show_plot:
        # reconstruct GMM pdf and band for plotting
        x = x_grid
        # x_col = x[None, :]
        gmm_pdf_values = gmm_prior(x_grid)
        gmm_pdf_values_pos_nom = gmm_pos_nom(x_grid)

        l = np.clip(gmm_pdf_values - B1, 0.0, None)
        u = gmm_pdf_values + B1

        l_pos = l_func(x_grid)
        u_pos = u_func(x_grid)

        set_publication_plot_style()
        fig, axes = plt.subplots(1, 1, sharex=True)
        ax0 = axes

        # --- PDF band plot ---
        ax0.plot(x, gmm_pdf_values, color="blue", linestyle="-", label="Pri Ref")
        ax0.fill_between(x, u, l, color="blue", alpha=0.2, label="Pri Set")

        # Outer credible region (band-based)
        ax0.axvspan(q_left, q_right, alpha=0.15,
                    label="95% region (pri-set)")
        
        ax0.axvline(
            x=x_true,
            color="black",
            linestyle="-",
            lw=2,
            label="true state"
        )
        ax0.axvline(
            x=y,
            color="black",
            linestyle="--",
            lw=1.5,
            label="observation"
        )
        
        ax0.plot(x, gmm_pdf_values_pos_nom, color="red", label="Pos Ref")
        ax0.axvspan(q_left_pos_nom, q_right_pos_nom, alpha=0.2,
                    color="red",
                    label="Pos Ref - 95% region")
        
        # Outer credible region (band-based)
        ax0.fill_between(x, u_pos, l_pos, color="green", alpha=0.5, label="Pos Set")
        ax0.axvspan(q_left_pos, q_right_pos, alpha=0.3,
                    color="green",
                    label="Pos Set - 95% region")
        
        ax0.set_ylim([0, 1])
        ax0.set_ylabel("PDF")
        ax0.set_xlabel("x")
        # ax0.set_title("PDF band, outer credible region, and GMM nominal interval")
        ax0.legend(loc="upper left", ncol=1)
        custom_save_plot(True, "figs/1d_credible_region_xtrue{:.1f}.pdf".format(x_true))
        plt.show()


def test_outer_credible_region_gmm_1d_with_measurement_mc(
    conf_level: float = 0.95,
    num_trials: int = 2000,
    show_plot: bool = False,
    x_true_method: int = 0,
    show_size_plot: bool = True,   # NEW: boxplot for sizes
):
    """
    Monte Carlo test for outer_credible_region_gmm_1d_with_measurement.

    Computes 4 credible regions each trial:
      1) prior set-based band interval
      2) prior nominal GMM interval
      3) posterior set-based band interval
      4) posterior nominal GMM interval

    Prints coverage frequencies and size stats (mean ± std).
    Also draws a boxplot of interval sizes if show_size_plot=True.
    """

    rng = np.random.default_rng(12345)

    # --- same GMM prior as in your original test ---
    weights = np.array([0.4, 0.6])
    means = np.array([[0.0],
                      [3.0]])
    covs = np.array([[[0.5 ** 2]],
                     [[0.8 ** 2]]])
    B1 = 0.03
    H = 1.0
    R = 0.5  # variance
    obs_model = (H, R)

    # --- PRIOR credible interval (set-based prior band) ---
    q_left_pri_set, q_right_pri_set, x_grid, F_upper, F_lower = outer_credible_region_gmm_1d(
        weights, means, covs, B1, conf_level=conf_level
    )
    x_min = x_grid[0]
    x_max = x_grid[-1]

    # --- PRIOR credible interval (single nominal GMM prior) ---
    gmm_cdf = _gmm_cdf_1d_for_test(x_grid, weights, means, covs)
    gmm_cdf = np.maximum.accumulate(gmm_cdf)
    gmm_cdf = np.clip(gmm_cdf, 0.0, 1.0)
    tail = 0.5 * (1.0 - conf_level)
    p_left = tail
    p_right = 1.0 - tail
    q_left = float(np.interp(p_left, gmm_cdf, x_grid))
    q_right = float(np.interp(p_right, gmm_cdf, x_grid))

    # Precompute for sampling x_true
    K = weights.shape[0]
    stds = np.sqrt(covs[:, 0, 0])

    # Counters
    count_pri_set = 0
    count_pri_nom = 0
    count_pos_set = 0
    count_pos_nom = 0

    # --- NEW: store interval sizes ---
    size_pri_set = np.zeros(num_trials)
    size_pri_nom = np.zeros(num_trials)
    size_pos_set = np.zeros(num_trials)
    size_pos_nom = np.zeros(num_trials)

    # Pre-sampled x_true candidates (depending on x_true_method)
    X_TRUE_SAMPLES, W_X_TRUE_SAMPLES = sample_density_ratio_class_gmm_1d(
        weights, means, covs, B1, num_trials, x_min=x_min, x_max=x_max
    )

    # Track the "most different" trial (posterior set vs nominal)
    best_diff = -np.inf
    best_x_true = None
    best_y = None
    best_q_left_pos = None
    best_q_right_pos = None
    best_q_left_pos_nom = None
    best_q_right_pos_nom = None

    for idx in range(num_trials):
        # --- 1) Sample x_true ---
        k = rng.choice(K, p=weights)

        if x_true_method == 0:
            x_true = rng.normal(loc=means[k, 0], scale=stds[k])
        elif x_true_method == 1:
            x_true = X_TRUE_SAMPLES[idx]
        elif x_true_method == 2:
            idx_x_true = rng.choice(len(X_TRUE_SAMPLES), p=W_X_TRUE_SAMPLES)
            x_true = X_TRUE_SAMPLES[idx_x_true]
        else:
            raise RuntimeError("Error x_true sampling method not implemented")

        # --- 2) Sample measurement y ---
        y = x_true + rng.normal(loc=0.0, scale=np.sqrt(R))

        # --- 3) Posterior nominal credible region ---
        weights_pos_nom, means_pos_nom, covs_pos_nom = data_assimilation_gmm_1d_prior(
            y, obs_model, weights, means, covs
        )
        gmm_cdf_pos = _gmm_cdf_1d_for_test(x_grid, weights_pos_nom, means_pos_nom, covs_pos_nom)
        gmm_cdf_pos = np.maximum.accumulate(gmm_cdf_pos)
        gmm_cdf_pos = np.clip(gmm_cdf_pos, 0.0, 1.0)
        q_left_pos_nom = float(np.interp(p_left, gmm_cdf_pos, x_grid))
        q_right_pos_nom = float(np.interp(p_right, gmm_cdf_pos, x_grid))

        # --- 4) Posterior set-based outer credible region ---
        gmm_prior = make_gmm_pdf_1d(weights, means, covs)
        gmm_pos_nom = make_gmm_pdf_1d(weights_pos_nom, means_pos_nom, covs_pos_nom)

        def l_func(x):
            x = np.asarray(x, dtype=float)
            g = gmm_prior(x)
            l_prior = np.clip(g - B1, 0.0, None)
            ratio = np.zeros_like(g)
            mask = g > 0
            ratio[mask] = l_prior[mask] / g[mask]
            return ratio * gmm_pos_nom(x)

        def u_func(x):
            x = np.asarray(x, dtype=float)
            g = gmm_prior(x)
            u_prior = g + B1
            ratio = np.zeros_like(g)
            mask = g > 0
            ratio[mask] = u_prior[mask] / g[mask]
            return ratio * gmm_pos_nom(x)

        q_left_pos, q_right_pos, _, _, _ = outer_credible_region_1d_from_bounds(
            l_func, u_func, conf_level=conf_level, x_min=x_min, x_max=x_max
        )

        # --- 5) Coverage checks ---
        if q_left_pri_set <= x_true <= q_right_pri_set:
            count_pri_set += 1
        if q_left <= x_true <= q_right:
            count_pri_nom += 1
        if q_left_pos <= x_true <= q_right_pos:
            count_pos_set += 1
        if q_left_pos_nom <= x_true <= q_right_pos_nom:
            count_pos_nom += 1

        # --- NEW: record sizes ---
        size_pri_set[idx] = q_right_pri_set - q_left_pri_set
        size_pri_nom[idx] = q_right - q_left
        size_pos_set[idx] = q_right_pos - q_left_pos
        size_pos_nom[idx] = q_right_pos_nom - q_left_pos_nom

        # --- 6) Track most different posterior trial ---
        diff_this = max(
            abs(q_left_pos_nom - q_left_pos),
            abs(q_right_pos_nom - q_right_pos),
        )
        if diff_this > best_diff:
            best_diff = diff_this
            best_x_true = x_true
            best_y = y
            best_q_left_pos = q_left_pos
            best_q_right_pos = q_right_pos
            best_q_left_pos_nom = q_left_pos_nom
            best_q_right_pos_nom = q_right_pos_nom

    # --- 7) Empirical coverage frequencies ---
    freq_pri_set = count_pri_set / num_trials
    freq_pri_nom = count_pri_nom / num_trials
    freq_pos_set = count_pos_set / num_trials
    freq_pos_nom = count_pos_nom / num_trials

    # --- NEW: size stats ---
    def mean_std(a):
        return float(np.mean(a)), float(np.std(a, ddof=1))

    m_pri_set, s_pri_set = mean_std(size_pri_set)
    m_pri_nom, s_pri_nom = mean_std(size_pri_nom)
    m_pos_set, s_pos_set = mean_std(size_pos_set)
    m_pos_nom, s_pos_nom = mean_std(size_pos_nom)

    print(f"num_trials                 : {num_trials}")
    print(f"conf_level                 : {conf_level:.3f}")
    print(f"Coverage prior     (set)   : {freq_pri_set:.3f}")
    print(f"Coverage prior     (ref)   : {freq_pri_nom:.3f}")
    print(f"Coverage posterior (set)   : {freq_pos_set:.3f}")
    print(f"Coverage posterior (ref)   : {freq_pos_nom:.3f}")
    print(f"Max endpoint diff (post set vs nom): {best_diff:.4f}")

    print("\nInterval size stats (mean ± 1 std):")
    print(f"  Prior set-based     : {m_pri_set:.4f} ± {s_pri_set:.4f}")
    print(f"  Prior nominal GMM   : {m_pri_nom:.4f} ± {s_pri_nom:.4f}")
    print(f"  Posterior set-based : {m_pos_set:.4f} ± {s_pos_set:.4f}")
    print(f"  Posterior nominal   : {m_pos_nom:.4f} ± {s_pos_nom:.4f}")

    # --- NEW: boxplot for size distributions ---
    if show_size_plot:
        set_publication_plot_style()
        plt.figure()
        plt.boxplot(
            [size_pri_nom, size_pri_set, size_pos_nom, size_pos_set],
            tick_labels=["Pri Ref", "Pri Set", "Pos Ref", "Pos Set"],
            showmeans=True
        )
        plt.ylabel("Interval size (width)")
        custom_save_plot(True, "figs/1d_credible_region_mc_boxplot.pdf")


    # Optional "most different" plot (your existing logic)
    if show_plot:
        x = x_grid
        x_col = x[None, :]
        vars_ = covs[:, 0, 0]
        vars_col = vars_[:, None]

        norm_consts = 1.0 / np.sqrt(2.0 * np.pi * vars_col)
        exponents = -0.5 * ((x_col - means) ** 2 / vars_col)
        comp_pdfs = norm_consts * np.exp(exponents)
        gmm_pdf_values = np.dot(weights, comp_pdfs)

        l = np.clip(gmm_pdf_values - B1, 0.0, None)
        u = gmm_pdf_values + B1

        set_publication_plot_style()
        fig, ax0 = plt.subplots(1, 1, sharex=True)

        ax0.plot(x, gmm_pdf_values, linestyle="-", label="Pri Ref")
        ax0.fill_between(x, u, l, color="gray", alpha=0.5, label="Pri Set")

        # ax0.axvspan(q_left, q_right, color="gray", alpha=0.15, label="95% region (prior)")
        ax0.axvspan(best_q_left_pos, best_q_right_pos, alpha=0.3, color="green",
                    label="Pos Ref - 95% region")
        ax0.axvspan(best_q_left_pos_nom, best_q_right_pos_nom, alpha=0.2, color="red",
                    label="Pos Set - 95% region")

        ax0.axvline(best_x_true, color="black", linestyle="-", lw=2,
                    label="true state")
        ax0.axvline(best_y, color="black", linestyle="--", lw=1.5,
                    label="observation")

        ax0.set_ylabel("PDF")
        # ax0.set_title(
        #     # f"Coverage: Pri Ref={freq_pri_nom:.3f}, Pri Set={freq_pri_set:.3f}, "
        #     f"Pos Ref={freq_pos_nom:.3f}, Pos Set={freq_pos_set:.3f}\n"
        #     "PDF band & credible regions (most-different trial)"
        # )
        ax0.legend(loc="upper left")
        custom_save_plot(True, "figs/1d_credible_region_mc_singletrial.pdf")


    if(show_plot):
        set_publication_plot_style()
        plt.figure()

        # x = avg size, y = coverage
        sizes_pri_ref = m_pri_nom
        sizes_pri_set = m_pri_set
        sizes_pos_ref = m_pos_nom
        sizes_pos_set = m_pos_set

        cov_pri_ref = freq_pri_nom
        cov_pri_set = freq_pri_set
        cov_pos_ref = freq_pos_nom
        cov_pos_set = freq_pos_set

        # plot each method separately so we can control color/marker
        plt.scatter(sizes_pri_ref, cov_pri_ref, s=70, c="black", marker="o", label="Pri Ref")
        plt.scatter(sizes_pri_set, cov_pri_set, s=70, c="blue",  marker="s", label="Pri Set")
        plt.scatter(sizes_pos_ref, cov_pos_ref, s=70, c="red",   marker="^", label="Pos Ref")
        plt.scatter(sizes_pos_set, cov_pos_set, s=70, c="green", marker="D", label="Pos Set")

        # horizontal line at desired confidence level
        plt.axhline(conf_level, linestyle="--", color="black", label=f"{conf_level:.2f} target")
        plt.legend(loc="best")
        plt.xlabel("Average credible region size")
        plt.ylabel("Coverage freq.")
        custom_save_plot(True, "figs/1d_credible_region_mc_summary.pdf")

    if(show_plot):
        plt.show()

    return (freq_pri_set, freq_pri_nom, freq_pos_set, freq_pos_nom,
            size_pri_set, size_pri_nom, size_pos_set, size_pos_nom)



if __name__ == "__main__":
    # test_outer_credible_region_gmm_1d(show_plot=True)

    test_outer_credible_region_gmm_1d_with_measurement(show_plot=True)
    
    # test_outer_credible_region_gmm_1d_with_measurement_mc(
    #     conf_level=0.95, num_trials=1000, show_plot=True, x_true_method=1)
