import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repo_root
sys.path.insert(0, str(ROOT))
from utilities._General.classic_gmm import GMMWhitenedModel, fit_classic_gmm, plot_1d_true_vs_gmm_marginals_model
from utilities._General.util import build_discretized_cells
from main import t1s, x_low, x_hig


SAMPLE_USED = 1000000


def generate_true_pdf_by_fitting_gmm():
    T_monte = np.round(t1s, 2)
    mu_whiten = np.array([0.])
    cov_whiten = np.array([[1.]])
    for t in T_monte:
        Xsamples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32).reshape((-1,1))
        print(Xsamples.shape)
        Xsamples = Xsamples[0:SAMPLE_USED, :]
        fit_classic_gmm(t, Xsamples, mu_whiten, cov_whiten, "data/", K_list=(1,2))


def show_fitted_gmm():
    T_monte = np.round(t1s, 2)
    T_show = [T_monte[0], T_monte[-1]]
    for t in T_show:
        Xsamples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32).reshape((-1,1))
        print(Xsamples.shape)
        Xsamples = Xsamples[0:SAMPLE_USED, :]
        model = GMMWhitenedModel.load("data/gmm_whitened_t{:.2f}.npz".format(t))
        title="True vs GMM (1D marginals) at t:{:.2f}".format(t)
        plot_1d_true_vs_gmm_marginals_model(model, Xsamples, title=title)
    plt.show()


def histogram_pdf_from_midpoints(mids, w, Xsamples, *, clip_outside=False):
    """
    Build a 1D histogram PDF over cells centered at `mids` with constant width `w`.

    Parameters
    ----------
    mids : array_like, shape (N,1) or (N,)
        Cell midpoints (assumed uniformly spaced with step ~= w).
    w : float
        Constant cell width.
    Xsamples : array_like, shape (M,1) or (M,)
        Drawn samples of the random variable X.
    clip_outside : bool, default False
        If False: samples outside the covered range are ignored (do not count).
        If True : samples outside are clipped into the end bins.

    Returns
    -------
    pdf : ndarray, shape (N,)
        Estimated PDF values per cell (so that pdf.sum() * w ≈ 1 if no samples are ignored).
    counts : ndarray, shape (N,)
        Raw counts in each cell.
    edges : ndarray, shape (N+1,)
        Bin edges used (half-open bins [edges[i], edges[i+1])).
    """
    mids = np.asarray(mids, dtype=float).reshape(-1)
    X = np.asarray(Xsamples, dtype=float).reshape(-1)
    N = mids.size

    # Construct edges that align with the midpoints for constant width w
    start = mids[0] - 0.5 * w
    edges = start + w * np.arange(N + 1)

    if clip_outside:
        # Clip samples into the covered range, then bin via digitize
        Xc = np.clip(X, edges[0], np.nextafter(edges[-1], -np.inf))
        # indices in [0..N-1]
        idx = np.digitize(Xc, edges, right=False) - 1
        counts = np.bincount(idx, minlength=N)
    else:
        # Ignore out-of-range samples using numpy.histogram
        counts, _ = np.histogram(X, bins=edges)

    pdf = counts.astype(float) / (X.size * w)
    return pdf, counts, edges


def generate_true_pdf_by_historgram():
    # Setup problem
    problem = {
        "D": 1,
        "N_x": 512,
        "X_dom": np.array([[x_low, x_hig]]),
        "X_event": np.array([[-2. , 1.]]),
    }
    problem["midpts"], problem["widths"], problem["mask"], problem["dV"] = build_discretized_cells(problem)
    np.save("data/midpts_512.npy", problem["midpts"])

    T_monte = np.round(t1s, 2)
    for t in T_monte:
        Xsamples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32).reshape((-1, 1))
        pdf, counts, _ = histogram_pdf_from_midpoints(problem["midpts"], problem["dV"], Xsamples)
        # plt.figure()
        # plt.plot(problem["midpts"], pdf)
        # plt.show()
        np.save("data/p_512_t{:.1f}.npy".format(t), pdf)


def main():
    # generate_true_pdf_by_fitting_gmm()
    # show_fitted_gmm()
    generate_true_pdf_by_historgram()


if __name__ == "__main__":
    main()