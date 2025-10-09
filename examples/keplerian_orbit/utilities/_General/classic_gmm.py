import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
import inspect
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp


# --------- small helpers ----------
def _whiten(X: NDArray[np.float64], mu: NDArray[np.float64], std: NDArray[np.float64]) -> NDArray[np.float64]:
    return (X - mu[None, :]) / std[None, :]


def _logsumexp(a: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class _Params:
    # stored in **z-space**
    weights: NDArray[np.float64]        # (K,)
    means_z: NDArray[np.float64]        # (K, D)
    covs_z: NDArray[np.float64]         # (K, D, D)   (full)
    # whitening stats for x<->z
    mu_x: NDArray[np.float64]           # (D,)
    std_x: NDArray[np.float64]          # (D,)


class GMMWhitenedModel:
    """
    Fit a GMM in whitened z-space, save portable params, load, and evaluate pdf(x) in physical space.

    x -> z = (x - mu_x) / std_x  (elementwise)
    p_x(x) = p_z(z) / prod(std_x)
    """
    def __init__(self,
                 K_list: Iterable[int] = (1, 2, 4, 8, 16),
                 reg_covar: float = 1e-6,
                 tol: float = 1e-5,
                 max_iter: int = 2000,
                 n_init: int = 10,
                 random_state: Optional[int] = 0):
        self.K_list = tuple(K_list)
        self.reg_covar = float(reg_covar)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = random_state
        self.params: Optional[_Params] = None
        self._inv_det_std: Optional[float] = None  # 1/prod(std_x)

    # ---------- training ----------
    def fit(self, X: NDArray[np.float64], mu_x: NDArray[np.float64], std_x: NDArray[np.float64],
            val_frac: float = 0.2) -> "GMMWhitenedModel":
        """
        Fit GMM on Z = whiten(X). Choose K by held-out log-likelihood with n_init restarts.
        """
        X = np.asarray(X, dtype=np.float64)
        mu_x = np.asarray(mu_x, dtype=np.float64)
        std_x = np.asarray(std_x, dtype=np.float64)
        Z = _whiten(X, mu_x, std_x)

        # split train/val
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(Z))
        n_val = int(round(len(Z) * val_frac))
        Z_tr, Z_val = Z[idx[n_val:]], Z[idx[:n_val]]

        # 2) train on a subset of data but evaluate over the entire data
        # Z_tr = Z[idx[n_val:]]
        # Z_val = Z

        rng_local = np.random.default_rng(self.random_state)
        best, best_score = None, -np.inf
        for K in self.K_list:
            # print(K)
            best_k, best_k_score = None, -np.inf
            for _ in range(self.n_init):
                seed = rng_local.integers(np.iinfo(np.int32).max)
                g = GaussianMixture(
                    n_components=K,
                    covariance_type="full",
                    reg_covar=self.reg_covar,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    init_params="kmeans",
                    random_state=seed,      # ← different each time
                    verbose=0,
                ).fit(Z_tr)
                s = g.score(Z_val).mean()
                # print(seed, s)
                if s > best_k_score:
                    best_k, best_k_score = g, s
            if best_k_score > best_score:
                best, best_score = best_k, best_k_score

        # stash params (in z-space) + whitening stats
        covs = best.covariances_
        self.params = _Params(
            weights=best.weights_.astype(np.float64),
            means_z=best.means_.astype(np.float64),
            covs_z=covs.astype(np.float64),
            mu_x=mu_x.astype(np.float64),
            std_x=std_x.astype(np.float64),
        )
        self._inv_det_std = float(1.0 / np.prod(std_x))
        return self

    # ---------- evaluation ----------
    def pdf(self, X: NDArray[np.float64], batch: int = 200_000) -> NDArray[np.float64]:
        """
        Evaluate the fitted pdf at physical-space points X. Returns shape (N,).
        """
        assert self.params is not None, "Model not fitted/loaded."
        p = self.params
        inv_det = self._inv_det_std if self._inv_det_std is not None else float(1.0 / np.prod(p.std_x))

        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)

        # Precompute Cholesky + logdet for each component in z
        K, D = p.means_z.shape
        Ls = np.empty_like(p.covs_z)
        logdets = np.empty(K, dtype=np.float64)
        for k in range(K):
            L = np.linalg.cholesky(p.covs_z[k])    # cov = L L^T
            Ls[k] = L
            logdets[k] = 2.0 * np.sum(np.log(np.diag(L)))

        const = -0.5 * (D * np.log(2.0 * np.pi))

        for i in range(0, X.shape[0], batch):
            j = min(i + batch, X.shape[0])
            Z = _whiten(X[i:j], p.mu_x, p.std_x)    # (B,D)
            # log N_k(z) for all k, vectorized over batch
            # quad_k = || L_k^{-1} (z - m_k) ||^2
            B = Z.shape[0]
            log_comp = np.empty((B, K), dtype=np.float64)
            for k in range(K):
                Y = (Z - p.means_z[k])            # (B,D)
                alpha = np.linalg.solve(Ls[k], Y.T)  # (D,B)
                quad = np.sum(alpha * alpha, axis=0) # (B,)
                log_comp[:, k] = const - 0.5 * (logdets[k] + quad)
            # log p_z(z) = logsumexp(log w_k + log N_k)
            lpz = _logsumexp(np.log(p.weights)[None, :] + log_comp, axis=1)  # (B,)
            out[i:j] = np.exp(lpz) * inv_det
        return out

    # ---------- sampling ----------
    def sample(self, N: int, rng: Optional[np.random.Generator] = None) -> NDArray[np.float64]:
        """
        Draw N samples in physical x-space from the fitted mixture.
        """
        assert self.params is not None, "Model not fitted/loaded."
        p = self.params
        if rng is None:
            rng = np.random.default_rng(0)
        K = len(p.weights)
        D = p.means_z.shape[1]

        # choose components
        ks = rng.choice(K, size=N, p=p.weights)
        X = np.empty((N, D), dtype=np.float64)
        for k in range(K):
            idx = np.where(ks == k)[0]
            if len(idx) == 0: continue
            L = np.linalg.cholesky(p.covs_z[k])
            z = p.means_z[k] + (L @ rng.normal(size=(D, len(idx)))).T
            X[idx] = z
        # unwhiten to x-space
        return X * p.std_x[None, :] + p.mu_x[None, :]

    # ---------- utilities: z->x parameter transform ----------
    def _x_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (weights, means_x, covs_x) where
            means_x[k] = mu_x + std_x * means_z[k]
            covs_x[k]  = diag(std_x) @ covs_z[k] @ diag(std_x)
        Shapes:
            weights: (K,)
            means_x: (K, D)
            covs_x : (K, D, D)
        """
        assert self.params is not None, "Model not fitted/loaded."
        p = self.params
        # means: elementwise scale + shift
        means_x = p.means_z * p.std_x[None, :] + p.mu_x[None, :]
        # covariances: A Σ_z A^T with A = diag(std_x)
        covs_x = np.einsum('kij,i,j->kij', p.covs_z, p.std_x, p.std_x)
        return p.weights.copy(), means_x, covs_x

    def print_x_params(self, decimals: int = 6, return_arrays: bool = True):
        """
        Print GMM weights and x-space means/covariances.
        Args:
            decimals: number of decimals to show.
            return_arrays: if True, also return a dict with arrays.
        """
        w, m_x, C_x = self._x_params()
        K, D = m_x.shape

        # # Pretty printing
        # old_opts = np.get_printoptions()
        # try:
        #     np.set_printoptions(precision=decimals, suppress=True, linewidth=120)
        #     print(f"[GMM x-space params] K={K}, D={D}")
        #     print("weights:", w)
        #     print("means_x:\n", m_x)
        #     print("covs_x:\n", C_x)
        # finally:
        #     np.set_printoptions(**old_opts)

        if return_arrays:
            return {"weights": w, "means_x": m_x, "covs_x": C_x}

    # ---------- persistence ----------
    def save(self, path: str):
        """
        Save to a single .npz (portable across sklearn versions).
        """
        assert self.params is not None, "Nothing to save; fit or load first."
        p = self.params
        np.savez(
            path,
            version=np.array([1], dtype=np.int64),
            weights=p.weights,
            means_z=p.means_z,
            covs_z=p.covs_z,
            mu_x=p.mu_x,
            std_x=p.std_x,
            reg_covar=np.array([self.reg_covar]),
            tol=np.array([self.tol]),
            max_iter=np.array([self.max_iter]),
            n_init=np.array([self.n_init]),
            random_state=np.array([-1 if self.random_state is None else self.random_state])
        )

    @classmethod
    def load(cls, path: str) -> "GMMWhitenedModel":
        data = np.load(path, allow_pickle=False)
        model = cls()
        model.params = _Params(
            weights=data["weights"].astype(np.float64),
            means_z=data["means_z"].astype(np.float64),
            covs_z=data["covs_z"].astype(np.float64),
            mu_x=data["mu_x"].astype(np.float64),
            std_x=data["std_x"].astype(np.float64),
        )
        model._inv_det_std = float(1.0 / np.prod(model.params.std_x))
        # (optional) restore trainer attrs if present
        for key in ("reg_covar", "tol", "max_iter", "n_init", "random_state"):
            if key in data.files:
                setattr(model, key, (int(data[key][0]) if key in ("max_iter", "n_init") else float(data[key][0]) if key in ("reg_covar", "tol") else int(data[key][0])))
        return model


def fit_classic_gmm(t_prime, X_tr, mu_whiten, cov_whiten, data_folder, K_list=(1, 2, 4, 8, 16)):
    # Whiten statistics
    mu  = np.asarray(mu_whiten, dtype=np.float64)
    std = np.sqrt(np.asarray(np.diag(cov_whiten), dtype=np.float64))

    # Draw training data from the push-forward in x
    n_train = len(X_tr)
    print("data size: ", n_train)

    # 1) Fit
    model = GMMWhitenedModel(K_list=K_list, n_init=3, tol=1e-5, max_iter=2000, reg_covar=1e-6, random_state=0)
    model.fit(X_tr, mu_x=mu, std_x=std)

    # 2) Save to disk
    model.save(data_folder+"gmm_whitened_t{:.2f}.npz".format(t_prime))


def make_gmm_pdf(weights, means, covs):
    """
    Return pdf(X, return_log=False) for a (possibly single-component) GMM.

    Inputs (float64 coerced):
      weights: (K,) or scalar
      means:   (K, D) or (D,)
      covs:    (K, D, D) or (D, D)

    The PDF is evaluated in correlation space:
      Σ = S R S,  where S = diag(std),  R = S^{-1} Σ S^{-1}.
      We evaluate N((X-μ)/S | 0, R) and back-transform by prod(S).
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    M = np.asarray(means,   dtype=np.float64)
    C = np.asarray(covs,    dtype=np.float64)

    # Coerce single-component to (K,D)/(K,D,D)
    if M.ndim == 1:
        M = M[None, :]
    if C.ndim == 2:
        C = C[None, :, :]

    K, D = M.shape
    if w.size == 1 and K > 1:
        w = np.full(K, w.item(), dtype=np.float64)
    if w.size != K:
        raise ValueError("weights must have length K to match means/covs.")

    # Validate cov shapes
    if C.shape != (K, D, D):
        raise ValueError("covs must have shape (K, D, D).")

    # Normalize weights
    s = w.sum()
    if s <= 0:
        raise ValueError("Sum of weights must be positive.")
    log_w = np.log(w / s)

    # Precompute per-component scaling and frozen RVs in correlation space
    tiny = np.finfo(np.float64).tiny
    scales = np.empty((K, D), dtype=np.float64)
    rvs = []
    log_scale_prod = np.empty(K, dtype=np.float64)

    for k in range(K):
        # Symmetrize defensively
        P = 0.5 * (C[k] + C[k].T)

        # Per-dim std with floor to avoid division by zero
        s_k = np.sqrt(np.clip(np.diag(P), tiny, None))
        scales[k] = s_k
        log_scale_prod[k] = np.log(s_k).sum()

        # Correlation-like matrix
        R = P / np.outer(s_k, s_k)

        # Build frozen Gaussian in scaled space (allow singular PSD)
        rvs.append(
            multivariate_normal(mean=np.zeros(D, dtype=np.float64),
                                cov=R, allow_singular=True)
        )

    def pdf(X, return_log=False):
        X = np.asarray(X, dtype=np.float64)

        # Accept (N,) when D==1; otherwise (N,D) or (D,) -> (1,D)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if D == 1 else X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        if X.shape[1] != D:
            raise ValueError(f"X must have shape (N, {D}).")

        # log p_k(X) = log N( (X-μ_k)/s_k | 0, R_k ) - log(prod s_k)
        logs = []
        for k in range(K):
            Z = (X - M[k]) / scales[k]                   # (N, D)
            logs.append(log_w[k] + rvs[k].logpdf(Z) - log_scale_prod[k])
        log_mix = logsumexp(np.vstack(logs), axis=0)     # (N,)
        return log_mix if return_log else np.exp(log_mix)

    return pdf


def plot_1d_true_vs_gmm_marginals_model(model,
                                        X_true,
                                        dims=None,
                                        bins="fd",
                                        qrange=(0.001, 0.999),
                                        figsize=(12, 6),
                                        save_path=None,
                                        title="True vs GMM (1D marginals)"):
    """
    Overlay histograms of true samples with analytical GMM 1D marginals (in x-space).

    model  : GMMWhitenedModel (must have model.params with weights, means_z, covs_z, mu_x, std_x)
    X_true : (N, D) true samples in x-space
    dims   : list of dims to plot (default: all)
    bins   : 'fd' | 'scott' | int
    qrange : robust plotting range from true-sample quantiles
    """
    assert hasattr(model, "params") and model.params is not None, "Model must be fitted/loaded."
    p = model.params

    X_true = np.asarray(X_true)
    N, D = X_true.shape
    if dims is None:
        dims = list(range(D))

    w     = p.weights              # (K,)
    m_z   = p.means_z              # (K,D)
    covs  = p.covs_z               # (K,D,D)
    mu_x  = p.mu_x                 # (D,)
    std_x = p.std_x                # (D,)

    # per-dim std in z (for analytic 1D marginals)
    var_z = np.stack([np.diag(C) for C in covs], axis=0)  # (K,D)
    sd_z  = np.sqrt(var_z + 0.0)                          # (K,D)

    ncols = min(3, len(dims))
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for idx, j in enumerate(dims):
        ax = axes[idx]
        # robust range from true data
        lo, hi = np.quantile(X_true[:, j], qrange[0]), np.quantile(X_true[:, j], qrange[1])
        xs = np.linspace(lo, hi, 600)

        # analytical 1D marginal in x:
        # z_j = (x_j - mu_x[j]) / std_x[j]
        # p_xj(x) = (1/std_x[j]) * Σ_k w_k * N(z_j | m_z[k,j], var_z[k,j])
        zj  = (xs - mu_x[j]) / std_x[j]
        mk  = m_z[:, j]          # (K,)
        sdk = sd_z[:, j]         # (K,)
        phi = (np.exp(-0.5 * ((zj[None, :] - mk[:, None]) / sdk[:, None])**2) /
               (sdk[:, None] * np.sqrt(2.0 * np.pi)))                 # (K, M)
        p_xj = (w[:, None] * phi).sum(axis=0) / std_x[j]              # (M,)

        ax.hist(X_true[:, j], bins=bins, range=(lo, hi), density=True, alpha=0.45, label="True")
        ax.plot(xs, p_xj, lw=2, label="GMM marginal")
        ax.set_title(f"x{j}")
        if idx == 0:
            ax.legend(frameon=False)

    # hide empty axes
    for k in range(len(dims), len(axes)):
        axes[k].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
