import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

@torch.no_grad()
def _normalize_pdf_on_grid(x_np, p_np):
    """Normalize p(x) on a grid using the trapezoidal rule."""
    # guard against negatives / NaNs
    p = np.clip(np.nan_to_num(p_np, nan=0.0, posinf=0.0, neginf=0.0), a_min=0.0, a_max=None)
    Z = np.trapz(p, x_np)
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError("p_init returned a non-positive or non-finite integral on the grid.")
    return p / Z

def _gmm_pdf(x, weights, means, sigmas):
    """Evaluate 1D GMM pdf at x (torch tensors). x: (M,), returns (M,)."""
    # x: (M,), means: (K,), sigmas: (K,), weights: (K,)
    # Compute all component densities at once
    xM = x.unsqueeze(-1)                  # (M,1)
    mK = means.unsqueeze(0)               # (1,K)
    sK = sigmas.unsqueeze(0)              # (1,K)
    # Gaussian pdf
    norm = 1.0 / (torch.sqrt(torch.tensor(2.0*np.pi, dtype=x.dtype, device=x.device)) * sK)
    exp_term = torch.exp(-0.5 * ((xM - mK) / sK)**2)
    comp = norm * exp_term                # (M,K)
    return (comp * weights.unsqueeze(0)).sum(dim=1)  # (M,)

def fit_gmm_to_p_init(
    p_init,
    n_components: int,
    x_range: tuple,
    n_grid: int = 4096,
    n_steps: int = 3000,
    lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    min_sigma: float = 1e-3,
    progress: bool = True,
    # --- new options ---
    fix_means: str | None = None,     # None | "uniform"
    fixed_means=None,                 # np.ndarray or torch.Tensor of shape (K,) if provided
    fixed_sigma: float | None = None  # if set, keep all sigmas equal to this value (no training)
):
    """
    If fix_means == 'uniform', component means are placed uniformly in x_range and *not* optimized.
    If fixed_means is provided (array-like of shape (K,)), those are used and *not* optimized.
    If fixed_sigma is provided (float), all sigmas are fixed to that value and *not* optimized.
    """
    rng = np.random.default_rng(seed)

    # ----- grid & weights -----
    x_min, x_max = float(x_range[0]), float(x_range[1])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("x_range must be finite with xmax > xmin.")
    x_np = np.linspace(x_min, x_max, num=n_grid, dtype=np.float64)
    p_raw = np.asarray(p_init(x_np), dtype=np.float64)
    p_np = p_raw  # assume p_init already normalized if you want strict CE; otherwise it just scales the loss

    dx = (x_max - x_min) / (n_grid - 1)
    w_np = np.ones_like(x_np) * dx
    w_np[0] *= 0.5
    w_np[-1] *= 0.5

    x = torch.tensor(x_np, device=device, dtype=dtype)
    p = torch.tensor(p_np, device=device, dtype=dtype)
    w = torch.tensor(w_np, device=device, dtype=dtype)

    # ----- means -----
    if fixed_means is not None:
        m0 = np.asarray(fixed_means, dtype=np.float64).reshape(-1)
        if m0.shape[0] != n_components:
            raise ValueError("fixed_means must have length n_components.")
        theta_m = torch.tensor(m0, device=device, dtype=dtype)   # not a Parameter
        train_means = False
    elif fix_means == "uniform":
        # put centers uniformly; inclusive ends gives nice coverage
        m0 = np.linspace(x_min, x_max, n_components, dtype=np.float64)
        theta_m = torch.tensor(m0, device=device, dtype=dtype)   # not a Parameter
        train_means = False
    else:
        # original stochastic init and train means
        probs = np.clip(p_np, 0, None)
        s = probs.sum()
        probs = probs / s if s > 0 else np.full_like(probs, 1.0 / len(probs))
        init_idx = rng.choice(n_grid, size=n_components, replace=False, p=probs) if n_components <= n_grid else rng.integers(0, n_grid, size=n_components)
        init_means = np.sort(x_np[init_idx])
        theta_m = nn.Parameter(torch.tensor(init_means, device=device, dtype=dtype))
        train_means = True

    # ----- sigmas -----
    if fixed_sigma is not None:
        if fixed_sigma <= 0:
            raise ValueError("fixed_sigma must be positive.")
        theta_s = torch.log(torch.tensor(fixed_sigma, device=device, dtype=dtype))  # scalar
        train_sigmas = False
    else:
        # heuristic: overlapping bells across the interval
        spread = 0.5 * (x_max - x_min) / max(n_components, 1)
        init_sigmas = np.clip(np.full(n_components, spread), min_sigma, None)
        theta_s = nn.Parameter(torch.log(torch.tensor(init_sigmas, device=device, dtype=dtype)))
        train_sigmas = True

    # ----- weights (always trained) -----
    theta_w = nn.Parameter(torch.zeros(n_components, device=device, dtype=dtype))

    # build optimizer with only trainable params
    params = [theta_w]
    if train_means:
        params.append(theta_m)
    if train_sigmas:
        params.append(theta_s)
    opt = torch.optim.Adam(params, lr=lr)

    for t in range(n_steps):
        opt.zero_grad()

        weights = torch.softmax(theta_w, dim=0)  # (K,)

        if train_sigmas:
            sigmas = torch.clamp(torch.exp(theta_s), min=min_sigma)  # (K,)
        else:
            # broadcast scalar or vector to (K,)
            if fixed_sigma is not None:
                sigmas = torch.full((n_components,), float(fixed_sigma), device=device, dtype=dtype)
            else:
                sigmas = torch.clamp(torch.exp(theta_s), min=min_sigma)

        means = theta_m if train_means else theta_m  # same tensor; not requires_grad if frozen

        q = _gmm_pdf(x, weights, means, sigmas)
        q = torch.clamp(q, min=1e-12)

        ce = torch.sum(p * (-torch.log(q)) * w)
        ce.backward()
        opt.step()

        if progress and (t % max(1, n_steps // 20) == 0 or t == n_steps - 1):
            print(f"[{t:5d}/{n_steps}] CE ≈ {ce.item():.6f}")

    with torch.no_grad():
        weights = torch.softmax(theta_w, dim=0).detach()
        means   = theta_m.detach() if isinstance(theta_m, torch.Tensor) else torch.tensor(theta_m, device=device, dtype=dtype)
        sigmas  = (torch.clamp(torch.exp(theta_s), min=min_sigma).detach()
                   if train_sigmas else torch.full((n_components,), float(fixed_sigma), device=device, dtype=dtype))

    def gmm_pdf_np(x_query):
        xt = torch.tensor(np.asarray(x_query), device=device, dtype=dtype).reshape(-1)
        with torch.no_grad():
            vals = _gmm_pdf(xt, weights, means, sigmas)
        return vals.cpu().numpy()

    return {"weights": weights, "means": means, "sigmas": sigmas, "pdf": gmm_pdf_np}


# ------------------------
# Visualization utilities
# ------------------------


def evaluate_on_grid(p_init, fit, x_range, n_grid=2000, device="cpu", dtype=torch.float32):
    """Return x grid, normalized p_init(x), fitted GMM q(x), and component curves."""
    x_min, x_max = map(float, x_range)
    x_np = np.linspace(x_min, x_max, n_grid, dtype=np.float64)

    # Normalize p_init on the plotting grid (trapezoidal)
    p_raw = np.asarray(p_init(x_np), dtype=np.float64)
    p_np = p_raw

    # Fitted mixture
    q_np = fit["pdf"](x_np)

    # Individual components (optional)
    with torch.no_grad():
        x_t = torch.tensor(x_np, device=device, dtype=dtype)
        w = fit["weights"].to(device=device, dtype=dtype)
        m = fit["means"].to(device=device, dtype=dtype)
        s = fit["sigmas"].to(device=device, dtype=dtype)

        xM = x_t.unsqueeze(-1)            # (M,1)
        mK = m.unsqueeze(0)               # (1,K)
        sK = s.unsqueeze(0)               # (1,K)
        norm = 1.0 / (torch.sqrt(torch.tensor(2.0*np.pi, dtype=dtype, device=device)) * sK)
        comp = norm * torch.exp(-0.5 * ((xM - mK)/sK)**2)  # (M,K)
        comps_np = (comp * w.unsqueeze(0)).cpu().numpy()   # scaled components, sum to q

    return x_np, p_np, q_np, comps_np

def plot_fit(
    p_init, fit, x_range, n_grid=2000,
    show_components=True, show_residual=True,
    title="GMM fit to p_init", save_path=None, device="cpu", dtype=torch.float32
):
    x, p, q, comps = evaluate_on_grid(p_init, fit, x_range, n_grid, device, dtype)

    # Basic metrics on the plotting grid
    dx = (x[-1] - x[0]) / (len(x) - 1)
    l1_err = np.sum(np.abs(p - q)) * dx
    ce = np.sum(p * (-np.log(np.clip(q, 1e-16, None)))) * dx  # cross-entropy ∫ p[-log q]

    if show_residual:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))

    # Top: densities
    ax_top = ax1
    ax_top.plot(x, p, lw=2.0, label="normalized $p_{\\mathrm{init}}(x)$")
    ax_top.plot(x, q, lw=2.0, ls="--", label="fitted GMM $q(x)$")
    if show_components:
        for k in range(comps.shape[1]):
            ax_top.plot(x, comps[:, k], lw=1.0, alpha=0.6)
    ax_top.set_ylabel("density")
    ax_top.set_title(f"{title}\nL1 ≈ {l1_err:.4e}   CE ≈ {ce:.4f}")
    ax_top.legend()
    ax_top.grid(True, alpha=0.25)

    # Bottom: residual
    if show_residual:
        ax_bot = ax2
        ax_bot.plot(x, p - q, lw=1.2)
        ax_bot.axhline(0.0, lw=1.0)
        ax_bot.set_xlabel("x")
        ax_bot.set_ylabel("p − q")
        ax_bot.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def make_gmm_pdf(weights, means, sigmas, device="cpu", dtype=torch.float32):
    """Return a callable pdf(x) for a 1D Gaussian mixture."""
    w = torch.tensor(weights, device=device, dtype=dtype)
    m = torch.tensor(means, device=device, dtype=dtype)
    s = torch.tensor(sigmas, device=device, dtype=dtype)

    def pdf(x_np):
        x_t = torch.tensor(np.asarray(x_np), device=device, dtype=dtype).reshape(-1)
        xM = x_t.unsqueeze(-1)
        mK = m.unsqueeze(0)
        sK = s.unsqueeze(0)
        norm = 1.0 / (torch.sqrt(torch.tensor(2.0*np.pi, device=device, dtype=dtype)) * sK)
        comp = norm * torch.exp(-0.5 * ((xM - mK)/sK)**2)
        return (comp * w.unsqueeze(0)).sum(dim=1).cpu().numpy()

    return pdf


# ------------------------
# Minimal example
# ------------------------
if __name__ == "__main__":
    # Example target: a bimodal 1D density (normalized automatically)
    def p_init(x):
        const_mu = -2.0
        const_std = 0.5
        # two Gaussians (unnormalized on purpose; fitter will normalize)
        return np.exp(-0.5*((x-const_mu)/const_std)**2) / (const_std*np.sqrt(2*np.pi))

    # fit = fit_gmm_to_p_init(
    #     p_init,
    #     n_components=26,
    #     x_range=(-6.0, 6.0),
    #     n_grid=512,
    #     n_steps=10000,
    #     lr=5e-3,
    #     device="cpu",
    #     dtype=torch.float32,
    #     progress=True,
    #     fix_means="uniform"
    # )

    # print("\nFitted parameters:")
    # print("weights:", fit["weights"].cpu().numpy())
    # print("means  :", fit["means"].cpu().numpy())
    # print("sigmas :", fit["sigmas"].cpu().numpy())

    # plot_fit(p_init, fit, x_range=(-6.0, 6.0),
    #          n_grid=512,
    #          show_components=True,
    #          show_residual=True,
    #          title="3-component GMM fit")
    
    # # After calling fit_gmm_to_p_init(...)
    # np.savez(
    #     "data/fitted_gmm_pinit.npz",
    #     weights=fit["weights"].cpu().numpy(),
    #     means=fit["means"].cpu().numpy(),
    #     sigmas=fit["sigmas"].cpu().numpy()
    # )
    # print("Saved GMM parameters to fitted_gmm.npz")

    # Example loading code
    data = np.load("data/fitted_gmm_pinit.npz")
    weights = data["weights"]
    means = data["means"]
    sigmas = data["sigmas"]
    pdf_gmm = make_gmm_pdf(weights, means, sigmas)

    x_vals = np.linspace(-6, 6, num=200, endpoint=True)
    p_init_vals = p_init(x_vals)
    p_init_gmm = pdf_gmm(x_vals)
    plt.figure()
    plt.plot(x_vals, p_init_vals)
    plt.plot(x_vals, p_init_gmm, linestyle="--")
    plt.show()

