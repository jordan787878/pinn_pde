import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
_PI = np.float32(math.pi)
_MU_EARTH = np.float32(3.9859e+14)
_R_EARTH  = np.float32(6.378e+6)
_A        = np.float32(4.2164e+7)
_W        = np.float32(np.sqrt(_MU_EARTH/_A**3))
_T        = np.float32(2*np.pi/_W)
_R        = np.float32(2e+6)
_THETA    = np.float32(0.015)
_PHI      = np.float32(0.0387)
_MEAN_I   = np.float32([_A, 0.5*_PI, 0.0, 0.0, 0.0, _W])
_N_MEAN_I = np.float32([_MEAN_I[0]/_R,
                        _MEAN_I[1]/_THETA, 
                        _MEAN_I[2]/_PHI,
                        _MEAN_I[3]/(_R/_T),
                        _MEAN_I[4]/(_THETA/_T),
                        (_MEAN_I[5]-_W)/(_PHI/_T)])  
_N_COV_I  = np.float32(np.diag([1e+11/(_R**2),
                                1e-5/(_THETA**2),
                                1e-4/(_PHI**2),
                                1e+3/(_R/_T)**2,
                                1e-13/(_THETA/_T)**2,
                                1e-12/(_PHI/_T)**2]))
_X1_RANGE = np.float32(np.array([20.1, 22.1]))
_X2_RANGE = np.float32(np.array([103.5, 105.9]))
_X3_RANGE = np.float32(np.array([-2.2, 2.2]))
_X4_RANGE = np.float32(np.array([-10., 10.]))
_X5_RANGE = np.float32(np.array([-6., 6.]))
_X6_RANGE = np.float32(np.array([-8., 8.]))


class Constants:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.X1_RANGE = torch.tensor(_X1_RANGE, dtype=dtype, device=device)
        self.X2_RANGE = torch.tensor(_X2_RANGE, dtype=dtype, device=device)
        self.X3_RANGE = torch.tensor(_X3_RANGE, dtype=dtype, device=device)
        self.X4_RANGE = torch.tensor(_X4_RANGE, dtype=dtype, device=device)
        self.X5_RANGE = torch.tensor(_X5_RANGE, dtype=dtype, device=device)
        self.X6_RANGE = torch.tensor(_X6_RANGE, dtype=dtype, device=device)
        self.N_MEAN_I = torch.tensor(_N_MEAN_I, dtype=dtype, device=device)
        self.N_COV_I = torch.tensor(_N_COV_I, dtype=dtype, device=device)

# ============================================================
# Time-conditioned Diagonal-GMM in N dimensions
# t -> {means(KxD), scales(KxD), weights(K)} ; x in R^D
# ============================================================
class TimeToDiagGMMND(nn.Module):
    """
    t -> K-component diagonal-covariance GMM in D dimensions.
    Supports input normalization using ranges from `constants`.
    Matches the structure of TimeToGMM1D.
    """
    def __init__(
        self,
        constants,
        x_dim: int,
        n_components: int = 16,
        hidden: int = 50,
        depth: int = 3,
        min_scale: float = 1e-4,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        assert n_components >= 1 and x_dim >= 1

        # Store constants & params
        self.constants = constants
        self.D = int(x_dim)
        self.K = int(n_components)
        self.min_scale = float(min_scale)

        # Backbone network: tanh MLP on time t
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.net = nn.Sequential(*layers)

        # Head produces [means(K*D), raw_scales(K*D), logits(K)]
        self.head = nn.Linear(hidden, self.K * (2 * self.D) + self.K)

        # Constants
        self.register_buffer(
            "LOG_2PI",
            torch.tensor(math.log(2.0 * math.pi), dtype=dtype)
        )

        # Move dtype/device
        self.to(device=device, dtype=dtype)

    # ------------------------- Utilities -------------------------
    @staticmethod
    def _col(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 1) if z.ndim == 1 else z

    def _ensure_x_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            if x.numel() != self.D:
                raise ValueError(
                    f"x is 1D with length {x.numel()}, but x_dim={self.D}. "
                    f"Pass (D,) or (N,D)."
                )
            return x.view(1, self.D)
        if x.ndim == 2 and x.shape[1] == self.D:
            return x
        raise ValueError(
            f"x must have shape (D,) or (N,{self.D}). Got {tuple(x.shape)}."
        )

    # ------------------------- Parameters -------------------------
    def params(self, t: torch.Tensor):
        """
        Inputs:
            t: (B,) or (B,1)
        Returns:
            means:  (B,K,D)
            scales: (B,K,D)   (positive)
            weights: (B,K)   (simplex)
        """
        w = self.head.weight
        t = self._col(t).to(dtype=w.dtype, device=w.device)
        h = self.net(t)
        out = self.head(h)  # (B, K*(2D)+K)

        # Split outputs
        split1 = self.K * self.D
        split2 = split1 + self.K * self.D
        m_flat = out[:, :split1]
        r_flat = out[:, split1:split2]
        g = out[:, split2:]

        means = m_flat.view(-1, self.K, self.D)
        scales = F.softplus(r_flat, beta=1.0, threshold=20.0).view(-1, self.K, self.D) + self.min_scale
        weights = F.softmax(g, dim=-1)
        return means, scales, weights

    # ------------------------- Component Log-PDF -------------------------
    def _diag_component_logpdf(self, x, means, scales):
        """
        x:      (B,N,1,D)
        means:  (B,1,K,D)
        scales: (B,1,K,D)
        Returns: (B,N,K)
        """
        z = (x - means) / scales
        return -0.5 * z.square().sum(dim=-1) \
               - torch.log(scales).sum(dim=-1) \
               - 0.5 * self.D * self.LOG_2PI

    # ------------------------- Normalization -------------------------
    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        c = self.constants
        ranges = torch.stack([
            c.X1_RANGE, c.X2_RANGE, c.X3_RANGE,
            c.X4_RANGE, c.X5_RANGE, c.X6_RANGE
        ])  # (D, 2)
        mids = 0.5 * (ranges[:, 1] + ranges[:, 0])
        half_ranges = 0.5 * (ranges[:, 1] - ranges[:, 0])
        return (x - mids) / half_ranges

    # ------------------------- Log-Prob -------------------------
    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self._ensure_x_2d(x)
        x = x.to(dtype=self.head.weight.dtype, device=self.head.weight.device)

        # Normalize x if constants are given
        if self.constants is not None:
            x = self._normalize_inputs(x)

        means, scales, weights = self.params(t)  # (B,K,D), (B,K,D), (B,K)

        # Broadcasting: if B==1, share params across all x; else require B==N
        if means.shape[0] == 1:
            B, N = 1, x.shape[0]
            xB = x.view(1, N, 1, self.D)
            mB = means.view(1, 1, self.K, self.D)
            sB = scales.view(1, 1, self.K, self.D)
            piB = weights.view(1, 1, self.K)
        else:
            assert means.shape[0] == x.shape[0], "Batched t requires len(t)==len(x)"
            B, N = x.shape[0], 1
            xB = x.view(B, N, 1, self.D)
            mB = means.view(B, 1, self.K, self.D)
            sB = scales.view(B, 1, self.K, self.D)
            piB = weights.view(B, 1, self.K)

        comp_logp = self._diag_component_logpdf(xB, mB, sB)  # (B,N,K)
        log_pi = torch.log(piB + 1e-12)
        log_mix = torch.logsumexp(log_pi + comp_logp, dim=-1)  # (B,N)
        return log_mix.view(-1, 1)

    def pdf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x, t))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.pdf(x, t)


# -------------------------
# Helper: uniform sampler over ND box
# -------------------------
def make_uniform_sampler(x_ranges: torch.Tensor):
    """
    x_ranges: (D,2) tensor with [lo, hi] per dimension (on correct device/dtype)
    returns sampler(N) -> (N,D) uniform
    """
    D = x_ranges.shape[0]
    lo = x_ranges[:, 0].unsqueeze(0)  # (1,D)
    hi = x_ranges[:, 1].unsqueeze(0)  # (1,D)
    def sampler(N: int):
        u = torch.rand(N, D, device=x_ranges.device, dtype=x_ranges.dtype)
        return lo + (hi - lo) * u
    return sampler

# -------------------------
# Train with *pdf values* (no target sampling)
# Weighted log-PDF regression over random x ~ Uniform(box)
# -------------------------
def train_with_pdf_values_nd(
    model: nn.Module,
    p_target_fn,                # callable: p_target_fn(x:[N,D]) -> pdf values [N] or [N,1]
    x_ranges,                   # (D,2) per-dim [lo, hi]
    steps: int = 6000,
    batch_size: int = 4096,
    val_size: int = 20000,
    lr: float = 2e-3,
    log_every: int = 250,
    eps: float = 1e-12,
    weight_by_pdf: bool = True, # emphasize high-density regions
    grad_clip: float | None = 5.0,
    patience: int = 30,
):
    """
    Minimizes:  E_{x~Uniform(box)} [ w(x) * ( log p_model(x|t=0) - log p_target(x) )^2 ]
    with w(x) ∝ p_target(x) if weight_by_pdf=True (like the 1D version).
    """
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # Ensure x_ranges lives with model
    if not isinstance(x_ranges, torch.Tensor):
        x_ranges = torch.tensor(x_ranges, device=device, dtype=dtype)
    else:
        x_ranges = x_ranges.to(device=device, dtype=dtype)
    assert x_ranges.ndim == 2 and x_ranges.shape[1] == 2, "x_ranges must be (D,2)"

    D = x_ranges.shape[0]
    sampler = make_uniform_sampler(x_ranges)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    # Fixed validation set (so we can early-stop)
    with torch.no_grad():
        X_val = sampler(val_size)
        t_val = torch.zeros(X_val.shape[0], device=device, dtype=dtype)
        p_t_val = p_target_fn(X_val).to(device=device, dtype=dtype).view(-1)
        p_t_val = torch.clamp(p_t_val, min=eps)
        log_p_t_val = torch.log(p_t_val)
        W_val = p_t_val / p_t_val.mean() if weight_by_pdf else torch.ones_like(p_t_val)

    @torch.no_grad()
    def eval_full_loss() -> float:
        log_pm = model.log_prob(X_val, t_val).view(-1)
        return ((log_pm - log_p_t_val) ** 2 * W_val).mean().item()

    for step in range(1, steps + 1):
        # Random batch from proposal (uniform box)
        X = sampler(batch_size)
        t0 = torch.zeros(X.shape[0], device=device, dtype=dtype)

        # Target log-pdf and weights
        p_t = p_target_fn(X).to(device=device, dtype=dtype).view(-1)
        p_t = torch.clamp(p_t, min=eps)
        log_p_t = torch.log(p_t)
        W = p_t / p_t.mean() if weight_by_pdf else torch.ones_like(p_t)

        # Model log-pdf
        log_pm = model.log_prob(X, t0).view(-1)

        loss = ((log_pm - log_p_t) ** 2 * W).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()

        if step % log_every == 0:
            val_loss = eval_full_loss()
            improved = val_loss + 1e-9 < best_loss
            if improved:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"[LogPDF-ND] step {step:5d} | train {loss.item():.6e} | val {val_loss:.6e} | "
                  f"best {best_loss:.6e} | patience {no_improve}/{patience}")
            if no_improve >= patience and best_state is not None:
                model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def make_target_mvn_6d(constants, device="cpu", dtype=torch.float32):
    """
    Returns a function that computes the PDF of a 6D Gaussian.
    constants must provide:
        - N_MEAN_I: (6,) mean vector
        - N_COV_I: (6,6) covariance matrix
    """
    mu = torch.as_tensor(constants.N_MEAN_I, dtype=dtype, device=device)    # (6,)
    cov = torch.as_tensor(constants.N_COV_I, dtype=dtype, device=device)    # (6,6)
    cov_inv = torch.linalg.inv(cov)                                        # (6,6)
    cov_det = torch.linalg.det(cov)                                        # scalar

    D = mu.shape[0]
    log_norm_const = -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device)) + torch.log(cov_det))

    def pdf_fn(X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate PDF at N points.
        Args:
            X: (..., D) tensor of samples.
        Returns:
            pdf: (...) tensor of PDF values.
        """
        diff = X - mu
        exponent = -0.5 * torch.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        log_pdf = log_norm_const + exponent
        return torch.exp(log_pdf)

    return pdf_fn


# ============================================================
# 6D validation: set the network to output a known 6D GMM at t=0
# ============================================================
if __name__ == "__main__":
    constants = Constants()
    # torch.set_default_dtype(torch.float32)
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # D, K = 6, 1
    # model = TimeToDiagGMMND(constants=constants, x_dim=D, n_components=K, hidden=64, depth=3, init_std=1.0).to(device)

    # # Define a *target* 6D diagonal-GMM (means, stds, weights)
    # torch.manual_seed(0)
    # means_target  = torch.randn(K, D, device=device) * 0.5 + torch.linspace(-1.0, 1.0, K, device=device).view(K,1)
    # stds_target   = 0.4 + 0.3 * torch.rand(K, D, device=device)
    # logits_target = torch.randn(K, device=device)
    # weights_target = torch.softmax(logits_target, dim=-1)

    # # Helper: set model to *constant* mapping so params(t) == target for any t
    # with torch.no_grad():
    #     # zero backbone so net(t)=0 for all t
    #     for m in model.net:
    #         if isinstance(m, nn.Linear):
    #             m.weight.zero_()
    #             if m.bias is not None:
    #                 m.bias.zero_()
    #     # zero head weights; encode all params in head.bias
    #     model.head.weight.zero_()

    #     # pack biases: [means(K*D), raw_scales(K*D), logits(K)]
    #     b = torch.zeros(model.head.out_features, device=device, dtype=model.head.bias.dtype)
    #     # means
    #     b[:K*D] = means_target.reshape(-1)
    #     # raw scales = softplus^{-1}(stds)
    #     b[K*D:2*K*D] = torch.log(torch.expm1(stds_target.reshape(-1)))
    #     # logits
    #     b[2*K*D:] = torch.log(weights_target + 1e-12)  # log-weights OK; softmax normalizes
    #     model.head.bias.copy_(b)

    # # Reference evaluator for the same diagonal GMM
    # def ref_diag_gmm_pdf(x: torch.Tensor) -> torch.Tensor:
    #     x = x.view(-1, D)
    #     xB = x.view(1, x.shape[0], 1, D)
    #     mB = means_target.view(1, 1, K, D)
    #     sB = stds_target.view(1, 1, K, D)
    #     log_2pi = torch.tensor(math.log(2.0 * math.pi), device=x.device, dtype=x.dtype)
    #     comp_logp = -0.5 * (((xB - mB) / sB) ** 2).sum(-1) - torch.log(sB).sum(-1) - 0.5 * D * log_2pi  # (1,N,K)
    #     log_mix = torch.logsumexp(torch.log(weights_target.view(1,1,K)) + comp_logp, dim=-1)             # (1,N)
    #     return torch.exp(log_mix).view(-1, 1)

    # # Generate random 6D samples and compare PDFs
    # X = torch.randn(100, D, device=device)
    # t0_single = torch.tensor([0.0], device=device)   # broadcast case
    # with torch.no_grad():
    #     p_model = model.pdf(X, t0_single)            # (1000,1)
    #     p_ref   = ref_diag_gmm_pdf(X)                # (1000,1)
    #     max_abs = (p_model - p_ref).abs().max().item()
    #     rel_mae = ( (p_model - p_ref).abs() / (p_ref + 1e-12) ).mean().item()
    #     print(f"[6D validation] max |p_model - p_ref| = {max_abs:.3e} | relative MAE = {rel_mae:.3e}")

    # # Also test batched t (len(t)==len(x))
    # t_batched = torch.zeros(X.shape[0], device=device)
    # with torch.no_grad():
    #     p_model_batched = model.pdf(X, t_batched)
    #     max_diff_batched = (p_model - p_model_batched).abs().max().item()
    #     print(f"[broadcast vs batched t] max difference = {max_diff_batched:.3e}")

    # # Print the parameters recovered from the model at t=0
    # with torch.no_grad():
    #     M, S, W = model.params(torch.tensor([0.0], device=device))
    #     print("\n--- Model params at t=0 (should match target up to fp error) ---")
    #     for k in range(K):
    #         print(f"comp {k:02d}: w={W[0,k].item():.6f}  mean={M[0,k,].tolist()}  std[]={S[0,k,].tolist()}")

    # Training Example

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use your TimeToDiagGMMND class defined earlier
    # Example config
    D, K = 6, 16
    model = TimeToDiagGMMND(constants=constants, x_dim=D, n_components=K, hidden=50, depth=3).to(device)

    # Target 6D Normal (evaluation only)
    p_target_fn = make_target_mvn_6d(constants, device)

    # Choose a bounding box for uniform sampling (should cover the mass)
    x_ranges = torch.stack([
        constants.X1_RANGE,
        constants.X2_RANGE,
        constants.X3_RANGE,
        constants.X4_RANGE,
        constants.X5_RANGE,
        constants.X6_RANGE
    ], dim=0).to(device)

    # Train with pdf values at random x ~ Uniform(box)
    model = train_with_pdf_values_nd(
        model,
        p_target_fn=p_target_fn,
        x_ranges=x_ranges,
        steps=10000,
        batch_size=4096,
        val_size=20000,
        lr=2e-3,
        log_every=1000,
        weight_by_pdf=True,
        patience=30,
    )

    # -------------------------
    # Evaluate on fresh random points (no sampling from target for training!)
    # -------------------------
    with torch.no_grad():
        # --- Total variation (TV) over the evaluation box ---
        sampler_eval = make_uniform_sampler(x_ranges)
        X_eval = sampler_eval(50_000)
        t_eval = torch.zeros(X_eval.shape[0], device=device, dtype=X_eval.dtype)

        # logp_model = model.log_prob(X_eval, t_eval).view(-1)
        # pm = torch.exp(logp_model)   # model pdf
        pm = model(X_eval, t_eval).view(-1)
        pt = p_target_fn(X_eval)

        # Volume of the evaluation region X = ∏_d (hi_d - lo_d)
        volume = torch.prod(x_ranges[:, 1] - x_ranges[:, 0])

        # TV ≈ (1/2) * ∫_X |p - q| dx  ≈ 0.5 * volume * E_{U~Unif(X)} |p(U) - q(U)|
        tv_est = 0.5 * volume * torch.mean(torch.abs(pm - pt))
        norm_worst_error = torch.max(torch.abs(pm - pt)).item() / torch.max(pt).item()

        print("\n=== Final evaluation on random uniform points (Total Variation) ===")
        print(f"TV estimate over box: {100.*tv_est.item():.3f} %")
        print(f"Norm. worst error estimate over box: {100.*norm_worst_error:.3f} %")

        # Print a brief mixture summary at t=0
        M, S, W = model.params(torch.tensor([0.0], device=device))
        print("\n--- Learned GMM at t=0 (first 3 comps preview) ---")
        for k in range(min(K, 3)):
            m3 = ", ".join(f"{v:+.3f}" for v in M[0, k, :3].tolist())
            s3 = ", ".join(f"{v:.3f}"  for v in S[0, k, :3].tolist())
            print(f"comp {k:02d}: w={W[0,k].item():.5f}  mean[:3]=[{m3}]  std[:3]=[{s3}]")
