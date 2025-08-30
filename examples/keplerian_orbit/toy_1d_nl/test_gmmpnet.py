import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math

# -------------------------
# Time-conditioned 1D GMM model
# -------------------------
class TimeToGMM1D(nn.Module):
    """
    t -> K-component GMM params: (means, scales, weights)
    and mixture log-pdf/pdf evaluators for x.
    """
    def __init__(
        self,
        scale = 1.0,
        n_components: int = 16,
        hidden: int = 50,
        depth: int = 3,
        min_scale: float = 1e-4,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.scale = scale
        assert n_components >= 1
        self.K = int(n_components)
        self.min_scale = float(min_scale)

        # layers = [nn.Linear(1, hidden), nn.SiLU(inplace=True)]
        # for _ in range(max(0, depth - 1)):
        #     layers += [nn.Linear(hidden, hidden), nn.SiLU(inplace=True)]
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.net = nn.Sequential(*layers)

        # Head outputs [means_K, raw_scales_K, logits_K] per t
        self.head = nn.Linear(hidden, 3 * self.K)

        # constants as buffers
        self.register_buffer(
            "LOG_2PI",
            torch.log(torch.tensor(2.0 * math.pi, dtype=dtype))
        )
        self.to(device=device, dtype=dtype)

    @staticmethod
    def _col(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 1) if z.ndim == 1 else z

    def params(self, t: torch.Tensor):
        """
        Inputs:
            t: (B,) or (B,1)
        Returns (each with shape (B,K)):
            means, scales>0, weights in simplex (sum=1 along K)
        """
        w = self.head.weight
        t = self._col(t).to(dtype=w.dtype, device=w.device)
        h = self.net(t)
        out = self.head(h)  # (B, 3K)
        m, r, g = torch.split(out, [self.K, self.K, self.K], dim=-1)
        # Positive scales
        s = F.softplus(r, beta=1.0, threshold=20.0) + self.min_scale
        # Mixture weights
        pi = F.softmax(g, dim=-1)
        return m, s, pi

    def _component_logpdf(self, x: torch.Tensor, means: torch.Tensor, scales: torch.Tensor):
        """
        x: (B,N,1) , means/scales: (B,1,K) -> returns (B,N,K)
        """
        z = (x - means) / scales
        return -0.5 * z.square() - torch.log(scales) - 0.5 * self.LOG_2PI

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (N,) or (N,1)
        t: (B,) or (B,1)
        If B==1, we broadcast params over N; else require B==N (per-sample t).
        Returns (N,1) log p(x|t)
        """
        x = self._col(x)
        m, s, pi = self.params(t)                 # (B,K),(B,K),(B,K)

        # Broadcast handling: either per-batch (B==N) or single t for all x (B==1)
        if m.shape[0] == 1:
            B, N, K = 1, x.shape[0], self.K
            xB = x.view(1, N, 1)
            mB = m.view(1, 1, K)
            sB = s.view(1, 1, K)
            piB = pi.view(1, 1, K)
        else:
            assert m.shape[0] == x.shape[0], "If t is batched, must have len(t)==len(x)"
            B, N, K = x.shape[0], 1, self.K
            xB = x.view(B, N, 1)
            mB = m.view(B, 1, K)
            sB = s.view(B, 1, K)
            piB = pi.view(B, 1, K)

        comp_logp = self._component_logpdf(xB, mB, sB)  # (B,N,K)
        log_pi = torch.log(piB + 1e-12)
        log_mix = torch.logsumexp(log_pi + comp_logp, dim=-1)  # (B,N)
        return log_mix.view(-1, 1)

    def pdf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x, t))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.pdf(x, t)


# -------------------------
# Route-B trainer (unchanged API)
# -------------------------
@torch.no_grad()
def _precompute_targets_on_grid(p_target_fn, x_lin, eps=1e-12, weight_by_pdf=True):
    p_t = p_target_fn(x_lin).to(dtype=torch.float32).view(-1)
    p_t = torch.clamp(p_t, min=eps)
    log_p_t = torch.log(p_t)
    w = p_t / p_t.mean() if weight_by_pdf else torch.ones_like(p_t)
    return log_p_t, w




# -------------------------
# Minimal example (bimodal target) + print params + plot
# -------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bimodal target to show mixture capacity
    def p_target_pdf(x):
        x = x.view(-1, 1)
        pi = torch.tensor(3.141592653589793, dtype=torch.float32, device=x.device)
        def n(mu, sig):
            return torch.exp(-0.5*((x - mu)/sig)**2) / (sig * torch.sqrt(2.0*pi))
        p = 0.6*n(-1.5, 0.5) + 0.4*n(1.8, 0.7)
        return p.squeeze(-1)

    # Model: K-component time-conditioned GMM
    K = 4
    model = TimeToGMM1D(n_components=K, hidden=64, depth=3).to(device)

    # Train at t = 0
    model = train_with_pdf_values(
        model, p_target_pdf,
        x_range=(-5.0, 5.0),
        grid_points=256,
        epochs=500,
        batch_size=256,
        lr=2e-3,
        log_every=200,
        weight_by_pdf=True
    )

    # Print learned mixture params at t = 0
    with torch.no_grad():
        t0 = torch.tensor([0.0], device=device)
        means, scales, weights = model.params(t0)  # shapes (1,K)
        print("\n--- Learned GMM at t=0 ---")
        for k in range(K):
            print(f" comp {k:02d}: weight={weights[0,k].item():.6f}  mean={means[0,k].item():+.6f}  std={scales[0,k].item():.6f}")

    # Visualization
    with torch.no_grad():
        xs = torch.linspace(-5, 5, 1200, device=device)
        ts = torch.zeros_like(xs)
        pt = p_target_pdf(xs).cpu().numpy()
        pm = model.pdf(xs, ts).squeeze().cpu().numpy()
        xs_np = xs.cpu().numpy()

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs_np, pt, label="p_target(x)", linewidth=2.0)
    plt.plot(xs_np, pm, label=f"p_model(x | t=0), K={K}", linewidth=2.0, linestyle="--")
    plt.xlabel("x"); plt.ylabel("density"); plt.title("Target vs Fitted GMM PDF at t = 0")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
