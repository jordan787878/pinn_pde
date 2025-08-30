import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class TimeToNormal1D(nn.Module):
    """
    Efficient: t -> (mean, scale) -> log p(x|t) / p(x|t)
    """
    def __init__(self, scale=1.0, hidden=64, depth=3, min_scale=1e-4, dtype=torch.float32, device=None):
        super().__init__()
        self.min_scale = float(min_scale)
        self.scale = scale
        layers = [nn.Linear(1, hidden), nn.SiLU(inplace=True)]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)  # [mean, raw_scale]

        # constants as buffers (on the right device/dtype)
        self.register_buffer("LOG_2PI", torch.tensor(1.8378770664093453, dtype=dtype))  # log(2π)

        # move module
        self.to(device=device, dtype=dtype)

    @staticmethod
    def _col(z: torch.Tensor) -> torch.Tensor:
        return z.reshape(-1, 1) if z.ndim == 1 else z

    def params(self, t: torch.Tensor):
        """
        t: (N,) or (N,1)
        returns mean (N,1), scale (N,1) with scale > 0
        """
        # ensure dtype/device match first layer
        w = self.head.weight
        t = self._col(t).to(dtype=w.dtype, device=w.device)
        h = self.net(t)
        out = self.head(h)
        mean = out[:, :1]
        # stable positive scale
        scale = F.softplus(out[:, 1:2], beta=1.0, threshold=20.0).add_(self.min_scale)
        return mean, scale

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (N,) or (N,1)
        t: (N,) or (N,1)
        returns log p(x|t) with shape (N,1)
        """
        x = self._col(x)
        mean, scale = self.params(t)
        z = (x - mean) / scale
        # log N(x|mean,scale^2) = -0.5 z^2 - log(scale) - 0.5 log(2π)
        lp = -0.5 * z.mul(z) - torch.log(scale) - 0.5 * self.LOG_2PI
        return lp

    def pdf(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x, t))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Alias to pdf for convenience
        return self.pdf(x, t)    


@torch.no_grad()
def _precompute_targets_on_grid(p_target_fn, x_lin, eps=1e-12, weight_by_pdf=True):
    p_t = p_target_fn(x_lin).to(dtype=torch.float32).view(-1)
    p_t = torch.clamp(p_t, min=eps)
    log_p_t = torch.log(p_t)
    w = p_t / p_t.mean() if weight_by_pdf else torch.ones_like(p_t)
    return log_p_t, w


def train_with_pdf_values(
    model: nn.Module,
    p_target_fn,               # callable: p_target_fn(x) -> (N,) or (N,1) tensor of positive pdf values
    x_range=(-6.0, 6.0),
    grid_points=4096,
    epochs=4000,
    batch_size=2048,
    lr=3e-3,
    log_every=200,
    eps=1e-12,
    weight_by_pdf=True,
):
    """
    Fit by minimizing weighted MSE on log-pdf over a uniform grid in x_range:
        L = E_x [ w(x) * ( log p_model(x|t=0) - log p_target(x) )^2 ]
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Grid over x and precompute targets/weights
    x_lin = torch.linspace(x_range[0], x_range[1], steps=grid_points, dtype=torch.float32, device=device)
    log_p_t, w = _precompute_targets_on_grid(p_target_fn, x_lin, eps=eps, weight_by_pdf=weight_by_pdf)

    # DataLoader over grid points (CPU storage to keep loader light)
    ds = TensorDataset(x_lin.cpu())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    patience = 30

    t0 = torch.zeros(1, dtype=torch.float32, device=device)  # scalar t=0

    @torch.no_grad()
    def eval_full_loss():
        t = t0.expand_as(x_lin)
        log_pm = model.log_prob(x_lin, t).view(-1)
        return ((log_pm - log_p_t) ** 2 * w).mean().item()

    for epoch in range(1, epochs + 1):
        for (x_batch_cpu,) in dl:
            x_batch = x_batch_cpu.to(device=device, dtype=torch.float32).view(-1)
            t = t0.expand_as(x_batch)

            # map x_batch to nearest grid indices (since targets were precomputed on x_lin)
            idx = ((x_batch - x_range[0]) / (x_range[1] - x_range[0]) * (grid_points - 1)).round().long().clamp(0, grid_points - 1)

            log_pm = model.log_prob(x_batch, t).view(-1)
            loss = ((log_pm - log_p_t[idx]) ** 2 * w[idx]).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

        if epoch % log_every == 0:
            full_loss = eval_full_loss()
            if full_loss + 1e-9 < best_loss:
                best_loss = full_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            print(f"[LogPDF] epoch {epoch:5d} | grid MSE {full_loss:.6e} | best {best_loss:.6e} | patience {no_improve}/{patience}")
            if no_improve >= patience and best_state is not None:
                model.load_state_dict(best_state)
                break

    return model


# -------------------------
# Minimal example
# -------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example target pdf evaluator (Gaussian mixture)
    def p_target_pdf(x):
        x = x.view(-1, 1)
        pi = torch.tensor(3.141592653589793, dtype=torch.float32, device=x.device)
        def n(mu, sig):
            return torch.exp(-0.5*((x - mu)/sig)**2) / (sig * torch.sqrt(2.0*pi))
        p = n(0.0, 0.6)
        return p.squeeze(-1)

    # Your model
    model = TimeToNormal1D(hidden=64, depth=3).to(device)

    # Train to match p_target(x) at t=0
    model = train_with_pdf_values(
        model, p_target_pdf,
        x_range=(-4.0, 5.0),
        grid_points=8192,
        epochs=500,
        batch_size=256,
        lr=2e-3,
        log_every=200,
        weight_by_pdf=True
    )

    # Quick check
    with torch.no_grad():
        xs = torch.linspace(-4, 5, 1000, device=device)
        ts = torch.zeros_like(xs)
        pm = model.pdf(xs, ts).squeeze()
        pt = p_target_pdf(xs)
        mae = torch.mean(torch.abs(pm - pt)).item()
        print(f"MAE(model pdf, target) on grid: {mae:.3e}")
        mean, std = model.params(torch.tensor([0.0], device=device))
        # mean, cov = model.params(0.0)
        print("fitted mean, std: ", mean, std)

    # -------------------------
    # Visualization (t = 0)
    # -------------------------
    with torch.no_grad():
        xs = torch.linspace(-4, 5, 1000, device=device)
        ts = torch.zeros_like(xs)
        pt = p_target_pdf(xs).cpu().numpy()
        pm = model.pdf(xs, ts).squeeze().cpu().numpy()
        xs_np = xs.cpu().numpy()

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs_np, pt, label="p_target(x)", linewidth=2.0)
    plt.plot(xs_np, pm, label="p_model(x | t=0)", linewidth=2.0, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Target vs Fitted PDF at t = 0")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
