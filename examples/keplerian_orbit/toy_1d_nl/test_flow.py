# pip install zuko
import torch
import torch.nn as nn
import zuko  # Zuko: normalizing flows in PyTorch


class PDF_Flow(nn.Module):
    """
    Time-conditioned normalizing flow using Zuko.
    p(x | t) is modeled with an NSF (neural spline flow), which is
    usually more expressive than affine MAF.

    Args:
        x_dim:      dimension of state x
        t_dim:      dimension of time/context t (often 1)
        scale:      (kept for compatibility; unused)
        neurons:    hidden width in the conditioner networks
        num_layers: number of autoregressive spline transforms
        bins:       number of spline bins per dim
    """
    def __init__(
        self,
        x_dim: int = 1,
        t_dim: int = 1,
        scale: float = 1.0,
        neurons: int = 64,
        num_layers: int = 10,
        bins: int = 8,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.t_dim = t_dim
        self.scale = scale

        # Zuko returns a *conditional* distribution when called with context.
        # You choose the architecture here (NSF, MAF, RealNVP, CNF, …).
        # hidden_features can be a list defining the MLP depth.
        self.flow = zuko.flows.NSF(
            features=x_dim,
            context=t_dim,
            transforms=num_layers,
            bins=bins,
            hidden_features=[neurons, neurons, neurons],
        )

    @staticmethod
    def _prep_context(t: torch.Tensor) -> torch.Tensor:
        # ensure (N, t_dim)
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 1:
            t = t[:, None]
        return t

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self._prep_context(t)
        return self.flow(c).log_prob(x)   # log p(x | c)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # match your original: return *density* not log-density
        return self.log_prob(x, t).unsqueeze(-1).exp()


# pip install zuko
import torch
import torch.nn as nn
import zuko  # Zuko: normalizing flows in PyTorch
from zuko.flows.continuous import FFJTransform
from zuko.lazy import Flow, UnconditionalDistribution
from zuko.distributions import DiagNormal


class PDF_Flow_CNF(nn.Module):
    """
    Time-conditioned normalizing flow using Zuko's FFJTransform (continuous flow).
    """
    def __init__(
        self,
        x_dim: int = 1,
        t_dim: int = 1,
        scale: float = 1.0,
        neurons: int = 64,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.t_dim = t_dim
        self.scale = scale

        # ---- build continuous transform explicitly via FFJTransform ----
        transform = FFJTransform(
            features=x_dim,
            context=t_dim,
            freqs=4,
            atol=1e-6,
            rtol=1e-5,
            exact=True,
            hidden_features=[neurons, neurons, neurons],
            activation=nn.ELU,
        )
        base = UnconditionalDistribution(
            DiagNormal,
            torch.zeros(x_dim),
            torch.ones(x_dim),
            buffer=True,
        )
        self.flow = Flow(transform, base)

    @staticmethod
    def _prep_context(t: torch.Tensor) -> torch.Tensor:
        # ensure (N, t_dim)
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 1:
            t = t[:, None]
        return t

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self._prep_context(t)
        return self.flow(c).log_prob(x)   # log p(x | c)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # return *density* not log-density
        return self.log_prob(x, t).unsqueeze(-1).exp()
