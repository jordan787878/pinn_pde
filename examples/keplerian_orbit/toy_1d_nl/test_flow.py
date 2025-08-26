import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm
from scipy.stats import multivariate_normal
import seaborn as sns

# Correct imports for modern nflows API
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import CompositeTransform
from nflows.transforms.permutations import RandomPermutation

# --------------------------
# Device & dtype
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------
# Save dir
# --------------------------
folder_path = "test_data"
os.makedirs(folder_path, exist_ok=True)

# --------------------------
# Problem constants
# --------------------------
x_low = np.float32(-6.0)
x_hig = np.float32(6.0)
mu = np.float32(-2.0)
std = np.float32(0.5)
const_a = np.float32(-0.1)
const_b = np.float32(0.1)
const_c = np.float32(0.5)
const_d = np.float32(0.5)
const_e = np.float32(0.8)
t0 = np.float32(0.0)
T_end = np.float32(5.0)

torch.manual_seed(0)
np.random.seed(0)


def p_init(x, mu=mu, std=std):
    """
    Standard normal pdf with mean=mu, std=std evaluated at x.
    Works with NumPy arrays/scalars and PyTorch tensors.
    - If any input is a torch.Tensor, returns a torch.Tensor on the right device/dtype.
    - Otherwise returns a NumPy array (or scalar) matching x's dtype.
    """
    is_xtorch = isinstance(x, torch.Tensor)
    is_mutorch = isinstance(mu, torch.Tensor)
    is_storch  = isinstance(std, torch.Tensor)
    use_torch  = is_xtorch or is_mutorch or is_storch

    if use_torch:
        # pick a reference tensor for device/dtype
        ref = x if is_xtorch else (mu if is_mutorch else std)
        device = ref.device
        dtype  = ref.dtype

        x_t   = x   if is_xtorch   else torch.as_tensor(x,   dtype=dtype, device=device)
        mu_t  = mu  if is_mutorch  else torch.as_tensor(mu,  dtype=dtype, device=device)
        std_t = std if is_storch   else torch.as_tensor(std, dtype=dtype, device=device)

        two = torch.as_tensor(2.0, dtype=dtype, device=device)
        pi  = torch.as_tensor(np.pi, dtype=dtype, device=device)

        z = (x_t - mu_t) / std_t
        return torch.exp(-0.5 * z**2) / (std_t * torch.sqrt(two * pi))

    else:
        # NumPy path
        x_np   = np.asarray(x)
        # promote to float if x is integer
        if not np.issubdtype(x_np.dtype, np.floating):
            x_np = x_np.astype(np.float32)
        mu_np  = np.asarray(mu,  dtype=x_np.dtype)
        std_np = np.asarray(std, dtype=x_np.dtype)

        return np.exp(-0.5 * ((x_np - mu_np) / std_np) ** 2) / (std_np * np.sqrt(2 * np.pi))


def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True, dtype=np.float32)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def u_residual_1d(x, t, flow):
    """
    x: (N,1) requires_grad=True
    t: (N,1) requires_grad=True
    flow.log_prob(x, context=t) returns scalar log-density per sample
    """
    # log-density u(x,t)
    u = flow.log_prob(x, t).view(-1, 1)  # u = log p

    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]   # (N,1)
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]   # (N,1)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]  # (N,1)

    # coefficients as tensors on the right device/dtype
    ref = x
    a = torch.as_tensor(const_a, dtype=ref.dtype, device=ref.device)
    b = torch.as_tensor(const_b, dtype=ref.dtype, device=ref.device)
    c = torch.as_tensor(const_c, dtype=ref.dtype, device=ref.device)
    d = torch.as_tensor(const_d, dtype=ref.dtype, device=ref.device)
    e = torch.as_tensor(const_e, dtype=ref.dtype, device=ref.device)

    mu      = a*x**3 + b*x**2 + c*x + d
    mu_prime= 3*a*x**2 + 2*b*x + c

    resid = u_t + mu_prime + mu*u_x - 0.5*(e**2)*(u_xx + u_x*u_x)
    return resid


# --------------------------
# Normalizing Flow Model
# --------------------------
from nflows.transforms import MaskedAffineAutoregressiveTransform

class PDF_Flow(nn.Module):
    def __init__(self, x_dim, t_dim=1, neurons=32, num_layers=2):
        super().__init__()
        self.base_dist = StandardNormal([x_dim])

        layers = []
        for _ in range(num_layers):
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=x_dim,
                    context_features=t_dim,
                    hidden_features=neurons,
                    num_blocks=3,                    # deeper conditioner
                    activation=nn.Tanh(),            # smooth (or nn.Softplus())
                    use_residual_blocks=False,
                    dropout_probability=0.0,
                    use_batch_norm=False
                )
            )
            layers.append(RandomPermutation(features=x_dim))
        self.flow = Flow(CompositeTransform(layers), self.base_dist)

    def log_prob(self, x, t):
        context = t.view(-1, 1)
        return self.flow.log_prob(inputs=x, context=context)

    def forward(self, x, t):
        log_p = self.log_prob(x, t).view(-1, 1)
        return log_p

    def sample(self, num_samples, t):
        context = t.view(-1, 1).expand(num_samples, -1)
        return self.flow.sample(num_samples, context=context)


def train_p_net(p_net, PATH, iterations=5000):
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-4)   # smaller LR
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    batch_size = 1000
    min_loss = float('inf')
    loss_history = []
    iterations_per_decay = 500

    normal_ic = torch.distributions.Normal(
        loc=torch.as_tensor(mu,  dtype=torch.float32, device=device),
        scale=torch.as_tensor(std, dtype=torch.float32, device=device)
    )

    for epoch in range(iterations):
        optimizer.zero_grad()

        # ---- IC at t0 (log domain, stable) ----
        x_bc = (torch.rand(batch_size, 1, device=device) * (x_hig - x_low) + x_low)
        t_bc = torch.full((batch_size, 1), t0, device=device)
        log_p_i   = normal_ic.log_prob(x_bc).view(-1)
        log_phat_i= p_net.log_prob(x_bc, t_bc).view(-1)
        loss_ic   = mse_cost_function(log_phat_i, log_p_i)

        # ---- PDE residual ----
        x = (torch.rand(batch_size, 1, device=device) * (x_hig - x_low) + x_low).requires_grad_(True)
        t = (torch.rand(batch_size, 1, device=device) * (T_end - t0) + t0).requires_grad_(True)
        res_out = u_residual_1d(x, t, p_net)
        mse_res = mse_cost_function(res_out, torch.zeros_like(res_out))

        lambda_pde = 1e-1

        loss = loss_ic + lambda_pde * mse_res

        # save best
        if loss.item() < 0.9 * min_loss:
            print(f"[SAVE] epoch={epoch}  loss={loss.item():.4e}  ic={loss_ic.item():.4e}  pde={(lambda_pde*mse_res).item():.4e}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': p_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'train_time': 0.0
            }, PATH)
            min_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(p_net.parameters(), 1.0)  # clip!
        optimizer.step()

        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def load_train_model(net, PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"best pnet epoch: {epoch}, loss: {loss:.4f}, train time: {checkpoint['train_time']:.2f}")
    return net

class E_Net(nn.Module):
    def __init__(self, neurons=32, scale=1.0, p_net=None):
        super().__init__()
        self.scale = scale
        self.p_net = p_net
        self.l1 = nn.Linear(2, neurons)
        self.l2 = nn.Linear(neurons, neurons)
        self.out = nn.Linear(neurons, 1)
    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = F.tanh(self.l1(h))
        h = F.tanh(self.l2(h))
        output_raw = self.scale * self.out(h)
        p_net_out = torch.exp(self.p_net(x, t))
        output = output_raw + p_net_out
        return output     # e(x,t)

# def flow_prob_residual(x_in, t_in, flow):
#     x = x_in.clone().detach().requires_grad_(True)
#     t = t_in.clone().detach().requires_grad_(True)

#     a = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
#     D_ = torch.as_tensor(D, dtype=x.dtype, device=x.device)

#     u = flow.log_prob(x, t).view(-1,1)
#     p = torch.exp(u)
#     p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#     p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#     p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
#     residual = p_t - a * (p_x * x + p) - D_ * p_xx
#     # if verbose:
#     #     print(p, residual)
#     return residual

# def diff_e_hat(x_in, t_in, e_net):
#     x = x_in.clone().detach().requires_grad_(True)
#     t = t_in.clone().detach().requires_grad_(True)

#     a = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
#     D_ = torch.as_tensor(D, dtype=x.dtype, device=x.device)

#     e = e_net(x, t)
#     e_t  = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
#     e_x  = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
#     e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]

#     return e_t - a*(x*e_x + e) - D_*e_xx

# def train_e_net(e_net, flow, iterations=4000, batch_size=500, path="e_net.pth"):
#     mse = torch.nn.MSELoss()
#     opt = torch.optim.Adam(e_net.parameters(), lr=1e-3)
#     sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

#     for p in flow.parameters():
#         p.requires_grad_(False)
#     flow.eval()

#     min_loss = float('inf')
#     for it in range(iterations):
#         opt.zero_grad()

#         # ---- IC in probability space: e(x,t0) = p_true - p_hat
#         x_ic = (torch.rand(batch_size,1,device=device)*(x_hig-x_low) + x_low)
#         t_ic = torch.full_like(x_ic, t0)
#         p_true_ic = p_init(x_ic)
#         p_hat_ic  = torch.exp(flow.log_prob(x_ic, t_ic).view(-1,1))
#         ic_pred   = e_net(x_ic, t_ic)
#         loss_ic   = mse(ic_pred, p_true_ic - p_hat_ic)

#         # ---- PDE: L[e] = - residual(p_hat)
#         x = (torch.rand(batch_size,1,device=device)*(x_hig-x_low) + x_low)
#         t = (torch.rand(batch_size,1,device=device)*(T_end - t0) + t0)
#         L_e   = diff_e_hat(x, t, e_net)
#         rhs   = - flow_prob_residual(x, t, flow).detach()
#         # (optional) normalize rhs to stabilize scale
#         loss_pde = mse(L_e, rhs)

#         loss = loss_ic + (T_end - t0)*loss_pde

#         if loss.item() < 0.9*min_loss:
#             min_loss = loss.item()
#             print(f"[E SAVE] it={it}, loss={loss.item():.4e}, ic={loss_ic.item():.4e}, pde={loss_pde.item():.4e}")
#             torch.save({
#                 'epoch': it,
#                 'model_state_dict': e_net.state_dict(),
#                 'optimizer_state_dict': opt.state_dict(),
#                 'loss': loss.item()
#             }, path)

#         loss.backward()
#         opt.step()
#         if (it+1) % 1000 == 0:
#             sch.step()

# def load_e_net(e_net, path):
#     ckpt = torch.load(path)
#     e_net.load_state_dict(ckpt['model_state_dict'])
#     print(f"best e epoch: {ckpt['epoch']}, loss: {ckpt['loss']:.4f}")
#     return e_net

# @torch.no_grad()
# def corrected_logp(flow, du_net, x, t):
#     return flow.log_prob(x, t).view(-1, 1) + du_net(x, t)

# @torch.no_grad()
# def corrected_pdf(flow, du_net, x, t):
#     return torch.exp(corrected_logp(flow, du_net, x, t))


@torch.no_grad()
def plot_p_surface(p_net, num=100, model_type="mlp",
                   n_samp=200, bins=50, e_net=None):
    plt.rcParams.update({"font.family": "serif",
                         "font.sans-serif": ["Times New Roman"],
                         "font.size": 18})
    t_eval = [t0, (t0+T_end)/2, T_end]
    x_grid = np.load("data/xsim.npy").astype(np.float32)

    fig, axs = plt.subplots(1, 3, figsize=(18,6), sharey=True)

    for i, t_val in enumerate(t_eval):
        t_grid = np.full((len(x_grid),1), t_val, dtype=np.float32)
        x_tensor = torch.from_numpy(x_grid).to(device).reshape(-1,1)
        t_tensor = torch.from_numpy(t_grid).to(device)
        print(x_tensor.shape, t_tensor.shape)

        B1 = None
        logp_pred = p_net(x_tensor, t_tensor)
        p_pred = torch.exp(torch.clamp(logp_pred, min=-60.0, max=30.0)).cpu().numpy()
        if(e_net is not None):
            e_pred = e_net(x_tensor, t_tensor).cpu().numpy().reshape(-1)
            B1 = np.max(np.abs(e_pred)) * 2.

        p_true = np.load("data/psim_t{:.1f}.npy".format(t_val)).astype(np.float32).reshape(-1,)

        ax = axs[i]
        ax.plot(x_grid, p_true, 'r-',  lw=2, label='Exact')
        ax.plot(x_grid, p_pred, 'b--', lw=2, label=f'{model_type.upper()}')
        if(B1 is not None):
            ax.fill_between(
                x_grid.flatten(),                 # x-coords
                (p_pred - B1).flatten(),          # lower bound
                (p_pred + B1).flatten(),          # upper bound
                color="green",
                alpha=0.2,                        # transparency of the band
                label="±B1 region"
            )


        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('x')
        if i == 0: ax.set_ylabel('p(x,t)')
        ax.set_xlim(float(x_low), float(x_hig))
        ax.legend(); ax.grid(True, linestyle='--')
    plt.tight_layout()

    # # Make the separate error figure if we have an error net
    # if model_type == "pflow" and e_net is not None:
    #     plot_error_surface_prob(p_net, e_net, num=num)

    plt.show()

# @torch.no_grad()
# def plot_error_surface_prob(p_net, e_net, num=200):
#     plt.rcParams.update({"font.family": "serif",
#                          "font.sans-serif": ["Times New Roman"],
#                          "font.size": 18})
#     t_eval = [t0, (t0+T_end)/2, T_end]
#     x_grid = np.linspace(x_low, x_hig, num=num, dtype=np.float32).reshape(-1,1)

#     fig, axs = plt.subplots(1,3, figsize=(18,5), sharey=True)

#     for i, t_val in enumerate(t_eval):
#         t_grid = np.full((num,1), t_val, dtype=np.float32)
#         x_tensor = torch.from_numpy(x_grid).to(device)
#         t_tensor = torch.from_numpy(t_grid).to(device)

#         p_true = p_exact(x_tensor, t_tensor)
#         p_hat  = torch.exp(p_net.log_prob(x_tensor, t_tensor).view(-1,1))
#         true_err = (p_true - p_hat).cpu().numpy()

#         e_hat = e_net(x_tensor, t_tensor).cpu().numpy()
#         B1 = np.max(np.abs(e_hat)) * 2.

#         ax = axs[i]
#         ax.plot(x_grid, true_err, 'r-', lw=2, label='True error: p_true - p_hat')
#         ax.plot(x_grid, e_hat,   'b--', lw=2, label='Learned error: e(x,t)')
#         ax.fill_between(
#                 x_grid.flatten(),                 # x-coords
#                 (e_hat*0. + B1).flatten(),          # lower bound
#                 (e_hat*0. - B1).flatten(),          # upper bound
#                 color="green",
#                 alpha=0.2,                        # transparency of the band
#                 label="±B1 region"
#             )
#         ax.axhline(0.0, color='k', ls=':', lw=1.0)
#         ax.set_title(f't = {t_val:.2f}')
#         ax.set_xlabel('x')
#         if i == 0: ax.set_ylabel('probability error')
#         ax.grid(True, linestyle='--'); ax.legend()

#     plt.tight_layout(); plt.show()


# =========================
# CHANGED: run_experiment trains flow, then δu (only for pflow)
# =========================
def run_experiment(model_type, iterations=20000, e_iterations=10000):
    print(f"--- Starting training for {model_type.upper()} PINN ---")
    x_dim = 1; t_dim = 1

    model = PDF_Flow(x_dim=x_dim, t_dim=t_dim).to(device)
    e_net = E_Net(neurons=32, scale=1.0).to(device)

    p_path = f"test_data/pflow.pth"
    e_path = f"test_data/e_net.pth"

    # 1) Train p-model
    train_p_net(model, p_path, iterations=iterations)
    loaded_model = load_train_model(model, p_path)

    # 2) If flow, train probability-space error e(x,t)
    e_loaded = None
    # if model_type == "pflow":
    #     print("--- Training probability-space error e(x,t) ---")
    #     e_net.p_net = loaded_model
    #     train_e_net(e_net, loaded_model, iterations=e_iterations, batch_size=500, path=e_path)
    #     e_loaded = load_e_net(e_net, e_path)

    # 3) Plot: PDFs + separate error figure
    plot_p_surface(loaded_model, model_type=model_type, e_net=e_loaded)



if __name__ == "__main__":
    # Run Normalizing Flow (PDF_Flow) experiment
    run_experiment("pflow")
