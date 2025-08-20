import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm
from scipy.linalg import expm, cholesky
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal
import seaborn as sns


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
folder_path = "data"
os.makedirs(folder_path, exist_ok=True)

# --------------------------
# Problem constants
# --------------------------
# computation domain
x_low = np.float32(-6.0)
x_hig = np.float32(6.0)

# Ref: https://link.springer.com/content/pdf/10.1007/978-1-4939-1323-7.pdf
# OU dynamics: dx = -\alpha x dt + \sqrt{2*D} dw, Eq. 4.19
# p(x,t) analytical solution, Eq. 4.22
# p(x,t|x_0) = \sqrt{\frac{\beta}{2 \pi D(1-e^{-2 \beta t})}} \exp \Big(-\frac{\beta(x-x_0e^{-\beta t})^2}{2D(1-e^{-2\beta t})} \Big)
# FP-PDE: Eq. 4.18a
x0    = np.float32(1.0)
alpha = np.float32(0.2)
D     = np.float32(0.2)

t0     = np.float32(1.0)
T_end  = np.float32(3.0)
dt     = np.float32(0.1)

# set fixed random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# Exact solution: NumPy (for reference) and Torch (for training)
# --------------------------
def p_exact(x, t):
    """Ornstein–Uhlenbeck exact PDF.
    Works with both NumPy arrays and Torch tensors.
    If `x` or `t` is a torch.Tensor, returns a torch.Tensor on the same device/dtype.
    Otherwise returns a NumPy array.
    """
    is_xtorch = isinstance(x, torch.Tensor)
    is_ttorch = isinstance(t, torch.Tensor)
    if is_xtorch or is_ttorch:
        ref = x if is_xtorch else t
        a   = torch.as_tensor(alpha, dtype=ref.dtype, device=ref.device)
        D_  = torch.as_tensor(D,      dtype=ref.dtype, device=ref.device)
        x0_ = torch.as_tensor(x0,     dtype=ref.dtype, device=ref.device)
        two = torch.as_tensor(2.0,    dtype=ref.dtype, device=ref.device)
        pi  = torch.as_tensor(np.pi,  dtype=ref.dtype, device=ref.device)
        x_t = x if is_xtorch else torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
        t_t = t if is_ttorch else torch.as_tensor(t, dtype=ref.dtype, device=ref.device)
        denom = two * D_ * (1 - torch.exp(-two * a * t_t))
        coef  = torch.sqrt(a / (two * pi * D_ * (1 - torch.exp(-two * a * t_t))))
        expo  = torch.exp(-a * (x_t - x0_ * torch.exp(-a * t_t))**2 / denom)
        return coef * expo
    else:
        return (np.sqrt(alpha/(2*np.pi*D*(1-np.exp(-2*alpha*t)))) * np.exp(-alpha*(x - x0*np.exp(-alpha*t))**2/(2*D*(1-np.exp(-2*alpha*t))))).astype(x.dtype)


def p_init(x):
  """
  mu_i = x0*exp(-alpha*t0)
  cov_i = D*(1-exp(-2*alpha*t0))/alpha
  p0(x) ~ N(x| mu_i, cov_i)
  """
  return p_exact(x, t0)


def get_p_normalize():
  """
  get the maximum p0(x) for normaliztion: improve training
  """
  x = np.linspace(x_low, x_hig, num=200, endpoint=True, dtype=np.float32)
  p0_true = p_init(x)
  return np.max(np.abs(p0_true))


# --------------------------
# Neural Network
# --------------------------
class Net(nn.Module):
    def __init__(self, p_init_func):
        neurons = 32
        self.scale = 1.0
        self.p_init_func = p_init_func
        super(Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x,t],axis=1)
        layer1_out = F.tanh(self.hidden_layer1(inputs))
        layer2_out = F.tanh(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out)
        output = F.softplus(output)
        # p_init_output = self.p_init_func(x)
        # dp = output*(t-t0)
        # output = dp + p_init_output
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


# --------------------------
# Residual (FP-PDE for OU)
# p_t - α*(p_x*x + p) - D*p_xx = 0
# --------------------------
def res_func(x, t, net, verbose=False):
    p = net(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t - alpha*(p_x*x + p) - D*p_xx
    if(verbose):
      print(p)
      print(residual)
      print(residual.shape)
    return residual


# --------------------------
# PINN Training
# --------------------------
def train_p_net(p_net, PATH):
  mse_cost_function = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(p_net.parameters())
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
  batch_size = 500
  iterations = 5000
  min_loss = np.inf
  loss_history = []
  normalize = p_net.scale
  iterations_per_decay = 1000
  start_time = time.time()

  for epoch in range(iterations):
      optimizer.zero_grad() # to make the gradients zero

      # Loss based on boundary conditions
      x_bc = (torch.rand(batch_size, 1) * (x_hig - x_low) + x_low).to(device)
      t_bc = (torch.ones(batch_size, 1) * t0).to(device)
      u_bc = p_init(x_bc).detach()
      net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
      mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)
      # print(u_bc[0:3,:].data)
      # print(net_bc_out[0:3,:].data)

      # Loss based on PDE
      x = (torch.rand(batch_size, 1, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
      t = (torch.rand(batch_size, 1, requires_grad=True) * (T_end - t0) + t0).to(device)
      all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
      res_out = res_func(x, t, p_net)
      mse_res = mse_cost_function(res_out/normalize, all_zeros)

      loss_ic = mse_u
      loss_r = (T_end-t0)*mse_res
      loss = loss_ic + loss_r
      loss_history.append(loss.item())

      # Save the min loss model
      if(loss.data < 0.9*min_loss):
          print("save epoch:", epoch, ", loss:", loss.item(), 'loss_ic:', loss_ic.item(), 'loss_r:', loss_r.item())
          torch.save({
                  'epoch': epoch,
                  'model_state_dict': p_net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss.item(),
                  'train_time': time.time() - start_time
                  }, PATH)
          min_loss = loss.item()

      loss.backward() # This is for computing gradients using backward propagation
      optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
      # Exponential learning rate decay
      if (epoch + 1) % iterations_per_decay == 0:
          scheduler.step()

# --------------------------
# Load best
# --------------------------
def load_train_model(net, PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best pnet epoch: ", epoch, ", loss:", loss, "train time:", checkpoint['train_time'])
    return net

# --------------------------
# Plotting
# --------------------------
def plot_p_surface(p_net, num=100):
    plt.rcParams.update({
    # General font settings
    "font.family": "serif",       # Use sans-serif font for non-math text
    "font.sans-serif": ["Times New Roman"],  # Prioritize Helvetica (must be installed on your system)
    "font.size": 18,                   # Base font size for non-math text
    # "figure.autolayout": True,

    # Math font settings
    "mathtext.fontset": "stix",        # STIX fonts for math symbols

    # Title and label sizes
    "axes.titlesize": 18,              # Title font size
    "axes.labelsize": 18,              # Axis label font size

    # Legend settings
    "legend.fontsize": 18,             # Legend text size
    "legend.title_fontsize": 18        # Legend title size (if you use legend titles)
    })

    t1s = [1.0, 1.5, 2.0, 2.5, 3.0]
    x = np.linspace(x_low, x_hig, num=num, dtype=np.float32)
    t = np.linspace(t0, T_end, num=num, dtype=np.float32)
    x_mesh, t_mesh = np.meshgrid(x,t)
    x_mesh_tensor = torch.from_numpy(x_mesh.reshape(-1,1)).to(device)
    t_mesh_tensor = torch.from_numpy(t_mesh.reshape(-1,1)).to(device)
    phat = p_net(x_mesh_tensor, t_mesh_tensor).detach().cpu().numpy().reshape(num, -1)

    p_list = []
    for t1 in t1s:
        p_true = p_exact(x, x*0+t1)
        p_list.append(p_true)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.8, label=r"$\hat{p}$")
    z_max = 1.5*np.max(np.abs(phat))
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x*0 + t1
        if i == 0:
            ax.plot(x, t1_monte, p_list[i], color="black", label=r"$p$")
        else:
            ax.plot(x, t1_monte, p_list[i], color="black")
    ax.view_init(20, -50)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$t$')
    ax.text2D(0.94, 0.77, "PDF", transform=ax.transAxes)
    ax.legend(loc='lower right', bbox_to_anchor=(0.32, 0.60))


def get_mu_cov_sol(t):
  mu = x0*np.exp(-alpha*t)
  cov = D*(1-np.exp(-2*alpha*t))/alpha
  return np.float32(mu), np.float32(cov)


def _get_Jacobian_expression():
    x1 = sp.symbols('x1', real=True)
    ALPHA = sp.symbols('ALPHA', real=True, positive=True)
    f1 = -ALPHA*x1
    f = sp.Matrix([f1])
    x = sp.Matrix([x1])

    # symbolic Jacobian
    J = sp.simplify(f.jacobian(x))
    print("[info] Jacobian matrix: ", J)


def get_Jacobian(x):
    J = -alpha
    return np.float32(J)


def ou_dyn(x):
  f1 = -alpha*x
  return f1


def linear_propagation(dt_precision=7, dt_save=1e-2, save_path=None):
    # --- initialization ---
    mu_i, cov_i = get_mu_cov_sol(t0)
    x = mu_i    # initial mean
    Px = cov_i     # initial covariance

    dtt = np.float64(10**(-1*dt_precision))
    tf = T_end
    ti = t0
    kf = int((tf-ti) / dtt)
    current_time = ti

    # --- storage arrays ---
    times = [current_time]
    means = [x.copy()]
    covs = [Px.copy()]
    t_to_save = current_time + dt_save

    # --- time stepping loop ---
    for k in tqdm(range(kf), desc="propagting over time"):
        # compute dynamics and Jacobian
        fx = ou_dyn(x)
        Jx = get_Jacobian(x)
        # propagate mean and covariance
        x = x + fx * dtt
        Px = Px + (Jx*Px + Jx*Px + 2*D)*dtt

        # update time
        current_time += dtt
        # current_time = np.round(current_time, dt_precision)
        if(abs(current_time-t_to_save) < dtt/2):
            # store results
            times.append(np.round(current_time,3))
            means.append(x.copy())
            covs.append(Px.copy())
            t_to_save += dt_save

    # --- convert lists to arrays ---
    times = np.array(times)
    means = np.array(means)       # shape: (kf+1, n)
    covs = np.array(covs)         # shape: (kf+1, n, n)
    np.savez(save_path,
             times=times,
             means=means,
             covs=covs,
             dt_precision=dt_precision)


def _unscented_sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0):
    """
    Generate Unscented Transform sigma points for N-dim state.

    Parameters
    ----------
    mean : (N,) array_like
        State mean vector.
    cov : (N,N) array_like
        State covariance (symmetric, PSD).
    alpha : float, optional
        Spread of the sigma set (small, e.g. 1e-3). Affects higher-order terms.
    beta : float, optional
        Prior knowledge about distribution; 2 is optimal for Gaussian.
    kappa : float, optional
        Secondary scaling (often 0 or 3-N).

    Returns
    -------
    X : (2N+1, N) ndarray
        Sigma points. X[0] is the mean, others are +/- columns of the scaled root.
    Wm : (2N+1,) ndarray
        Weights for computing the mean.
    Wc : (2N+1,) ndarray
        Weights for computing the covariance.

    """
    m = np.asarray(mean, dtype=float).reshape(-1)
    N = m.size
    P = np.asarray(cov, dtype=float).reshape(N, N)
    assert P.shape == (N, N), "cov must be (N,N)"

    lam = alpha**2 * (N + kappa) - N
    c = N + lam
    if c <= 0:
        raise ValueError("N + lambda must be positive; adjust alpha/kappa.")
    S = cholesky(P, lower=True)
    S *= np.sqrt(c)  # scale by sqrt(N+lambda)

    # Sigma points
    X = np.empty((2*N + 1, N), dtype=float)
    X[0] = m
    X[1:N+1]     = m + S.T   # columns of S
    X[N+1:2*N+1] = m - S.T

    # Weights
    Wm = np.full(2*N + 1, 1.0/(2.0*c), dtype=float)
    Wc = np.full(2*N + 1, 1.0/(2.0*c), dtype=float)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return X, Wm, Wc

def unscent_propagation(dt_precision=7, dt_save=1e-2, save_path=None):
    # --- initialization ---
    mu_i, cov_i = get_mu_cov_sol(t0)
    x = mu_i    # initial mean
    Px = cov_i     # initial covariance

    dtt = np.float64(10**(-1*dt_precision))
    tf = T_end
    ti = t0
    kf = int((tf-ti) / dtt)
    current_time = ti

    # --- storage arrays ---
    times = [current_time]
    means = [x.copy()]
    covs = [Px.copy()]
    t_to_save = current_time + dt_save

    # --- time stepping loop ---
    for k in tqdm(range(kf), desc="propagting over time"):
        # compute sigma points
        _sigma_pts, _w_mean, _w_cov = _unscented_sigma_points(x, Px)

        for i in range(_sigma_pts.shape[0]):
            _x_i = _sigma_pts[i, :]
            fx_i = ou_dyn(_x_i)
            _sigma_pts[i, :] = _x_i + fx_i * dtt

        x = _w_mean @ _sigma_pts
        X_diff = _sigma_pts - x
        Px = X_diff.T @ (_w_cov[:, None] * X_diff) + (2*D)*dtt

        # update time
        current_time += dtt
        current_time = np.round(current_time, dt_precision)
        if(abs(current_time-t_to_save) < dtt/2):
            # store results
            times.append(current_time)
            means.append(x.item())
            covs.append(Px.item())
            t_to_save += dt_save

    # --- convert lists to arrays ---
    times = np.array(times)
    means = np.array(means)       # shape: (kf+1, n)
    covs = np.array(covs)         # shape: (kf+1, n, n)
    np.savez(save_path,
             times=times,
             means=means,
             covs=covs,
             dt_precision=dt_precision)


class PropagationData:
    def __init__(self, path: str):
        self.data = np.load(path)
    def get(self, time):
        times = self.data["times"]
        means = self.data["means"]
        covs = self.data["covs"]
        # dt_precision = self.data["dt_precision"]
        # time_threshold = 10. *10**(-1. *dt_precision)
        time = np.round(time, 3)
        idx = np.where(abs(time-times)< 1e-3)[0]
        if len(idx) == 0:
            assert("The linear propagation result does not have data at this time")
        idx = idx[0]
        return times[idx], means[idx], covs[idx]


def p_total_variation(p1, p2):
    vol_est = x_hig - x_low
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    tv = 0.5 * np.mean(np.abs(p1-p2)) * vol_est
    return 100.*np.clip(tv, 0., 1.)


def p_rel_worst_error(p1, p2):
    p1 = p1.reshape(-1,)
    p2 = p2.reshape(-1,)
    # print(np.max(p1), np.max(p2))
    rel_error = np.max(np.abs(p1-p2)) / np.max(p2)
    return 100.*rel_error.item()


def run_baseline(lp_path, ut_path):
  # NOTE float64 precision is used due to ensure small error
  linear_propagation(save_path=lp_path, dt_precision=6)
  unscent_propagation(save_path=ut_path, dt_precision=6)


# --------------------------
# Main
# --------------------------
def main(TRAIN_FLAG=False, RUN_BASELINE=False):
    p_net = Net(p_init).to(device)
    p_net.apply(init_weights)

    # e1_net = E1Net().to(device)
    # e1_net.apply(init_weights)

    p_net.scale = get_p_normalize()
    PATH = os.path.join(folder_path, "p_net.pt")
    if(TRAIN_FLAG):
      train_p_net(p_net, PATH); print("p_net train complete")

    p_net = load_train_model(p_net, PATH)
    plot_p_surface(p_net)

    _get_Jacobian_expression()
    lp_path = os.path.join(folder_path, "lp_np64_dt6.npz") # the last label specifies the precision used
    ut_path = os.path.join(folder_path, "ut_np64_dt6.npz")
    if(RUN_BASELINE):
        run_baseline(lp_path, ut_path)

    data_lp = PropagationData(lp_path)
    data_ut = PropagationData(ut_path)
    print(data_lp.data["times"], data_ut.data["times"])

    # --- Visual ---
    x = np.linspace(x_low, x_hig, num=300, endpoint=True, dtype=np.float32)
    x_tensor = torch.from_numpy(x.reshape(-1,1)).to(device)
    t_span = data_lp.data["times"].astype(x.dtype)
    idx_show = 0

    metrics = {
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_pinn": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_pinn": [],
        "t": t_span
    }

    colors = sns.color_palette("husl", 3)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for t in t_span:
        _, _mu_lp, _cov_lp = data_lp.get(t)
        pdf_func = multivariate_normal(mean=_mu_lp, cov=_cov_lp)
        pdf_lp = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)

        _, _mu_ut, _cov_ut = data_ut.get(t)
        pdf_func = multivariate_normal(mean=_mu_ut, cov=_cov_ut)
        pdf_ut = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)

        pdf_sol = p_exact(x, t)

        _t = np.ones(x.shape[0], dtype=x.dtype)*t
        t_tensor = torch.from_numpy(_t.reshape(-1,1)).to(device)
        pdf_pinn = p_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_sol.shape)

        rel_error_lp = p_rel_worst_error(pdf_lp, pdf_sol)
        tv_lp = p_total_variation(pdf_lp, pdf_sol)
        rel_error_ut = p_rel_worst_error(pdf_ut, pdf_sol)
        tv_ut = p_total_variation(pdf_ut, pdf_sol)
        rel_error_pinn = p_rel_worst_error(pdf_pinn, pdf_sol)
        tv_pinn = p_total_variation(pdf_pinn, pdf_sol)
        if(idx_show % 25 == 0):
            print("time {:.2f} total variation,  PINN: {:.3f} %,  LP: {:.3f} %,  UT: {:.3f} %".format(
            t, tv_pinn, tv_lp, tv_ut
            ))
            print("time {:.2f} worst rel. error, PINN: {:.3f} %,  LP: {:.3f} %,  UT: {:.3f} %".format(
                t, rel_error_pinn, rel_error_lp, rel_error_ut
            ))
            ax.plot(np.full_like(x, t), x, pdf_sol,
                    color="black", linestyle="-")
            ax.plot(np.full_like(x, t), x, pdf_pinn,
                    color=colors[0], linestyle="--")
            ax.plot(np.full_like(x, t), x, pdf_lp,
                    color=colors[1], linestyle="--")
            ax.plot(np.full_like(x, t), x, pdf_ut,
                    color=colors[2], linestyle=":", lw=3)
        metrics["rel_error_lp"].append(rel_error_lp)
        metrics["rel_error_ut"].append(rel_error_ut)
        metrics["rel_error_pinn"].append(rel_error_pinn)
        metrics["tv_lp"].append(tv_lp)
        metrics["tv_ut"].append(tv_ut)
        metrics["tv_pinn"].append(tv_pinn)
        idx_show += 1
            
    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", label=r"$p$ MC"),
        Line2D([0], [0], color=colors[0], linestyle="--", label=r"$\hat{p}$ PINN"),
        Line2D([0], [0], color=colors[1], linestyle="--",  label=r"$p$ Linear Prop."),
        Line2D([0], [0], color=colors[2], linestyle=":", lw=3, label=r"$p$ Unscent Trans.")
    ]
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend(handles=legend_elements, loc="best", frameon=True)
    plt.title("1D OU Process")

    plt.figure()
    plt.plot(metrics["t"], metrics["rel_error_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["rel_error_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["rel_error_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("worst rel. error %")

    plt.figure()
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("total variation %")

    plt.show()



if __name__ == "__main__":
    main(TRAIN_FLAG=False,
         RUN_BASELINE=False)

# from scipy.stats import multivariate_normal
# def test_p_sol(t):
#   x = np.linspace(x_low, x_hig, num=200, dtype=np.float32)
#   mu, cov = get_mu_cov_sol(t)
#   print(t, mu, cov)
#   pdf_func = multivariate_normal(mean=mu, cov=cov)
#   p_check = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
#   p_sol = p_exact(x, t)
#   plt.figure()
#   plt.plot(x, p_sol)
#   plt.plot(x, p_check, linestyle="--")
#   plt.show()
# test_p_sol(3.)