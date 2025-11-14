import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pinn_model import E1Net
from test_gmmpnet import TimeToGMM1D
import sys
sys.path.insert(0, '../utilities/')
from _General.util import set_publication_plot_style, custom_save_plot, apply_default_locators

# --------------------------
# Device & dtype
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

const_mu = -2.0
const_std = 0.5
const_a = -0.1
const_b = 0.1
const_c = 0.5
const_d = 0.5
const_e = 0.8
x_low = -6
x_hig = 6
t0 = 0.0
T_end = 5.0
t1s = np.arange(0.0, 5.0 + 0.5, 0.5)
# t1s = np.arange(0.0, 5.0 + 1.0, 1.0)

colors_4set = sns.color_palette([
    "#FF008C",  # GA
    "#00FBFF",  # UT
    "#FF8400",  # GMM
    "#000000",  # PINN-GMM
])

# 7 high-contrast linestyle/marker pairs (index-bound)
linestyles_4set = [
    (0, (5, 2)),        # GA
    (0, (7, 2, 3, 2)),  # UT
    "--",               # GMM
    "-",                # PINN-GMM
]
markers_4set = ['o', # GA
                's', # UT
                '^', # GMM
                'None', # PINN-GMM
                ] 


def p_init(x):
    """
    taks numpy.array or torch.tensor as inputs
    """
    if isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        exponent = -0.5 * ((x - const_mu_tensor) / const_std_tensor) ** 2
        normalization = const_std_tensor * torch.sqrt(torch.tensor(2 * torch.pi))
        return torch.exp(exponent) / normalization
    elif isinstance(x, np.ndarray):
        return np.exp(-0.5*((x-const_mu)/const_std)**2) / (const_std*np.sqrt(2*np.pi))
    else:
        return "Neither PyTorch Tensor nor NumPy Array"


def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def get_e1_normalize(pnet):
    x = np.linspace(x_low, x_hig, num=200, endpoint=True).reshape(-1,1)
    p0_true = p_init(x)
    pt_x = torch.from_numpy(x).float()
    pt_t = pt_x*0.0 + t0
    p0_hat = pnet(pt_x, pt_t).data.cpu().numpy()
    e0_true = p0_true - p0_hat
    return np.max(np.abs(e0_true))


def load_train_model(net, PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    # print("load pinn from: ", PATH)
    # print("best epoch: ", epoch, ", loss:", loss, "train time:", checkpoint['train_time'])
    net.eval()
    return net
       

def plot_error_at_time_points():
    """
    LOAD_PRIOR (True): the best from prior work
    Base: the baseline MLP model
    Normal: PINN-Normal model
    """
    torch.manual_seed(0); np.random.seed(0)

    p_net = TimeToGMM1D().to(device)
    p_net = load_train_model(p_net, PATH="data/p_net(pinn-gmm).pth")

    e1_net = E1Net(scale=get_e1_normalize(p_net),
                   normalize=get_e1_normalize(p_net),
                   ).to(device)
    e1_net = load_train_model(e1_net, PATH="data/e1_net(pinn-gmm).pth")

    # --- Evaluation ---
    p_net.eval(); e1_net.eval()

    # --- grid in x ---
    x = np.load("data/xsim.npy").astype(np.float32)
    x_tensor = torch.from_numpy(x.reshape(-1, 1)).to(device)

    # choose 3 time points you want to show
    # (just make sure you have corresponding files xsamples_t{:.1f}.npy, psim_t{:.1f}.npy)
    t_plot = [1.0, 3.0, 5.0]   # change if you want other times

    set_publication_plot_style(save_tight_pad=0.3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    colors = sns.color_palette("husl", 4)

    gap = 6  # subsample for e_true markers so the plot isn't too dense

    for i, t in enumerate(t_plot):
        ax = axes[i]

        # load MC reference samples and PDF at this time
        x_mc_samples = np.load(f"data/xsamples_t{t:.1f}.npy").astype(np.float32)
        x_mc_samples = x_mc_samples[:1_000_000]

        pdf_mc = np.load(f"data/psim_t{t:.1f}.npy").astype(np.float32).reshape(-1,)
        if np.isclose(t, 0.0, atol=1e-8):
            pdf_mc = p_init(x).reshape(-1,)

        _t = np.full_like(x, t, dtype=x.dtype)
        t_tensor = torch.from_numpy(_t.reshape(-1, 1)).to(device)

        # --- PINN PDF + error ---
        pdf_pinn_grid = p_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_mc.shape)
        e1_true = pdf_mc - pdf_pinn_grid
        e1_pinn_grid = e1_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_mc.shape)

        # --- plot on this subplot ---
        # true error (subsampled, with markers)
        ax.plot(x[::gap], e1_true[::gap],
                color="#0066FF", linestyle="-", marker="o", markersize=3,
                label=r"$e(x,t)$")

        # PINN error approximation (full line)
        ax.plot(x, e1_pinn_grid,
                color=colors_4set[3], linestyle=linestyles_4set[3],
                label=r"$\hat e(x,t)$")

        ax.set_title(rf"$t = {t:.1f}$", pad=4)
        ax.set_xlabel(r"$x$")

        if i == 0:
            ax.set_ylabel(r"Error")
            ax.legend(loc="upper right")

        ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)

    # single shared legend
    custom_save_plot(True, "figs/errors.pdf")


def plot_error_bound_at_time_points():
    torch.manual_seed(0); np.random.seed(0)

    p_net = TimeToGMM1D().to(device)
    p_net = load_train_model(p_net, PATH="data/p_net(pinn-gmm).pth")

    e1_net = E1Net(scale=get_e1_normalize(p_net),
                   normalize=get_e1_normalize(p_net),
                   ).to(device)
    e1_net = load_train_model(e1_net, PATH="data/e1_net(pinn-gmm).pth")

    # --- Evaluation ---
    p_net.eval(); e1_net.eval()

    # --- grid in x ---
    x = np.load("data/xsim.npy").astype(np.float32)
    x_tensor = torch.from_numpy(x.reshape(-1, 1)).to(device)

    # choose 3 time points you want to show
    # (just make sure you have corresponding files xsamples_t{:.1f}.npy, psim_t{:.1f}.npy)
    t_plot = [1.0, 3.0, 5.0]   # change if you want other times

    set_publication_plot_style(save_tight_pad=0.3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    colors = sns.color_palette("husl", 4)

    gap = 1  # subsample for e_true markers so the plot isn't too dense

    for i, t in enumerate(t_plot):
        ax = axes[i]

        # load MC reference samples and PDF at this time
        x_mc_samples = np.load(f"data/xsamples_t{t:.1f}.npy").astype(np.float32)
        x_mc_samples = x_mc_samples[:1_000_000]

        pdf_mc = np.load(f"data/psim_t{t:.1f}.npy").astype(np.float32).reshape(-1,)
        if np.isclose(t, 0.0, atol=1e-8):
            pdf_mc = p_init(x).reshape(-1,)

        _t = np.full_like(x, t, dtype=x.dtype)
        t_tensor = torch.from_numpy(_t.reshape(-1, 1)).to(device)

        # --- PINN PDF + error ---
        pdf_pinn_grid = p_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_mc.shape)
        e1_true = pdf_mc - pdf_pinn_grid
        e1_pinn_grid = e1_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_mc.shape)
        B1 = np.max(np.abs(e1_pinn_grid)).item() * 2.0
        print(B1)

        # --- plot on this subplot ---
        # true error (subsampled, with markers)
        ax.plot(x[::gap], pdf_mc[::gap],
                color="#0066FF", linestyle="-", markersize=3,
                label=r"$p(x,t)$")
        
        ax.fill_between(x[::gap], 
                pdf_pinn_grid[::gap] - B1,  # Lower bound
                pdf_pinn_grid[::gap] + B1,  # Upper bound
                color="black", 
                alpha=0.3,
                label=r"$\hat p \pm B_1(t)$")

        # # PINN error approximation (full line)
        # ax.plot(x, e1_pinn_grid,
        #         color=colors_4set[3], linestyle=linestyles_4set[3],
        #         label=r"PINN-GMM $\hat e(x,t)$")

        ax.set_title(rf"$t = {t:.1f}$", pad=4)
        ax.set_xlabel(r"$x$")

        if i == 0:
            ax.set_ylabel(r"PDF")
            ax.legend(loc="upper right")

        # ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)

    # single shared legend
    custom_save_plot(True, "figs/error_bounds.pdf")


def main():
    plot_error_at_time_points()
    plot_error_bound_at_time_points()
    plt.show()


if __name__ == "__main__":
    main()