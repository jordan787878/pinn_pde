import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sympy as sp
from tqdm import tqdm
import os
from scipy.stats import multivariate_normal
from scipy.linalg import expm, cholesky
import seaborn as sns

# --------------------------
# Device & dtype
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

mu = -2.0
std = 0.5
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


def p_init(x):
    return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


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


class PNet(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(PNet, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = F.softplus((self.hidden_layer1(inputs)))
        layer2_out = F.softplus((self.hidden_layer2(layer1_out)))
        layer3_out = F.softplus((self.hidden_layer3(layer2_out)))
        output = F.softplus( self.output_layer(layer3_out) )
        return output
    

class E1Net(nn.Module):
    def __init__(self, scale=1.0): 
        neurons = 50
        self.scale = scale
        super(E1Net, self).__init__()
        self.hidden_layer1 = (nn.Linear(2,neurons))
        self.hidden_layer2 = (nn.Linear(neurons,neurons))
        self.hidden_layer3 = (nn.Linear(neurons,neurons))
        self.hidden_layer4 = (nn.Linear(neurons,neurons))
        self.hidden_layer5 = (nn.Linear(neurons,neurons))
        self.hidden_layer6 = (nn.Linear(neurons,neurons))
        self.output_layer =  (nn.Linear(neurons,1))
        self.activation = nn.GELU()
    def forward(self, x, t):
        inputs = torch.cat([x, t],axis=1)
        layer1_out = self.activation((self.hidden_layer1(inputs)))
        layer2_out = self.activation((self.hidden_layer2(layer1_out)))
        layer3_out = self.activation((self.hidden_layer3(layer2_out)))
        layer4_out = self.activation((self.hidden_layer4(layer3_out)))
        layer5_out = self.activation((self.hidden_layer5(layer4_out)))
        layer6_out = self.activation((self.hidden_layer6(layer5_out)))
        output = self.output_layer(layer6_out)
        output = self.scale * output
        return output


def load_train_model(net, PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("best pnet epoch: ", epoch, ", loss:", loss, "train time:", checkpoint['train_time'])
    net.eval()
    return net


class PropagationData:
    def __init__(self, path=None, prop_method=None):
        if(prop_method == "LP"):
            self.data = None
            # self._get_Jacobian_expression()
            self.linear_propagation(save_path=path, dt_precision=6)
            self.data = np.load(path)
        if(prop_method == "UT"):
            self.data = None
            # self._get_Jacobian_expression()
            self.unscent_propagation(save_path=path, dt_precision=6)
            self.data = np.load(path)
        else:
            self.data = np.load(path)

    def get(self, time):
        if(self.data is None):
            raise("data has not been loaded") 
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
    
    def _get_Jacobian_expression(self):
        x1 = sp.symbols('x1', real=True)
        A, B, C, D = sp.symbols('A B C D' , real=True, positive=True)
        f1 = A*x1**3 + B*x1**2 + C*x1 + D
        f = sp.Matrix([f1])
        x = sp.Matrix([x1])
        # symbolic Jacobian
        J = sp.simplify(f.jacobian(x))
        print("[info] Jacobian matrix: ", J)

    def get_Jacobian(self, x):
        x1 = x
        J = 3*const_a*x1**2 + 2*const_b*x1 + const_c
        return np.float32(J)
    
    def nl_dyn(self, x):
        x1 = x
        A, B, C, D = const_a, const_b, const_c, const_d
        f1 = A*x1**3 + B*x1**2 + C*x1 + D
        return f1
    
    def linear_propagation(self, dt_precision=6, dt_save=0.1, save_path=None):
        # --- initialization ---
        mu_i, cov_i = mu, std**2
        x = np.float64(mu_i)    # initial mean
        Px = np.float64(cov_i)     # initial covariance

        dtt = np.float64(10**(-1*dt_precision))
        tf = T_end
        ti = t0
        kf = int(np.ceil((tf-ti) / dtt))
        print(kf)
        current_time = ti

        # --- storage arrays ---
        times = [current_time]
        means = [x]
        covs = [Px]
        t_to_save = current_time + dt_save

        # --- time stepping loop ---
        for k in tqdm(range(kf), desc="propagting over time"):
            # compute dynamics and Jacobian
            fx = self.nl_dyn(x)
            Jx = self.get_Jacobian(x)
            # propagate mean and covariance
            x = x + fx * dtt
            Px = Px + (Jx*Px + Jx*Px + const_e**2)*dtt

            # update time
            current_time += dtt
            # current_time = np.round(current_time, dt_precision)
            if(abs(current_time-t_to_save) < dtt/2):
                # store results
                times.append(np.round(current_time,2))
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
        
    def _unscented_sigma_points(self, mean, cov, alpha=1.0, beta=2.0, kappa=0.0):
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
    
    def unscent_propagation(self, dt_precision=6, dt_save=0.1, save_path=None):
        # --- initialization ---
        mu_i, cov_i = mu, std**2
        x = np.float64(mu_i)    # initial mean
        Px = np.float64(cov_i)     # initial covariance

        dtt = np.float64(10**(-1*dt_precision))
        tf = T_end
        ti = t0
        kf = int(np.ceil((tf-ti) / dtt))
        print(kf)
        current_time = ti

        # --- storage arrays ---
        times = [current_time]
        means = [x.copy()]
        covs = [Px.copy()]
        t_to_save = current_time + dt_save

        # --- time stepping loop ---
        for k in tqdm(range(kf), desc="propagting over time"):
            # compute sigma points
            _sigma_pts, _w_mean, _w_cov = self._unscented_sigma_points(x, Px)
            for i in range(_sigma_pts.shape[0]):
                _x_i = _sigma_pts[i, :]
                fx_i = self.nl_dyn(_x_i)
                _sigma_pts[i, :] = _x_i + fx_i * dtt

            x = _w_mean @ _sigma_pts
            X_diff = _sigma_pts - x
            Px = X_diff.T @ (_w_cov[:, None] * X_diff) + (const_e**2)*dtt
            # x = (_w_mean[:, None] * _sigma_pts).sum(axis=0)
            # diffs = _sigma_pts - x
            # Px = diffs.T @ (diffs * _w_cov[:, None]) + (const_e**2)*dtt

            # update time
            current_time += dtt
            current_time = np.round(current_time, dt_precision)
            if(abs(current_time-t_to_save) < dtt/2):
                # store results
                times.append(np.round(current_time,3))
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

    def indices_in_range(self, x, x_range):
        a, b = x_range
        start_idx = np.searchsorted(x, a, side="left")
        end_idx = np.searchsorted(x, b, side="right")

        # If everything is inside the range, return all indices
        if start_idx == 0 and end_idx == len(x):
            return np.arange(len(x))

        # If the range is completely outside x, return empty array
        if start_idx >= end_idx:
            return np.array([], dtype=int)

        return np.arange(start_idx, end_idx)


def empirical_moments(x, p, N=1):
    # p = p/np.sum(p)
    dx = x[1]-x[0]
    mean = np.sum(p*x)*dx
    if(N == 1):
        return mean
    else:
        return np.sum(p*(x-mean)**2)*dx


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


def p_rel_KL(p_at_x):
    return np.mean(-np.log(p_at_x))


def test_MC_accuracy(x):
    pdf_true = p_init(x)
    pdf_mc = np.load("data/psim_t0.0.npy")
    rel_error = p_rel_worst_error(pdf_mc, pdf_true)
    tv = p_total_variation(pdf_mc, pdf_true)
    print("[test] Validate MC at t0 --- rel. error {:.5f} %, tv {:.5f} %".format(
        rel_error, tv))


def main(TRAIN_FLAG=False, RUN_BASELINE=False):
    p_net = PNet().to(device)
    p_net.scale = get_p_normalize()
    p_net = load_train_model(p_net, PATH="data/p_net.pth")

    e1_net = E1Net().to(device)
    e1_net.scale = get_e1_normalize(p_net)
    e1_net = load_train_model(e1_net, PATH="data/e1_net.pth")

    # --- Visual ---
    x = np.load("data/xsim.npy").astype(np.float32)

    test_MC_accuracy(x)

    lp_path = os.path.join("data", "lp_np64_dt6.npz") # the last label specifies the precision used
    ut_path = os.path.join("data", "ut_np64_dt6.npz") # the last label specifies the precision used
    ut_path_alpha0_1 = os.path.join("data", "ut_np64_dt6_alpha0.1.npz") # the last label specifies the precision used
    if(RUN_BASELINE):
        data_lp = PropagationData(path=lp_path, prop_method="LP")
        data_ut = PropagationData(path=ut_path, prop_method="UT")
        data_ut_alpha0_1 = PropagationData(path=ut_path_alpha0_1, prop_method="UT")
    data_lp = PropagationData(path=lp_path)
    data_ut = PropagationData(path=ut_path)
    data_ut_alpha0_1 = PropagationData(path=ut_path_alpha0_1)
    print(data_ut_alpha0_1.data["times"])

    colors = sns.color_palette("husl", 3)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    x_tensor = torch.from_numpy(x.reshape(-1,1)).to(device)
    t_span = np.array(t1s).astype(np.float32)
    metrics = {
        "rel_error_pinn": [],
        "rel_error_lp": [],
        "rel_error_ut": [],
        "rel_error_ut_alpha_0_1": [],
        "tv_pinn": [],
        "tv_lp": [],
        "tv_ut": [],
        "tv_ut_alpha_0_1": [],
        "normalize_B1": [],
        "t": t_span
    }
    x_mc_samples = None
    for t in t_span:
        x_mc_samples = np.load("data/xsamples_t{:.1f}.npy".format(t)).astype(np.float32)
        pdf_mc = np.load("data/psim_t{:.1f}.npy".format(t)).astype(np.float32).reshape(-1,)
        _t = np.ones(x.shape[0], dtype=x.dtype)*t
        t_tensor = torch.from_numpy(_t.reshape(-1,1)).to(device)

        _, _mu_lp, _cov_lp = data_lp.get(t)
        pdf_func = multivariate_normal(mean=_mu_lp, cov=_cov_lp)
        pdf_lp = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
        # compute coverage mass
        idx_in_one_std = data_lp.indices_in_range(x, np.array([_mu_lp-(_cov_lp)**0.5, _mu_lp+(_cov_lp)**0.5]))
        if(len(idx_in_one_std) > 0):
            x_in_one_std = x[idx_in_one_std]
            pdf_mass_lp = 100. *np.mean(pdf_mc[idx_in_one_std]) * (x_in_one_std[-1] - x_in_one_std[0])
        else:
            pdf_mass_lp = 0.0
        if(x_mc_samples is not None):
            _p_at_x_lp = pdf_func.pdf(x_mc_samples).reshape(-1,)
            rel_kl_lp = p_rel_KL(_p_at_x_lp)

        _, _mu_ut, _cov_ut = data_ut.get(t)
        pdf_func = multivariate_normal(mean=_mu_ut, cov=_cov_ut)
        pdf_ut = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)
        # compute coverage mass
        idx_in_one_std = data_ut.indices_in_range(x, np.array([_mu_ut-(_cov_ut)**0.5, _mu_ut+(_cov_ut)**0.5]))
        if(len(idx_in_one_std) > 0):
            x_in_one_std = x[idx_in_one_std]
            pdf_mass_ut = 100. * np.mean(pdf_mc[idx_in_one_std]) * (x_in_one_std[-1] - x_in_one_std[0])
        else:
            pdf_mass_ut = 0.0
        if(x_mc_samples is not None):
            _p_at_x_ut = pdf_func.pdf(x_mc_samples).reshape(-1,)
            rel_kl_ut = p_rel_KL(_p_at_x_ut)

        _, _mu_ut_alpha0_1, _cov_ut_alpha0_1 = data_ut_alpha0_1.get(t)
        pdf_func = multivariate_normal(mean=_mu_ut_alpha0_1, cov=_cov_ut_alpha0_1)
        pdf_ut_alpha0_1 = pdf_func.pdf(x).reshape(-1,).astype(x.dtype)

        pdf_pinn = p_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(pdf_mc.shape)
        if(x_mc_samples is not None):
            x_mc_samples_tensor = torch.from_numpy(x_mc_samples.reshape(-1,1)).to(device)
            _t_mc_samples = np.ones(x_mc_samples_tensor.shape[0], dtype=x.dtype)*t
            t_mc_samples_tensor = torch.from_numpy(_t_mc_samples.reshape(-1,1)).to(device)
            _p_at_x_pinn = p_net(x_mc_samples_tensor, t_mc_samples_tensor).detach().cpu().numpy().reshape(-1,)
            rel_kl_pinn = p_rel_KL(_p_at_x_pinn)

        e1_pinn = e1_net(x_tensor, t_tensor).detach().cpu().numpy().reshape(-1,)
        B1 = 2.* np.max(np.abs(e1_pinn)).item()
        p_true_max = np.max(pdf_mc).item()
        normalize_B1 = 100.*B1/p_true_max

        rel_error_pinn = p_rel_worst_error(pdf_pinn, pdf_mc)
        tv_pinn = p_total_variation(pdf_pinn, pdf_mc)
        rel_error_lp = p_rel_worst_error(pdf_lp, pdf_mc)
        tv_lp = p_total_variation(pdf_lp, pdf_mc)
        rel_error_ut = p_rel_worst_error(pdf_ut, pdf_mc)
        tv_ut = p_total_variation(pdf_ut, pdf_mc)
        rel_error_ut_alpha_0_1 = p_rel_worst_error(pdf_ut_alpha0_1, pdf_mc)
        tv_ut_alpha_0_1 = p_total_variation(pdf_ut_alpha0_1, pdf_mc)

        print("time {:.2f} total variation , PINN: {:.5f} %,  LP: {:.5f} %,  UT: {:.5f} %,  UT (alpha=0.1): {:.5f} %".format(
            t, tv_pinn, tv_lp, tv_ut, tv_ut_alpha_0_1
        ))
        print("time {:.2f} worst rel. error, PINN: {:.5f} %,  LP: {:.5f} %,  UT: {:.5f} %,  UT (alpha=0.1): {:.5f} %".format(
            t, rel_error_pinn, rel_error_lp, rel_error_ut, rel_error_ut_alpha_0_1
        ))
        if(x_mc_samples is not None):
            print("time {:.2f} rel. KL, PINN: {:.5f},  LP: {:.5f},  UT: {:.5f}".format(
                t, rel_kl_pinn, rel_kl_lp, rel_kl_ut
            ))

        metrics["rel_error_pinn"].append(rel_error_pinn)
        metrics["rel_error_lp"].append(rel_error_lp)
        metrics["rel_error_ut"].append(rel_error_ut)
        metrics["rel_error_ut_alpha_0_1"].append(rel_error_ut_alpha_0_1)

        metrics["tv_pinn"].append(tv_pinn)
        metrics["tv_lp"].append(tv_lp)
        metrics["tv_ut"].append(tv_ut)
        metrics["tv_ut_alpha_0_1"].append(tv_ut_alpha_0_1)

        metrics["normalize_B1"].append(normalize_B1)

        mean_mc = empirical_moments(x, pdf_mc)
        cov_mc = empirical_moments(x, pdf_mc, N=2)
        print("Mean, MC: {:.3f}, LP: {:.3f}, UT: {:.3f}".format(
            mean_mc, _mu_lp, _mu_ut
        ))
        print("Cov , MC: {:.3f}, LP: {:.3f}, UT: {:.3f}".format(
            cov_mc, _cov_lp, _cov_ut
        ))
        print("probability mass within +/- 1 std, LP: {:.3f} %, UT: {:.3f} %".format(
            pdf_mass_lp, pdf_mass_ut))

        ax.plot(np.full_like(x, t), x, pdf_mc,
                color="black", linestyle="-")
        ax.plot(np.full_like(x, t), x, pdf_pinn,
                color=colors[0], linestyle="--")
        ax.plot(np.full_like(x, t), x, pdf_lp,
                color=colors[1], linestyle="--")
        ax.plot(np.full_like(x, t), x, pdf_ut,
                color=colors[2], lw=3, linestyle=":")

    legend_elements = [
        Line2D([0], [0], color="black", linestyle="-", label=r"$p$ MC"),
        Line2D([0], [0], color=colors[0], linestyle="--", label=r"$\hat{p}$ PINN"),
        Line2D([0], [0], color=colors[1], linestyle="--",  label=r"$p$ Linear Prop."),
        Line2D([0], [0], color=colors[2], linestyle=":", lw=3, label=r"$p$ Unscent Trans.")
    ]
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend(handles=legend_elements, loc="best", frameon=True)

    plt.figure()
    plt.plot(metrics["t"], metrics["rel_error_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    # plt.plot(metrics["t"], metrics["normalize_B1"], color=colors[0], label=r"$Error Bound$ PINN")
    plt.fill_between(
        metrics["t"],
        metrics["rel_error_pinn"],
        metrics["normalize_B1"],
        color=colors[0],
        alpha=0.2,
        label="PINN Error Bound"
    )
    
    plt.plot(metrics["t"], metrics["rel_error_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["rel_error_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["rel_error_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("worst rel. error %")

    plt.figure()
    plt.plot(metrics["t"], metrics["tv_pinn"], color=colors[0], label=r"$\hat{p}$ PINN")
    plt.plot(metrics["t"], metrics["tv_lp"], color=colors[1],   label=r"$p$ Linear Prop.")
    plt.plot(metrics["t"], metrics["tv_ut"], color=colors[2],   label=r"$p$ Unscent Trans.")
    plt.plot(metrics["t"], metrics["tv_ut_alpha_0_1"], color=colors[2], marker="o", label=r"$p$ Unscent Trans. $(\alpha=0.1)$")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("total variation %")

    plt.show()


if __name__ == "__main__":
    main(TRAIN_FLAG=False, 
         RUN_BASELINE=False)