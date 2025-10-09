# hohmann_planar.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp

# -----------------------------
# Constants (km, s, km^3/s^2)
# -----------------------------
MU_EARTH = 398600.4418     # Earth's GM
R_EARTH  = 6378.137        # Earth mean equatorial radius

# -----------------------------
# Two-body planar dynamics
# y = [x, y, vx, vy]
# -----------------------------

def two_body_ode(t, y, mu):
    x, y_, vx, vy = y
    r3 = (x*x + y_*y_)**1.5
    ax = -mu * x / r3
    ay = -mu * y_ / r3
    return [vx, vy, ax, ay]


# --- Nondimensionalization (component form for [x, y, vx, vy]) ----------------
# Scales (choose DU = a, the Hohmann semi-major axis):
#   DU = a                      # distance unit
#   TU = sqrt(a^3 / μ)          # time unit
#   VU = DU / TU = sqrt(μ / a)  # velocity unit
#
# Component-wise conversions:
#   x~  = x  / DU        y~  = y  / DU        τ = t / TU
#   vx~ = vx / VU        vy~ = vy / VU
#   x  = x~ * DU         y  = y~ * DU         t = τ * TU
#   vx = vx~ * VU        vy = vy~ * VU
#
# Nondimensional dynamics (μ-free):
#   r~ = sqrt(x~^2 + y~^2)
#   dx~/dτ  = vx~
#   dy~/dτ  = vy~
#   dvx~/dτ = - x~ / (r~^3)
#   dvy~/dτ = - y~ / (r~^3)
#
# Notes:
# - With DU = a, the Hohmann half-transfer time is exactly τ = π.
# - Δv in ND → physical via Δv = Δv~ * VU.
# - For covariance on [x,y,vx,vy], scale with
#     S = diag(1/DU, 1/DU, 1/VU, 1/VU),   C~ = S @ C @ S.T
#   (equivalently: σ_x~ = σ_x/DU, σ_y~ = σ_y/DU, σ_vx~ = σ_vx/VU, σ_vy~ = σ_vy/VU)
# ------------------------------------------------------------------------------ 

def _scales_from_a(mu: float, a: float):
    """Return canonical scales for DU/TU/VU with DU=a."""
    DU = a
    TU = np.sqrt(a**3 / mu)
    VU = np.sqrt(mu / a)
    return DU, TU, VU


def two_body_ode_nd(tau, y_nd):
    """μ-free nondimensional two-body ODE: r'' = -r / ||r||^3."""
    x, y, vx, vy = y_nd
    r3 = (x*x + y*y)**1.5
    ax = -x / r3
    ay = -y / r3
    return [vx, vy, ax, ay]


def hohmann_planar(r1=20000.0, r2=384400.0, mu=MU_EARTH, t_fac=1.0, nondim: bool = False):
    """
    Returns trajectory of the Hohmann transfer from r1 to r2 (both circular about Earth).
    - If nondim=True: integrates the μ-free ODE in canonical units (DU=a), then converts back.
    - Outputs are always in physical units (km, km/s, s).
    """
    # Geometry
    a = 0.5 * (r1 + r2)                      # semimajor axis (km)
    T_transfer = np.pi * np.sqrt(a**3 / mu) * t_fac  # half-period (s)
    print("Time of transfer: ", int(T_transfer/t_fac), " sec")

    # Physical speeds (km/s)
    v_circ_1 = np.sqrt(mu / r1)
    v_circ_2 = np.sqrt(mu / r2)
    v_perigee = np.sqrt(mu * (2.0/r1 - 1.0/a))
    v_apogee  = np.sqrt(mu * (2.0/r2 - 1.0/a))
    dv1 = v_perigee - v_circ_1
    dv2 = v_circ_2 - v_apogee

    if not nondim:
        # Physical integration (original path)
        y0 = np.array([r1, 0.0, 0.0, v_perigee], dtype=float)
        t_span = (0.0, T_transfer)
        t_eval = np.linspace(0.0, T_transfer, 3000)
        sol = solve_ivp(two_body_ode, t_span, y0, args=(mu,),
                        t_eval=t_eval, rtol=1e-9, atol=1e-9, max_step=T_transfer/2000)
        x, y, vx, vy, t = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.t
    else:
        # Nondimensional integration with DU=a, TU=sqrt(a^3/mu), VU=sqrt(mu/a)
        DU, TU, VU = _scales_from_a(mu, a)
        # ND initial state at perigee
        x0_nd = r1 / DU
        y0_nd = 0.0
        vper_nd = v_perigee / VU
        y0_nd_vec = np.array([x0_nd, y0_nd, 0.0, vper_nd], float)
        # ND time grid: half transfer is exactly π in ND
        tau_final = t_fac * np.pi
        tau_eval = np.linspace(0.0, tau_final, 3000)
        sol_nd = solve_ivp(two_body_ode_nd, (0.0, tau_final), y0_nd_vec,
                           t_eval=tau_eval, rtol=1e-9, atol=1e-9, max_step=np.pi/2000)
        # Back to physical units
        x  = sol_nd.y[0] * DU
        y  = sol_nd.y[1] * DU
        vx = sol_nd.y[2] * VU
        vy = sol_nd.y[3] * VU
        t  = sol_nd.t * TU

    return {
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "T_transfer_s": T_transfer,
        "T_transfer_days": T_transfer / 86400.0,
        "dv1_kms": dv1,
        "dv2_kms": dv2,
        "r1": r1, "r2": r2, "a": a
    }


def plot_transfer(traj):
    x, y = traj["x"], traj["y"]
    r1, r2 = traj["r1"], traj["r2"]
    print(x)    

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, y, lw=2, label="Hohmann transfer arc")

    # mark start (perigee) and apogee
    ax.scatter([x[0], x[-1]], [y[0], y[-1]], s=40, zorder=5,
               label="Perigee / Apogee")

    # draw Earth
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_EARTH*np.cos(theta), R_EARTH*np.sin(theta), 'k-', lw=1, label="Earth")

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.grid(True, ls=":")
    ax.set_title(
        f"Planar Hohmann Transfer: r1={r1:,.0f} km → r2={r2:,.0f} km\n"
        f"T ≈ {traj['T_transfer_days']:.2f} days,  Δv1={traj['dv1_kms']:.3f} km/s,  Δv2={traj['dv2_kms']:.3f} km/s"
    )
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


# ============================
# Normalized rotating polar (τ = t / S_T)
# ============================
from dataclasses import dataclass

@dataclass
class Scale:
    R: float = 1.0     # S_R [km]
    THETA: float = 1.0 # S_Θ [rad]
    W: float = 0.0     # S_Ω [rad/s] (physical rotation)
    T: float = 1.0     # S_T [s], tau = t / S_T


# --- Cartesian <-> Polar helpers (planar) ------------------------------------
def _cart2polar_state_xyv(x, y, vx, vy):
    r  = np.hypot(x, y)
    th = np.arctan2(y, x)
    rd = (x*vx + y*vy) / r
    thd = (x*vy - y*vx) / (r*r)
    return np.array([r, th, rd, thd], float)

def _polar2cart_state_xyv(r, th, rd, thd):
    c, s = np.cos(th), np.sin(th)
    x, y  = r*c, r*s
    vx = rd*c - r*thd*s
    vy = rd*s + r*thd*c
    return np.array([x, y, vx, vy], float)


# --- Forward & inverse transforms (include time scaling) ---------------------
def _to_rot_norm(t, y_phys, s: Scale):
    """
    (r,θ, rdot, θdot) at physical time t  -> (ρ, φ, ρ', φ') where
      ρ' = dρ/dτ, φ' = dφ/dτ, τ = t/S_T.
    """
    r, th, rd, thd = y_phys
    SR, ST, SW, STi = s.R, s.THETA, s.W, s.T
    return np.array([
        r / SR,
        (th - SW * t) / ST,
        (STi / SR) * rd,
        (STi / ST) * (thd - SW)
    ], float)

def _from_rot_norm(t, yp, s: Scale):
    """
    (ρ, φ, ρ', φ') at τ=t/S_T -> (r,θ, rdot, θdot) (physical).
    """
    rho, phi, rhod, phid = yp
    SR, ST, SW, STi = s.R, s.THETA, s.W, s.T
    return np.array([
        SR * rho,
        ST * phi + SW * t,
        (SR / STi) * rhod,
        SW + (ST / STi) * phid
    ], float)


# --- Normalized rotating polar RHS (pure 2BP, no thrust) ---------------------
def _rhs_rot_norm_polar(tau, yp, mu, s: Scale):
    """
    yp = [ρ, φ, ρ', φ'], integrated in τ = t/S_T.
    Two-body EOM in polar; rotation appears via θ̇ = S_Ω + (S_Θ/S_T) φ'.
    """
    rho, phi, rhod, phid = yp
    SR, ST, SW, STi = s.R, s.THETA, s.W, s.T

    # Map to physical kinematics
    r = SR * rho
    rdot = (SR / STi) * rhod
    thetadot = SW + (ST / STi) * phid

    # Physical dynamics
    rddot     = r * thetadot**2 - mu / (r*r)
    thetaddot = - 2.0 * rdot * thetadot / r

    # Back to τ-accelerations
    rhodd = (STi**2 / SR) * rddot
    phidd = (STi**2 / ST) * thetaddot
    return np.array([rhod, phid, rhodd, phidd], float)


# --- Integrate normalized rotating polar on the Hohmann time grid ------------
def simulate_rot_norm_for_hohmann(traj_nominal: dict,
                                  mu: float,
                                  s: Scale,
                                  rtol=1e-9,
                                  atol=1e-9):
    """
    Use the nominal Hohmann trajectory's initial state & time grid.
    Integrate normalized rotating polar dynamics in τ, then map back to Cartesian.
    Returns:
      dict with keys:
        - 't'        : (T,) physical times
        - 'Yp'       : (T,4) normalized states [ρ, φ, ρ', φ']
        - 'x_back'   : (T,), 'y_back',(T,), 'vx_back',(T,), 'vy_back',(T,)
    """
    t = np.asarray(traj_nominal["t"], float)
    # Initial Cartesian -> polar
    x0, y0, vx0, vy0 = traj_nominal["x"][0], traj_nominal["y"][0], traj_nominal["vx"][0], traj_nominal["vy"][0]
    y0_pol = _cart2polar_state_xyv(x0, y0, vx0, vy0)

    # Polar -> normalized at t0, integrate in τ on same physical times
    yp0 = _to_rot_norm(t[0], y0_pol, s)
    tau = t / s.T

    sol = solve_ivp(lambda tau_k, yp_k: _rhs_rot_norm_polar(tau_k, yp_k, mu, s),
                    (tau[0], tau[-1]), yp0, t_eval=tau,
                    rtol=rtol, atol=atol, method="RK45")
    if not sol.success:
        raise RuntimeError(sol.message)

    # Back to physical polar & then Cartesian along t-grid
    Yp = sol.y.T
    x_b = np.empty_like(t); y_b = np.empty_like(t)
    vx_b = np.empty_like(t); vy_b = np.empty_like(t)
    for k, tk in enumerate(t):
        y_pol_k = _from_rot_norm(tk, Yp[k], s)
        xk, yk, vxk, vyk = _polar2cart_state_xyv(*y_pol_k)
        x_b[k], y_b[k], vx_b[k], vy_b[k] = xk, yk, vxk, vyk

    return dict(t=t, Yp=Yp, x_back=x_b, y_back=y_b, vx_back=vx_b, vy_back=vy_b)


def print_min_max(Y_back, names=("r'", "theta'", "rdot'", "thetadot'"), use_nan=False):
    """
    Print per-dimension min/max for Y_back (shape: [T, 4]).
    Returns a dict {name: (min, max)}.
    """
    amin = np.nanmin if use_nan else np.min
    amax = np.nanmax if use_nan else np.max

    out = {}
    for j, name in enumerate(names):
        vmin = float(amin(Y_back[:, j]))
        vmax = float(amax(Y_back[:, j]))
        out[name] = (vmin, vmax)
        print(f"{name:9s}: min = {vmin:.12g},  max = {vmax:.12g}")
    return out


# --- Validation & visualization ----------------------------------------------
def validate_rot_norm_against_cart(traj_nominal: dict, back: dict, title_suffix=""):
    """
    Compare back-transformed normalized-rotating trajectory to the original.
    Prints max|error| and RMS for [x,y,vx,vy]. Produces XY overlay and error plots.
    """
    t   = traj_nominal["t"]
    x   = np.asarray(traj_nominal["x"]);    y   = np.asarray(traj_nominal["y"])
    vx  = np.asarray(traj_nominal["vx"]);   vy  = np.asarray(traj_nominal["vy"])
    xb  = np.asarray(back["x_back"]);       yb  = np.asarray(back["y_back"])
    vxb = np.asarray(back["vx_back"]);      vyb = np.asarray(back["vy_back"])
    print_min_max(back["Yp"])

    ex, ey   = xb - x,   yb - y
    evx, evy = vxb - vx, vyb - vy

    def _stats(e):
        return np.max(np.abs(e)), np.sqrt(np.mean(e*e))

    stats = {
        "x": _stats(ex), "y": _stats(ey),
        "vx": _stats(evx), "vy": _stats(evy)
    }
    print("[Validation] max|err|,  rms(err)")
    for k in ["x","y","vx","vy"]:
        print(f"  {k:>2s}: {stats[k][0]:.3e},  {stats[k][1]:.3e}")

    # XY overlay
    plt.figure(figsize=(6.8, 6.2))
    plt.plot(x, y,  'k-',  lw=2, label="Original (Cartesian)")
    plt.plot(xb, yb, 'r--', lw=2, label="Back from rot-norm polar")
    th = np.linspace(0, 2*np.pi, 360)
    plt.plot(R_EARTH*np.cos(th), R_EARTH*np.sin(th), 'k-', lw=1, label="Earth")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.grid(True, ls=":")
    plt.title(f"XY Trajectory Overlay {title_suffix}")
    plt.legend(); plt.tight_layout(); plt.show()

    # Time series of position & velocity errors
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t/3600.0, np.hypot(ex, ey), lw=2)
    axs[0].set_ylabel(r"$\|\,\Delta r\,\|$ (km)"); axs[0].grid(True, ls=":")
    axs[1].plot(t/3600.0, np.hypot(evx, evy), lw=2)
    axs[1].set_ylabel(r"$\|\,\Delta v\,\|$ (km/s)"); axs[1].set_xlabel("t (hr)")
    axs[1].grid(True, ls=":")
    plt.tight_layout(); plt.show()

    return stats


# ---------- Monte-Carlo core ----------
def run_monte_carlo(
    traj_nominal,
    mu=MU_EARTH,
    N=300,
    std_xyzv=(100.0, 100.0, 1e-4, 1e-4),   # [km, km, km/s, km/s]
    cov_4x4=None,                          # optional full covariance (physical units)
    seed=42,
    rtol=1e-9,
    atol=1e-9,
    nondim: bool = False,
):
    """
    Monte-Carlo propagation for two-body planar dynamics with uncertain initial state.

    If nondim=True, sampling and integration are performed in canonical units
    (DU=a, TU=sqrt(a^3/mu), VU=sqrt(mu/a)) and results are converted back to physical units.
    Outputs are always returned in physical units.
    """
    rng = np.random.default_rng(seed)
    t_phys = traj_nominal["t"]
    T = len(t_phys)
    # Nominal initial state (physical)
    x0_nom = np.array([traj_nominal["x"][0],
                       traj_nominal["y"][0],
                       traj_nominal["vx"][0],
                       traj_nominal["vy"][0]], dtype=float)

    # Build covariance in the working space (physical or ND)
    if cov_4x4 is None:
        sx, sy, svx, svy = std_xyzv
        Cov_phys = np.diag([sx**2, sy**2, svx**2, svy**2])
    else:
        Cov_phys = np.asarray(cov_4x4, float)
        if Cov_phys.shape != (4,4):
            raise ValueError("cov_4x4 must be a 4x4 matrix")

    if not nondim:
        # ---- Physical MC path ----
        L = np.linalg.cholesky(Cov_phys)
        zeta = rng.standard_normal((N, 4))
        Y0 = x0_nom[None, :] + zeta @ L.T    # (N,4)

        t_span = (t_phys[0], t_phys[-1])
        max_step = (t_phys[-1] - t_phys[0]) / 2000.0
        samples = np.empty((N, T, 4), dtype=float)

        for i in tqdm(range(N), desc="Propagating samples"):
            sol_i = solve_ivp(two_body_ode, t_span, Y0[i], args=(mu,),
                              t_eval=t_phys, rtol=rtol, atol=atol, max_step=max_step)
            if not sol_i.success:
                raise RuntimeError(f"MC sample {i} integration failed: {sol_i.message}")
            samples[i, :, 0] = sol_i.y[0]
            samples[i, :, 1] = sol_i.y[1]
            samples[i, :, 2] = sol_i.y[2]
            samples[i, :, 3] = sol_i.y[3]

        mean_t = samples.mean(axis=0)
        std_t  = samples.std(axis=0, ddof=1)
        return dict(t=t_phys, samples=samples, mean=mean_t, std=std_t, x0_nom=x0_nom)

    # ---- Nondimensional MC path ----
    # Use a from the nominal trajectory (constructed by hohmann_planar)
    if "a" not in traj_nominal:
        raise ValueError("traj_nominal must contain 'a' when nondim=True.")
    a = float(traj_nominal["a"])
    DU, TU, VU = _scales_from_a(mu, a)

    # Scale covariance to ND: S = diag(1/DU, 1/DU, 1/VU, 1/VU)
    S = np.diag([1.0/DU, 1.0/DU, 1.0/VU, 1.0/VU])
    Cov_nd = S @ Cov_phys @ S.T

    # Sample ND initial conditions around ND nominal x0
    x0_nd = np.array([traj_nominal["x"][0]/DU,
                      traj_nominal["y"][0]/DU,
                      traj_nominal["vx"][0]/VU,
                      traj_nominal["vy"][0]/VU], dtype=float)
    L_nd = np.linalg.cholesky(Cov_nd)
    zeta = rng.standard_normal((N, 4))
    Y0_nd = x0_nd[None, :] + zeta @ L_nd.T

    # ND time grid aligned to the provided physical t grid
    tau_eval = t_phys / TU
    t_span_nd = (tau_eval[0], tau_eval[-1])
    max_step_nd = (tau_eval[-1] - tau_eval[0]) / 2000.0

    samples = np.empty((N, T, 4), dtype=float)
    for i in tqdm(range(N), desc="Propagating samples (ND)"):
        sol_i = solve_ivp(two_body_ode_nd, t_span_nd, Y0_nd[i],
                          t_eval=tau_eval, rtol=rtol, atol=atol, max_step=max_step_nd)
        if not sol_i.success:
            raise RuntimeError(f"MC sample {i} (ND) integration failed: {sol_i.message}")
        # Back to physical units
        samples[i, :, 0] = sol_i.y[0] * DU
        samples[i, :, 1] = sol_i.y[1] * DU
        samples[i, :, 2] = sol_i.y[2] * VU
        samples[i, :, 3] = sol_i.y[3] * VU

    mean_t = samples.mean(axis=0)
    std_t  = samples.std(axis=0, ddof=1)
    return dict(t=t_phys, samples=samples, mean=mean_t, std=std_t, x0_nom=x0_nom)


# ---------- Visualization helpers ----------
def plot_compare_time_bands(traj_nominal, mc, lin):
    """
    Overlay MC mean±1σ and linearized mean±1σ for [x,y,vx,vy] over time.
    All inputs must be in physical units.
    """
    t = mc["t"]
    mean_mc, std_mc = mc["mean"], mc["std"]
    mean_lin = lin["mean"]
    std_lin  = np.sqrt(np.stack([lin["cov"][:,0,0],
                                 lin["cov"][:,1,1],
                                 lin["cov"][:,2,2],
                                 lin["cov"][:,3,3]], axis=1))

    labels = ["x (km)", "y (km)", "vx (km/s)", "vy (km/s)"]
    nom = [traj_nominal["x"], traj_nominal["y"], traj_nominal["vx"], traj_nominal["vy"]]

    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for k, ax in enumerate(axs):
        # MC mean ±1σ
        ax.plot(t, mean_mc[:, k], lw=2, label="MC mean")
        ax.fill_between(t, mean_mc[:,k]-std_mc[:,k], mean_mc[:,k]+std_mc[:,k],
                        alpha=0.25, label="MC ±1σ" if k==0 else None)

        # Linear mean ±1σ
        ax.plot(t, mean_lin[:, k], lw=1.8, linestyle="--", label="Linear mean")
        ax.fill_between(t, mean_lin[:,k]-std_lin[:,k], mean_lin[:,k]+std_lin[:,k],
                        alpha=0.18, hatch=None, label="Linear ±1σ" if k==0 else None)

        # Nominal
        ax.plot(t, nom[k], lw=1.2, linestyle=":", label="Nominal" if k==0 else None)

        ax.grid(True, ls=":")
        ax.set_ylabel(labels[k])
        ax.legend(loc="best")
    axs[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


def _ellipse_from_cov(mu2, C2, nsig=1.0, npts=256):
    """Return ellipse points for 2×2 covariance C2 at mean mu2."""
    vals, vecs = np.linalg.eigh(C2)
    vals = np.clip(vals, 0.0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rx, ry = nsig * np.sqrt(vals[0]), nsig * np.sqrt(vals[1])
    R = vecs
    phi = np.linspace(0, 2*np.pi, npts)
    circ = np.vstack([rx*np.cos(phi), ry*np.sin(phi)]).T
    return mu2[None,:] + circ @ R.T


def plot_compare_xy(traj_nominal, mc, lin, n_plot=200, nsig=3.0):
    """
    XY overlay of MC sample trajectories, nominal path, linear mean path,
    and both final-time covariance ellipses (MC vs Linear).
    """
    samples = mc["samples"]          # (N,T,4)
    t = mc["t"]
    N = samples.shape[0]
    idx = np.linspace(0, N-1, min(n_plot, N)).astype(int)

    plt.figure(figsize=(6.8, 6.4))

    # MC sample trajectories (thin)
    for i in idx:
        plt.plot(samples[i,:,0], samples[i,:,1], alpha=0.08, lw=1)

    # Nominal and linear mean trajectories
    plt.plot(traj_nominal["x"], traj_nominal["y"], "k--", lw=2, label="Nominal")
    plt.plot(lin["mean"][:,0], lin["mean"][:,1], lw=2, linestyle="-.", label="Linear mean")

    # Earth
    th = np.linspace(0, 2*np.pi, 200)
    plt.plot(R_EARTH*np.cos(th), R_EARTH*np.sin(th), "k-", lw=1, label="Earth")

    # Final-time MC covariance ellipse
    Xf = samples[:, -1, 0:2]
    mu_mc = Xf.mean(axis=0)
    Cf_mc = np.cov(Xf.T, ddof=1)
    ell_mc = _ellipse_from_cov(mu_mc, Cf_mc, nsig=nsig)
    plt.plot(ell_mc[:,0], ell_mc[:,1], lw=2, label=f"MC {nsig}σ ellipse")
    plt.scatter(*mu_mc, s=40, zorder=5, label="MC final mean")

    # Final-time Linear covariance ellipse
    mu_lin = lin["mean"][-1, 0:2]
    C_lin = lin["cov"][-1, 0:2, 0:2]
    ell_lin = _ellipse_from_cov(mu_lin, C_lin, nsig=nsig)
    plt.plot(ell_lin[:,0], ell_lin[:,1], lw=2, linestyle="--", label=f"Linear {nsig}σ ellipse")
    plt.scatter(*mu_lin, s=40, zorder=5, marker="x", label="Linear final mean")

    plt.gca().set_aspect("equal", "box")
    plt.grid(True, ls=":")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# --- Baseline Method ---
def _A_nd(x, y):
    """
    Jacobian A = ∂f_nd/∂y for y=[x,y,vx,vy] in non-dimensional space.
    f_nd = [vx, vy, -x/r^3, -y/r^3],  r = sqrt(x^2 + y^2)
    """
    r2 = x*x + y*y
    r = np.sqrt(r2)
    r5 = r2 * r**3
    cxx = (3.0*x*x - r2) / r5
    cxy = 3.0*x*y / r5
    cyy = (3.0*y*y - r2) / r5
    # rows: [x, y, vx, vy]
    A = np.array([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [cxx,  cxy,  0.0, 0.0],
                  [cxy,  cyy,  0.0, 0.0]], dtype=float)
    return A


def linear_propagation(
    traj_nominal,
    mu=MU_EARTH,
    std_xyzv=(100.0, 100.0, 1e-4, 1e-4),  # [km, km, km/s, km/s]
    cov_4x4=None,                         # optional 4x4 covariance in physical units
    rtol=1e-10,
    atol=1e-12,
    nondim: bool = True,
):
    """
    Propagate mean and covariance using linearized (time-varying) dynamics about the nominal.
    Core integration is done in nondimensional space (DU=a, TU=√(a³/μ), VU=√(μ/a));
    results are returned in physical units to match Monte-Carlo outputs.

    Returns dict with:
        - 't'      : (T,) time grid [s]
        - 'mean'   : (T,4) propagated mean in [x,y,vx,vy] (km, km, km/s, km/s)
        - 'cov'    : (T,4,4) propagated covariance in physical units
        - 'std'    : (T,4)  sqrt(diag(cov))
    """
    # Time & nominal (physical)
    t = np.asarray(traj_nominal["t"], float)
    x = np.asarray(traj_nominal["x"], float)
    y = np.asarray(traj_nominal["y"], float)
    vx = np.asarray(traj_nominal["vx"], float)
    vy = np.asarray(traj_nominal["vy"], float)
    T = len(t)
    if "a" not in traj_nominal:
        raise ValueError("traj_nominal must contain 'a' (semi-major axis).")
    a = float(traj_nominal["a"])

    # Scales and ND nominal
    DU, TU, VU = _scales_from_a(mu, a)
    tau = t / TU
    x_nd, y_nd = x / DU, y / DU
    vx_nd, vy_nd = vx / VU, vy / VU

    # Initial covariance (physical) -> ND space via S
    if cov_4x4 is None:
        sx, sy, svx, svy = std_xyzv
        P0_phys = np.diag([sx**2, sy**2, svx**2, svy**2])
    else:
        P0_phys = np.asarray(cov_4x4, float)
        if P0_phys.shape != (4, 4):
            raise ValueError("cov_4x4 must be (4,4).")
    S = np.diag([1.0/DU, 1.0/DU, 1.0/VU, 1.0/VU])  # phys -> ND
    P0_nd = S @ P0_phys @ S.T

    # Build continuous interpolants for nominal ND trajectory (for A(tau))
    # (linear is sufficient; you can swap to cubic if desired)
    from scipy.interpolate import interp1d
    fx = interp1d(tau, x_nd, kind="linear", fill_value="extrapolate", assume_sorted=True)
    fy = interp1d(tau, y_nd, kind="linear", fill_value="extrapolate", assume_sorted=True)

    # Variational equation: dΦ/dτ = A(τ) Φ, Φ(τ0)=I
    def var_ode(tau_k, phi_flat):
        Phi = phi_flat.reshape(4, 4)
        xk, yk = float(fx(tau_k)), float(fy(tau_k))
        Ak = _A_nd(xk, yk)
        dPhi = Ak @ Phi
        return dPhi.ravel()

    # Integrate Φ on the provided grid
    Phi0 = np.eye(4).ravel()
    sol_var = solve_ivp(
        var_ode, (tau[0], tau[-1]), Phi0,
        t_eval=tau, rtol=rtol, atol=atol, max_step=max(1e-6, (tau[-1]-tau[0])/2000.0)
    )
    if not sol_var.success:
        raise RuntimeError(f"Variational integration failed: {sol_var.message}")

    # Propagate covariance in ND and map back to physical
    cov_phys = np.empty((T, 4, 4), dtype=float)
    for k in range(T):
        Phi = sol_var.y[:, k].reshape(4, 4)
        P_nd = Phi @ P0_nd @ Phi.T
        # ND -> physical via T = diag(DU, DU, VU, VU)
        Tmap = np.diag([DU, DU, VU, VU])
        P_phys = Tmap @ P_nd @ Tmap.T
        cov_phys[k] = P_phys

    # Mean propagation: with zero-mean perturbations about the nominal,
    # linear mean stays on the nominal trajectory.
    mean_phys = np.stack([x, y, vx, vy], axis=1)
    std_phys = np.sqrt(np.clip(np.stack([cov_phys[:, 0, 0],
                                         cov_phys[:, 1, 1],
                                         cov_phys[:, 2, 2],
                                         cov_phys[:, 3, 3]], axis=1), 0.0, np.inf))

    return dict(t=t, mean=mean_phys, cov=cov_phys, std=std_phys)


if __name__ == "__main__":
    # --- nominal trajectory of the Hohmann transfer ---
    traj = hohmann_planar(r1=7000.0, r2=40000.0, mu=MU_EARTH, t_fac=1.0, nondim=True)

    # --- visualize the nominal trajctory ---
    plot_transfer(traj)

    # --- Normalized rotating polar consistency check (minimal choices) ---
    # Choose scales; simplest is to reuse canonical DU/TU with no rotation.
    DU, TU, VU = _scales_from_a(MU_EARTH, traj["a"])
    s = Scale(R=DU, THETA=3.14, W=0.0, T=TU)   # rotation off; you can try W ≈ v_perigee/r1

    back = simulate_rot_norm_for_hohmann(traj, MU_EARTH, s)
    _ = validate_rot_norm_against_cart(traj, back, title_suffix="(Hohmann)")

    # --- Do Monte-Carlo for N_MC samples with initial uncertainty ---
    std_xyzv = (10.0,10.0,1e-3,1e-3)
    # x: std=10 km)
    # y: std=10 km)
    # vx: 1 m/s)
    # vy: 1 m/s)
    mc = run_monte_carlo(
        traj_nominal=traj,
        mu=MU_EARTH,
        N=100,                         # adjust as you like
        std_xyzv=std_xyzv,
        seed=123,
        nondim=True
    )

    # --- Baseline Method ---
    lin = linear_propagation(traj, mu=MU_EARTH, std_xyzv=std_xyzv, nondim=True)

    # --- Visualize ---
    plot_compare_time_bands(traj, mc, lin)
    plot_compare_xy(traj, mc, lin)

    # Mean error vs MC:
    mean_err = mc["mean"] - lin["mean"]            # (T,4)
    # Covariance (diag) comparison:
    std_ratio = mc["std"] / (lin["std"] + 1e-15)   # (T,4)
