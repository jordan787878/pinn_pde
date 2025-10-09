# --- Planar 2BP in polar coordinates with velocity-aligned external force ---
# State y = [r, theta, rdot, thetadot]
# Units: km, s (so mu in km^3/s^2).  External "thrust" is interpreted as
#        a constant force magnitude; accel = thrust / mass_kg  [km/s^2].
#        If you prefer to pass acceleration directly, just set mass_kg=1.

from dataclasses import dataclass
import numpy as np
from numpy import sqrt, sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ----------------------------
# Parameters
# ----------------------------
@dataclass
class Params:
    mu_km3s2: float = 398600.4418  # Earth's GM [km^3/s^2]
    mass_kg: float = 100.0         # spacecraft mass [kg]
    thrust: float = 0.0            # constant force magnitude (same direction as v)

    def accel_along_v(self) -> float:
        """Acceleration magnitude along v-hat [km/s^2]."""
        return (self.thrust / self.mass_kg) * 1e-3


# ----------------------------
# Coordinate transforms
# ----------------------------
def cart2polar_state(r_xy: np.ndarray, v_xy: np.ndarray) -> np.ndarray:
    """
    Cartesian (x,y,vx,vy) -> polar state [r, theta, rdot, thetadot].
    """
    x, y = r_xy
    vx, vy = v_xy
    r = sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    rdot = (x*vx + y*vy) / r
    thetadot = (x*vy - y*vx) / (r*r)  # z-component of angular momentum / r^2
    return np.array([r, theta, rdot, thetadot], dtype=float)


def polar2cart(r: float, theta: float) -> np.ndarray:
    """(r, theta) -> (x, y)."""
    return np.array([r*cos(theta), r*sin(theta)], dtype=float)


# ----------------------------
# Kinematics
# ----------------------------
def planar_speed(r: float, rdot: float, thetadot: float) -> float:
    """|v| for polar coordinates in a plane."""
    return sqrt(rdot*rdot + (r*thetadot)**2)


# ----------------------------
# Dynamics (planar polar)
# ----------------------------
def rhs_planar(_t: float, y: np.ndarray, p: Params) -> np.ndarray:
    """
    y = [r, theta, rdot, thetadot]
    Two-body + thrust aligned with velocity:
      a_r     = aT * rdot / V
      a_theta = aT * r*thetadot / V
    EOM:
      rddot      = r*thetadot^2 - mu/r^2 + a_r
      thetaddot  = (a_theta - 2*rdot*thetadot)/r
    """
    r, theta, rdot, thetadot = y
    mu = p.mu_km3s2

    V  = max(planar_speed(r, rdot, thetadot), 1e-16)
    aT = p.accel_along_v()

    a_r     = aT * (rdot / V)
    a_theta = aT * (r * thetadot / V)

    rddot     = r*(thetadot**2) - mu/(r*r) + a_r
    thetaddot = (a_theta - 2.0*rdot*thetadot) / r

    return np.array([rdot, thetadot, rddot, thetaddot], dtype=float)


# ----------------------------
# Simulation wrapper
# ----------------------------
def simulate_planar(p: Params,
                    r0_xy: np.ndarray,
                    v0_xy: np.ndarray,
                    t0: float,
                    tf: float,
                    dt_out: float = 10.0,
                    rtol: float = 1e-9,
                    atol: float = 1e-9):
    """
    Integrate planar EOM from Cartesian initial state (x0,y0,vx0,vy0).
    Returns (T, Y) with Y columns: [r, theta, rdot, thetadot].
    """
    y0 = cart2polar_state(np.asarray(r0_xy, float), np.asarray(v0_xy, float))
    t_eval = np.arange(t0, tf + 0.5*dt_out, dt_out)

    sol = solve_ivp(lambda t, y: rhs_planar(t, y, p),
                    (t0, tf), y0, t_eval=t_eval,
                    rtol=rtol, atol=atol, method="RK45")
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


# --- Scaling constants --------------------------------------------------------
@dataclass
class Scale:
    R: float = 1.0        # distance scale S_R [km]
    THETA: float = 1.0    # angle scale S_Theta [rad]
    W: float = 0.0        # constant rotation rate S_Omega [rad/s] (physical)
    T: float = 1.0        # time scale S_T [s], tau = t / S_T


# --- Normalized rotating RHS (integrated w.r.t. tau) --------------------------
def rhs_rot_norm(tau: float, yp: np.ndarray, p: Params, s: Scale) -> np.ndarray:
    """
    Normalized rotating polar dynamics integrated in normalized time tau = t / S_T.
    yp = [rho, phi, rhodot, phidot], where rhodot = d(rho)/d(tau), phidot = d(phi)/d(tau).
    """
    rho, phi, rhod, phid = yp
    SR, ST, SW, STi = s.R, s.THETA, s.W, s.T
    mu = p.mu_km3s2
    aT = p.accel_along_v()  # [km/s^2]

    # Map normalized state (tau-based rates) -> physical kinematics (t-based)
    r      = SR * rho
    rdot   = (SR / STi) * rhod
    thetadot = SW + (ST / STi) * phid

    # Physical speed magnitude
    V = np.sqrt(rdot*rdot + (r*thetadot)**2)
    V = max(V, 1e-16)

    # Thrust components (physical)
    a_r     = aT * (rdot / V)
    a_theta = aT * (r * thetadot / V)

    # Physical dynamics
    rddot      = r * thetadot**2 - mu / (r*r) + a_r
    thetaddot  = (a_theta - 2.0 * rdot * thetadot) / r

    # Map physical second derivatives -> tau-second-derivatives
    rhodd = (STi**2 / SR) * rddot
    phidd = (STi**2 / ST) * thetaddot

    return np.array([rhod, phid, rhodd, phidd], dtype=float)


# --- Forward & inverse transforms (now consistent with tau-scaling) -----------
def to_rot_norm(t: float, y: np.ndarray, s: Scale) -> np.ndarray:
    """
    (r,theta,rdot,thetadot) at physical time t -> (rho,phi,rhod,phid) where
    rhod = d(rho)/d(tau), phid = d(phi)/d(tau), tau = t / S_T
    """
    r, th, rd, thd = y
    SR, ST, SW, STi = s.R, s.THETA, s.W, s.T
    return np.array([
        r / SR,
        (th - SW * t) / ST,
        (STi / SR) * rd,
        (STi / ST) * (thd - SW)
    ], dtype=float)


def from_rot_norm(t: float, yp: np.ndarray, s: Scale) -> np.ndarray:
    """
    (rho,phi,rhod,phid) at physical time t (only used for SW*t in angle),
    where rhod, phid are tau-derivatives, -> (r,theta,rdot,thetadot) in physical.
    """
    rho, phi, rhod, phid = yp
    SR, ST, SW, STi = s.R, s.THETA, s.W, s.T
    return np.array([
        SR * rho,
        ST * phi + SW * t,
        (SR / STi) * rhod,
        SW + (ST / STi) * phid
    ], dtype=float)


# --- Simulation in normalized rotating frame (internal tau integration) -------
def simulate_rot_norm(p: Params,
                      s: Scale,
                      r0_xy: np.ndarray,
                      v0_xy: np.ndarray,
                      t0: float,
                      tf: float,
                      dt_out: float = 10.0,
                      rtol: float = 1e-9,
                      atol: float = 1e-9):
    """
    Integrate the normalized rotating system in tau, but keep I/O in physical time.
    Returns:
      t_eval (physical) : (T,)
      Yp (normalized)   : (T,4) [rho,phi,rhod,phid] (tau-based rates)
      Y_phys_back       : (T,4) back-transformed physical polar state
    """
    # Initial state in polar (physical) -> normalized (tau-based)
    y0_phys = cart2polar_state(np.asarray(r0_xy, float), np.asarray(v0_xy, float))
    yp0 = to_rot_norm(t0, y0_phys, s)

    # Physical sampling grid, mapped to tau
    t_eval = np.arange(t0, tf + 0.5*dt_out, dt_out)
    tau0, tauf = t0 / s.T, tf / s.T
    tau_eval = t_eval / s.T

    # Integrate in tau
    sol = solve_ivp(lambda tau, yp: rhs_rot_norm(tau, yp, p, s),
                    (tau0, tauf), yp0,
                    t_eval=tau_eval, rtol=rtol, atol=atol, method="RK45")
    if not sol.success:
        raise RuntimeError(sol.message)

    # Back to physical polar along the physical time grid (t = tau*s.T)
    Y_phys = np.empty((t_eval.size, 4))
    for k, tauk in enumerate(tau_eval):
        tk = tauk * s.T
        Y_phys[k] = from_rot_norm(tk, sol.y[:, k], s)

    return t_eval, sol.y.T, Y_phys  # physical times, normalized states, physical-back


# --- Validation against nominal planar propagation ---------------------------
def _polar_to_cart_series(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized (r,theta,rdot,thetadot) -> (x,y) and (vx,vy) over time."""
    r = Y[:, 0]; th = Y[:, 1]; rd = Y[:, 2]; thd = Y[:, 3]
    x = r * np.cos(th); y = r * np.sin(th)
    vx = rd * np.cos(th) - r * thd * np.sin(th)
    vy = rd * np.sin(th) + r * thd * np.cos(th)
    return np.column_stack([x, y]), np.column_stack([vx, vy])


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


def validate_rot_norm(p: Params,
                      s: Scale,
                      r0_xy: np.ndarray,
                      v0_xy: np.ndarray,
                      t0: float,
                      tf: float,
                      dt_out: float = 10.0,
                      rtol: float = 1e-9,
                      atol: float = 1e-9):
    """
    Run both: (1) nominal planar integrator and (2) normalized-rotating integrator,
    back-transform the latter, and report max errors in polar and Cartesian states.
    Returns a dict of error metrics.
    """
    # nominal
    t_nom, Y_nom = simulate_planar(p, r0_xy, v0_xy, t0, tf, dt_out, rtol, atol)
    # normalized-rotating (with back-transform)
    t_rot, Y_back_norm, Y_back = simulate_rot_norm(p, s, r0_xy, v0_xy, t0, tf, dt_out, rtol, atol)
    print_min_max(Y_back_norm)

    if t_nom.shape != t_rot.shape or not np.allclose(t_nom, t_rot):
        raise ValueError("Time grids differ. Use identical t0, tf, dt_out.")

    # Angle wrapping can cause 2π jumps; compare also in Cartesian
    # Polar errors (wrap theta difference to [-pi,pi))
    dY = Y_back - Y_nom
    dY[:,1] = (dY[:,1] + np.pi) % (2*np.pi) - np.pi

    # Cartesian comparison
    X_nom, V_nom = _polar_to_cart_series(Y_nom)
    X_b,   V_b   = _polar_to_cart_series(Y_back)

    err = {
        "t": t_nom,
        "max_abs_polar": np.max(np.abs(dY), axis=0),         # [r, theta, rdot, thetadot]
        "rms_polar": np.sqrt(np.mean(dY**2, axis=0)),
        "max_abs_pos_xy": np.max(np.abs(X_b - X_nom), axis=0),  # [x,y]
        "max_abs_vel_xy": np.max(np.abs(V_b - V_nom), axis=0),  # [vx,vy]
    }

    # XY ground-track style
    plt.figure(figsize=(6.6, 6.2))
    plt.title("Validating Dynamics in Normalized Rotating Coordinates")
    plt.plot(X_nom[:,0], X_nom[:,1], lw=2, label="Trajectory")
    plt.plot(X_b[:,0], X_b[:,1], lw=2, linestyle="--", label="Trajectory (NROT)")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("X (km)"); plt.ylabel("Y (km)")
    plt.grid(True, ls=":")
    plt.legend(); plt.tight_layout()
    plt.show()

    return err


def monte_carlo_planar(p: Params,
                       r0_xy: np.ndarray,
                       v0_xy: np.ndarray,
                       t0: float,
                       tf: float,
                       dt_out: float = 10.0,
                       N: int = 100,
                       std_diag=(1.0, 1.0, 1e-3, 1e-3),  # [km, km, km/s, km/s]
                       seed: int | None = 0,
                       rtol: float = 1e-9,
                       atol: float = 1e-9):
    """
    Monte Carlo propagation of planar 2BP with velocity-aligned thrust.

    Initial state is Cartesian mean [x0, y0, vx0, vy0] with diagonal covariance
    diag(std_diag^2). Each sample is integrated on the same time grid and
    converted to polar state [r, theta, rdot, thetadot].

    Returns:
      dict with
        - 't'       : (T,) time grid
        - 'samples' : (N, T, 4) stacked polar states
        - 'mean'    : (T, 4) timewise mean over samples
        - 'std'     : (T, 4) timewise std over samples (ddof=1)
        - 'x0_cart' : (4,) nominal Cartesian initial state used
        - 'std_diag': (4,) std used for sampling
    """
    rng = np.random.default_rng(seed)

    # common time grid
    t_eval = np.arange(t0, tf + 0.5*dt_out, dt_out)
    T = t_eval.size

    # nominal Cartesian initial state
    x0_cart = np.array([r0_xy[0], r0_xy[1], v0_xy[0], v0_xy[1]], dtype=float)
    std_diag = np.asarray(std_diag, dtype=float)

    # allocate
    samples = np.empty((N, T, 4), dtype=float)

    for i in range(N):
        # sample Cartesian ICs then convert to polar state
        delta = rng.standard_normal(4) * std_diag
        x0_i = x0_cart + delta
        y0_i = cart2polar_state(x0_i[:2], x0_i[2:])

        sol = solve_ivp(lambda t, y: rhs_planar(t, y, p),
                        (t0, tf), y0_i, t_eval=t_eval,
                        rtol=rtol, atol=atol, method="RK45")
        if not sol.success:
            raise RuntimeError(f"Sample {i} integration failed: {sol.message}")

        samples[i] = sol.y.T  # (T,4)

    # timewise stats
    mean_t = samples.mean(axis=0)
    std_t  = samples.std(axis=0, ddof=1)

    return {
        "t": t_eval,
        "samples": samples,
        "mean": mean_t,
        "std": std_t,
        "x0_cart": x0_cart,
        "std_diag": std_diag,
    }


# ----------------------------
# Utilities for plotting
# ----------------------------
def polar_traj_to_cart(Y: np.ndarray) -> np.ndarray:
    XY = np.zeros((Y.shape[0], 2))
    for k, (r, theta) in enumerate(Y[:, :2]):
        XY[k] = polar2cart(r, theta)
    return XY


def plot_planar(T, Y, R_earth=6378.137):
    XY = polar_traj_to_cart(Y)

    # XY ground-track style
    plt.figure(figsize=(6.6, 6.2))
    plt.plot(XY[:,0], XY[:,1], lw=2, label="Trajectory")
    th = np.linspace(0, 2*np.pi, 300)
    plt.plot(R_earth*np.cos(th), R_earth*np.sin(th), "k-", lw=1, label="Earth")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("X (km)"); plt.ylabel("Y (km)")
    plt.grid(True, ls=":")
    plt.legend(); plt.tight_layout()

    # Time series
    r   = Y[:,0]
    thw = np.unwrap(Y[:,1])
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(T/3600.0, r - R_earth, lw=2); axs[0].set_ylabel("Altitude (km)")
    axs[1].plot(T/3600.0, thw, lw=2);        axs[1].set_ylabel(r"$\theta$ (rad)")
    axs[1].set_xlabel("t (hr)")
    for ax in axs: ax.grid(True, ls=":")
    plt.tight_layout()
    plt.show()


def _ellipse_from_cov(mu2, C2, nsig=1.0, npts=256):
    """
    Return points of the nsig–sigma covariance ellipse for 2×2 covariance C2 at mean mu2.
    """
    vals, vecs = np.linalg.eigh(C2)
    vals = np.clip(vals, 0.0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rx, ry = nsig * np.sqrt(vals[0]), nsig * np.sqrt(vals[1])  # principal radii
    phi = np.linspace(0, 2*np.pi, npts)
    circ = np.vstack([rx*np.cos(phi), ry*np.sin(phi)]).T       # (npts,2)
    return mu2[None, :] + circ @ vecs.T                        # rotate + shift


def plot_mc_xy_plane(mc,
                     R_earth=6378.137,
                     nsig=3.0,
                     n_plot=200,
                     t_indices=None,
                     n_ellipses=5,
                     title="Monte-Carlo in x–y (planar)"):
    """
    Visualize Monte-Carlo results on x–y plane.
      - mc: dict from monte_carlo_planar(...)  [expects 'samples' and 't']
      - Draw thin sample trajectories (subset n_plot)
      - Draw MC mean trajectory
      - Draw nsig-sigma covariance ellipses at selected time indices

    If t_indices is None, n_ellipses evenly spaced indices (including final) are used.
    """

    samples = mc["samples"]           # (N, T, 4) in polar [r, theta, rdot, thetadot]
    t = mc["t"]                       # (T,)
    N, T, _ = samples.shape

    # Convert whole ensemble to Cartesian (vectorized)
    r   = samples[:, :, 0]
    th  = samples[:, :, 1]
    X   = r * np.cos(th)              # (N, T)
    Y   = r * np.sin(th)              # (N, T)

    # Mean trajectory in Cartesian space (mean of x & y over samples)
    Xm = X.mean(axis=0)               # (T,)
    Ym = Y.mean(axis=0)               # (T,)

    # Choose which times to show ellipses
    if t_indices is None:
        # include t[0] and t[-1] and ~even spacing
        if n_ellipses < 2:
            t_indices = [T-1]
        else:
            t_indices = np.linspace(0, T-1, n_ellipses, dtype=int)
            t_indices = np.unique(t_indices).tolist()

    # Plot
    plt.figure(figsize=(7.2, 6.6))

    # Thin sample trajectories (subset for clarity)
    idx_subset = np.linspace(0, N-1, min(n_plot, N), dtype=int)
    for i in idx_subset:
        plt.plot(X[i], Y[i], color="black", lw=0.5, alpha=0.25)

    # MC mean trajectory
    plt.plot(Xm, Ym, 'k-', lw=2.2, label="MC mean")

    # Earth circle
    th = np.linspace(0, 2*np.pi, 300)
    plt.plot(R_earth*np.cos(th), R_earth*np.sin(th), 'k-', lw=1, label="Earth")

    # Covariance ellipses at selected times
    for k in t_indices:
        XYk = np.vstack([X[:, k], Y[:, k]])          # (2, N)
        mu = XYk.mean(axis=1)                        # (2,)
        C  = np.cov(XYk, ddof=1)                     # (2,2)
        ell = _ellipse_from_cov(mu, C, nsig=nsig)
        lab = f"{nsig}σ @ t={t[k]/3600:.2f} hr"
        plt.plot(ell[:, 0], ell[:, 1], lw=2, label=lab)
        plt.scatter(mu[0], mu[1], s=30, zorder=5)

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.grid(True, ls=":")
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


def set_initial_state(p, method=0):
    if method == 0:
        r1=7000.0; r2=40000.0
        a = 0.5 * (r1 + r2)                      # semimajor axis (km)
        mu = p.mu_km3s2
        v_circ_1 = np.sqrt(mu / r1)
        v_perigee = np.sqrt(mu * (2.0/r1 - 1.0/a))
        r0_xy = np.array([r1, 0.0])             # km
        v0_xy = np.array([0.0, v_circ_1])       # km/s
    if method == 1:
        r0_xy = np.array([6628.0, 0.0])          # km
        v0_xy = np.array([0.0, 7.7])             # km/s (pick something LEO-like)
    return r0_xy, v0_xy


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    p = Params(mu_km3s2=398600.4418, mass_kg=1.0, thrust=0.2)

    r0_xy, v0_xy = set_initial_state(p)

    t0, tf, dt_out = 0.0, 2.0*3600.0, 60.0
    
    # T, Y = simulate_planar(p, r0_xy, v0_xy, t0, tf, dt_out=dt_out)
    # plot_planar(T, Y)

    # Choose scales (e.g., normalize radius by 7000 km and rotate at mean motion)
    n = 0.5*np.sqrt(p.mu_km3s2 / 7000.0**3)    # mean motion ~ rad/s for circular example
    s = Scale(R=7000.0, THETA=1.0, W=n, T=3600.0)

    report = validate_rot_norm(p, s, r0_xy, v0_xy, t0, tf, dt_out)
    print("Max abs polar errors [r,theta,rdot,thetadot]:", report["max_abs_polar"])
    print("Max abs position errors [x,y] (km):         ", report["max_abs_pos_xy"])
    print("Max abs velocity errors [vx,vy] (km/s):     ", report["max_abs_vel_xy"])

    mc = monte_carlo_planar(
        p, r0_xy, v0_xy,
        t0=t0, tf=tf, dt_out=dt_out,
        N=200,
        std_diag=(1.0, 1.0, 1e-3, 1e-3),  # 1 km pos, 1 m/s vel sigmas
        seed=123
    )

    plot_mc_xy_plane(mc)
