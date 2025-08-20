import numpy as np

# --------------------------
# Problem setup
# --------------------------
def f(x):
    """Drift: -0.1*x^3 + 0.1*x^2 + 0.5*x + 0.5 (vectorized)."""
    return -0.1*x**3 + 0.1*x**2 + 0.5*x + 0.5

q = 0.8  # additive diffusion (constant), so process noise variance per step is q^2 * dt


# --------------------------
# Sigma points (generic N)
# --------------------------
def unscented_sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0, jitter=1e-12):
    """
    Generate UKF sigma points for N-dim Gaussian (mean, cov).
    Returns X (2N+1,N), Wm (2N+1,), Wc (2N+1,)
    """
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    N = m.size
    P = np.asarray(cov, dtype=np.float64).reshape(N, N)
    assert P.shape == (N, N), "cov must be (N,N)"

    lam = alpha**2 * (N + kappa) - N
    c = N + lam
    if c <= 0:
        raise ValueError("N + lambda must be positive; adjust alpha/kappa.")

    # Jittered Cholesky for numerically PSD matrices
    I = np.eye(N, dtype=np.float64)
    try:
        S = np.linalg.cholesky(P + jitter * I)
    except np.linalg.LinAlgError:
        eps = 1e-9 * max(1.0, float(np.max(np.diag(P))))
        S = np.linalg.cholesky(P + eps * I)

    S *= np.sqrt(c)  # scale by sqrt(N + lambda)

    # Build sigma points using columns of S
    X = np.empty((2 * N + 1, N), dtype=np.float64)
    X[0] = m
    X[1:N+1] = (m + S.T)
    X[N+1:2*N+1] = (m - S.T)

    # Weights
    Wm = np.full(2 * N + 1, 1.0 / (2.0 * c), dtype=np.float64)
    Wc = np.full(2 * N + 1, 1.0 / (2.0 * c), dtype=np.float64)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)

    return X, Wm, Wc


# --------------------------
# One UT step (additive noise)
# --------------------------
def ut_step_additive(mu, P, dt, alpha=1.0, beta=2.0, kappa=0.0):
    """
    One UT step for the discretized SDE with additive noise:
      x_{k+1} = x_k + f(x_k)*dt + q*sqrt(dt)*xi
    Since the noise is additive (q constant), we propagate the deterministic part
    with UT and then add Q = q^2 * dt to the variance.
    """
    X, Wm, Wc = unscented_sigma_points([mu], [[P]], alpha=alpha, beta=beta, kappa=kappa)
    X = X[:, 0]  # (2N+1,) for N=1

    Y = X + f(X) * dt  # deterministic EM part

    mu_new = float(np.sum(Wm * Y))
    P_new = float(np.sum(Wc * (Y - mu_new)**2) + (q * q) * dt)  # Add-Q
    return mu_new, P_new


# --------------------------
# Rollout with substepping
# --------------------------
def ut_rollout(mu0, P0, T, dt_ut, substeps=5, alpha=1.0, beta=2.0, kappa=0.0):
    """
    Propagate (mu, P) over [0, T] using UT with substeps per dt_ut
    to reduce per-step nonlinearity.
    """
    mu, P = float(mu0), float(P0)
    t = [0.0]; mus = [mu]; Ps = [P]
    nsteps = int(np.round(T / dt_ut))
    h = dt_ut / substeps

    for _ in range(nsteps):
        for __ in range(substeps):
            mu, P = ut_step_additive(mu, P, h, alpha=alpha, beta=beta, kappa=kappa)
        t.append(t[-1] + dt_ut); mus.append(mu); Ps.append(P)

    return np.array(t), np.array(mus), np.array(Ps)


# --------------------------
# Monte Carlo reference (Euler–Maruyama)
# --------------------------
def mc_reference(mu0, P0, T, dt_em, N=200_000, seed=42):
    """
    High-fidelity MC baseline using small EM time step dt_em.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(mu0, np.sqrt(P0), size=N)
    steps = int(np.round(T / dt_em))

    means = np.empty(steps + 1, dtype=np.float64)
    vars_ = np.empty(steps + 1, dtype=np.float64)
    times = np.linspace(0.0, steps * dt_em, steps + 1)

    for k in range(steps + 1):
        means[k] = x.mean()
        vars_[k] = x.var(ddof=0)
        if k < steps:
            xi = rng.normal(0.0, 1.0, size=N)
            x = x + f(x) * dt_em + q * np.sqrt(dt_em) * xi

    return times, means, vars_


# --------------------------
# Linear interpolation helper
# --------------------------
def interp_to_times(t_src, y_src, t_tgt):
    """
    Piecewise-linear interpolation of y(t) from (t_src, y_src) to t_tgt.
    Assumes t_src is increasing and covers [t_tgt.min(), t_tgt.max()].
    """
    out = np.empty_like(t_tgt, dtype=np.float64)
    for j, t in enumerate(t_tgt):
        i = np.searchsorted(t_src, t)
        i = np.clip(i, 1, len(t_src) - 1)
        w = (t - t_src[i - 1]) / (t_src[i] - t_src[i - 1])
        out[j] = (1 - w) * y_src[i - 1] + w * y_src[i]
    return out


# --------------------------
# Main experiment
# --------------------------
if __name__ == "__main__":
    # Initial Gaussian
    mu0, P0 = -2.0, 0.5

    # Horizon and steps
    T = 4.0
    dt_ut = 1e-2          # "macro" step for UT
    substeps = 5          # split each dt_ut into smaller steps for accuracy
    dt_em = 1e-4          # fine EM step for MC
    N_mc = 200_000        # number of particles

    # Monte Carlo baseline
    t_mc, mu_mc, P_mc = mc_reference(mu0, P0, T, dt_em, N=N_mc, seed=42)

    # UT rollout (alpha=1.0 crucial in 1D to avoid tiny spread)
    t_ut, mu_ut, P_ut = ut_rollout(mu0, P0, T, dt_ut, substeps=substeps,
                                   alpha=1.0, beta=2.0, kappa=0.0)

    # Interpolate MC to UT times
    mu_mc_at_ut = interp_to_times(t_mc, mu_mc, t_ut)
    P_mc_at_ut  = interp_to_times(t_mc,  P_mc, t_ut)

    print(mu_mc_at_ut[-1], mu_ut[-1])
    print(P_mc_at_ut[-1], P_ut[-1])

    # Errors
    mu_err = np.abs(mu_mc_at_ut - mu_ut)
    P_err  = np.abs(P_mc_at_ut  - P_ut)

    print(f"Mean absolute mean error: {mu_err.mean():.3e}")
    print(f"Mean absolute var  error: {P_err.mean():.3e}")
    print(f"Max  absolute mean error: {mu_err.max():.3e}")
    print(f"Max  absolute var  error: {P_err.max():.3e}")

    # Reasonable thresholds for this setup (tune to taste)
    assert mu_err.mean() < 5e-3, "UT mean deviates too much from MC"
    assert P_err.mean()  < 8e-3, "UT variance deviates too much from MC"

    # Optional: print weight sanity check for first step
    X0, Wm0, Wc0 = unscented_sigma_points([mu0], [[P0]], alpha=1.0, beta=2.0, kappa=0.0)
    print(f"sum Wm: {Wm0.sum():.6f}, sum Wc: {Wc0.sum():.6f}")
