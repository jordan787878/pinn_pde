import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional, Tuple, Any, Union, Dict
from scipy.integrate import solve_ivp
from numpy.linalg import cholesky, LinAlgError


""" Common wrappers for other signatures

# If you have f(x, c) and A(x, c) (no explicit time):
dyn_fcn = lambda t, x, c: f(x, c)
jac_fcn = lambda t, x, c: A(x, c)

# If you have f(x) and A(x) (no constants, no time):
dyn_fcn = lambda t, x, c: f(x)
jac_fcn = lambda t, x, c: A(x)

# If you have f(t, x) and A(t, x) (no constants):
dyn_fcn = lambda t, x, c: f(t, x)
jac_fcn = lambda t, x, c: A(t, x)

# If your original ones are orbit_dyn(constants, x) / get_Jacobian(constants, x):
dyn_fcn = lambda t, x, c: orbit_dyn(c, x)
jac_fcn = lambda t, x, c: get_Jacobian(c, x)

"""


class PropagationData:
    def __init__(self, path: str):
        self.data = np.load(path)
    def get(self, time):
        times = self.data["times"]
        means = self.data["means"]
        covs = self.data["covs"]
        time = np.round(time, 3)
        idx = np.where(abs(time-times)< 1e-3)[0]
        if len(idx) == 0:
            assert("The propagation result does not have data at this time")
        idx = idx[0]
        if self.data["method"] == "gmm":
            weights = self.data["weights"]
            return times[idx], weights, means[idx, :, :], covs[idx, :, :, :]
        return times[idx], np.array([1.]), means[idx, :], covs[idx, :, :]
    

def linear_propagation_master(
    x0: NDArray[np.float64],
    P0: NDArray[np.float64],
    dyn_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],
    jac_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],
    constants: Any,
    t_span: Tuple[float, float],
    dt_save: float = 1e-2,
    Q: Optional[Union[NDArray[np.float64], Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]]]] = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    solver: str = "RK45",
    save_path: Optional[str] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Linear (time-varying) uncertainty propagation via Lyapunov dynamics ONLY.

    Integrates:
        ẋ = f(t, x)
        Ṗ = A(t,x) P + P A(t,x)^T + Q(t,x)
    If Q is None, Q ≡ 0.

    Parameters
    ----------
    x0 : (n,)
    P0 : (n,n)
    dyn_fcn : f(t, x, constants) -> (n,)
    jac_fcn : A(t, x, constants) -> (n,n)
    constants : user-defined params passed through to dyn_fcn/jac_fcn
    t_span : (t0, tf)
    dt_save : output sampling step
    Q : None, (n,n), or Q(t, x, constants) -> (n,n)
    rtol, atol, solver : solve_ivp controls
    save_path : if given, saves .npz with times, means, covs, method="lyap"

    Returns
    -------
    dict with keys: times (T,), means (T,n), covs (T,n,n), method ("lyap")
    """
    # ---- sanitize ----
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    n = x0.size
    P0 = np.asarray(P0, dtype=np.float64).reshape((n, n))
    # P0 = 0.5 * (P0 + P0.T)

    # t0, tf = float(t_span[0]), float(t_span[1])
    # if tf < t0:
    #     raise ValueError("t_span must have tf >= t0")

    # # output grid
    # t_eval = np.round(np.arange(t0, tf + 0.5 * dt_save, dt_save, dtype=np.float64), 2) 
    t0 = np.float64(np.round(t_span[0], 2))
    tf = np.float64(np.round(t_span[1], 2))
    if tf < t0:
        raise ValueError("t_span must have tf >= t0")
    if dt_save <= 0:
        raise ValueError("dt_save must be positive.")

    # output grid (rounded to 2; saved rounded to 3)
    t_eval = np.round(np.arange(t0, tf + 0.5*dt_save, dt_save, dtype=np.float64), 2)

    # Q evaluator
    def Q_eval(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if Q is None:
            return np.zeros((n, n), dtype=np.float64)
        M = Q(t, x, constants) if callable(Q) else Q
        M = np.asarray(M, dtype=np.float64).reshape((n, n))
        return M

    # augmented state: [x; vec(P)]
    y0 = np.concatenate([x0, P0.ravel()])

    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        x = y[:n]
        P = y[n:].reshape(n, n)
        fx = dyn_fcn(t, x, constants)          # (n,)
        A  = jac_fcn(t, x, constants)          # (n,n)
        Qt = Q_eval(t, x)                      # (n,n)
        dP = A @ P + P @ A.T + Qt
        # dP = 0.5 * (dP + dP.T)                 # keep symmetry numerically
        return np.concatenate([fx, dP.ravel()])

    sol = solve_ivp(rhs, (t0, tf), y0, t_eval=t_eval,
                    rtol=rtol, atol=atol, method=solver)
    if not sol.success:
        raise RuntimeError(f"Integration failed (Lyap): {sol.message}")

    times = sol.t
    means = sol.y[:n, :].T
    covs  = np.empty((times.size, n, n), dtype=np.float64)
    for k in range(times.size):
        Pk = sol.y[n:, k].reshape(n, n)
        covs[k] = Pk #0.5 * (Pk + Pk.T)

    if save_path is not None:
        np.savez(save_path,
                 times=np.round(times, 3),
                 means=means,
                 covs=covs,
                 method="lp")


def unscent_propagation_master(
    x0: NDArray[np.float64],
    P0: NDArray[np.float64],
    dyn_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],
    jac_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],   # unused
    constants: Any,
    G: NDArray[np.float64],  # constant diffusion matrix (n x m_w)
    t_span: Tuple[float, float],
    dt_save: float = 1e-2,
    Q: Optional[NDArray[np.float64]] = None,  # covariance of dW_t; defaults to I
    rtol: float = 1e-9,
    atol: float = 1e-12,
    solver: str = "RK45",
    save_path: Optional[str] = None,
    # Sigma-point / UT hyperparameters for the Gaussian integration rule:
    alpha: float = 1e-3,
    beta: float  = 2.0,   # kept for API compatibility; not used explicitly in Alg. 9.5
    kappa: float = 0.0,
) -> Dict[str, NDArray[np.float64]]:
    """
    Sigma-point (Algorithm 9.5) propagation of mean and covariance for the SDE

        dx = f(x, t) dt + G dW_t,
        E[dW_t dW_t^T] = Q dt,

    where G is a constant (n x m_w) diffusion matrix and Q is an (m_w x m_w)
    covariance matrix of the driving Wiener process.

    We approximate the moment dynamics via Algorithm 9.5:

        dm/dt ≈ Σ_i W^(i) f(m + sqrt(P) ξ_i, t)

        dP/dt ≈ Σ_i W^(i) [
                           f_i ξ_i^T sqrt(P)^T
                         + sqrt(P) ξ_i f_i^T
                         ]
                + Σ_i W^(i) G Q G^T,

    where ξ_i and W^(i) form a Gaussian integration rule for N(0, I),
    sqrt(P) is a matrix square root of P (here, Cholesky S with P = S S^T),
    and f_i = f(m + sqrt(P) ξ_i, t).
    """
    # ---- init mean/cov ----
    m = np.asarray(x0, dtype=np.float64).reshape(-1)
    n = m.size
    P = np.asarray(P0, dtype=np.float64).reshape(n, n)
    P = 0.5 * (P + P.T)  # enforce symmetry

    # diffusion matrix G (n x m_w)
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != n:
        raise ValueError(f"G must have shape (n, m_w) with n={n}, got {G.shape}")
    m_w = G.shape[1]

    # Brownian covariance Q_w (m_w x m_w)
    if Q is None:
        Q_w = np.eye(m_w, dtype=np.float64)
    else:
        Q_w = np.asarray(Q, dtype=np.float64).reshape(m_w, m_w)
        Q_w = 0.5 * (Q_w + Q_w.T)

    # effective diffusion in x: Q_eff = G Q_w G^T (n x n)
    Q_eff = G @ Q_w @ G.T
    Q_eff = 0.5 * (Q_eff + Q_eff.T)

    # ----- Sigma-point rule for N(0, I) (Algorithm 9.5) -----
    # Scaled UT canonical sigma vectors ξ_i and weights W^(i)
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    if c <= 0:
        raise ValueError("n + lambda must be positive; adjust alpha/kappa.")

    Xi = np.zeros((2 * n + 1, n), dtype=np.float64)  # ξ_i
    Xi[1:n+1]     =  np.sqrt(c) * np.eye(n, dtype=np.float64)
    Xi[n+1:2*n+1] = -np.sqrt(c) * np.eye(n, dtype=np.float64)

    # Weights for Gaussian expectations under N(0, I)
    W = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
    W[0] = lam / c
    # (These W^(i) satisfy Σ W^(i) = 1 and Σ W^(i) ξ_i ξ_i^T = I.)

    # ---- time grid ----
    t0 = float(t_span[0])
    tf = float(t_span[1])
    if tf < t0:
        raise ValueError("t_span must have tf >= t0")
    if dt_save <= 0:
        raise ValueError("dt_save must be positive.")

    times = np.round(
        np.arange(t0, tf + 0.5 * dt_save, dt_save, dtype=np.float64),
        2,
    )

    means = np.empty((times.size, n), dtype=np.float64)
    covs  = np.empty((times.size, n, n), dtype=np.float64)
    means[0] = m
    covs[0]  = P

    # ---- time stepping: integrate (m, P) over each [t_k, t_{k+1}] with solve_ivp ----
    for k in range(1, times.size):
        ta = float(times[k - 1])
        tb = float(times[k])

        # pack current (m, P) into a single vector for solve_ivp
        y0 = np.concatenate([m, P.ravel()])

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            # unpack state
            m_t = y[:n]
            P_t = y[n:].reshape(n, n)
            P_t = 0.5 * (P_t + P_t.T)  # keep symmetric

            # sqrt(P_t) via Cholesky: P_t = S S^T
            try:
                S = cholesky(P_t)
            except LinAlgError:
                eps = 1e-10 * np.eye(n, dtype=np.float64)
                S = cholesky(P_t + eps)

            # Algorithm 9.5 sigma points:
            # χ_i(t) = m_t + sqrt(P_t) ξ_i = m_t + S ξ_i
            # Vectorized: (2n+1, n) = (1,n) + (2n+1,n) @ (n,n)^T
            X = m_t[None, :] + Xi @ S.T

            # Drift at each sigma point: f_i = f(χ_i(t), t)
            F_vals = np.empty_like(X)
            for i in range(2 * n + 1):
                F_vals[i] = np.asarray(
                    dyn_fcn(t, X[i], constants),
                    dtype=np.float64,
                ).reshape(n)

            # Mean ODE: dm/dt = Σ_i W^(i) f(χ_i(t), t)
            dm_dt = W @ F_vals  # (n,)

            # Covariance ODE drift term (Alg. 9.5):
            #
            # dP/dt |_drift = Σ_i W^(i) [
            #       f_i ξ_i^T S^T + S ξ_i f_i^T
            # ]
            dP_drift = np.zeros((n, n), dtype=np.float64)
            for i in range(2 * n + 1):
                xi = Xi[i].reshape(n, 1)   # (n,1)
                fi = F_vals[i].reshape(n, 1)  # (n,1)

                term1 = fi @ xi.T @ S.T    # f_i ξ_i^T sqrt(P)^T
                term2 = S @ xi @ fi.T      # sqrt(P) ξ_i f_i^T

                dP_drift += W[i] * (term1 + term2)

            # Diffusion term: Σ_i W^(i) G Q G^T = G Q G^T = Q_eff
            dP_dt = dP_drift + Q_eff

            return np.concatenate([dm_dt, dP_dt.ravel()])

        sol = solve_ivp(
            rhs,
            (ta, tb),
            y0,
            t_eval=[tb],
            method=solver,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Sigma-point mean/cov integration failed: {sol.message}")

        y_end = sol.y[:, -1]
        m = y_end[:n]
        P = y_end[n:].reshape(n, n)
        P = 0.5 * (P + P.T)

        means[k] = m
        covs[k]  = P

    if save_path is not None:
        np.savez(
            save_path,
            times=np.round(times, 3),
            means=means,
            covs=covs,
            method="sigma_point_alg9_5",
        )

    return {
        "times": np.round(times, 3),
        "means": means,
        "covs": covs,
    }


# def unscent_propagation_master(
#     x0: NDArray[np.float64],
#     P0: NDArray[np.float64],
#     dyn_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],
#     jac_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],   # used for Qd integration
#     constants: Any,
#     t_span: Tuple[float, float],
#     dt_save: float = 1e-2,
#     Q: Optional[Union[NDArray[np.float64], Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]]]] = None,
#     rtol: float = 1e-9,
#     atol: float = 1e-12,
#     solver: str = "RK45",
#     save_path: Optional[str] = None,
#     # UT hyperparameters:
#     alpha: float = 1e-3,
#     beta: float  = 2.0,
#     kappa: float = 0.0,
# ) -> Dict[str, NDArray[np.float64]]:
#     """
#     UT propagation using solve_ivp per sigma point on each [t_k, t_{k+1}] interval.
#     Process-noise contribution Q_d is computed by integrating dS = A S + S A^T + Q
#     with S(t_a)=0 over the same interval, using A from jac_fcn (local linearization).
#     """
#     # ---- init ----
#     m = np.asarray(x0, dtype=np.float64).reshape(-1)
#     n = m.size
#     P = np.asarray(P0, dtype=np.float64).reshape(n, n)
#     # P = 0.5*(P + P.T)

#     t0 = np.float64(np.round(t_span[0], 2))
#     tf = np.float64(np.round(t_span[1], 2))
#     if tf < t0:
#         raise ValueError("t_span must have tf >= t0")
#     if dt_save <= 0:
#         raise ValueError("dt_save must be positive.")

#     # output grid (rounded to 2; saved rounded to 3)
#     times = np.round(np.arange(t0, tf + 0.5*dt_save, dt_save, dtype=np.float64), 2)

#     def Q_eval(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
#         if Q is None:
#             return np.zeros((n, n), dtype=np.float64)
#         M = Q(t, x, constants) if callable(Q) else Q
#         return np.asarray(M, dtype=np.float64).reshape(n, n)

#     means = np.empty((times.size, n), dtype=np.float64)
#     covs  = np.empty((times.size, n, n), dtype=np.float64)
#     means[0] = m
#     covs[0]  = P

#     for k in range(1, times.size):
#         ta = float(times[k-1])
#         tb = float(times[k])

#         # 1) sigma points at (m, P)
#         X, Wm, Wc = _unscented_sigma_points(m, P, alpha=alpha, beta=beta, kappa=kappa)

#         # 2) propagate each sigma point ta->tb with solve_ivp
#         Xp = np.empty_like(X)
#         for i in range(X.shape[0]):
#             x_i0 = X[i]

#             def rhs(t, x):
#                 return np.asarray(dyn_fcn(t, x, constants), dtype=np.float64)

#             sol = solve_ivp(rhs, (ta, tb), x_i0, rtol=rtol, atol=atol, method=solver)
#             if not sol.success:
#                 raise RuntimeError(f"Sigma point integration failed: {sol.message}")
#             Xp[i] = sol.y[:, -1]

#         # 3) recombine flow part: Φ P Φ^T via sigma points
#         m_flow = Wm @ Xp
#         Xm = Xp - m_flow
#         P_flow = (Xm.T * Wc) @ Xm

#         # 4) integrate noise accumulator S over [ta, tb]:
#         #    dS = A S + S A^T + Q(t, m_ref),   S(ta)=0
#         # choose reference mean for linearization (use previous mean m)
#         # A_ref = np.asarray(jac_fcn(ta, m, constants), dtype=np.float64).reshape(n, n)
#         t_mid = 0.5*(ta + tb)
#         A_ref = np.asarray(jac_fcn(t_mid, m_flow, constants), dtype=np.float64).reshape(n, n)

#         def rhs_S(t, s_vec):
#             S = s_vec.reshape(n, n)
#             # evaluate Q at the (deterministic) mean; you could also use m_flow
#             # Qt = Q_eval(t, m)
#             Qt = Q_eval(t, m_flow)   # <-- use m_flow, not previous m
#             dS = A_ref @ S + S @ A_ref.T + Qt
#             return dS.ravel()

#         if Q is None:   
#             S_end = np.zeros((n, n), dtype=np.float64)
#         else:
#             s0 = np.zeros(n*n, dtype=np.float64)
#             solS = solve_ivp(rhs_S, (ta, tb), s0, rtol=rtol, atol=atol, method=solver)
#             if not solS.success:
#                 raise RuntimeError(f"Noise integral failed: {solS.message}")
#             S_end = solS.y[:, -1].reshape(n, n)

#         # 5) assemble, symmetrize
#         m = m_flow
#         P = P_flow + S_end
#         # P = 0.5 * (P + P.T)

#         means[k] = m
#         covs[k]  = P

#     if save_path is not None:
#         np.savez(save_path,
#                  times=np.round(times, 3),
#                  means=means,
#                  covs=covs,
#                  method="ut")


def gmm_propagation_master(
    weights: NDArray[np.float64],                 # (K,)
    means0:  NDArray[np.float64],                 # (K, n)
    covs0:   NDArray[np.float64],                 # (K, n, n)
    dyn_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],
    jac_fcn: Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]],
    constants: Any,
    t_span: Tuple[float, float],
    dt_save: float = 1e-2,
    Q: Optional[Union[NDArray[np.float64],
                      Callable[[float, NDArray[np.float64], Any], NDArray[np.float64]]]] = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    solver: str = "RK45",
    save_path: Optional[str] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Propagate a Gaussian Mixture under linearized (Lyapunov) dynamics per component.

    For each component k:
        ẋ = f(t, x)
        Ṗ = A(t,x) P + P A(t,x)^T + Q(t,x)
    with its own (x0_k, P0_k). Mixture weights stay constant.

    Saves: times (T,), weights (K,), means (T,K,n), covs (T,K,n,n), method="gmm"
    """
    # ---- sanitize inputs (float64, shapes) ----
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    K = w.size
    M0 = np.asarray(means0, dtype=np.float64)
    C0 = np.asarray(covs0,  dtype=np.float64)
    if M0.ndim != 2:
        raise ValueError("means0 must be (K, n).")
    if C0.ndim != 3:
        raise ValueError("covs0 must be (K, n, n).")
    if M0.shape[0] != K or C0.shape[0] != K:
        raise ValueError("Leading dim of means0/covs0 must equal weights.size.")
    n = M0.shape[1]
    if C0.shape[1:] != (n, n):
        raise ValueError("Each covariance must be (n,n).")
    # (Optional) normalize weights defensively (keep exact zeros if any)
    s = np.sum(w)
    if s > 0:
        w = w / s

    # time span / grid
    t0 = np.float64(np.round(t_span[0], 2))
    tf = np.float64(np.round(t_span[1], 2))
    if tf < t0:
        raise ValueError("t_span must have tf >= t0.")
    if dt_save <= 0:
        raise ValueError("dt_save must be positive.")
    t_eval = np.round(np.arange(t0, tf + 0.5*dt_save, dt_save, dtype=np.float64), 2)

    # Q evaluator (per component, at its mean)
    def Q_eval(t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if Q is None:
            return np.zeros((n, n), dtype=np.float64)
        M = Q(t, x, constants) if callable(Q) else Q
        return np.asarray(M, dtype=np.float64).reshape(n, n)

    # storage
    Tm = t_eval.size
    means = np.empty((Tm, K, n), dtype=np.float64)
    covs  = np.empty((Tm, K, n, n), dtype=np.float64)

    # integrate each component independently
    for k in range(K):
        x0 = M0[k].reshape(-1)
        P0 = C0[k]
        # augmented initial condition: [x; vec(P)]
        y0 = np.concatenate([x0, P0.ravel()])

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            x = y[:n]
            P = y[n:].reshape(n, n)
            fx = dyn_fcn(t, x, constants)               # (n,)
            A  = jac_fcn(t, x, constants)               # (n,n)
            Qt = Q_eval(t, x)
            dP = A @ P + P @ A.T + Qt
            # (Optional) symmetrize dP: dP = 0.5*(dP + dP.T)
            return np.concatenate([fx, dP.ravel()])

        sol = solve_ivp(rhs, (t0, tf), y0, t_eval=t_eval,
                        rtol=rtol, atol=atol, method=solver)
        if not sol.success:
            raise RuntimeError(f"GMM component {k} integration failed: {sol.message}")

        # unpack to outputs
        means[:, k, :] = sol.y[:n, :].T
        for i in range(Tm):
            Pk = sol.y[n:, i].reshape(n, n)
            covs[i, k, :, :] = Pk   # or 0.5*(Pk + Pk.T) if you prefer symmetry guard

    if save_path is not None:
        np.savez(save_path,
                 times=np.round(t_eval, 3),
                 weights=w,
                 means=means,
                 covs=covs,
                 method="gmm")