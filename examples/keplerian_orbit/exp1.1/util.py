import numpy as np
from scipy.optimize import fsolve


mu_gravity = 398600.4418 # (km^3/s^2)
I_vec = np.array([1.0, 0.0, 0.0])
J_vec = np.array([0.0, 1.0, 0.0])
K_vec = np.array([0.0, 0.0, 1.0])


def COE2RV(a, e, i, RAAN, w, nu, lonper):
    p = a*(1-e**2)
    if(abs(i) < 1e-10):
        RAAN = 0.0
        w    = lonper
    r_vec_PQW = np.array([p*np.cos(nu)/(1+e*np.cos(nu)),
                      p*np.sin(nu)/(1+e*np.cos(nu)),
                      0.0])
    v_vec_PQW = np.array([-np.sqrt(mu_gravity/p)*np.sin(nu),
                          np.sqrt(mu_gravity/p)*(e+np.cos(nu)),
                          0.0])
    Rot = np.array([[np.cos(RAAN)*np.cos(w) - np.sin(RAAN)*np.sin(w)*np.cos(i),
                     -np.cos(RAAN)*np.sin(w) - np.sin(RAAN)*np.cos(w)*np.cos(i),
                     np.sin(RAAN)*np.sin(i)],
                    [np.sin(RAAN)*np.cos(w) + np.cos(RAAN)*np.sin(w)*np.cos(i),
                     -np.sin(RAAN)*np.sin(w) + np.cos(RAAN)*np.cos(w)*np.cos(i),
                     -np.cos(RAAN)*np.sin(i)],
                    [np.sin(w)*np.sin(i),
                     np.cos(w)*np.sin(i),
                     np.cos(i)]])
    r_IJK = np.matmul(Rot, r_vec_PQW)
    v_IJK = np.matmul(Rot, v_vec_PQW)
    return r_IJK, v_IJK


def RV2COE(r_IJK, v_IJK):
    h_vec = np.cross(r_IJK, v_IJK)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross(K_vec, h_vec) ###
    n = np.linalg.norm(n_vec)
    r = np.linalg.norm(r_IJK)
    v = np.linalg.norm(v_IJK)
    e_vec = ((v**2 - mu_gravity/r)*r_IJK - np.dot(r_IJK, v_IJK)*v_IJK)/mu_gravity
    e = np.linalg.norm(e_vec)
    xi = v**2/2 - mu_gravity/r
    a = -mu_gravity/(2*xi)
    i = np.arccos(h_vec[2]/h)
    if(abs(i) < 1e-10):
        RAAN = 0.0
        lonper = np.arccos(e_vec[0]/e)
        if(e_vec[1] < 0):
            lonper = 2.0*np.pi - lonper
        w = lonper
    else:
        RAAN = np.arccos(n_vec[0]/n)
        if(n_vec[1] < 0):
            RAAN = 2.0*np.pi - RAAN
        w = np.arccos(np.dot(n_vec, e_vec)/ (n*e))
        if(e_vec[2] < 0):
            w = 2.0*np.pi - w
        lonper = RAAN + w

    nu_cos = np.dot(e_vec, r_IJK)/ (e*r)
    nu_cos = np.clip(nu_cos, -1.0, 1.0)
    nu = np.arccos(nu_cos)
    if(np.dot(r_IJK, v_IJK) < 0):
        nu = 2.0*np.pi - nu
    return a, e, i, RAAN, w, nu, lonper


def COE2MEO(a, e, i, RAAN, nu, lonper):
    p = a*(1-e**2)
    f = e*np.cos(lonper)
    g = e*np.sin(lonper)
    h = np.tan(i/2)*np.cos(RAAN)
    k = np.tan(i/2)*np.sin(RAAN)
    L = lonper + nu
    return np.array([p, f, g, h, k, L])


# def MEO2RV(meo, mu_earth=mu_gravity):
#     p = meo[0]
#     f = meo[1]
#     g = meo[2]
#     h = meo[3]
#     k = meo[4]
#     L = meo[5]
#     alpha = np.sqrt(h**2 - k**2)
#     s = np.sqrt(1 + h**2 + k**2)
#     w = 1 + f * np.cos(L) + g * np.sin(L)
#     r = p / w
#     r_vec = np.array([
#         r / s**2 * (np.cos(L) + alpha**2 * np.cos(L) + 2 * h * k * np.sin(L)),
#         r / s**2 * (np.sin(L) - alpha**2 * np.sin(L) + 2 * h * k * np.cos(L)),
#         2 * r / s**2 * (h * np.sin(L) - k * np.cos(L))
#     ])
#     v_vec = np.array([
#         -1 / s**2 * np.sqrt(mu_earth / p) * (np.sin(L) + alpha**2 * np.sin(L) - 2 * h * k * np.cos(L) + g - 2 * f * h * k + alpha**2 * g),
#         -1 / s**2 * np.sqrt(mu_earth / p) * (-np.cos(L) + alpha**2 * np.cos(L) + 2 * h * k * np.sin(L) - f + 2 * g * h * k + alpha**2 * f),
#         2 / s**2 * np.sqrt(mu_earth / p) * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
#     ])
#     return r_vec, v_vec


# def propagate_meo(meo, t):
#     p = meo[0]
#     f = meo[1]
#     g = meo[2]
#     h = meo[3]
#     k = meo[4]
#     L = meo[5]
#     _w = 1+f*np.cos(L)+g*np.sin(L)
#     dLdt = np.sqrt(mu_gravity*p)*((_w/p)**2)
#     meo[5] = meo[5] + dLdt*t
#     return meo


def true_to_mean_anomaly(e, nu):
    if(1-e**2 < 0):
        print("[debug] not valid eccentricity: ", e)
    E = np.arctan2(np.sin(nu)*np.sqrt(1-e**2), np.cos(nu)+e)
    M = E - e*np.sin(E)
    return M


def solve_kepler(e, M):
    # Define the function to solve
    def fun(psi):
        return psi - e * np.sin(psi) - M

    # Improved initial guess
    psi0 = M if M < np.pi else M - 2 * np.pi  # Adjust based on the mean anomaly

    # Solve the equation
    psi_solution = fsolve(fun, psi0)[0]

    # Calculate f
    f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(psi_solution / 2))

    # Adjust f to be in the range [0, 2*pi]
    if f < 0:
        f += 2 * np.pi

    return f


def main():
    # unit testing 
    r_IJK, v_IJK = COE2RV(8000.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0)
    print(r_IJK, v_IJK)

    # unit testing of MOE analytical propagation over t seconds
    # t = 0.0, tests the correctness of forward and backward mapping
    t = 86400.0
    mean_0 = np.array([r_IJK, v_IJK]).reshape(-1,)
    cov_0 = np.diag([(1.0)**2, (1.0)**2, 0.0, (1e-4)**2, (1e-4)**2, 0.0])

    Error = np.zeros(6)
    R_stat = np.zeros(3)
    V_stat = np.zeros(3)
    N_trials = 100
    for idx in range(N_trials):
        rv = np.random.multivariate_normal(mean_0, cov_0).T
        _r = np.array((rv[0], rv[1], rv[2])).reshape(-1,)
        _v = np.array((rv[3], rv[4], rv[5])).reshape(-1,)
        _a, _e, _i, _RAAN, _w, _nu, _lonper = RV2COE(_r, _v)
        _M = true_to_mean_anomaly(_e, _nu)
        _M = _M + np.sqrt(mu_gravity/_a**3)*t
        _nu = solve_kepler(_e, _M)
        r_k, v_k = COE2RV(_a, _e, _i, _RAAN, _w, _nu, _lonper)
        Error = Error + np.array((_r[0]-r_k[0], _r[1]-r_k[1], _r[2]-r_k[2],
                                  _v[0]-v_k[0], _v[1]-v_k[1], _v[2]-v_k[2])).reshape(-1,)
        R_stat = R_stat + r_k
        V_stat = V_stat + v_k
    if(t == 0.0):
        print("Error after Mapping: ", Error/N_trials)
    print("Expected R(t): ", R_stat/N_trials)
    print("Expected V(t): ", V_stat/N_trials)
        

if __name__ == "__main__":
    main()