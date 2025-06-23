import numpy as np

pi = np.float32(np.pi)
n_d = 2
a_0 = np.float32(8000.0)
lam_0 = np.float32(0.0)

# initial distribution (normal)
mu_0 = np.float32(np.array([a_0, lam_0]).reshape(2,))
cov_0 = np.float32(np.array([[100.0**2, 0.0], [0.0, (pi/10)**2]]))

# initial distribution (beta)
# mu_0 = np.float32(np.array([8365.203032312347, -1.3972667555306468]).reshape(2,))
# cov_0 = np.float32(np.array([[39218.53730297353**2, 0.0], [0.0, (0.6880488700740756)**2]]))

x1_low = np.float32(7000.0)
x1_hig = np.float32(9000.0)
x2_low = np.float32(-2.0 * pi)
x2_hig = np.float32(4.0 * pi)
ti = np.float32(0.0)
tf = np.float32(3.0 * 3600)
t1s = np.float32([0.0, 3600.0, 7200.0, 10800.0])
mu_gravity = np.float32(398600.4418)
t_orbit = np.float32(2 * pi * np.sqrt(a_0**3 / mu_gravity))
print("nominal orbit period: ", t_orbit)


def nonlinear_mean_dynamics(x):
    xdot = np.zeros(2)
    xdot[1] = np.sqrt(mu_gravity/x[0]**3)
    return xdot
    # A = np.zeros((2,2))
    # A[1,0] = np.sqrt(mu_gravity/(x[0]**3))
    # return np.matmul(A, x)


def linear_cov_dynamics(p, x):
    A = np.zeros((2,2))
    A[1,0] = mu_gravity**0.5*(-1.5)*(x[0]**(-2.5))
    p_dot = np.matmul(p, A.T) + np.matmul(A, p)
    return p_dot


def propagate_guassian():
    # t1s = [3600.0*10.0, 3600.0*24, 3600.0*48]
    dt = 0.1
    T_end = t1s[-1]
    K = int(T_end/dt)
    k1s = np.array(t1s)/dt
    # print(k1s)
    x = mu_0
    p = cov_0
    k_store = 1
    X = [x]
    for k in range(K):
        # Note that both numerical methods give same results due to the dynamics
        # euler forward
        xnew = x + nonlinear_mean_dynamics(x) * dt
        pnew = p + linear_cov_dynamics(p, x) * dt

        # runge-kutta
        # h1 = linear_dynamics(x)
        # h2 = linear_dynamics(x+0.5*h1*dt)
        # h3 = linear_dynamics(x+0.5*h2*dt)
        # h4 = linear_dynamics(x+1.0*h3*dt)
        # xnew = x + dt*(h1 + 2.0*h2 + 2.0*h3 + h4)/6.0

        x = xnew
        p = pnew

        if(k_store < len(k1s)):
            if( abs(k+1 - k1s[k_store]) < 1e-5 ):
                # print("store at k: ", k)
                k_store = k_store + 1
                X.append(x)
                print((k+1)*dt)
                print(x)
                print(p)
    X = np.array(X).T
    # print(X)
    # print(x)
    # print(p)



def main():
    propagate_guassian()



if __name__ == "__main__":
    main()