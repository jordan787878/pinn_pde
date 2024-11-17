import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.linalg import expm


class MyConstants3D:
    _DIM = 3
    _A = np.float32(0.3*np.diag([1.0, 1.0, 1.0]))
    _A[2,0] = np.float32(-0.1)

    _L = np.float32(np.diag([0.05, 0.05, 0.05]))
    _MEAN_I   = np.float32([-0.2, 0.2, 0.0])
    _COV_I    = np.float32(np.diag([0.1, 0.1, 0.1]))
    _X_RANGE = np.float32(np.array([-1.0, 1.0]))
    _TI = np.float32(0.0)
    _TF = np.float32(1.0)
    _T_SPAN = np.float32(np.array([_TI, 0.3*_TF, 0.7*_TF, _TF]))
    _A_TENSOR = torch.tensor(_A)
    _L_TENSOR = torch.tensor(_L)

    _FOLDER = "exp/3d/"
    _PATH_PNET = _FOLDER+"output/p_net.pth"
    _PATH_PNET_LOSS = _FOLDER+"output/p_net_train_loss.npy"
    _PATH_E1NET = _FOLDER+"output/e1_net.pth"
    _PATH_E1NET_LOSS = _FOLDER+"output/e1_net_train_loss.npy"

    @property
    def DIM(self):
        return self._DIM

    @property
    def A(self):
        return self._A
    
    @property
    def A_TENSOR(self):
        return self._A_TENSOR
    
    @property
    def L(self):
        return self._L
    
    @property
    def L_TENSOR(self):
        return self._L_TENSOR

    @property
    def MEAN_I(self):
        return self._MEAN_I
    
    @property
    def COV_I(self):
        return self._COV_I
    
    @property
    def X_RANGE(self):
        return self._X_RANGE
    
    @property
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
    @property
    def T_SPAN(self):
        return self._T_SPAN
    
    def p_init(self, x):
        pdf_func = multivariate_normal(mean=self.MEAN_I, cov=self.COV_I)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def p_sol(self, x, t):
        exp_At = expm(self.A*t)
        mu_t = np.dot(exp_At, self.MEAN_I)
        cov_t = exp_At @ self.COV_I @ (exp_At.T)

        # (if noise is present)
        if(t > self.TI):
            dt = 0.001
            tspan = np.arange(self.TI, t+dt, dt)
            integral_sum = 0.0*self.COV_I
            for tt in tspan:
                exponent = expm(self.A*(t-tt))
                integrand = dt * exponent @ self.L @ (self.L.T) @ (exponent.T) 
                integral_sum = integral_sum + integrand
            # print("[check] additional cov: ", integral_sum)
            cov_t = cov_t + integral_sum

        pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def prepare_gridpoints(self, grid_num=50):
        x1s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x2s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x3s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x1_grid, x2_grid, x3_grid = np.meshgrid(x1s, x2s, x3s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel()]).T
        return [x1s, x1_grid, x2s, x2_grid, x3s, x3_grid, grid_points]
    
    def get_pinit_max(self):
        grid_points = self.prepare_gridpoints()[-1]
        p_i = self.p_init(grid_points)
        p_i_max = np.max(p_i)
        print("[check] p0 max: {:.3f}".format(p_i_max))
        return p_i_max
    

class MyConstants5D:
    _DIM = 5
    # _A = np.float32(np.diag([1.0, 1.0, 1.0]))
    _A = np.float32(np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.05, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.05, 0.0, 0.0, 0.0]
                              ]))
    _L = np.float32(np.diag([0.05, 0.05, 0.05, 0.05, 0.05]))
    _MEAN_I   = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
    _COV_I    = np.float32(np.diag([0.1, 0.1, 0.1, 0.1, 0.1]))
    _X_RANGE = np.float32(np.array([-1.0, 1.0]))
    _TI = np.float32(0.0)
    _TF = np.float32(1.0)
    _T_SPAN = np.float32(np.array([_TI, 0.3*_TF, 0.7*_TF, _TF]))
    _A_TENSOR = torch.tensor(_A)
    _L_TENSOR = torch.tensor(_L)

    _FOLDER = "exp/5d/"
    _PATH_PNET = _FOLDER+"output/p_net.pth"
    _PATH_PNET_LOSS = _FOLDER+"output/p_net_train_loss.npy"
    _PATH_E1NET = _FOLDER+"output/e1_net.pth"
    _PATH_E1NET_LOSS = _FOLDER+"output/e1_net_train_loss.npy"

    @property
    def DIM(self):
        return self._DIM

    @property
    def A(self):
        return self._A
    
    @property
    def A_TENSOR(self):
        return self._A_TENSOR
    
    @property
    def L(self):
        return self._L
    
    @property
    def L_TENSOR(self):
        return self._L_TENSOR

    @property
    def MEAN_I(self):
        return self._MEAN_I
    
    @property
    def COV_I(self):
        return self._COV_I
    
    @property
    def X_RANGE(self):
        return self._X_RANGE
    
    @property
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
    @property
    def T_SPAN(self):
        return self._T_SPAN
    
    def p_init(self, x):
        pdf_func = multivariate_normal(mean=self.MEAN_I, cov=self.COV_I)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def p_sol(self, x, t):
        exp_At = expm(self.A*t)
        mu_t = np.dot(exp_At, self.MEAN_I)
        cov_t = exp_At @ self.COV_I @ (exp_At.T)

        # (if noise is present)
        if(t > self.TI):
            dt = 0.001
            tspan = np.arange(self.TI, t+dt, dt)
            integral_sum = 0.0*self.COV_I
            for tt in tspan:
                exponent = expm(self.A*(t-tt))
                integrand = dt * exponent @ self.L @ (self.L.T) @ (exponent.T) 
                integral_sum = integral_sum + integrand
            # print("[check] additional cov: ", integral_sum)
            cov_t = cov_t + integral_sum

        pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def prepare_gridpoints(self, grid_num=40):
        x1s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x2s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x3s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x4s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x5s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x1_grid, x2_grid, x3_grid, x4_grid, x5_grid = np.meshgrid(x1s, x2s, x3s, x4s, x5s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel()]).T
        return [x1s, x2s, x3s, x4s, x5s, grid_points]
    
    def get_pinit_max(self):
        grid_point_structure = self.prepare_gridpoints()
        p_i = self.p_init(grid_point_structure[-1])
        p_i_max = np.max(p_i)
        print("[check] p0 max: {:.3f}".format(p_i_max))
        return p_i_max
    

    def generate_random_samples(self, num_samples=1000000):
        x = np.column_stack([
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
        ])
        return x
        # x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        # t = x[:,0]*0.0 + tau
        # t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        # return x, t, x_tensor, t_tensor


class MyConstants7D:
    _DIM = 7

    _A = np.float32(0.3*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    _A[6,0] = np.float32(-0.01)

    _L = np.float32(np.diag([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
    _MEAN_I   = np.float32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _COV_I    = np.float32(np.diag([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]))
    _X_RANGE = np.float32(np.array([-1.0, 1.0]))
    _TI = np.float32(0.0)
    _TF = np.float32(1.0)
    _T_SPAN = np.float32(np.array([_TI, 0.3*_TF, 0.7*_TF, _TF]))
    _A_TENSOR = torch.tensor(_A)
    _L_TENSOR = torch.tensor(_L)

    _FOLDER = "exp/7d_2/"
    _PATH_PNET = _FOLDER+"output/p_net.pth"
    _PATH_PNET_LOSS = _FOLDER+"output/p_net_train_loss.npy"
    _PATH_E1NET = _FOLDER+"output/e1_net.pth"
    _PATH_E1NET_LOSS = _FOLDER+"output/e1_net_train_loss.npy"

    @property
    def DIM(self):
        return self._DIM

    @property
    def A(self):
        return self._A
    
    @property
    def A_TENSOR(self):
        return self._A_TENSOR
    
    @property
    def L(self):
        return self._L
    
    @property
    def L_TENSOR(self):
        return self._L_TENSOR

    @property
    def MEAN_I(self):
        return self._MEAN_I
    
    @property
    def COV_I(self):
        return self._COV_I
    
    @property
    def X_RANGE(self):
        return self._X_RANGE
    
    @property
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
    @property
    def T_SPAN(self):
        return self._T_SPAN
    
    def p_init(self, x):
        pdf_func = multivariate_normal(mean=self.MEAN_I, cov=self.COV_I)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def p_sol(self, x, t):
        exp_At = expm(self.A*t)
        mu_t = np.dot(exp_At, self.MEAN_I)
        cov_t = exp_At @ self.COV_I @ (exp_At.T)

        # (if noise is present)
        if(t > self.TI):
            dt = 0.001
            tspan = np.arange(self.TI, t+dt, dt)
            integral_sum = 0.0*self.COV_I
            for tt in tspan:
                exponent = expm(self.A*(t-tt))
                integrand = dt * exponent @ self.L @ (self.L.T) @ (exponent.T) 
                integral_sum = integral_sum + integrand
            # print("[check] additional cov: ", integral_sum)
            cov_t = cov_t + integral_sum

        pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    # [explode for high dimension]
    # def prepare_gridpoints(self, grid_num=40):
    #     x1s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x2s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x3s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x4s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x5s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x6s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x7s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
    #     x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid, x7_grid = np.meshgrid(x1s, x2s, x3s, x4s, x5s, x6s, x7s, indexing="ij") # the indexing is very important
    #     grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel(), x6_grid.ravel(), x7_grid.ravel()]).T
    #     return [x1s, x2s, x3s, x4s, x5s, x6s, x7s, grid_points]
    
    def get_pinit_max(self):
        grid_points = self.generate_random_samples()
        p_i = self.p_init(grid_points)
        p_i_max = np.max(p_i)
        print("[check] p0 max: {:.3f}".format(p_i_max))
        return p_i_max
    

    def generate_random_samples(self, num_samples=1000000):
        x = np.column_stack([
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
        ])
        return x
    

class MyConstants10D:
    _DIM = 10

    _A = np.float32(0.05*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    _A[9,0] = np.float32(-0.01)

    _L = np.float32(np.diag([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
    _MEAN_I   = np.float32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _COV_I    = np.float32(0.11*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    _X_RANGE = np.float32(np.array([-1.0, 1.0]))
    _TI = np.float32(0.0)
    _TF = np.float32(1.0)
    _T_SPAN = np.float32(np.array([_TI, 0.3*_TF, 0.7*_TF, _TF]))
    _A_TENSOR = torch.tensor(_A)
    _L_TENSOR = torch.tensor(_L)

    _FOLDER = "exp/10d_2/"
    _PATH_PNET = _FOLDER+"output/p_net.pth"
    _PATH_PNET_LOSS = _FOLDER+"output/p_net_train_loss.npy"
    _PATH_E1NET = _FOLDER+"output/e1_net.pth"
    _PATH_E1NET_LOSS = _FOLDER+"output/e1_net_train_loss.npy"

    @property
    def DIM(self):
        return self._DIM

    @property
    def A(self):
        return self._A
    
    @property
    def A_TENSOR(self):
        return self._A_TENSOR
    
    @property
    def L(self):
        return self._L
    
    @property
    def L_TENSOR(self):
        return self._L_TENSOR

    @property
    def MEAN_I(self):
        return self._MEAN_I
    
    @property
    def COV_I(self):
        return self._COV_I
    
    @property
    def X_RANGE(self):
        return self._X_RANGE
    
    @property
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
    @property
    def T_SPAN(self):
        return self._T_SPAN
    
    def p_init(self, x):
        pdf_func = multivariate_normal(mean=self.MEAN_I, cov=self.COV_I)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def p_sol(self, x, t):
        exp_At = expm(self.A*t)
        mu_t = np.dot(exp_At, self.MEAN_I)
        cov_t = exp_At @ self.COV_I @ (exp_At.T)

        # (if noise is present)
        if(t > self.TI):
            dt = 0.001
            tspan = np.arange(self.TI, t+dt, dt)
            integral_sum = 0.0*self.COV_I
            for tt in tspan:
                exponent = expm(self.A*(t-tt))
                integrand = dt * exponent @ self.L @ (self.L.T) @ (exponent.T) 
                integral_sum = integral_sum + integrand
            # print("[check] additional cov: ", integral_sum)
            cov_t = cov_t + integral_sum

        pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def get_pinit_max(self):
        # grid_points = self.generate_random_samples()
        # p_i = self.p_init(grid_points)
        p_i_max = self.p_init(self.MEAN_I)
        p_i_max = np.max(p_i_max)
        print("[check] p0 max: {:.6f}".format(p_i_max))
        return p_i_max
    
    def generate_random_samples(self, num_samples=10000000):
        x = np.column_stack([
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
            np.random.uniform(self.X_RANGE[0], self.X_RANGE[1], num_samples),
        ])
        return x
    

# Time-dependent OU process
class MyConstants3D_Time:
    _DIM = 3
    _A = np.float32(0.3*np.diag([1.0, 1.0, 1.0]))
    _A[2,0] = np.float32(-0.1)
    _DA = np.float32([[0.0, 0.5, 0.0],
                      [0.0, 0.0, 0.5],
                      [0.0, -0.3, 0.0]])

    _L = np.float32(np.diag([0.05, 0.05, 0.05]))
    _MEAN_I   = np.float32([-0.2, 0.2, 0.0])
    _COV_I    = np.float32(np.diag([0.1, 0.1, 0.1]))
    _X_RANGE = np.float32(np.array([-1.0, 1.0]))
    _TI = np.float32(0.0)
    _TF = np.float32(1.0)
    _T_SPAN = np.float32(np.array([_TI, 0.3*_TF, 0.7*_TF, _TF]))
    _A_TENSOR = torch.tensor(_A)
    _DA_TENSOR = torch.tensor(_DA)
    _L_TENSOR = torch.tensor(_L)

    _FOLDER = "exp/3d_2/"
    _PATH_PNET = _FOLDER+"output/p_net.pth"
    _PATH_PNET_LOSS = _FOLDER+"output/p_net_train_loss.npy"
    _PATH_E1NET = _FOLDER+"output/e1_net.pth"
    _PATH_E1NET_LOSS = _FOLDER+"output/e1_net_train_loss.npy"

    @property
    def DIM(self):
        return self._DIM

    @property
    def A(self):
        return self._A
    
    @property
    def DA(self):
        return self._DA
    
    @property
    def A_TENSOR(self):
        return self._A_TENSOR
    
    @property
    def DA_TENSOR(self):
        return self._DA_TENSOR
    
    @property
    def L(self):
        return self._L
    
    @property
    def L_TENSOR(self):
        return self._L_TENSOR

    @property
    def MEAN_I(self):
        return self._MEAN_I
    
    @property
    def COV_I(self):
        return self._COV_I
    
    @property
    def X_RANGE(self):
        return self._X_RANGE
    
    @property
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
    @property
    def T_SPAN(self):
        return self._T_SPAN
    
    def get_A_time(self, t):
        return self.A + self.DA*np.exp(-t**3)
    
    def p_init(self, x):
        pdf_func = multivariate_normal(mean=self.MEAN_I, cov=self.COV_I)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def p_sol(self, x, t):
        print("[check] new p_sol")

        # 1. Compute mu(t): x_dot = A(t) x + G w
        mu_t = self.MEAN_I # init
        if(t > self.TI):
            dt = 0.0001
            tspan = np.arange(self.TI, t+dt, dt)
            for tt in tspan:
                mu_t = mu_t + self.get_A_time(tt) @ mu_t * dt

        # 2. Compute P(t): P_dot = PA^T + AP + L L^T
        cov_t = self.COV_I # init
        if(t > self.TI):
            dt = 0.0001
            tspan = np.arange(self.TI, t+dt, dt)
            for tt in tspan:
                cov_t = cov_t + (cov_t @ (self.get_A_time(tt).T) + self.get_A_time(tt) @ cov_t + self.L @ self.L.T)*dt

        pdf_func = multivariate_normal(mean=mu_t, cov=cov_t)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def prepare_gridpoints(self, grid_num=50):
        x1s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x2s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x3s = np.linspace(self.X_RANGE[0], self.X_RANGE[1], num=grid_num, endpoint=True).astype(np.float32)
        x1_grid, x2_grid, x3_grid = np.meshgrid(x1s, x2s, x3s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel()]).T
        return [x1s, x1_grid, x2s, x2_grid, x3s, x3_grid, grid_points]
    
    def get_pinit_max(self):
        grid_points = self.prepare_gridpoints()[-1]
        p_i = self.p_init(grid_points)
        p_i_max = np.max(p_i)
        print("[check] p0 max: {:.3f}".format(p_i_max))
        return p_i_max
  