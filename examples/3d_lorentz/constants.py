import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.linalg import expm
from tqdm import tqdm


class MyConstantsLorentz:
    _DIM = 3
    _BETA  = np.float32(8/3)
    _SIGMA = np.float32(10.0)
    _RHO =   np.float32(28.0)
    _L = np.float32(np.diag([0.3, 0.3, 0.3]))
    _MEAN_I   = np.float32([-6.0, -8.0, 24.0])
    _COV_I    = np.float32(np.diag([0.5, 0.5, 0.5]))
    _X_RANGE = np.float32(np.array([[-13.0, -2.0],
                                    [-15.0, -2.0],
                                    [20.0, 35.0]]))
    _TI = np.float32(0.0)
    _TF = np.float32(1.0)
    _T_SPAN = np.float32(np.array([_TI, 0.2*_TF, 0.5*_TF, _TF]))
    # _A_TENSOR = torch.tensor(_A)
    _L_TENSOR = torch.tensor(_L)

    _FOLDER = "exp/1/"
    _PATH_PNET = _FOLDER+"output/p_net.pth"
    _PATH_PNET_LOSS = _FOLDER+"output/p_net_train_loss.npy"
    _PATH_E1NET = _FOLDER+"output/e1_net.pth"
    _PATH_E1NET_LOSS = _FOLDER+"output/e1_net_train_loss.npy"

    @property
    def DIM(self):
        return self._DIM
    
    @property
    def BETA(self):
        return self._BETA
    
    @property
    def SIGMA(self):
        return self._SIGMA
    
    @property
    def RHO(self):
        return self._RHO
    
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
    
    def f_sde(self, x):
        f1 = self.SIGMA*(x[1] - x[0])
        f2 = x[0]*(self.RHO - x[2]) - x[1]
        f3 = x[0]*x[1] - self.BETA*x[2]
        return np.array([f1, f2, f3]).astype(x.dtype)
    
    def p_init(self, x):
        pdf_func = multivariate_normal(mean=self.MEAN_I, cov=self.COV_I)
        pdf_eval = pdf_func.pdf(x).reshape(-1,1).astype(x.dtype)
        return pdf_eval
    
    def prepare_gridpoints(self, grid_num=50):
        x1s = np.linspace(self.X_RANGE[0,0], self.X_RANGE[0,1], num=grid_num, endpoint=True).astype(np.float32)
        x2s = np.linspace(self.X_RANGE[1,0], self.X_RANGE[1,1], num=grid_num, endpoint=True).astype(np.float32)
        x3s = np.linspace(self.X_RANGE[2,0], self.X_RANGE[2,1], num=grid_num, endpoint=True).astype(np.float32)
        x1_grid, x2_grid, x3_grid = np.meshgrid(x1s, x2s, x3s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel()]).T
        return [x1s, x1_grid, x2s, x2_grid, x3s, x3_grid, grid_points]

    def get_pinit_max(self):
        grid_points = self.prepare_gridpoints()[-1]
        p_i = self.p_init(grid_points)
        p_i_max = np.max(p_i)
        print("[check] p0 max: {:.3f}".format(p_i_max))
        return p_i_max

    def construct_p_sol(self):
        for t in self.T_SPAN:
            x1s, x2s, x3s, pdf = self.p_sol_monte(t)
            np.save(self._FOLDER+"data/pdf_t{:.1f}.npy".format(t), pdf)
            if(t == 0):
                np.save(self._FOLDER+"data/x1s.npy", x1s)
                np.save(self._FOLDER+"data/x2s.npy", x2s)
                np.save(self._FOLDER+"data/x3s.npy", x3s)

    def load_gridpoints_from_monte(self):
        x1s = np.load(self._FOLDER+"data/x1s.npy")
        x2s = np.load(self._FOLDER+"data/x2s.npy")
        x3s = np.load(self._FOLDER+"data/x3s.npy")
        x1_grid, x2_grid, x3_grid = np.meshgrid(x1s, x2s, x3s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel()]).T
        return [x1s, x1_grid, x2s, x2_grid, x3s, x3_grid, grid_points]
    
    def load_p_sol_monte(self, t):
        pdf = np.load(self._FOLDER+"data/pdf_t{:.1f}.npy".format(t))
        return pdf

    def p_sol_monte(self, t=0.0, linespace_num=81, stat_sample=100000000):
        X = np.random.multivariate_normal(self.MEAN_I, self.COV_I, size=stat_sample).astype(np.float32)
        
        # NOTE: numerical integration to time t
        dtt = 0.0005
        t_span = np.arange(self.TI, t, dtt)
        for i in tqdm(range(stat_sample), desc="Processing samples"):
            x = X[i,:]
            for t in t_span:
                w1 = np.random.normal(0, np.sqrt(dtt))
                w2 = np.random.normal(0, np.sqrt(dtt))
                w3 = np.random.normal(0, np.sqrt(dtt))
                w = np.array([w1, w2, w3]).reshape(3,)
                x = x + self.f_sde(x)*dtt + np.matmul(self.L, w)
            X[i,:] = x
        
        # Define bins for each dimension
        bins_x1 = np.linspace(self.X_RANGE[0,0], self.X_RANGE[0,1], num=linespace_num, endpoint=True).astype(np.float32)
        bins_x2 = np.linspace(self.X_RANGE[1,0], self.X_RANGE[1,1], num=linespace_num, endpoint=True).astype(np.float32)
        bins_x3 = np.linspace(self.X_RANGE[2,0], self.X_RANGE[2,1], num=linespace_num, endpoint=True).astype(np.float32)

        # Digitize to find bin indices for each dimension
        bin_indices_x1 = np.digitize(X[:, 0], bins_x1) - 1
        bin_indices_x2 = np.digitize(X[:, 1], bins_x2) - 1
        bin_indices_x3 = np.digitize(X[:, 2], bins_x3) - 1

        # Initialize frequency array for 4D
        pdf = np.zeros((len(bins_x1) - 1, len(bins_x2) - 1, len(bins_x3) - 1)).astype(np.float32)

        # Count occurrences in each bin
        for i in tqdm(range(stat_sample), desc="Counting samples"):
            idx_x1 = bin_indices_x1[i]
            idx_x2 = bin_indices_x2[i]
            idx_x3 = bin_indices_x3[i]

            # Check if the indices are valid
            if (0 <= idx_x1 < pdf.shape[0] and
                0 <= idx_x2 < pdf.shape[1] and
                0 <= idx_x3 < pdf.shape[2]):
                pdf[idx_x1, idx_x2, idx_x3] += 1

        # Normalize the frequency to get the probability density
        dx1 = bins_x1[1] - bins_x1[0]
        dx2 = bins_x2[1] - bins_x2[0]
        dx3 = bins_x3[1] - bins_x3[0]
        pdf /= (dx1 * dx2 * dx3 * stat_sample) 

        # Check the sum of the probability density function
        print("[check] sum pdf(monte) = 1.0", np.sum(pdf) * dx1 * dx2 * dx3)

        # Calculate the midpoints for bins (optional, depending on your needs)
        midpoints_x1 = (bins_x1[:-1] + bins_x1[1:]) / 2
        midpoints_x2 = (bins_x2[:-1] + bins_x2[1:]) / 2
        midpoints_x3 = (bins_x3[:-1] + bins_x3[1:]) / 2
        return midpoints_x1, midpoints_x2, midpoints_x3, pdf