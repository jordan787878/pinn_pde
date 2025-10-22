import numpy as np
import torch
from scipy.stats import qmc


class Case2_4D_Constants:
    """
    define the constants of Case 2 in the ref. paper
    the state X = [r', phi', r'_dot, phi'_dot] is the normalized shperical coordinates
    """
    _MU_EARTH = np.float32(3.9859e+14)
    _R_EARTH  = np.float32(6.378e+6)
    _A        = np.float32(4.2164e+7)
    _W        = np.float32(np.sqrt(_MU_EARTH/_A**3))
    _T        = np.float32(2*np.pi/_W)
    _R        = np.float32(2e+6)
    _THETA    = np.float32(0.015)
    _PHI      = np.float32(0.0387)
    _MEAN_I   = np.float32([_A, 0.0, 0.0, _W])
    _N_MEAN_I = np.float32([_MEAN_I[0]/_R, 
                            _MEAN_I[1]/_PHI,
                            _MEAN_I[2]/(_R/_T),
                            (_MEAN_I[3]-_W)/(_PHI/_T)])  
    _N_COV_I  = np.float32(np.diag([1e+11/(_R**2),
                                    1e-4/(_PHI**2),
                                    1e+3/(_R/_T)**2,
                                    1e-12/(_PHI/_T)**2]))
    _J2 = np.float32(1.0826e-3)
    _J2_VR = 2.0*(3*_T**2 * _J2 * _MU_EARTH * _R_EARTH**2)/(2*_R**5)

    # Domain of TF = 0.2*T
    _N_X1_RANGE = np.float32(np.array([18.0, 24.0]))
    _N_X2_RANGE = np.float32(np.array([-2.0, 3.0]))
    _N_X3_RANGE = np.float32(np.array([-20.0, 20.0]))
    _N_X4_RANGE = np.float32(np.array([-22.0, 32.0]))
    
    _T_PRIME_SPAN   = np.float32(np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20]))

    _N_Q_NOISE = np.array([1e-8/(_R**2*_T**3), 
                           1e-22/(_PHI**2*_T**3)])
    
    _N_Q_NOISE_TENSOR = torch.tensor(_N_Q_NOISE, dtype=torch.float32, device="cpu")
    
    _NX_RANGE_NP = np.array([
        _N_X1_RANGE,
        _N_X2_RANGE,
        _N_X3_RANGE,
        _N_X4_RANGE,
    ])

    _NX_RANGE = torch.from_numpy(_NX_RANGE_NP)

    _N_MEAN_I_TENSOR = torch.as_tensor(_N_MEAN_I, dtype=torch.float32, device="cpu")
    _N_COV_I_TENSOR = torch.as_tensor(_N_COV_I, dtype=torch.float32, device="cpu")
    _D = 4
    _COV_INV_TENSOR = torch.linalg.inv(_N_COV_I_TENSOR)
    _COV_DET_TENSOR = torch.linalg.det(_N_COV_I_TENSOR)
    _NORM_CONST = 1.0 / torch.sqrt((2 * torch.pi) ** _D * _COV_DET_TENSOR)

    def p_init_torch(self, x):
        diff = x - self._N_MEAN_I_TENSOR
        mahal = torch.einsum("ni,ij,nj->n", diff, self._COV_INV_TENSOR, diff)
        pdf_eval = self._NORM_CONST * torch.exp(-0.5 * mahal)
        return pdf_eval.view(-1, 1)
    
    @property
    def N_Q_NOISE_TENSOR(self):
        return self._N_Q_NOISE_TENSOR

    @property
    def MU_EARTH(self):
        return self._MU_EARTH
    
    @property
    def R_EARTH(self):
        return self._R_EARTH
    
    @property
    def W(self):
        return self._W
    
    @property
    def T(self):
        return self._T
    
    @property
    def R(self):
        return self._R
    
    @property
    def THETA(self):
        return self._THETA
    
    @property
    def PHI(self):
        return self._PHI
    
    @property
    def N_MEAN_I(self):
        return self._N_MEAN_I
    
    @property
    def N_COV_I(self):
        return self._N_COV_I
    
    @property
    def J2(self):
        return self._J2
    
    @property
    def J2_VR(self):
        return self._J2_VR
    
    @property
    def N_X1_RANGE(self):
        return self._N_X1_RANGE
    
    @property
    def N_X2_RANGE(self):
        return self._N_X2_RANGE
    
    @property
    def N_X3_RANGE(self):
        return self._N_X3_RANGE
    
    @property
    def N_X4_RANGE(self):
        return self._N_X4_RANGE
    
    @property
    def T_PRIME_SPAN(self):
        return self._T_PRIME_SPAN
    
    @property
    def N_Q_NOISE(self):
        return self._N_Q_NOISE

    @property
    def NX_RANGE(self):
        return self._NX_RANGE
    
    def test_printout(self):
        print("MU_EARTH: ", self.MU_EARTH)
        print("W: ", self.W)
        print("T: ", self.T)
        print("R: ", self.R)
        print("THETA: ", self.THETA)
        print("PHI: ", self.PHI)
        print("TI: ", self.TI)
        print("TF: ", self.TF)
        print("N_MEAN_I: ", self.N_MEAN_I)
        print("N_COV_I: ", self.N_COV_I)
        print("N X1_RANGE: ", self.N_X1_RANGE)
        print("N X2_RANGE: ", self.N_X2_RANGE)
        print("N X3_RANGE: ", self.N_X3_RANGE)
        print("N X4_RANGE: ", self.N_X4_RANGE)

    def _saturate_to_range(self, x):
        """
        Clamp samples to per-dimension ranges in NX_RANGE.
        Supports np.ndarray (N,D) and torch.Tensor (N,D).
        """
        if isinstance(x, np.ndarray):
            lo = self._NX_RANGE_NP[:, 0]          # shape (D,)
            hi = self._NX_RANGE_NP[:, 1]          # shape (D,)
            return np.clip(x, lo, hi, out=x)      # in-place, returns x
        elif torch.is_tensor(x):
            lo = self._NX_RANGE[:, 0].to(x.dtype).to(x.device)  # shape (D,)
            hi = self._NX_RANGE[:, 1].to(x.dtype).to(x.device)  # shape (D,)
            return torch.max(torch.min(x, hi), lo)              # broadcasting clamp
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def sample_init_points(self, N_samples):
        N_nom = int(0.5 * N_samples)
        N_uni = N_samples - N_nom
        _x_bc_normal = self._saturate_to_range(np.random.multivariate_normal(self.N_MEAN_I, self.N_COV_I, size=N_nom).astype(np.float32))
        _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32)
        _x_bc = np.column_stack([
            np.random.uniform(self.N_X1_RANGE[0], self.N_X1_RANGE[1], N_uni),
            np.random.uniform(self.N_X2_RANGE[0], self.N_X2_RANGE[1], N_uni),
            np.random.uniform(self.N_X3_RANGE[0], self.N_X3_RANGE[1], N_uni),
            np.random.uniform(self.N_X4_RANGE[0], self.N_X4_RANGE[1], N_uni),
        ])
        _x_bc = torch.tensor(_x_bc, dtype=torch.float32)
        x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
        t_bc = (torch.ones(len(x_bc), 1) * self.T_PRIME_SPAN[0])
        return x_bc, t_bc
    
    def sample_res_points_bias(self, N_samples):
        N_nom = int(0.5 * N_samples)
        N_uni = N_samples - N_nom
        _x_normal = self._saturate_to_range(np.random.multivariate_normal(self.N_MEAN_I, self.N_COV_I, size=N_nom).astype(np.float32))
        _x_normal = torch.tensor(_x_normal, dtype=torch.float32)
        _x = np.column_stack([
            np.random.uniform(self.N_X1_RANGE[0], self.N_X1_RANGE[1], N_uni),
            np.random.uniform(self.N_X2_RANGE[0], self.N_X2_RANGE[1], N_uni),
            np.random.uniform(self.N_X3_RANGE[0], self.N_X3_RANGE[1], N_uni),
            np.random.uniform(self.N_X4_RANGE[0], self.N_X4_RANGE[1], N_uni),
        ])
        _x = torch.tensor(_x, dtype=torch.float32)
        x = torch.cat((_x_normal, _x), dim=0)
        t = np.random.uniform(self.T_PRIME_SPAN[0], self.T_PRIME_SPAN[-1], len(x))
        t = torch.tensor(t, dtype=torch.float32).view(-1,1)
        return x, t
    
    def sample_res_points_uniform(self, N_samples):
        _x = np.column_stack([
            np.random.uniform(self.N_X1_RANGE[0], self.N_X1_RANGE[1], N_samples),
            np.random.uniform(self.N_X2_RANGE[0], self.N_X2_RANGE[1], N_samples),
            np.random.uniform(self.N_X3_RANGE[0], self.N_X3_RANGE[1], N_samples),
            np.random.uniform(self.N_X4_RANGE[0], self.N_X4_RANGE[1], N_samples),
        ])
        x = torch.tensor(_x, dtype=torch.float32)
        t = np.random.uniform(self.T_PRIME_SPAN[0], self.T_PRIME_SPAN[-1], len(x))
        t = torch.tensor(t, dtype=torch.float32).view(-1,1)
        return x, t
    
    def sample_x_uniform(self, N_samples):
        X = np.column_stack([
            np.random.uniform(self.N_X1_RANGE[0], self.N_X1_RANGE[1], N_samples),
            np.random.uniform(self.N_X2_RANGE[0], self.N_X2_RANGE[1], N_samples),
            np.random.uniform(self.N_X3_RANGE[0], self.N_X3_RANGE[1], N_samples),
            np.random.uniform(self.N_X4_RANGE[0], self.N_X4_RANGE[1], N_samples),
        ])
        return X
    