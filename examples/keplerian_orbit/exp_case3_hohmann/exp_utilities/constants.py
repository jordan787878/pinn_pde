import numpy as np
import torch


class Case2_Planar_Transfer:
    _MU_EARTH = np.float32(398600.4418) # km3/s2
    _A        = np.float32(7000.0)
    _W        = np.float32(np.sqrt(_MU_EARTH / _A**3))
    _T        = np.float32(2*np.pi/_W)
    _R        = np.float32(7000.0)
    _PHI      = np.float32(1e-2)
    _THRUST_ACCEL = np.float32(0.2/1000.0) # km/s2

    _N_MEAN_I = np.array([1.0, -1.67459435e-05, -2.45221932e-06,  6.95868034e-04], dtype=np.float32)
    _N_COV_I  = np.array([[ 2.0525357247940360e-07, -4.6723394355242229e-09, -1.6435472554973773e-08, -1.2867636903456074e-04],
                [-4.6723394355242229e-09,  2.0313095740378804e-03,  1.2741515425425539e-04, -2.3081932630343769e-04],
                [-1.6435472554973773e-08,  1.2741515425425539e-04,  7.7740487012882304e-05,  2.7601120338146077e-06],
                [-1.2867636903456074e-04, -2.3081932630343769e-04,  2.7601120338146077e-06,  7.7202150591966046e-01]], dtype=np.float32)

    # Domain of TF = 0.2*T
    _N_X1_RANGE = np.float32(np.array([18.0, 24.0]))
    _N_X2_RANGE = np.float32(np.array([-2.0, 3.0]))
    _N_X3_RANGE = np.float32(np.array([-20.0, 20.0]))
    _N_X4_RANGE = np.float32(np.array([-22.0, 32.0]))
    
    _T_PRIME_SPAN   = np.float32(np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0]))
    
    _NX_RANGE_NP = np.array([
        _N_X1_RANGE,
        _N_X2_RANGE,
        _N_X3_RANGE,
        _N_X4_RANGE,
    ])
    _NX_RANGE = torch.from_numpy(_NX_RANGE_NP)

    # _N_MEAN_I_TENSOR = torch.as_tensor(_N_MEAN_I, dtype=torch.float32, device="cpu")
    # _N_COV_I_TENSOR = torch.as_tensor(_N_COV_I, dtype=torch.float32, device="cpu")
    # _D = 4
    # _COV_INV_TENSOR = torch.linalg.inv(_N_COV_I_TENSOR)
    # _COV_DET_TENSOR = torch.linalg.det(_N_COV_I_TENSOR)
    # _NORM_CONST = 1.0 / torch.sqrt((2 * torch.pi) ** _D * _COV_DET_TENSOR)

    # def p_init_torch(self, x):
    #     diff = x - self._N_MEAN_I_TENSOR
    #     mahal = torch.einsum("ni,ij,nj->n", diff, self._COV_INV_TENSOR, diff)
    #     pdf_eval = self._NORM_CONST * torch.exp(-0.5 * mahal)
    #     return pdf_eval.view(-1, 1)

    @property
    def MU_EARTH(self):
        return self._MU_EARTH
    
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
    def PHI(self):
        return self._PHI
    
    @property
    def THRUST_ACCEL(self):
        return self._THRUST_ACCEL
    
    @property
    def N_MEAN_I(self):
        return self._N_MEAN_I
    
    @property
    def N_COV_I(self):
        return self._N_COV_I
    
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
    def NX_RANGE(self):
        return self._NX_RANGE
    
    def dyn_f1(self, x):
        v_rho = x[:,2]
        return v_rho
    
    def dyn_f2(self, x):
        v_phi = x[:,3]
        return v_phi
    
    def dyn_f3(self, x):
        rho = x[:,0]
        phi = x[:,1]
        v_rho = x[:,2]
        v_phi = x[:,3]
        # R, PHI, W, T, MU_EARTH, THRUST_ACCEL

        r = self._R * rho
        rdot = (self._R / self._T) * v_rho
        thetadot = self._W + (self.PHI / self._T) * v_phi

        # Physical speed magnitude
        V = np.sqrt(rdot*rdot + (r*thetadot)**2)

        # Thrust components (physical)
        a_r     = self._THRUST_ACCEL * (rdot / V)

        # Physical dynamics
        rddot      = r * thetadot**2 - self._MU_EARTH / (r*r) + a_r
        return (self._T**2 / self._R) * rddot
    
    def dyn_f4(self, x):
        rho = x[:,0]
        phi = x[:,1]
        v_rho = x[:,2]
        v_phi = x[:,3]
        # R, PHI, W, T, MU_EARTH, THRUST_ACCEL

        r = self._R * rho
        rdot = (self._R / self._T) * v_rho
        thetadot = self._W + (self.PHI / self._T) * v_phi

        # Physical speed magnitude
        V = np.sqrt(rdot*rdot + (r*thetadot)**2)

        # Thrust components (physical)
        a_theta = self._THRUST_ACCEL * (r * thetadot / V)

        # Physical dynamics
        thetaddot  = (a_theta - 2.0 * rdot * thetadot) / r
        return (self._T**2 / self._PHI) * thetaddot
    
    
    def test_printout(self):
        print("MU_EARTH: ", self.MU_EARTH)
        print("W: ", self.W)
        print("T: ", self.T)
        print("R: ", self.R)
        print("PHI: ", self.PHI)
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
    
    def cart2polar_state(self, x_cart):
        """
        Cartesian (x,y,vx,vy) -> polar state [r, theta, rdot, thetadot].
        """
        x, y, vx, vy = x_cart
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        rdot = (x*vx + y*vy) / r
        thetadot = (x*vy - y*vx) / (r*r)  # z-component of angular momentum / r^2
        return np.array([r, theta, rdot, thetadot])
    
    def to_rot_norm(self, t, x_polar):
        """
        (r,theta,rdot,thetadot) at physical time t -> (rho,phi,rhod,phid) where
        rhod = d(rho)/d(tau), phid = d(phi)/d(tau), tau = t / S_T
        """
        r, th, rd, thd = x_polar
        SR, ST, SW, STi = self.R, self.PHI, self.W, self.T
        return np.array([
            r / SR,
            (th - SW * t) / ST,
            (STi / SR) * rd,
            (STi / ST) * (thd - SW)
        ])
    
    def get_initial_mean_covariance_normalized(self, N_samples=1000000):
        mean_vector = np.array([7000.0, 0., 0., 7.54605329]).astype(np.float32)
        covariance_matrix = np.diag([10. , 10., 1e-4, 1e-4]).astype(np.float32)
        samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=N_samples)
        samples_nsph = np.empty_like(samples)
        for idx, x in enumerate(samples):
            _x = self.to_rot_norm(0., self.cart2polar_state(x))
            samples_nsph[idx, :] = _x
        # Compute mean and covariance after the loop
        mean_nsph = np.mean(samples_nsph, axis=0)
        cov_nsph = np.cov(samples_nsph, rowvar=False)

        def mvn_pdf_at_mean(C):
            s, ld = np.linalg.slogdet(C)
            if s <= 0: raise np.linalg.LinAlgError("covariance must be PD")
            d = C.shape[0]
            return float(np.exp(-0.5*(d*np.log(2*np.pi)+ld)))
        
        print(mean_nsph)
        s = np.array2string(cov_nsph, precision=17, separator=', ', max_line_width=10**9)
        print(f"cov_nsph = np.array({s}, dtype=np.float64)")
        print(mvn_pdf_at_mean(cov_nsph))
        print(np.linalg.cond(cov_nsph))
