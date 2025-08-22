import numpy as np
import torch
import math
# from scipy.stats import qmc


class Case1_6D_Constants_Equin:
    """
    define the constants of Case 2 in the ref. paper
    the state X = [r', phi', r'_dot, phi'_dot] is the normalized shperical coordinates
    """
    _NAME = "Case1-6D-Equinoctial"
    _PI = np.float32(math.pi)
    _MU_EARTH = np.float32(3.9859e+14)
    _R_EARTH  = np.float32(6.378e+6)
    _A        = np.float32(4.2164e+7)
    _W        = np.float32(np.sqrt(_MU_EARTH/_A**3))
    _T        = np.float32(2*np.pi/_W)
    _R        = np.float32(2e+6)
    _THETA    = np.float32(0.015)
    _PHI      = np.float32(0.0387)
    _TI       = np.float32(0.0)
    _TF       = np.float32(0.1*_T)
    _CONSTANTS_DATA = np.load("data/constants.npz")
    _MEAN_I   = _CONSTANTS_DATA["mu_vec"]
    _COV_I    = _CONSTANTS_DATA["cov_diag"]
    _J2 = np.float32(1.0826e-3)
    _J2_VR = 2.0*(3*_T**2 * _J2 * _MU_EARTH * _R_EARTH**2)/(2*_R**5)

    # Domain of TF = 0.1*T
    _X1_RANGE = np.float32(np.array([_CONSTANTS_DATA["min_vec"][0], _CONSTANTS_DATA["max_vec"][0]]))
    _X2_RANGE = np.float32(np.array([_CONSTANTS_DATA["min_vec"][1], _CONSTANTS_DATA["max_vec"][1]]))
    _X3_RANGE = np.float32(np.array([_CONSTANTS_DATA["min_vec"][2], _CONSTANTS_DATA["max_vec"][2]]))
    _X4_RANGE = np.float32(np.array([_CONSTANTS_DATA["min_vec"][3], _CONSTANTS_DATA["max_vec"][3]]))
    _X5_RANGE = np.float32(np.array([_CONSTANTS_DATA["min_vec"][4], _CONSTANTS_DATA["max_vec"][4]]))
    _X6_RANGE = np.float32(np.array([-0.25, 1.])) # by MC
    
    # _MAX_PX1  = np.float32(3.0)
    # _MAX_PX2  = np.float32(1.8)
    # _MAX_PX3  = np.float32(0.35)
    # _MAX_PX4  = np.float32(0.2)
    _T_PRIME_SPAN   = np.float32(np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1]))

    @property
    def NAME(self):
        return self._NAME

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
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
    @property
    def MEAN_I(self):
        return self._MEAN_I
    
    @property
    def COV_I(self):
        return self._COV_I
    
    @property
    def J2(self):
        return self._J2
    
    @property
    def J2_VR(self):
        return self._J2_VR
    
    @property
    def X1_RANGE(self):
        return self._X1_RANGE
    
    @property
    def X2_RANGE(self):
        return self._X2_RANGE
    
    @property
    def X3_RANGE(self):
        return self._X3_RANGE
    
    @property
    def X4_RANGE(self):
        return self._X4_RANGE
    
    @property
    def X5_RANGE(self):
        return self._X5_RANGE
    
    @property
    def X6_RANGE(self):
        return self._X6_RANGE
    
    @property
    def T_PRIME_SPAN(self):
        return self._T_PRIME_SPAN
    
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
        print("X1_RANGE: ", self.X1_RANGE)
        print("X2_RANGE: ", self.X2_RANGE)
        print("X3_RANGE: ", self.X3_RANGE)
        print("X4_RANGE: ", self.X4_RANGE)

    def sample_init_points(self, N_samples):
        _x_bc_normal = np.random.multivariate_normal(self.MEAN_I, self.COV_I, size=N_samples).astype(np.float32)
        _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
        _x_bc = np.column_stack([
            np.random.uniform(self.X1_RANGE[0], self.X1_RANGE[1], N_samples),
            np.random.uniform(self.X2_RANGE[0], self.X2_RANGE[1], N_samples),
            np.random.uniform(self.X3_RANGE[0], self.X3_RANGE[1], N_samples),
            np.random.uniform(self.X4_RANGE[0], self.X4_RANGE[1], N_samples),
            np.random.uniform(self.X5_RANGE[0], self.X5_RANGE[1], N_samples),
            np.random.uniform(self.X6_RANGE[0], self.X6_RANGE[1], N_samples),
        ])
        _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)
        x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
        t_bc = (torch.ones(len(x_bc), 1, dtype=torch.float32) * self.TI)
        return x_bc, t_bc
    
    def sample_res_points(self, N_samples):
        _x_normal = np.random.multivariate_normal(self.MEAN_I, self.COV_I, size=N_samples).astype(np.float32)
        _x_normal = torch.tensor(_x_normal, dtype=torch.float32, requires_grad=True)
        _x = np.column_stack([
            np.random.uniform(self.X1_RANGE[0], self.X1_RANGE[1], N_samples),
            np.random.uniform(self.X2_RANGE[0], self.X2_RANGE[1], N_samples),
            np.random.uniform(self.X3_RANGE[0], self.X3_RANGE[1], N_samples),
            np.random.uniform(self.X4_RANGE[0], self.X4_RANGE[1], N_samples),
            np.random.uniform(self.X5_RANGE[0], self.X5_RANGE[1], N_samples),
            np.random.uniform(self.X6_RANGE[0], self.X6_RANGE[1], N_samples),
        ])
        _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
        x = torch.cat((_x_normal, _x), dim=0)
        t = np.random.uniform(self._T_PRIME_SPAN[0], self._T_PRIME_SPAN[-1], len(x))
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
        return x, t
    
    def get_xinputs_on_grids(self, grid_folder):
        x1s = np.load(grid_folder+"x1s.npy")
        x2s = np.load(grid_folder+"x2s.npy")
        x3s = np.load(grid_folder+"x3s.npy")
        x4s = np.load(grid_folder+"x4s.npy")
        x5s = np.load(grid_folder+"x5s.npy")
        x6s = np.load(grid_folder+"x6s.npy")
        x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid = np.meshgrid(x1s, x2s, x3s, x4s, x5s, x6s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel(), x6_grid.ravel()]).T
        x_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False).reshape(-1,6)
        return x_tensor
    

    def get_grid_numpy(self, grid_folder):
        x1s = np.load(grid_folder+"x1s.npy").astype(np.float32)
        x2s = np.load(grid_folder+"x2s.npy").astype(np.float32)
        x3s = np.load(grid_folder+"x3s.npy").astype(np.float32)
        x4s = np.load(grid_folder+"x4s.npy").astype(np.float32)
        x5s = np.load(grid_folder+"x5s.npy").astype(np.float32)
        x6s = np.load(grid_folder+"x6s.npy").astype(np.float32)
        x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid = np.meshgrid(x1s, x2s, x3s, x4s, x5s, x6s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel(), x6_grid.ravel()]).T
        return grid_points


class Case1_6D_Constants:
    """
    define the constants of Case 2 in the ref. paper
    the state X = [r', phi', r'_dot, phi'_dot] is the normalized shperical coordinates
    """
    _PI = np.float32(math.pi)
    _MU_EARTH = np.float32(3.9859e+14)
    _R_EARTH  = np.float32(6.378e+6)
    _A        = np.float32(4.2164e+7)
    _W        = np.float32(np.sqrt(_MU_EARTH/_A**3))
    _T        = np.float32(2*np.pi/_W)
    _R        = np.float32(2e+6)
    _THETA    = np.float32(0.015)
    _PHI      = np.float32(0.0387)
    _TI       = np.float32(0.0)
    _TF       = np.float32(0.1*_T)
    _MEAN_I   = np.float32([_A, 0.5*_PI, 0.0, 0.0, 0.0, _W])
    _N_MEAN_I = np.float32([_MEAN_I[0]/_R,
                            _MEAN_I[1]/_THETA, 
                            _MEAN_I[2]/_PHI,
                            _MEAN_I[3]/(_R/_T),
                            _MEAN_I[4]/(_THETA/_T),
                            (_MEAN_I[5]-_W)/(_PHI/_T)])  
    _N_COV_I  = np.float32(np.diag([1e+11/(_R**2),
                                    1e-5/(_THETA**2),
                                    1e-4/(_PHI**2),
                                    1e+3/(_R/_T)**2,
                                    1e-13/(_THETA/_T)**2,
                                    1e-12/(_PHI/_T)**2]))
    _J2 = np.float32(1.0826e-3)
    _J2_VR = 2.0*(3*_T**2 * _J2 * _MU_EARTH * _R_EARTH**2)/(2*_R**5)

    # Domain of TF = 0.1*T
    _X1_RANGE = np.float32(np.array([20.1, 22.1]))
    _X2_RANGE = np.float32(np.array([103.5, 105.9]))
    _X3_RANGE = np.float32(np.array([-2.2, 2.2]))
    _X4_RANGE = np.float32(np.array([-10., 10.]))
    _X5_RANGE = np.float32(np.array([-6., 6.]))
    _X6_RANGE = np.float32(np.array([-8., 8.]))
    
    # _MAX_PX1  = np.float32(3.0)
    # _MAX_PX2  = np.float32(1.8)
    # _MAX_PX3  = np.float32(0.35)
    # _MAX_PX4  = np.float32(0.2)
    _T_PRIME_SPAN   = np.float32(np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1]))

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
    def TI(self):
        return self._TI
    
    @property
    def TF(self):
        return self._TF
    
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
    def X1_RANGE(self):
        return self._X1_RANGE
    
    @property
    def X2_RANGE(self):
        return self._X2_RANGE
    
    @property
    def X3_RANGE(self):
        return self._X3_RANGE
    
    @property
    def X4_RANGE(self):
        return self._X4_RANGE
    
    @property
    def X5_RANGE(self):
        return self._X5_RANGE
    
    @property
    def X6_RANGE(self):
        return self._X6_RANGE
    
    # @property
    # def MAX_PX1(self):
    #     return self._MAX_PX1
    
    # @property
    # def MAX_PX2(self):
    #     return self._MAX_PX2
    
    # @property
    # def MAX_PX3(self):
    #     return self._MAX_PX3
    
    # @property
    # def MAX_PX4(self):
    #     return self._MAX_PX4
    
    @property
    def T_PRIME_SPAN(self):
        return self._T_PRIME_SPAN
    
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
        print("X1_RANGE: ", self.X1_RANGE)
        print("X2_RANGE: ", self.X2_RANGE)
        print("X3_RANGE: ", self.X3_RANGE)
        print("X4_RANGE: ", self.X4_RANGE)

    def sample_init_points(self, N_samples):
        _x_bc_normal = np.random.multivariate_normal(self.N_MEAN_I, self.N_COV_I, size=N_samples).astype(np.float32)
        _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
        _x_bc = np.column_stack([
            np.random.uniform(self.X1_RANGE[0], self.X1_RANGE[1], N_samples),
            np.random.uniform(self.X2_RANGE[0], self.X2_RANGE[1], N_samples),
            np.random.uniform(self.X3_RANGE[0], self.X3_RANGE[1], N_samples),
            np.random.uniform(self.X4_RANGE[0], self.X4_RANGE[1], N_samples),
            np.random.uniform(self.X5_RANGE[0], self.X5_RANGE[1], N_samples),
            np.random.uniform(self.X6_RANGE[0], self.X6_RANGE[1], N_samples),
        ])
        _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)
        x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
        t_bc = (torch.ones(len(x_bc), 1, dtype=torch.float32) * self.TI)
        return x_bc, t_bc
    
    def sample_res_points(self, N_samples):
        _x_normal = np.random.multivariate_normal(self.N_MEAN_I, self.N_COV_I, size=N_samples).astype(np.float32)
        _x_normal = torch.tensor(_x_normal, dtype=torch.float32, requires_grad=True)
        _x = np.column_stack([
            np.random.uniform(self.X1_RANGE[0], self.X1_RANGE[1], N_samples),
            np.random.uniform(self.X2_RANGE[0], self.X2_RANGE[1], N_samples),
            np.random.uniform(self.X3_RANGE[0], self.X3_RANGE[1], N_samples),
            np.random.uniform(self.X4_RANGE[0], self.X4_RANGE[1], N_samples),
            np.random.uniform(self.X5_RANGE[0], self.X5_RANGE[1], N_samples),
            np.random.uniform(self.X6_RANGE[0], self.X6_RANGE[1], N_samples),
        ])
        _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
        x = torch.cat((_x_normal, _x), dim=0)
        t = np.random.uniform(self._T_PRIME_SPAN[0], self._T_PRIME_SPAN[-1], len(x))
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
        return x, t
    
    def get_xinputs_on_grids(self, grid_folder):
        x1s = np.load(grid_folder+"x1s.npy")
        x2s = np.load(grid_folder+"x2s.npy")
        x3s = np.load(grid_folder+"x3s.npy")
        x4s = np.load(grid_folder+"x4s.npy")
        x5s = np.load(grid_folder+"x5s.npy")
        x6s = np.load(grid_folder+"x6s.npy")
        x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid = np.meshgrid(x1s, x2s, x3s, x4s, x5s, x6s, indexing="ij") # the indexing is very important
        grid_points = np.vstack([x1_grid.ravel(), x2_grid.ravel(), x3_grid.ravel(), x4_grid.ravel(), x5_grid.ravel(), x6_grid.ravel()]).T
        x_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        return x_tensor


    # # [new sampling methods]
    # def sample_init_points_seq(self, N_samples, T_seq, sobol_seed=0):
    #     _x_bc_normal = np.random.multivariate_normal(self.N_MEAN_I, self.N_COV_I, size=N_samples).astype(np.float32)
    #     _x_bc_normal = torch.tensor(_x_bc_normal, dtype=torch.float32, requires_grad=False)
    #     _x_bc = np.column_stack([
    #         np.random.uniform(self.X1_RANGE[0], self.X1_RANGE[1], N_samples),
    #         np.random.uniform(self.X2_RANGE[0], self.X2_RANGE[1], N_samples),
    #         np.random.uniform(self.X3_RANGE[0], self.X3_RANGE[1], N_samples),
    #         np.random.uniform(self.X4_RANGE[0], self.X4_RANGE[1], N_samples),
    #     ])
    #     _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)

    #     # sobol sequence 
    #     lower_bounds = np.array([self.X1_RANGE[0], self.X2_RANGE[0], self.X3_RANGE[0], self.X4_RANGE[0]])
    #     upper_bounds = np.array([self.X1_RANGE[1], self.X2_RANGE[1], self.X3_RANGE[1], self.X4_RANGE[1]])
    #     # Create a Sobol sequence sampler for 4 dimensions
    #     sobol_sampler = qmc.Sobol(d=4, scramble=True, seed=sobol_seed)
    #     # Generate samples in the unit hypercube [0, 1]^4
    #     samples_unit = sobol_sampler.random_base2(m=10)
    #     # Scale the samples to the specified ranges for each dimension
    #     _x_bc_sol = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    #     _x_bc_sol = torch.tensor(_x_bc_sol, dtype=torch.float32, requires_grad=False)
    #     x_bc = torch.cat((_x_bc_normal, _x_bc, _x_bc_sol), dim=0)
    #     # x_bc = torch.cat((_x_bc_normal, _x_bc), dim=0)
    #     t_bc = (torch.ones(len(x_bc), 1) * T_seq[0]/self.T)
    #     return x_bc, t_bc
    
    # def sample_res_points_seq(self, N_samples, T_seq, sobol_seed=0):
    #     _x_normal = np.random.multivariate_normal(self.N_MEAN_I, self.N_COV_I, size=N_samples).astype(np.float32)
    #     _x_normal = torch.tensor(_x_normal, dtype=torch.float32, requires_grad=True)
    #     _x = np.column_stack([
    #         np.random.uniform(self.X1_RANGE[0], self.X1_RANGE[1], N_samples),
    #         np.random.uniform(self.X2_RANGE[0], self.X2_RANGE[1], N_samples),
    #         np.random.uniform(self.X3_RANGE[0], self.X3_RANGE[1], N_samples),
    #         np.random.uniform(self.X4_RANGE[0], self.X4_RANGE[1], N_samples),
    #     ])
    #     _x = torch.tensor(_x, dtype=torch.float32, requires_grad=True)
        
    #     # sobol sequence 
    #     lower_bounds = np.array([self.X1_RANGE[0], self.X2_RANGE[0], self.X3_RANGE[0], self.X4_RANGE[0]])
    #     upper_bounds = np.array([self.X1_RANGE[1], self.X2_RANGE[1], self.X3_RANGE[1], self.X4_RANGE[1]])
    #     # Create a Sobol sequence sampler for 4 dimensions
    #     sobol_sampler = qmc.Sobol(d=4, scramble=True, seed=sobol_seed)
    #     # Generate samples in the unit hypercube [0, 1]^4
    #     samples_unit = sobol_sampler.random_base2(m=10)
    #     # Scale the samples to the specified ranges for each dimension
    #     _x_sol = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    #     _x_sol = torch.tensor(_x_sol, dtype=torch.float32, requires_grad=True)
    #     x = torch.cat((_x_normal, _x, _x_sol), dim=0)
    #     # x = torch.cat((_x_normal, _x), dim=0)
        
    #     portion_of_time_boundary = 0.05
    #     N_total = len(x)
    #     # Number of boundary samples (20% of total)
    #     N_boundary = int(portion_of_time_boundary * N_total)
    #     N_internal = N_total - N_boundary
    #     # Half boundary samples at T_seq[0], half at T_seq[1]
    #     N_boundary_half = N_boundary // 2
    #     # Boundary samples
    #     t_boundary_start = np.full(N_boundary_half, T_seq[0]/self.T)
    #     t_boundary_end = np.full(N_boundary - N_boundary_half, T_seq[1]/self.T)
    #     # Internal uniform samples
    #     t_internal = np.random.uniform(T_seq[0]/self.T, T_seq[1]/self.T, N_internal)
    #     # Combine boundary and internal samples
    #     t_combined = np.concatenate([t_boundary_start, t_boundary_end, t_internal])
    #     # Shuffle the combined samples
    #     np.random.shuffle(t_combined)
    #     # Convert to tensor
    #     t = torch.tensor(t_combined, dtype=torch.float32, requires_grad=True).view(-1, 1)
    #     return x, t
    
    # def sample_points(self, N_samples, bounds):
    #     _x_bc = np.column_stack([
    #         np.random.uniform(bounds[0,0], bounds[0,1], N_samples),
    #         np.random.uniform(bounds[1,0], bounds[1,1], N_samples),
    #         np.random.uniform(bounds[2,0], bounds[2,1], N_samples),
    #         np.random.uniform(bounds[3,0], bounds[3,1], N_samples),
    #     ])
    #     _x_bc = torch.tensor(_x_bc, dtype=torch.float32, requires_grad=False)
    #     return _x_bc
    
    # def quasi_sample_points(self, N_samples, bounds):
    #     # sobol sequence 
    #     lower_bounds = bounds[:,0]
    #     upper_bounds = bounds[:,1]
    #     # Create a Sobol sequence sampler for 4 dimensions
    #     sobol_sampler = qmc.Sobol(d=4, scramble=False)
    #     # Generate samples in the unit hypercube [0, 1]^4
    #     samples_unit = sobol_sampler.random(N_samples)
    #     # Scale the samples to the specified ranges for each dimension
    #     _x_bc_sol = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    #     _x_bc_sol = torch.tensor(_x_bc_sol, dtype=torch.float32, requires_grad=False)
    #     return _x_bc_sol
 