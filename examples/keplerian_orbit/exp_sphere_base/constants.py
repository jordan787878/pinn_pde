import numpy as np

class Case2_4D_Constants:
    """
    define the constants of Case 2 in the ref. paper
    the state X = [r', phi', r'_dot, phi'_dot] is the normalized shperical coordinates
    """
    _MU_EARTH = np.float32(3.9859e+14)
    _A        = np.float32(4.2164e+7)
    _W        = np.float32(np.sqrt(_MU_EARTH/_A**3))
    _T        = np.float32(2*np.pi/_W)
    _R        = np.float32(2e+6)
    _THETA    = np.float32(0.015)
    _PHI      = np.float32(0.0387)
    _TI       = np.float32(0.0)
    _TF       = np.float32(0.1*_T)
    _MEAN_I   = np.float32([_A, 0.0, 0.0, _W])
    _N_MEAN_I = np.float32([_MEAN_I[0]/_R, 
                            _MEAN_I[1]/_PHI,
                            _MEAN_I[2]/(_R/_T),
                            (_MEAN_I[3]-_W)/(_PHI/_T)])  
    _N_COV_I  = np.float32(np.diag([1e+11/(_R**2),
                                    1e-4/(_PHI**2),
                                    1e+3/(_R/_T)**2,
                                    1e-12/(_PHI/_T)**2]))
    _X1_RANGE = np.float32(np.array([20.1, 22.1]))
    _X2_RANGE = np.float32(np.array([-2.2, 2.2]))
    _X3_RANGE = np.float32(np.array([-10.0, 10.0]))
    _X4_RANGE = np.float32(np.array([-8.0, 8.0]))
    _MAX_PX1  = np.float32(3.0)
    _MAX_PX2  = np.float32(1.8)
    _MAX_PX3  = np.float32(0.35)
    _MAX_PX4  = np.float32(0.2)
    _T_PRIME_SPAN   = np.float32(np.array([_TI, 0.2*_TF, 0.4*_TF, 0.6*_TF, 0.8*_TF, _TF])/_T)

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
    def MAX_PX1(self):
        return self._MAX_PX1
    
    @property
    def MAX_PX2(self):
        return self._MAX_PX2
    
    @property
    def MAX_PX3(self):
        return self._MAX_PX3
    
    @property
    def MAX_PX4(self):
        return self._MAX_PX4
    
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
    
    