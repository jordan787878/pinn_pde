import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, Independent, Normal


class TorchGMM(nn.Module):
    def __init__(self, constants, num_components=64, n_features=4, min_std=1e-2): #min_std=1e-2
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        dtype = torch.float32
        # self.min_std = min_std
        
        self.ranges =  np.array([
            constants.X1_RANGE,
            constants.X2_RANGE,
            constants.X3_RANGE,
            constants.X4_RANGE,
        ], dtype=np.float32)
        span = torch.tensor(self.ranges[:, 1] - self.ranges[:, 0], dtype=dtype)
        self.min_std = 0.02 * span  # shape [F]
        
        # self.logits = nn.Parameter(torch.randn(num_components))
        self.logits = nn.Parameter(torch.zeros(num_components))

        self.means  = nn.Parameter(torch.randn(num_components, n_features, dtype=dtype) + torch.from_numpy(constants.N_MEAN_I))

        # self.raw_scales = nn.Parameter(torch.randn(num_components, n_features))
        self.raw_scales = nn.Parameter(torch.ones(num_components, n_features, dtype=dtype))

    def get_weights(self, tau=1.0):
        # # K = self.num_components
        # # min_weight = 0.0
        # w2 = self.raw_weights.pow(2)               # square
        # p_i = w2 / w2.sum()                        # normalize
        # weights = p_i
        # # weights = min_weight + (1 - K*min_weight) * p_i
        # # sum_weights = weights.sum().detach().numpy()
        # # np.testing.assert_almost_equal(sum_weights, 1.0, decimal=5)
        # return weights
        return F.softmax(self.logits / tau, dim=0)

    def get_distribution(self):
        # ensure all std-devs are positive
        scales = F.softplus(self.raw_scales) + self.min_std      # [num_components, n_features]

        # 2) turn raw_weights into non-negative mixture probs
        weights = self.get_weights()

        # 3) build the Mixture distribution
        cat  = Categorical(probs=weights)                       # batch_shape=[num_components]
        comp = Independent(
            Normal(loc=self.means, scale=scales),
            1                                                    # event_dim = n_features
        )
        return MixtureSameFamily(cat, comp)

    def forward(self, x):
        """
        x: [batch_size, n_features]
        returns: [batch_size] densities
        """
        x = x.to(dtype=torch.float32)
        gmm = self.get_distribution()
        return gmm.log_prob(x).exp()
    
    def get_gmm_paramters(self):
        # 1) Directly from your parameters:
        # ------------------------------------------------
        # (a) mixture weights
        weights = self.get_weights()

        # (b) component means
        means = self.means                      # [num_components, n_features]

        # (c) component std-dev’s and covariances
        scales = F.softplus(self.raw_scales) + self.min_std       # std-dev’s, shape [C, F]
        covariances = scales.pow(2)                      # variances on the diagonal

        # Now if you want NumPy arrays
        w_np  = weights.detach().cpu().numpy()
        m_np  = means.detach().cpu().numpy()
        cov_np = covariances.detach().cpu().numpy()
        return (w_np, m_np, cov_np)
    
    def region_prob(self, region_bounds):
        """
        region_bounds: torch tensor of shape [n_features, 2],
                       where region_bounds[j,0] = lower bound on dim j,
                             region_bounds[j,1] = upper bound on dim j
        Returns the total probability mass in that hyper‐rectangle.
        """
        # 1) unpack parameters
        weights = self.get_weights()
        means  = self.means                  # [C, D]
        scales = F.softplus(self.raw_scales) + self.min_std # [C, D]

        # 2) get bounds (broadcasted)
        #    lb, ub shape [1, D] → will broadcast to [C, D]
        lb = region_bounds[:, 0].unsqueeze(0)
        ub = region_bounds[:, 1].unsqueeze(0)

        # 3) standardized coords
        #    z_low = (lb - μ) / (σ * sqrt(2)), same shape [C, D]
        denom = scales * (2**0.5)
        z_low  = (lb - means) / denom
        z_high = (ub - means) / denom

        # 4) 1D Gaussian CDF difference per component per dim
        #    Φ(ub) - Φ(lb) = 0.5 * [erf(z_high) - erf(z_low)]
        cdf_diff = 0.5 * (torch.erf(z_high) - torch.erf(z_low))  # [C, D]

        # 5) joint probability in D dims = product over dims
        #    (independence across features)
        comp_mass = torch.prod(cdf_diff, dim=1)  # [C]

        # 6) weight‐sum over components
        total_mass = torch.dot(weights, comp_mass)  # scalar

        return total_mass
    
   