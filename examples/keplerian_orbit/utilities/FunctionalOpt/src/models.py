import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, Independent, Normal


class TorchGMM(nn.Module):
    def __init__(self, constants, num_components=64, n_features=4):
        super().__init__()
        self.num_components = num_components
        self.n_features = n_features
        
        # mixture weights
        self.logits = nn.Parameter(torch.zeros(num_components))
        # means [num_components, n_features]
        self.means  = nn.Parameter(torch.randn(num_components, n_features) + 
                                   torch.from_numpy(constants.N_MEAN_I))
        # one “raw scale” per component per feature
        self.raw_scales = nn.Parameter(torch.randn(num_components, n_features))

    def get_distribution(self):
        # ensure all std-devs are positive
        scales = F.softplus(self.raw_scales)      # [num_components, n_features]
        
        # mixture category
        cat  = Categorical(logits=self.logits)    # batch_shape=[num_components]
        # independent multivariate normal with diagonal cov
        comp = Independent(
                   Normal(loc=self.means, scale=scales),
                   1
               )                                   # event_shape=[n_features]

        return MixtureSameFamily(cat, comp)

    def forward(self, x):
        """
        x: [batch_size, n_features]
        returns: [batch_size] densities
        """
        gmm = self.get_distribution()
        return gmm.log_prob(x).exp()
    
    def get_gmm_paramters(self):
        # 1) Directly from your parameters:
        # ------------------------------------------------
        # (a) mixture weights
        raw_logits = self.logits               # [num_components]
        weights     = torch.softmax(raw_logits, dim=0)    # normalized → probabilities

        # (b) component means
        means = self.means                      # [num_components, n_features]

        # (c) component std-dev’s and covariances
        scales      = F.softplus(self.raw_scales)       # std-dev’s, shape [C, F]
        covariances = scales.pow(2)                      # variances on the diagonal

        # Now if you want NumPy arrays
        w_np  = weights.detach().cpu().numpy()
        m_np  = means.detach().cpu().numpy()
        cov_np = covariances.detach().cpu().numpy()
        return (w_np, m_np, cov_np)


# class RBFDensity(nn.Module):
#     def __init__(self, input_dim=4, num_basis=20):
#         """
#         Approximates a function p(x) as a weighted sum of normalized Gaussian RBFs,
#         designed so that p(x) is a valid pdf:    
#            p(x) = sum_i w_i * phi_i(x)      
#         with phi_i(x) defined as a normalized Gaussian over R^4.     
#         The centers of each RBF are adjusted by adding constant offsets defined in
#         constants.N_MEAN_I.     
#         Args:
#             input_dim (int): Dimensionality of input (should be 4).
#             num_basis (int): Number of RBF basis functions.
#         """
#         super(RBFDensity, self).__init__()
#         self.input_dim = input_dim   # This should be 4.
#         self.num_basis = num_basis     
#         # Learnable centers: shape [num_basis, input_dim]
#         self.centers = nn.Parameter(torch.randn(num_basis, input_dim))    
#         # Learnable log-bandwidths (one per basis). Use softplus later to ensure positivity.
#         self.covs = nn.Parameter(torch.ones(num_basis))      
#         # Learnable logits for weights; using softmax will enforce nonnegativity and sum-to-one.
#         self.A = nn.Parameter(torch.ones(num_basis)/num_basis)
#     def forward(self, x):
#         """
#         Evaluate the density p(x) for a batch of input points x (shape [batch, input_dim]).     
#         Returns:
#             p (Tensor): The pdf evaluated at x, shape [batch].
#         """
#         batch = x.shape[0]
#         K = self.num_basis      
#         # Get sigma with softplus for numerical stability.
#         covs = torch.square(self.covs) + 1e-10    
#         # Compute normalized weights via softmax.
#         A = self.A + 1e-10
#         sum_A = torch.sum(A**2)
#         weights = A**2 / sum_A  # shape: [K]      
#         # Adjust centers by adding the constant offset.
#         offset = torch.tensor(constants.N_MEAN_I, device=self.centers.device, dtype=self.centers.dtype)
#         effective_centers = self.centers + offset.unsqueeze(0)  # shape: [K, input_dim]
#         pdf = torch.zeros(batch)
#         for i in range(K):
#             mean_i = effective_centers[i, :]
#             cov_i  = covs[i]
#             m = torch.distributions.MultivariateNormal(
#                 loc=mean_i,
#                 covariance_matrix=cov_i * torch.eye(4)
#             )
#             pdf_values = m.log_prob(x).exp()
#             pdf = pdf + weights[i] * pdf_values
#         return pdf
   