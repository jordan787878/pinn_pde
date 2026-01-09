import numpy as np
from scipy.stats import norm


def like_func(y, x, obs_model):
    """
    return the likelihood of y is
    y ~ N(H*x, R),
        H: mapping from state to measurement
        R: covariance of the measurement noise

    Assume: x in R^1
    """
    H, R = obs_model
    var = R
    mean = H * x                    # (N,1)
    diff = y - mean                 # (N,1)
    norm_const = np.sqrt(2.0 * np.pi * var)
    like = np.exp(-0.5 * (diff ** 2) / var) / norm_const  # (N,1)
    return like


def data_assimilation_gmm_1d_prior(y, obs_model, ws_prior, mus_prior, covs_prior):
    H, R = obs_model
    K = len(ws_prior) # Number of components

    # Initialize updated parameters
    mus_posterior = np.zeros_like(mus_prior)
    covs_posterior = np.zeros_like(covs_prior)
    ws_posterior = np.zeros_like(ws_prior)
    betas = np.zeros_like(ws_prior)

    # --- Kalman update for each component ---
    for i in range(K):
        mu_i_prior = mus_prior[i]
        P_i_prior = covs_prior[i]

        # 1D implementation notes:
        # H*mu is H*mu
        # H*P*H^T is H*P*H
        # (H*P*H + R)^-1 is 1 / (H*P*H + R)

        # Eq (c): Kalman Gain K_i
        # K_i = P_i(t|y_{0:t-1}) * H * (H * P_i(t|y_{0:t-1}) * H^T + R)^{-1}
        denominator = H * P_i_prior * H + R
        K_i = P_i_prior * H / denominator # Shape (1, 1)

        # Eq (a): Update Mean mu_i
        # mu_i(t|y_t) = mu_i(t| y_{0:t-1}) + K_i * (y_t - H * mu_i(t| y_{0:t-1}) )
        mus_posterior[i] = mu_i_prior + K_i * (y - H * mu_i_prior)

        # Eq (b): Update Covariance P_i
        # P_i(t|y_t) = (I - K_i * H) * P_i(t|y_{0:t-1})
        I_minus_KH = 1.0 - K_i * H # In 1D, I is 1
        covs_posterior[i] = I_minus_KH * P_i_prior

        # Eq (e): Calculate Likelihood beta_i(t)
        # beta_i(t) = N(y_t; H*mu_i(t| y_{0:t-1}), H*P_i(t|y_{0:t-1})*H^T + R)
        # This is the likelihood of the observation given the component i's prior state
        prediction_mean = H * mu_i_prior
        prediction_cov = H * P_i_prior * H + R
        
        # Use scipy.stats.norm PDF since it's 1D
        betas[i] = norm.pdf(
            y, 
            loc=prediction_mean.item(), 
            scale=np.sqrt(prediction_cov.item())
        )
    
    # --- Update Weights (Normalization Step) ---

    # Eq (d): w_i(t|y_t) = (w_i(t|y_{0:t-1}) * beta_i(t)) / Sum(...)
    numerator_weights = ws_prior * betas
    denominator_weights = np.sum(numerator_weights)
    
    # Check for numerical stability issues (e.g., if denominator_weights is near zero)
    if denominator_weights < 1e-10:
        print(f"Warning: Low likelihoods, Using uniform weights for posterior.")
        ws_posterior = np.ones_like(ws_prior) / K
    else:
        ws_posterior = numerator_weights / denominator_weights
    return ws_posterior, mus_posterior, covs_posterior