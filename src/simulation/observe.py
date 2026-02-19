import numpy as np

def observe(p_exp, sigma, drop_prob=0.0):
    """
    Generate noisy observations.
    
    INPUTS:
    - p_exp: (N, state_dim) ground-truth states
    - sigma: scalar or array-like, observation noise std per observed coordinate
    - drop_prob: float [0,1], probability that an observation is dropped at each timestep
    
    OUTPUT
    - pos_idx: indices of state vector that are observed (e.g., [0] or [0,1])
    - R: covariance (scalar or matrix)  for observed coordinates
    """
    N, state_dim = p_exp.shape

    if state_dim in [2, 3]:        # 1D CV or CA
        pos_idx = [0]

    elif state_dim in [4, 6]:      # 2D CV or CA
        pos_idx = [0, 1]

    else:
        raise ValueError("Unsupported state dimension")

    # Ensure sigma is array with proper shape
    sigma = np.atleast_1d(sigma)
    if sigma.size == 1:
        sigma = np.full(len(pos_idx), sigma.item())
    elif sigma.size != len(pos_idx):
        raise ValueError("sigma size must match number of observed coordinates")

    # Covariance scalar/matrix for observed coordinate(s)
    R = np.diag(sigma**2)

    # Observed positions with noise
    p_obs = p_exp[:, pos_idx] + np.random.normal(scale=sigma, size=(N, len(pos_idx)))

    # Apply random dropout
    if drop_prob > 0.0:
        #mask = np.random.rand(N, len(pos_idx)) <= drop_prob  # True where dropout occurs
        mask = np.random.rand(N) <= drop_prob
        mask[0] = False   # ensure first frame is always observed
        mask[-1] = False  # ensure last frame is always observed
        
        # hide True indexes in p_obs array
        if p_obs.ndim == 1:
            p_obs[mask] = np.nan
        else:  # 2D
            p_obs[mask, :] = np.nan
    return p_obs, R

