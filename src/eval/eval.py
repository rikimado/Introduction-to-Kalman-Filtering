import numpy as np

def cRMSE(pos_filt, pos_exp):
    """
    Compute cumulative root-mean-square error (cRMSE).
    INPUTS
    - pos_filt: (N, dim) filtered positions (e.g., 1D or 2D)
    - pos_exp:  (N, dim) ground-truth positions

    OUTPUT
    - crmse_vals: (N,) cumulative RMSE at each time step
    """
    # Compute squared error point-wise
    diff = pos_exp - pos_filt

    # If 2D positions, sum over coordinates
    if diff.ndim == 2:
        err_sq = np.sum(diff**2, axis=1)

    else:  # 1D positions
        err_sq = diff**2

    valid = ~np.isnan(err_sq)
    err_sq = err_sq[valid]
    crmse_vals = np.sqrt(np.cumsum(err_sq) / np.arange(1, len(err_sq)+1))
    return crmse_vals


def thresAcc(p_filt, p_exp, epsilon=0.1):
    """
    Compute threshold-based accuracy (% of time error <= epsilon [m])
    """
    diff = p_filt - p_exp

    if diff.ndim == 2:  # 2D
        errors = np.linalg.norm(diff, axis=1)
    else:  # 1D
        errors = np.abs(diff)

    acc = np.mean(errors <= epsilon)
    return acc

def cep95(p_filt, p_exp):
    """
    Compute the 95% Circular Error Probable (CEP95) radius [m] in 1D or 2D.
    
    INPUTS
    - p_filt: (N,) or (N,2) filtered positions
    - p_exp:  (N,) or (N,2) ground-truth positions

    OUTPUT
    - radius_95: 95th percentile of the error [m]
    """
    diff = p_filt - p_exp

    if diff.ndim == 1:       # 1D case
        errors = np.abs(diff)
    else:                    # 2D case
        errors = np.linalg.norm(diff, axis=1)
    # ignora NaN
    errors = errors[~np.isnan(errors)]
    if len(errors) == 0:
        return np.nan
    radius_95 = np.percentile(errors, 95)
    return radius_95
