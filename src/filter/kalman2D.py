import numpy as np

def kalmanFilter2D(fps, A, p_obs, R, sigma_v0, sigma_Q):
    """
    INPUTS: see 1D case
    OUTPUTS: see 1D case
    """
    dt = 1 / fps
    Nframes = len(p_obs)
    state_len = A.shape[0] 
    
    # Measurement matrix (observe position only)
    H = np.zeros((2, state_len))
    H[0, 0] = 1
    H[1, 1] = 1
    
    if state_len == 4:   # CV
        Q = sigma_Q**2 * np.array([
            [dt**4/4, 0,       dt**3/2, 0],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0],
            [0,       dt**3/2, 0,       dt**2]
        ])

    elif state_len == 6:   # CA
        Q1D = sigma_Q**2 * np.array([
            [dt**5/20, dt**4/8, dt**3/6],
            [dt**4/8,  dt**3/3, dt**2/2],
            [dt**3/6,  dt**2/2, dt]
        ])

        Q = np.zeros((6,6))
        Q[:3,:3] = Q1D
        Q[3:,3:] = Q1D
    
    else:
        raise ValueError("Unsupported state dimension")

    s_pred = np.zeros(state_len)
    s_pred[0:2] = p_obs[0]
    prior_preds = np.zeros((Nframes, state_len))
    prior_preds[0] = s_pred
    filt_preds = np.zeros((Nframes, state_len))
    filt_preds[0] = prior_preds[0]
    P = np.zeros((state_len, state_len)) 
    P[0,0] = R[0,0]       # initial x uncertainty = sensor error var_x
    P[1,1] = R[1,1]       # initial y uncertainty = sensor error var_y
    P[2,2] = sigma_v0**2  # initial v0x uncertainty = var_v0
    P[3,3] = sigma_v0**2  # initial v0y uncertainty = var_v0
    
    # initialize a0x and a0y uncertainty = var_Q
    if state_len == 6:
        P[4,4] = sigma_Q**2  
        P[5,5] = sigma_Q**2
    
    # Estimation error covariance
    P_prior = np.zeros((Nframes, state_len, state_len))
    P_post = np.zeros((Nframes, state_len, state_len))
    P_post[0] = P
    
    I = np.eye(state_len)

    for k in range(1, Nframes):
        
        # Predict state at frame k based on the motion model
        s_pred = A @ s_pred
        prior_preds[k] = s_pred

        # Predict error covariance at frame k:
        # Propaghi l’incertezza nel tempo + aggiungi rumore di processo.
        # - Più tempo passa → meno sono sicuro → P cresce
    
        P = A @ P @ A.T + Q
        P_prior[k]=P

        # Retrieve observation at frame k
        zk = p_obs[k]

        if np.isnan(zk).any():            # Missing observation → skip update
            filt_preds[k] = s_pred  # prediction only
            P_post[k] = P           # covariance stays as predicted
        else:
            # Compute the difference (innovation/residual) between the measurement 
            # and the predicted state projected into the measurement 
            # space (H @ s_pred), in this case only the position
            # it expresses Quanto ho sbagliato la previsione
            diff = zk - H @ s_pred

            # difference covariance, meaning how much I trust the difference
            S = H @ P @ H.T + R

            # Kalman Gain
            K = P @ H.T @ np.linalg.inv(S)

            # Update state estimate through the weighted difference
            s_pred = s_pred + K @ diff
            filt_preds[k] = s_pred

            # Update error covariance
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
            P_post[k] = P

    return prior_preds, P_prior, filt_preds, P_post

