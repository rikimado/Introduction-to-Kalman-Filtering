import numpy as np

def kalmanFilter1D(fps, A, p_obs, R, sigma_v0, sigma_Q):
    """
    INPUTS
    - fps: frame per second
    - A: state transition matrix
    - p_obs: (N,) observations array
    - R: sensor error covariance (std^2)
    - sigma_v0
    - sigma_Q

    OUTPUTS
    - prior_preds: (N,state_len) predicted states
    - P_prior: (N,state_len,state_len) predicted covariance
    - filt_preds: (N,state_len) filtered states
    - P_post: (N,state_len,state_len) updated covariance
    """

    dt = 1 / fps
    Nframes = len(p_obs)
    state_len = A.shape[0] 
    
    # Measurement matrix (observe position only)
    H = np.zeros((1, state_len))
    H[0, 0] = 1 
    
    if state_len == 2:   # CV
        Q = sigma_Q**2 * np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ])

    elif state_len == 3:   # CA
         Q = sigma_Q**2 * np.array([
            [dt**5/20, dt**4/8, dt**3/6],
            [dt**4/8,  dt**3/3, dt**2/2],
            [dt**3/6,  dt**2/2, dt]
        ])
    
    else:
        raise ValueError("Unsupported state dimension")

    # Prediction prior to observation update
    s_pred = np.zeros(state_len)
    
    # first prediction = first observation
    s_pred[0] = p_obs[0]
    
    # Store predictions before updates
    prior_preds = np.zeros((Nframes, state_len))
    prior_preds[0] = s_pred  # s_pred_0 = [p=obs_0, v=0]

    # Predictions updated 
    filt_preds = np.zeros((Nframes, state_len))

    # first = prior since the difference with the prediction is null as s_pred[0] == first obervation
    filt_preds[0] = prior_preds[0]

    # Estimation uncertainty covariance
    P = np.zeros((state_len, state_len)) 
    P[0,0] = R[0,0]      # initial position uncertainty = sensor error var
    P[1,1] = sigma_v0**2 # initial velocity uncertainty
    
    if state_len == 3:
        P[2,2] = sigma_Q**2 # initial acceleration uncertainty
    
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
        
        if np.isnan(zk):            # Missing observation → skip update
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

    sigma_prior = np.sqrt(P_prior[:,0,0])
    sigma_post = np.sqrt(P_post[:,0,0])
    
    return prior_preds, sigma_prior, filt_preds, sigma_post