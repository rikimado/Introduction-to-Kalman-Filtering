import numpy as np

def propagate(fps, L, x0, v0, motion="CV", acc=0.0):
    """
    Generate ground-truth 1D state by moving a point according to a defined motion model.

    INPUTS
    - fps: frame per second (float)
    - L: trajectory length [m] (float)
    - x0: initial position [m] (float)
    - v0: initial velocity magnitude [m/s] (float)
    - motion: str
        "CV" -> constant velocity
        "CA" -> constant acceleration
    - acc: acceleration magnitude [m/s^2] (only for CA)

    OUTPUTS
    - p_exp: (N, state_dim) ground-truth states
    - A: state transition matrix
    - dt: sampling time [s]
    """

    dt = 1 / fps
    T = L / v0  # # total acquisition time
    N = int(np.round(T*fps)) + 1 # n° of frames (or states)

    # Select motion model
    if motion == "CV":
        # State = [x, v]
        A = np.array([
            [1, dt], # x_{k} = x_{k-1} + v_{k-1}*dt
            [0, 1]   # v_{k} = v_{k-1} (const)
        ])
        
        s0 = np.array([x0, v0]) # initial state
    
    elif motion == "CA":
        # State = [x, v, a]
        A = np.array([
            [1, dt, 0.5*dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        # initial state
        s0 = np.array([x0, v0, acc]) # initial state
    else:
        raise ValueError("Unsupported motion type. Use 'CV' or 'CA'.")

    state_dim = len(s0)
    p_exp = np.zeros((N, state_dim))
    s_k = s0.copy()

    for k in range(N):
        p_exp[k] = s_k
        s_k = A @ s_k

    return p_exp, A