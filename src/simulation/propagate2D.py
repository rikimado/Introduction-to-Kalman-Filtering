import numpy as np
def propagate(fps, L, p0, v_mag, motion="CV", a_mag=0.0, traj_shape="curve", curve_rate=0, elbow_frame=None):

    """
    Generate ground-truth state by moving a point according to a motion model.

    INPUTS
    - p0 (2,): initial position coordinates [x0, y0] [m] (float)
    - v_mag (float): speed magnitude [m/s]
    - L (int): trajectory length [m]
    - fps (float): sampling rate
    - traj_shape (str): geometric trajectory shape 
        "curve" -> continuous curvature according to a curv_rate [deg/frame]
        "elbow" -> instant 90 deg turn at specified elbow_frame
    - elbow_frame (int): frame index at which the 90 deg turn occurs (only valid for "elbow")
    - curve_rate (float): curvature rate in deg/frame (only valid for "curve")
    - motion (str):  decribe state propagation according to motion equations
        "CV" -> constant velocity
        "CA" -> constant acceleration
    - a_mag (float): initial acceleration magnitude (only for CA model)

    OUTPUTS:
    - p_exp (N,state_dim): ground-truth states array
    - A (state_dim, state_dim): state transition matrix 
    - dt (float): sampling time [s]
    """

    T = L/v_mag               # time to cover distance L at speed v_mag
    N=int(np.round(T*fps))+1  # number of frames (add 1 for initial frame)
    dt = 1 / fps              # sampling time

    # Random initial direction
    angle = np.random.uniform(0, 2*np.pi)
    
    # velocity decomposition
    v0x = v_mag * np.cos(angle)
    v0y = v_mag * np.sin(angle)

    # Select motion model to propagate the state
    if motion == "CV":
        
        A = np.array([
            [1, 0, dt, 0], # x_{k} = x_{k-1} + vx_{k-1}*dt
            [0, 1, 0, dt], # y_{k} = y_{k-1} + vy_{k-1}*dt
            [0, 0, 1, 0],  # vx_{k} = vx_{k-1} (const)
            [0, 0, 0, 1]   # vy_{k} = vy_{k-1} (const)
        ])

        # initial state = [x0, y0, v0x, v0y]
        s0 = np.array([p0[0], p0[1], v0x, v0y])
        acc_idx = None

    elif motion == "CA":
        # State = [x, y, vx, vy, ax, ay]
        A = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        # initial acc vector along same velocity direction
        a0x = a_mag * np.cos(angle)
        a0y = a_mag * np.sin(angle)
        
        # initial state = [x0, y0, v0x, v0y, a0x, a0y]
        s0 = np.array([p0[0], p0[1], v0x, v0y, a0x, a0y])
        acc_idx = slice(4, 6)

    else:
        raise ValueError("Unsupported motion type. Use 'CV' or 'CA'.")
 
    vel_idx = slice(2, 4)
    state_dim = len(s0)
    p_exp = np.zeros((N, state_dim))
    s_k=s0.copy()

    if traj_shape == "curve":
        curve_angle = np.deg2rad(curve_rate)
        c = np.cos(curve_angle)
        s = np.sin(curve_angle)

    # Propagate according to chosen model and trajectory type
    if acc_idx is None:

        for k in range(N):
            
            if traj_shape == "elbow" and k == elbow_frame:
                vx, vy = s_k[vel_idx]
                s_k[vel_idx] = [-vy, vx]
            
            elif traj_shape == "curve":
                vx, vy = s_k[vel_idx]
                s_k[vel_idx] = [c*vx - s*vy, s*vx + c*vy]

            p_exp[k] = s_k
            s_k = A @ s_k

    else:

        for k in range(N):
            
            if traj_shape == "elbow" and k == elbow_frame:
                vx, vy = s_k[vel_idx]
                ax, ay = s_k[acc_idx]
                
                s_k[vel_idx] = [-vy, vx]
                s_k[acc_idx] = [-ay, ax]
            
            elif traj_shape == "curve":
                vx, vy = s_k[vel_idx]
                ax, ay = s_k[acc_idx]

                s_k[vel_idx] = [c*vx - s*vy, s*vx + c*vy]    
                s_k[acc_idx] = [c*ax - s*ay, s*ax + c*ay]

            p_exp[k] = s_k
            s_k = A @ s_k

    return p_exp, A
