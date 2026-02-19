import numpy as np
from simulation.observe import observe
from eval.eval import cRMSE, thresAcc, cep95

def run_test_sQ(dim, p_exp, A, kalman_fn, sigma_R, sigma_vel, sigma_Q_vals, fps):

    if dim == 1:
        coords_idxs = (slice(None), 0)
        acc_thresh = sigma_R
    else:
        coords_idxs = (slice(None), slice(0, 2))
        acc_thresh = sigma_R[0]

    p_obs, R = observe(p_exp, sigma_R, drop_prob=0.1)

    # Observation references
    obs_cep = cep95(p_obs[coords_idxs], p_exp[coords_idxs])    
    obs_crmse = cRMSE(p_obs[coords_idxs], p_exp[coords_idxs])[-1]
    obs_acc = thresAcc(p_obs[coords_idxs], p_exp[coords_idxs], acc_thresh)

    first_cep95_cross_sq = None  # keep track of sigmaQ value where filter cep95 first surpasses obs cep95
    first_crmse_cross_sq = None  # keep track of sigmaQ value where filter crmse first surpasses obs crmse
    first_acc_cross_sq = None    # keep track of sigmaQ value where filter accuracy first surpasses obs accuracy

    found_cep95_cross = False
    found_crmse_cross = False
    found_acc_cross = False

    # Prepare lists for test
    filt_list = []

    crmse_list = []
    acc_list = []
    cep_list = []

    for sq in sigma_Q_vals:
    
        _, _, filt_preds, _ = kalman_fn(fps, A, p_obs, R, sigma_vel, sq)

        # store filtration for current sigma_Q value
        filt_list.append(filt_preds[coords_idxs])

        last_crmse = cRMSE(filt_preds[coords_idxs], p_exp[coords_idxs])[-1]
        acc_val = thresAcc(filt_preds[coords_idxs], p_exp[coords_idxs], acc_thresh)
        cep = cep95(filt_preds[coords_idxs], p_exp[coords_idxs])

        crmse_list.append(last_crmse)
        acc_list.append(acc_val)
        cep_list.append(cep)

        # check if the cep95 for this sigma_Q is lower than obs. ref.
        if (not found_cep95_cross) and (cep <= obs_cep):
            first_cep95_cross_sq = sq
            found_cep95_cross = True
        
        # check if the minimum cRMSE for this sigma_Q is lower than obs. ref.
        if (not found_crmse_cross) and (last_crmse <= obs_crmse): 
            first_crmse_cross_sq = sq
            found_crmse_cross = True
    
        # check if the accuracy for this sigma_Q is higher than obs. ref.
        if (not found_acc_cross) and (acc_val >= obs_acc): 
            first_acc_cross_sq = sq
            found_acc_cross = True

    return dict(
        crmse_vs_sQ=np.array(crmse_list),
        acc_vs_sQ=np.array(acc_list),
        cep_vs_sQ=np.array(cep_list),
        filt_vs_sQ = filt_list,
        obs_crmse = obs_crmse,
        obs_cep = obs_cep,
        obs_acc = obs_acc,
        first_crmse_cross = first_crmse_cross_sq,
        first_cep_cross = first_cep95_cross_sq,
        first_acc_cross = first_acc_cross_sq,
        p_obs = p_obs
        
)


def run_test_sv(dim, p_exp, A, kalman_fn, sigma_R, sigma_v_vals, sigma_Q, n_test, fps):

    if dim == 1:
        coords_idxs = (slice(None), 0)
        acc_thresh = sigma_R
    else:
        coords_idxs = (slice(None), slice(0, 2))
        acc_thresh = sigma_R[0]

    best_sv_vals = [] # store best sigmaQ value for each test 
    first_cross_sv_vals = []  # store sigmaQ values where filter cep95 first surpasses obs cep95
    cep_obs_list = [] # store observations of all tests

    global_best_crmse = np.inf  # save the mean crmse relative to the best test
    global_best_p_obs = None # save the obs relative to the best test
    global_best_filt = None  # save the best filtration for best test

    # <-- initialize arrays to safe defaults
    crmse_vs_sV = np.array([])
    accuracy_vs_sV = np.array([])
    cep_vs_sV = np.array([])
    filt_vs_sV = []

    for test in range(n_test):

        filt_list = []
        crmse_list = []
        accuracy_list = []
        cep_list = []

        first_cross_sv = None
        found_cross = False
        
        # Initialize metrics to be updated with the results from best sigma_Q value for current test
        best_test_crmse = np.inf
        best_test_filt = None
        best_test_sv = None

        p_obs, R = observe(p_exp, sigma_R, drop_prob=0.1)

        # observation reference for this test
        cep_obs = cep95(p_obs[coords_idxs], p_exp[coords_idxs])
        cep_obs_list.append(cep_obs)

        for sv in sigma_v_vals:

            _, _, filt_preds, _ = kalman_fn(fps, A, p_obs, R, sv, sigma_Q)

            filt_list.append(filt_preds[coords_idxs])

            crmse = cRMSE(filt_preds[coords_idxs], p_exp[coords_idxs])
            last_crmse = crmse[-1]
            acc_val = thresAcc(filt_preds[coords_idxs], p_exp[coords_idxs], acc_thresh)
            cep = cep95(filt_preds[coords_idxs], p_exp[coords_idxs])

            if (not found_cross) and (cep <= cep_obs):
                first_cross_sv = sv
                found_cross = True

            crmse_list.append(crmse)
            accuracy_list.append(acc_val)
            cep_list.append(cep)

            if last_crmse < best_test_crmse:
                best_test_sv = sv
                best_test_filt = filt_preds[coords_idxs]
                best_test_crmse = last_crmse
                

        best_sv_vals.append(best_test_sv)
        first_cross_sv_vals.append(first_cross_sv)

        if best_test_crmse < global_best_crmse:
            global_best_p_obs = p_obs
            global_best_crmse = best_test_crmse
            global_best_filt = best_test_filt

            crmse_vs_sV = np.array([c[-1] for c in crmse_list])
            accuracy_vs_sV = np.array(accuracy_list)
            cep_vs_sV = np.array(cep_list)
            filt_vs_sV = filt_list.copy()

    return dict(
        best_sv_vals=best_sv_vals,
        first_cross_sv_vals=first_cross_sv_vals,
        cep_obs_list=cep_obs_list,
        global_best_crmse=global_best_crmse,
        global_best_p_obs=global_best_p_obs,
        global_best_filt=global_best_filt,
        crmse_vs_sV=crmse_vs_sV,
        accuracy_vs_sV=accuracy_vs_sV,
        cep_vs_sV=cep_vs_sV,
        filt_vs_sV=filt_vs_sV
    )

