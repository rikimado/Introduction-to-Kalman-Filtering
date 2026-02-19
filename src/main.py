import numpy as np
from filter.test import run_test_sQ

dim = 2     # 1D or 2D
fps = 30 
L = 10e-2   # total travelled path [m]
v0 = 0.05   # initial true velocity [m/s]
a0 = 5e-2  # true acceleration [m/s^2]

if dim == 1:
    from simulation.propagate1D import propagate 
    from filter.kalman1D import kalmanFilter1D
    p0 = 0.0           # initial position [m]
    sigma_R = 2e-3     # sensor error std [m]
    kalman_fn = kalmanFilter1D
    p_exp, A = propagate(fps, L, p0, v0, motion="CA", acc=a0)

else:
    
    from simulation.propagate2D import propagate 
    from filter.kalman2D import kalmanFilter2D

    p0 = [0.0, 0.0]                 
    sigma_R = np.array([2, 2])*1e-3
    kalman_fn = kalmanFilter2D
    p_exp, A = propagate(fps, L, p0, v_mag=v0, motion="CA", a_mag=a0, traj_shape="elbow", elbow_frame=int(0.5*fps))


scaling = np.arange(0.0, 10.01, 0.01)

# Rule of thumb: 
#   -σ_v ~ 5–10% of v0, 
#   -σ_acc ~ 5–20% of a0

#sigma_Q = 0.0*abs(a0)
#sigma_v_vals = percentages * abs(v0)

sigma_vel = 0.0*abs(v0)
sigma_Q_vals = scaling * abs(a0)

results = run_test_sQ(
    dim=dim,
    p_exp=p_exp,
    A=A,
    kalman_fn = kalman_fn,
    sigma_R=sigma_R,
    sigma_vel=sigma_vel,
    sigma_Q_vals=sigma_Q_vals,
    fps=fps,
)

crmse_vs_sQ = results["crmse_vs_sQ"]
acc_vs_sQ = results["acc_vs_sQ"]
cep_vs_sQ = results["cep_vs_sQ"]
filt_vs_sQ = results["filt_vs_sQ"]
obs_crmse = results["obs_crmse"]
obs_cep = results["obs_cep"]
obs_acc = results["obs_acc"]
first_crmse_cross = results["first_crmse_cross"]
first_cep_cross = results["first_cep_cross"]
first_acc_cross = results["first_acc_cross"]
p_obs = results["p_obs"]

## ====================================================================================
##                                      VISUALIZATION 
## ====================================================================================
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

t = np.arange(len(p_exp)) / fps

# 1) ground truth + obs 
missing_idx = np.isnan(p_obs[:,0])
valid_idx = ~missing_idx

plt.figure(figsize=(10,4))
if dim == 1:
    plt.plot(t, p_exp[:, 0]*1e3, 'b', zorder=2)
    plt.scatter(t[valid_idx], p_obs[valid_idx,0]*1e3, c='k', zorder=1)
    plt.plot(t, p_obs[:, 0]*1e3, 'k')
    plt.scatter(t[missing_idx], p_exp[missing_idx,0]*1e3, facecolors='none', edgecolors='red', zorder=1)
    plt.xlabel('Time [s]')
    plt.ylabel('x [mm]')

else:
    plt.plot(p_exp[:, 0]*1e3, p_exp[:, 1]*1e3, 'b', zorder=2)
    plt.scatter(p_obs[valid_idx,0]*1e3, p_obs[valid_idx,1]*1e3, c='k', zorder=1)
    plt.plot(p_obs[:, 0]*1e3, p_obs[:, 1]*1e3, 'k')
    plt.scatter(p_exp[missing_idx,0]*1e3, p_exp[missing_idx,1]*1e3, facecolors='none', edgecolors='red', zorder=1)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('equal')

legend_elements = [
    Line2D([0], [0], color='b', lw=1, label='ground truth'),
    Line2D([0], [0], color='k', lw=1, marker='o', markersize=6, markerfacecolor='black', label='observation'),
    Line2D([0], [0], color='r', lw=0, marker='o', markersize=6, markerfacecolor='none', markeredgecolor='red', label='dropouts')
]

plt.legend(handles=legend_elements)
plt.grid(True, alpha=0.3)
plt.show()

# 2) Metrics vs sigma_Q of for the best test (lowest mean CRMSE)
# CRMSE in mm
plt.figure(figsize=(6,5))
plt.plot(scaling, crmse_vs_sQ*1e3, label="filters")
plt.axhline(obs_crmse*1e3, color='r', linestyle='--', lw=1.5, label="obs")
plt.text(scaling[0] * 100 * 0.98, obs_crmse * 1e3,
         f"obs. cRMSE: {obs_crmse * 1e3:.2f} mm",
         color='k',
         ha='left',
         va='bottom',
         fontsize=9,
        )

if first_crmse_cross is not None:
    cross_crmse = first_crmse_cross/abs(a0)
    idx = np.argmin(np.abs(sigma_Q_vals-first_crmse_cross))
    plt.axvline(cross_crmse, linestyle=":", color='k', label=f"$σ_Q$ = {cross_crmse:.1f}$|a_0|$")
    plt.plot(cross_crmse, crmse_vs_sQ[idx]*1e3, 'kx')

else:
    print("Filter never better than sensor")

plt.xlabel("scaling factor")
plt.ylabel("cRMSE [mm]")
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# CEP95 in mm
plt.figure(figsize=(7,5))
plt.plot(scaling, cep_vs_sQ*1e3, label="filters")
plt.axhline(obs_cep*1e3, color='r', linestyle="--", label="obs")
plt.text(scaling[0] * 0.98, obs_cep * 1e3,
         f"obs. CEP95: {obs_cep * 1e3:.2f} mm",
         color='k',
         ha='left',
         va='bottom',
         fontsize=9,
        )

if first_cep_cross is not None:
    cross_cep = first_cep_cross/abs(a0)
    idx = np.argmin(np.abs(sigma_Q_vals-first_cep_cross))
    plt.axvline(cross_cep, linestyle=":", color='k', label=f"$σ_Q$ = {cross_cep:.1f}$|a_0|$")
    plt.plot(cross_cep, cep_vs_sQ[idx]*1e3, 'kx')

else:
    print("Filter never better than sensor.")

plt.xlabel("scaling factor")
plt.ylabel("CEP95 [mm]")
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Accuracy %
plt.figure(figsize=(7,5))
plt.plot(scaling, acc_vs_sQ*100, label='filters')
plt.axhline(obs_acc*100, color='r', linestyle="--", label="obs")
plt.text(scaling[0] * 0.05, obs_acc * 100,
         f"obs. accuracy: {obs_acc * 100:.2f}%",
         color='k',
         ha='left',
         va='bottom',
         fontsize=9,
        )

if first_acc_cross is not None:
    cross_acc = first_acc_cross/abs(a0)
    idx = np.argmin(np.abs(sigma_Q_vals-first_acc_cross))
    plt.axvline(cross_acc, linestyle=":", color='k', label=f"$σ_Q$ = {cross_acc:.1f}$|a_0|$")
    plt.plot(cross_acc, acc_vs_sQ[idx]*100, 'kx')

else:
    print("Filter never better than sensor.")

plt.xlabel("scaling factor")
plt.ylabel("Accuracy [%]")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 4) Filtration results + best filter
best_idx = np.argmin(crmse_vs_sQ)
best_sigmaQ = sigma_Q_vals[best_idx]
best_filter = filt_vs_sQ[best_idx]

plt.figure(figsize=(10,4))

if dim == 1:
    plt.plot(t, p_exp[:,0]*1e3, 'b--', lw=1, label="ground truth", zorder=1)
    
    for filt in filt_vs_sQ:
        plt.plot(t, filt*1e3, 'r', lw=0.5, alpha=0.25, zorder=1)
    
    plt.plot(t, best_filter*1e3, 'g', lw=1.5, label=f'best filtration ($σ_Q = {best_sigmaQ/abs(a0):.1f}|a_0|$)')
    plt.xlabel("Time [s]")
    plt.ylabel("x [mm]")

else: 
    plt.plot(p_exp[:,0]*1e3, p_exp[:,1]*1e3, 'b--', lw=1, label="ground truth", zorder=1)
    
    for filt in filt_vs_sQ:
        plt.plot(filt[:,0]*1e3, filt[:,1]*1e3, 'r', lw=0.5, alpha=0.25, zorder=1)
    
    plt.plot(best_filter[:,0]*1e3, best_filter[:,1]*1e3, 'g', lw=1.5, label=f'best filtration ($σ_Q = {best_sigmaQ/abs(a0):.1f}|a_0|$)')
    
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.axis('equal')

# dummy line for the legend
plt.plot([], [], 'r', alpha=0.3, label='test filters')
plt.grid(True, alpha=0.3)

plt.legend()
plt.show()

# 5) gt + obs + best filter

plt.figure(figsize=(10,4))
if dim == 1:
    plt.plot(t, p_exp[:, 0]*1e3, 'b--', linewidth=1, label = "ground truth")
    plt.plot(t, p_obs[:, 0]*1e3, 'k', linewidth=1.5, label = "observation")
    plt.plot(t, best_filter*1e3, 'r', lw=1, label=f'best filtration ($σ_Q = {best_sigmaQ/abs(a0):.1f}|a_0|$)')
    plt.xlabel('Time [s]')
    plt.ylabel('x [mm]')

else:
    plt.plot(p_exp[:, 0]*1e3, p_exp[:, 1]*1e3, 'b--', linewidth=1, label = "ground truth")
    plt.plot(p_obs[:, 0]*1e3, p_obs[:, 1]*1e3, 'k', linewidth=1.5, label = "observation")
    plt.plot(best_filter[:,0]*1e3, best_filter[:,1]*1e3, 'r', lw=1, label=f'best filtration ($σ_Q = {best_sigmaQ/abs(a0):.1f}|a_0|$)')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]') 
    plt.axis('equal')   

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


'''
best_sv_vals = results["best_sv_vals"]
global_best_filt = results["global_best_filt"]
global_best_p_obs = results["global_best_p_obs"]
crmse_vs_sV = results["crmse_vs_sV"]
cep_vs_sV = results["cep_vs_sV"]
filt_vs_sV = results["filt_vs_sV"]
first_cross_sv_vals=results["first_cross_sv_vals"]
cep_obs_list=results["cep_obs_list"]
global_best_crmse=results["global_best_crmse"]
accuracy_vs_sV=results["accuracy_vs_sV"]

'''
