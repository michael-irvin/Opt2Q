# MW Irvin -- Lopez Lab -- 2019-02-02

"""
===============================================
Apoptosis Model Calibrated to Fluorescence Data
===============================================

Plot results of the calibration of an apoptosis model to fluorescence data
"""
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.quantitative_example.fluorescence_likelihood_fn import fluorescence_data, likelihood_fn

calibrated_parameters = np.array([-3.61356002, -4.30641673, -6.18174898,  0.06471329, -8.59553348, -5.44259363])

# ------- Data -------
fluorescence_data_time_hrs = fluorescence_data['time'].apply(lambda x: x/3600)  # convert to hrs for plot.

cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.fill_between(fluorescence_data_time_hrs,
                 fluorescence_data['norm_IC-RP']+np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                 fluorescence_data['norm_IC-RP']-np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                 color=cm.colors[0], alpha=0.25)
plt.fill_between(fluorescence_data_time_hrs,
                 fluorescence_data['norm_EC-RP']+np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                 fluorescence_data['norm_EC-RP']-np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                 color=cm.colors[1], alpha=0.25)

# ======== Starting Parameters ========
# -------- Simulator --------
sim_results_0 = likelihood_fn.simulator.run()

# -------- Dynamics ---------
measurement_results_0 = likelihood_fn.measurement_model.run()
measurement_results_time_hrs = measurement_results_0['time'].apply(lambda x:x/3600)

plt.plot(measurement_results_time_hrs, measurement_results_0['BID_obs'], '--',
         color=cm.colors[0], alpha=0.5, label='tBID (starting params)')
plt.plot(measurement_results_time_hrs, measurement_results_0['cPARP_obs'], '--',
         color=cm.colors[1], alpha=0.5, label='cPARP (starting params)')

# ======== Calibrate Parameters ========
# -------- Simulator --------
likelihood_fn(calibrated_parameters)  # load calibrated parameters into likelihood function
sim_results = likelihood_fn.simulator.run()

# -------- Dynamics ---------
likelihood_fn.measurement_model.update_simulation_result(sim_results)
measurement_results_0 = likelihood_fn.measurement_model.run()

plt.plot(measurement_results_time_hrs, measurement_results_0['BID_obs'],
         color=cm.colors[0], alpha=0.5, label='tBID')
plt.plot(measurement_results_time_hrs, measurement_results_0['cPARP_obs'],
         color=cm.colors[1], alpha=0.5, label='cPARP')

plt.legend(loc="upper left")
plt.xlabel('time [hrs]')
plt.ylabel('fluorescence')
plt.title("Initiator and Effector Caspase Activity Reporter Fluorescence")
plt.savefig('fig3.png')
fig.show()

# ========= Plot Simulation Results wo Normalization =========
results_df_0 = sim_results_0.opt2q_dataframe
results_df_0 = results_df_0.reset_index()

results_df = sim_results.opt2q_dataframe
results_df = results_df.reset_index()
results_df_time_hrs = results_df['time'].apply(lambda x: x/3600)  # convert to hrs for plot

fig = plt.figure()
plt.plot(results_df_time_hrs, results_df['BID_obs'], color=cm.colors[0], alpha=0.5, label='calibrated params')
plt.plot(results_df_time_hrs, results_df_0['BID_obs'], '--', color=cm.colors[0], alpha=0.5, label='starting params')
plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results (tBID)')
plt.legend()
plt.savefig('fig1.png')
fig.show()

fig = plt.figure()
plt.plot(results_df_time_hrs, results_df['cPARP_obs'], color=cm.colors[1], alpha=0.5, label='calibrated params')
plt.plot(results_df_time_hrs, results_df_0['cPARP_obs'], '--', color=cm.colors[1], alpha=0.5, label='starting params')
plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results (cPARP)')
plt.legend()
plt.savefig('fig2.png')
fig.show()