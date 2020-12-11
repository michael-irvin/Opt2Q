import os
import pandas as pd
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


# ================ UPDATE THIS PART ==================
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'fluorescence_calibration_results'
calibration_date = '2020113'  # calibration file name contains date string
calibration_tag = 'fluorescence_data'

# ====================================================
# Load data
cal_args = (script_dir, calibration_folder, calibration_date, calibration_tag)

gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args)

# Parameter names
model_param_names = utils.get_model_param_names(model)
population_param = utils.get_population_param('fluorescence')
measurement_param_names = utils.get_measurement_param_names('fluorescence')

# Starting params
model_param_start = utils.get_model_param_start(model)

# True params
model_param_true = utils.get_model_param_true()

# Priors
model_param_priors = utils.get_model_param_priors()

# Post-Burn-in Parameters
burn_in = 50000
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

# Sample for Ensemble Simulations
sample_size = 10
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# =====================================================
# Run Simulations
sim = utils.set_up_simulator('fluorescence', model)

# Simulate Starting Params
sim.param_values = pd.DataFrame([model_param_start], columns=model_param_names)
sim_res_param_start = sim.run()

# Simulate True Params
sim.param_values = pd.DataFrame([10**model_param_true], columns=model_param_names)
sim_res_param_true = sim.run()

# Simulate Ensemble Params
ensemble_parameters = pd.DataFrame(10**parameter_sample, columns=model_param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()

ensemble_parameters_10ng_ml = pd.DataFrame(10**parameter_sample, columns=model_param_names)
ensemble_parameters_10ng_ml.reset_index(inplace=True)
ensemble_parameters_10ng_ml = ensemble_parameters_10ng_ml.rename(columns={'index': 'simulation'})


sim.param_values = ensemble_parameters_10ng_ml
sim_res_param_ensemble_10ng_ml = sim.run()

# =====================================================
# ======================= Plots =======================

cm = plt.get_cmap('tab10')

# Plot GR and Parameter Value Estimates
fig = plt.figure(1)
gs = gridspec.GridSpec(1, 5)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:])

plot.gelman_rubin_values(ax1, model_param_names, gr_values[:len(model_param_names)])
plot.model_param_estimates_relative_to_prior(ax2, parameter_traces, model_param_priors)

ax2.set_title('Posterior Estimates Relative to Prior')
ax2.axes.get_yaxis().set_visible(False)
plt.suptitle('Convergence and Parameter Values for Fluorescence Data Calibration')
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot parameter traces and histograms
fig = plt.figure(1, figsize=(9, 11))
gs = gridspec.GridSpec(len(model_param_names), 4, hspace=0.1)
ax_trace_list = []
ax_hist_list = []
for i, param in enumerate(model_param_names):
    ax_trace_list.append(fig.add_subplot(gs[i, :3]))
    ax_hist_list.append(fig.add_subplot(gs[i, 3]))

    plot.parameter_traces(ax_trace_list[i], parameter_traces_burn_in,
                          burnin=burn_in, param_idx=i, labels=False)
    plot.parameter_traces_histogram(ax_hist_list[i], parameter_traces_burn_in,
                                    param_idx=i, labels=False)

    ax_trace_list[i].set_yticks([])
    ax_trace_list[i].set_ylabel(model_param_names[i])
    ax_hist_list[i].axes.get_yaxis().set_visible(False)

ax_trace_list[0].set_title('Parameter Traces')
ax_trace_list[-1].set_xlabel('Iteration')

ax_hist_list[0].set_title('Parameter Histogram')
ax_hist_list[-1].set_xlabel('Parameter Value')
plt.suptitle('Parameter Value Traces and Histograms \n Fluorescence Data Calibration', y=0.95, fontsize=16)
plt.show()

fig, ax = plt.subplots()
ax.set_title('cPARP Simulation using Random Sample from Posterior \n of Model Trained to Fluorescence Data')
plot.plot_simulation_results(ax, sim_res_param_ensemble, 'cPARP_obs', alpha=0.4, color=cm.colors[0])
plt.show()

fig, ax = plt.subplots()
ax.set_title('tBID Simulation using Random Sample from Posterior \n of Model Trained to Fluorescence Data')
plot.plot_simulation_results(ax, sim_res_param_ensemble, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plt.show()


quit()

# =====================================================
# =============INDIVIDUAL PLOTS =======================
# Plot Simulations

quit()

# # Plot convergence figures
# fig_01, ax = plt.subplots()
# ax.set_title('Gelman-Rubin Statistic for Calibration to Fluorescence Data')
# plot.gelman_rubin_values(ax, model_param_names, gr_values[:len(model_param_names)])
# plt.show()
#
# fig_02, ax = plt.subplots()
# ax.set_title('Histogram Gelman-Rubin for Calibration to Fluorescence Data')
# plot.gelman_rubin_histogram(ax, gr_values[:len(model_param_names)])
# plt.show()
#
# fig_04, ax = plt.subplots()
# ax.set_title('Parameter Value Trace')
# plot.parameter_traces(ax, parameter_traces, param_idx=0)
# plt.show()
#
# fig_05, ax = plt.subplots()
# ax.set_title('Parameter Gelman-Rubin Trace')
# plot.parameter_traces_gr(ax, parameter_traces, param_idx=0)
# plt.show()
#
# fig_06, ax = plt.subplots()
# ax.set_title('Parameter Kernel Density Estimates of Each Chain')
# plot.parameter_traces_kde(ax, parameter_traces_burn_in, param_idx=0)
# plt.show()
#
# fig_07, ax = plt.subplots()
# ax.set_title('Parameter Histograms of Each Chain')
# plot.parameter_traces_histogram(ax, parameter_traces_burn_in, param_idx=0)
# plt.show()
#
# fig_08, ax = plt.subplots()
# ax.set_title('Multi-chain Autocorrelation Trace')
# plot.parameter_traces_acf(ax, parameter_traces_burn_in, param_idx=0, burnin=burn_in)
# plt.show()
#
# fig_09, ax = plt.subplots()
# ax.set_title('Parameter Value Traces (without Burn-in)')
# plot.parameter_traces(ax, parameter_traces_burn_in, param_idx=0, burnin=burn_in)
# plt.show()
#
# # =====================================================
# # Plot convergence figures
# fig_03, ax = plt.subplots()
# ax.set_title('Posterior Parameter Estimates Relative to Prior')
# plot.model_param_estimates_relative_to_prior(ax, parameter_traces, model_param_priors)
# plt.show()
