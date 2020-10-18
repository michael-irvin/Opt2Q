import os
import pandas as pd
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


# ================ UPDATE THIS PART ==================
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'cell_death_data_calibration_results'
calibration_date = '2020910'  # calibration file name contains date string
calibration_tag = 'fmm_extra_and_missing'

# ====================================================
# Load data
cal_args = (script_dir, calibration_folder, calibration_date, calibration_tag)

gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)

# Parameter names
model_param_names = utils.get_model_param_names(model, include_extra_reactions=True)
population_param = utils.get_population_param('cell_death_data')
measurement_param_names = utils.get_measurement_param_names('cell_death_data')

# Starting params
model_param_start = utils.get_model_param_start(model, include_extra_reactions=True)

# True params
model_param_true = utils.get_model_param_true(include_extra_reactions=True)

# Priors
model_param_priors = utils.get_model_param_priors(include_extra_reactions=True)

# Post-Burn-in Parameters
burn_in = 80000
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

# Sample of Max Posterior Parameters
sample_size = 10
best_parameter_sample, best_log_p_sample, best_indices = utils.get_max_posterior_parameters(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# # Random Sample from Posterior
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# =====================================================
# Run Simulations
sim = utils.set_up_simulator('cell_death_data', model)

# Simulate Starting Params
sim.param_values = pd.DataFrame([model_param_start], columns=model_param_names)
sim_res_param_start = sim.run()

# Simulate True Params
sim.param_values = pd.DataFrame([10**model_param_true], columns=model_param_names)
sim_res_param_true = sim.run()

# Simulate Best Params (wo extrinsic noise)
best_parameters = pd.DataFrame(10**best_parameter_sample[:, :-1], columns=model_param_names)
best_parameters.reset_index(inplace=True)
best_parameters = best_parameters.rename(columns={'index': 'simulation'})

sim.param_values = best_parameters
sim_res_param_best = sim.run()

best_parameters_10ng_ml = pd.DataFrame(10**best_parameter_sample[:, :-1], columns=model_param_names)
best_parameters_10ng_ml.reset_index(inplace=True)
best_parameters_10ng_ml = best_parameters_10ng_ml.rename(columns={'index': 'simulation'})
best_parameters_10ng_ml['L_0'] = 600

sim.param_values = best_parameters_10ng_ml
sim_res_param_best_10ng_ml = sim.run()

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :-1], columns=model_param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()

ensemble_parameters_10ng_ml = pd.DataFrame(10**parameter_sample[:, :-1], columns=model_param_names)
ensemble_parameters_10ng_ml.reset_index(inplace=True)
ensemble_parameters_10ng_ml = ensemble_parameters_10ng_ml.rename(columns={'index': 'simulation'})
ensemble_parameters_10ng_ml['L_0'] = 600

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
plt.suptitle('Convergence and Parameter Values for Cell Death Data \n '
             'Calibration (Fixed Measurement Model w/ Extra and Missing Feature)')
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.93])
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
plt.suptitle('Parameter Value Traces and Histograms \n '
             'Cell Death Data Calibration (Fixed Measurement Model w/ Extra and Missing Feature)', y=0.95, fontsize=16)
plt.show()

fig, ax = plt.subplots()
ax.set_title('tBID Simulation using Random Sample from Posterior of Model Trained \n to Cell Death Data '
             '(Fixed Measurement Model w/ Extra and Missing Feature)')
plot.plot_simulation_results(ax, sim_res_param_ensemble, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_ensemble_10ng_ml, 'tBID_obs', alpha=0.2, color='k')
plt.show()

fig2, ax = plt.subplots()
ax.set_title('tBID Simulation using Maximum Posterior Sample of Model Trained  \n to Cell Death Data '
             '(Fixed Measurement Model w/ Extra and Missing Feature)')
plot.plot_simulation_results(ax, sim_res_param_best, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_best_10ng_ml, 'tBID_obs', alpha=0.2, color='k')
plt.show()
