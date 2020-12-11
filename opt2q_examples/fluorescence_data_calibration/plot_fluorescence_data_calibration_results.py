# MW Irvin -- Lopez Lab -- 2019-09-16

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from opt2q_examples.apoptosis_model import model
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q.measurement import Fluorescence
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ================ UPDATE THIS PART ==================
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'fluorescence_calibration_results'
calibration_date = '2020113'  # calibration file name contains date string
calibration_tag = 'fluorescence'

# ====================================================
# Load data
fluorescence_data = utils.load_cell_death_data(script_dir, 'fluorescence_data.csv')

cal_args = (script_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args)

# Parameter names
model_param_names = utils.get_model_param_names(model)
population_param_name = utils.get_population_param('fluorescence')
measurement_param_names = utils.get_measurement_param_names('fluorescence')

# Starting params
model_param_start = utils.get_model_param_start(model)
population_param_start = utils.get_population_param_start('fluorescence')
measurement_model_param_start = utils.get_measurement_model_true_params('fluorescence')

# True params
model_param_true = utils.get_model_param_true()
population_param_true = population_param_start

# Priors
model_param_priors = utils.get_model_param_priors()
population_param_prior = utils.get_population_param_priors('fluorescence')

# Post-Burn-in Parameters
burn_in = 20000
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

# Sample of Max Posterior Parameters
sample_size = 100
best_parameter_sample, best_log_p_sample, best_indices = utils.get_max_posterior_parameters(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Posterior
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)

if calibration_folder == 'cell_death_data_calibration_results':
    prior_parameter_sample = np.column_stack([prior_parameter_sample, population_param_prior[0].dist.rvs(sample_size)])

if 'opt2q' in calibration_tag:
    best_measurement_model_parameters = best_parameter_sample[:, -5:]
    random_post_measurement_model_parameters = parameter_sample[: -5:]

# =====================================================
# Run Simulations
sim = utils.set_up_simulator('fluorescence', model)

# Simulate Starting Params
sim.param_values = pd.DataFrame([model_param_start], columns=model_param_names)
sim_res_param_start = sim.run()
sim_res_param_start_normed = Fluorescence(sim_res_param_start, observables=['tBID_obs', 'cPARP_obs']).run()

# Simulate True Params
sim.param_values = pd.DataFrame([10**model_param_true], columns=model_param_names)
sim_res_param_true = sim.run()
sim_res_param_true_normed = Fluorescence(sim_res_param_true, observables=['tBID_obs', 'cPARP_obs']).run()


# Simulate Best Params (wo extrinsic noise)
best_parameters = pd.DataFrame(10**best_parameter_sample, columns=model_param_names)
best_parameters.reset_index(inplace=True)
best_parameters = best_parameters.rename(columns={'index': 'simulation'})

sim.param_values = best_parameters
sim_res_param_best = sim.run()
sim_res_param_best_normed = Fluorescence(sim_res_param_best, observables=['tBID_obs', 'cPARP_obs']).run()


# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample, columns=model_param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()
sim_res_param_ensemble_normed = Fluorescence(sim_res_param_ensemble, observables=['tBID_obs', 'cPARP_obs']).run()


# Simulate Random Ensemble of Prior Parameters (wo extrinsic noise)
prior_ensemble_parameters = pd.DataFrame(10**prior_parameter_sample, columns=model_param_names)
prior_ensemble_parameters.reset_index(inplace=True)
prior_ensemble_parameters = prior_ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = prior_ensemble_parameters
prior_sim_res_param_ensemble = sim.run()
prior_sim_res_param_ensemble_normed = Fluorescence(prior_sim_res_param_ensemble,
                                                   observables=['tBID_obs', 'cPARP_obs']).run()


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
plt.suptitle('Convergence and Parameter Values for Fluorescence Data')
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
             'Fluorescence Data Calibration', y=0.95, fontsize=16)
plt.show()

# Plot Simulations tBID 90% Credible Interval Calculation
prior_sim_res_low_quantile, prior_sim_res_high_quantile = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble, [0.05, 0.95])

fig4, ax = plt.subplots()
ax.set_title('tBID Credible Interval using Random Sample from Posterior \n of Model Trained to Fluorescence Data')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='posterior')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble.opt2q_dataframe, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, prior_sim_res_low_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior')
ax.set_xlabel('time [s]')
ax.set_ylabel('Copies per Cell')
plt.legend()
plt.show()

# Plot Simulations tBID 90% Credible Interval Calculation
prior_sim_res_low_quantile_normed, prior_sim_res_high_quantile_normed = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_normed, [0.05, 0.95])

fig5, ax = plt.subplots()
ax.set_title('Normalized tBID 90% Credible Interval using Random Sample from Posterior '
             '\n of Model Trained to Fluorescence Data')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='posterior')

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, prior_sim_res_low_quantile_normed, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile_normed, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
plt.legend()
plt.show()

fig6, ax = plt.subplots()
ax.set_title('Random Sample from Posterior of Model Trained \n to Fluorescence Data, tBID Dynamics')
plot.plot_simulation_results(ax, sim_res_param_ensemble_normed, 'tBID_obs', alpha=0.1, color=cm.colors[1],
                             label='predicted')
ax.plot(fluorescence_data['# Time']*60, fluorescence_data['norm_IC-RP'], ':', label='tBID Data', color=cm.colors[1])
ax.fill_between(fluorescence_data['# Time']*60,
                fluorescence_data['norm_IC-RP'] - np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                fluorescence_data['norm_IC-RP'] + np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                color=cm.colors[1], alpha=0.3)
legend_elements = [Line2D([0], [0], color=cm.colors[1], alpha=0.3, label='tBID Predictions'),
                   Line2D([0], [0], color=cm.colors[1], linestyle=':', label='tBID Data')]
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
plt.legend(handles=legend_elements)
plt.show()

# Plot the Data
fig7, ax = plt.subplots()
ax.set_title('Quantitative Time Course Dataset of tBID Concentration')

ax.plot(fluorescence_data['# Time']*60, fluorescence_data['norm_IC-RP'], ':', label='tBID Data', color=cm.colors[1])
ax.fill_between(fluorescence_data['# Time']*60,
                fluorescence_data['norm_IC-RP'] - np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                fluorescence_data['norm_IC-RP'] + np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                color=cm.colors[1], alpha=0.3)
legend_elements = [Line2D([0], [0], color=cm.colors[1], linestyle=':', label='tBID Data')]
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
plt.legend(handles=legend_elements)
plt.show()
