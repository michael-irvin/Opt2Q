import os
import pandas as pd
import numpy as np
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from opt2q.measurement.base import ScaleToMinMax


# ================ UPDATE THIS PART ==================
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'fluorescence_calibration_results'
calibration_date = '2020126'  # calibration file name contains date string
calibration_tag = 'time_at_half_max'
# ====================================================

# ====================================================
# Load data
cal_args = (script_dir, calibration_folder, calibration_date, calibration_tag)

gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)

# Parameter names
model_param_names = utils.get_model_param_names(model, include_extra_reactions=True)
population_param_name = utils.get_population_param('cell_death_data')
measurement_param_names = ['smoothing_term']

# Starting params
model_param_start = utils.get_model_param_start(model, include_extra_reactions=True)
population_param_start = utils.get_population_param_start('cell_death_data')
measurement_model_param_start = [240.0]

# True params
model_param_true = utils.get_model_param_true(include_extra_reactions=True)
population_param_true = population_param_start
measurement_model_param_true = measurement_model_param_start

# Priors
model_param_priors = utils.get_model_param_priors(include_extra_reactions=True)
population_param_prior = utils.get_population_param_priors()

# Post-Burn-in Parameters
burn_in = 70000
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
prior_parameter_sample = np.column_stack([prior_parameter_sample, population_param_prior[0].dist.rvs(sample_size)])

if 'opt2q' in calibration_tag:
    best_measurement_model_parameters = best_parameter_sample[:, -1:]
    random_post_measurement_model_parameters = parameter_sample[: -1:]

# =====================================================
# Run Simulations
sim = utils.set_up_simulator('cell_death_data', model)

# Simulate Starting Params
sim.param_values = pd.DataFrame([model_param_start], columns=model_param_names)
sim_res_param_start = sim.run()
sim_res_param_start_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'],
                                           do_fit_transform=True).transform(sim_res_param_start.opt2q_dataframe)

# Simulate True Params
sim.param_values = pd.DataFrame([10**model_param_true], columns=model_param_names)
sim_res_param_true = sim.run()
sim_res_param_true_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'],
                                          do_fit_transform=True).transform(sim_res_param_true.opt2q_dataframe)


# Simulate Best Params (wo extrinsic noise)
best_parameters = pd.DataFrame(10**best_parameter_sample[:, :-2], columns=model_param_names)
best_parameters.reset_index(inplace=True)
best_parameters = best_parameters.rename(columns={'index': 'simulation'})

sim.param_values = best_parameters
sim_res_param_best = sim.run()
sim_res_param_best.opt2q_dataframe = sim_res_param_best.opt2q_dataframe.reset_index()
sim_res_param_best_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'], groupby=['simulation'],
                                          do_fit_transform=True).transform(sim_res_param_best.opt2q_dataframe)

best_parameters_10ng_ml = pd.DataFrame(10**best_parameter_sample[:, :-2], columns=model_param_names)
best_parameters_10ng_ml.reset_index(inplace=True)
best_parameters_10ng_ml = best_parameters_10ng_ml.rename(columns={'index': 'simulation'})
best_parameters_10ng_ml['L_0'] = 600

sim.param_values = best_parameters_10ng_ml
sim_res_param_best_10ng_ml = sim.run()
sim_res_param_best_10ng_ml.opt2q_dataframe = sim_res_param_best_10ng_ml.opt2q_dataframe.reset_index()
sim_res_param_best_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'], groupby=['simulation'],
                                                  do_fit_transform=True)\
    .transform(sim_res_param_best_10ng_ml.opt2q_dataframe)

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :-2], columns=model_param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()
sim_res_param_ensemble.opt2q_dataframe = sim_res_param_ensemble.opt2q_dataframe.reset_index()
sim_res_param_ensemble_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'], groupby=['simulation'],
                                              do_fit_transform=True).transform(sim_res_param_ensemble.opt2q_dataframe)

ensemble_parameters_10ng_ml = pd.DataFrame(10**parameter_sample[:, :-2], columns=model_param_names)
ensemble_parameters_10ng_ml.reset_index(inplace=True)
ensemble_parameters_10ng_ml = ensemble_parameters_10ng_ml.rename(columns={'index': 'simulation'})
ensemble_parameters_10ng_ml['L_0'] = 600

sim.param_values = ensemble_parameters_10ng_ml
sim_res_param_ensemble_10ng_ml = sim.run()
sim_res_param_ensemble_10ng_ml.opt2q_dataframe = sim_res_param_ensemble_10ng_ml.opt2q_dataframe.reset_index()
sim_res_param_ensemble_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'],
                                                      groupby=['simulation'], do_fit_transform=True)\
    .transform(sim_res_param_ensemble_10ng_ml.opt2q_dataframe)

# Simulate Random Ensemble of Prior Parameters (wo extrinsic noise)
prior_ensemble_parameters = pd.DataFrame(10**prior_parameter_sample[:, :34], columns=model_param_names)
prior_ensemble_parameters.reset_index(inplace=True)
prior_ensemble_parameters = prior_ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = prior_ensemble_parameters
prior_sim_res_param_ensemble = sim.run()
prior_sim_res_param_ensemble.opt2q_dataframe = prior_sim_res_param_ensemble.opt2q_dataframe.reset_index()
prior_sim_res_param_ensemble_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'], groupby=['simulation'],
                                                    do_fit_transform=True)\
    .transform(prior_sim_res_param_ensemble.opt2q_dataframe)

prior_ensemble_parameters_10ng_ml = pd.DataFrame(10**prior_parameter_sample[:, :34], columns=model_param_names)
prior_ensemble_parameters_10ng_ml.reset_index(inplace=True)
prior_ensemble_parameters_10ng_ml = prior_ensemble_parameters_10ng_ml.rename(columns={'index': 'simulation'})
prior_ensemble_parameters_10ng_ml['L_0'] = 600

sim.param_values = prior_ensemble_parameters_10ng_ml
prior_sim_res_param_ensemble_10ng_ml = sim.run()
prior_sim_res_param_ensemble_10ng_ml.opt2q_dataframe = prior_sim_res_param_ensemble_10ng_ml.opt2q_dataframe.reset_index()
prior_sim_res_param_ensemble_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'],
                                                            groupby=['simulation'], do_fit_transform=True)\
    .transform(prior_sim_res_param_ensemble_10ng_ml.opt2q_dataframe)

# =====================================================
# # Simulate extrinsic noise
# parameter_sample_size = 10
#
# try:
#     features_best_populations = pd.read_csv(
#         f'features_best_populations_{calibration_tag}.csv', index_col=0)
#     features_random_post_populations = pd.read_csv(
#         f'features_random_post_populations_{calibration_tag}.csv', index_col=0)
#     features_priors_populations = pd.read_csv(
#         f'features_priors_populations_{calibration_tag}.csv', index_col=0)
#
# except FileNotFoundError:
#     # Simulate populations based on best params from posterior
#     best_parameter_sample_en, best_log_p_sample_en, best_indices_en = utils.get_max_posterior_parameters(
#         parameter_traces_burn_in, log_p_traces_burn_in, sample_size=parameter_sample_size)
#     best_param_populations = calc.simulate_population_multi_params(best_parameter_sample_en)
#
#     sim.param_values = best_param_populations
#     sim_res_best_populations = sim.run()
#     features_best_populations = calc.pre_process_simulation(sim_res_best_populations)
#     features_best_populations.to_csv(f'features_best_populations_{calibration_tag}.csv')
#
#     # Simulate populations based on random sample of posterior
#     parameter_sample_en, log_p_sample_en = utils.get_parameter_sample(
#         parameter_traces_burn_in, log_p_traces_burn_in, sample_size=parameter_sample_size)
#     param_populations = calc.simulate_population_multi_params(parameter_sample_en)
#
#     sim.param_values = param_populations
#     sim_res_random_post_populations = sim.run()
#     features_random_post_populations = calc.pre_process_simulation(sim_res_random_post_populations)
#     features_random_post_populations.to_csv(f'features_random_post_populations_{calibration_tag}.csv')
#
#     prior_param_populations, sim_res_priors_populations_ = utils.sample_model_priors_for_feature_processing(
#         model_param_priors, population_param_prior, sim, 10)
#
#     sim.param_values = prior_param_populations
#     sim_res_priors_populations = sim.run()
#
#     features_priors_populations = calc.pre_process_simulation(sim_res_priors_populations)
#     features_priors_populations.to_csv(f'features_priors_populations_{calibration_tag}.csv')
#
# cols = features_best_populations.columns
# data_and_features_best_populations = pd.DataFrame(
#     np.column_stack([features_best_populations[cols].values,
#                      np.tile(dataset['apoptosis'].values, parameter_sample_size)]),
#     columns=cols.append(pd.Index(['apoptosis'])))
#
# data_and_features_random_post_populations = pd.DataFrame(
#     np.column_stack([features_random_post_populations[cols].values,
#                      np.tile(dataset['apoptosis'].values, parameter_sample_size)]),
#     columns=cols.append(pd.Index(['apoptosis'])))


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
plt.suptitle('Convergence and Parameter Values for Time at Half-Max tBID Data \n '
             'Calibration')
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.93])
plt.show()

# population parameter and classifier coefficient
fig4, ax = plt.subplots()
plot.gelman_rubin_values(ax, population_param_name + measurement_param_names, gr_values[len(model_param_names):])
ax.set_title('Gelman-Ruben Diagnostic for population parameter \n and Opt2Q Measurement Model Parameters')
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
             'Time at Half-Max tBID Data Calibration', y=0.95, fontsize=16)
plt.show()


fig = plt.figure(1, figsize=(9, 5.5))
gs = gridspec.GridSpec(len(population_param_name+measurement_param_names), 4, hspace=0.1)
ax_trace_list = []
ax_hist_list = []
for i, param in enumerate(population_param_name+measurement_param_names):
    ax_trace_list.append(fig.add_subplot(gs[i, :3]))
    ax_hist_list.append(fig.add_subplot(gs[i, 3]))
    plot.parameter_traces(ax_trace_list[i], parameter_traces_burn_in,
                          burnin=burn_in, param_idx=i+len(model_param_names), labels=False)
    plot.parameter_traces_histogram(ax_hist_list[i], parameter_traces_burn_in,
                                    param_idx=i+len(model_param_names), labels=False)
    ax_trace_list[i].set_yticks([])
    ax_trace_list[i].set_ylabel((population_param_name + measurement_param_names)[i], rotation=0, ha='right')
    ax_hist_list[i].axes.get_yaxis().set_visible(False)
ax_trace_list[0].set_title('Parameter Traces')
ax_trace_list[-1].set_xlabel('Iteration')
ax_hist_list[0].set_title('Parameter Histogram')
ax_hist_list[-1].set_xlabel('Parameter Value')
fig.subplots_adjust(top=0.8, left=0.23)
plt.suptitle('Parameter Value Traces and Histograms \n '
             'Time at Half-Max tBID Data Calibration', y=0.97, fontsize=16)
plt.show()

fig, ax = plt.subplots()
ax.set_title('tBID Simulation using Random Sample from Posterior \n of Model Trained to Time at Half-Max tBID Data')
plot.plot_simulation_results(ax, sim_res_param_ensemble, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_ensemble_10ng_ml, 'tBID_obs', alpha=0.3, color='k')
plt.show()

fig2, ax = plt.subplots()
ax.set_title('tBID Simulation using Maximum Posterior Sample \n of Model Trained to Time at Half-Max tBID Data')
plot.plot_simulation_results(ax, sim_res_param_best, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_best_10ng_ml, 'tBID_obs', alpha=0.3, color='k')
plt.show()

# tBID 90% Credible Interval Calculation
fig3, ax = plt.subplots()
ax.set_title('tBID Credible Interval using Maximum Posterior Sample \n of Model Trained to Time at Half-Max tBID Data')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_best, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_best_10ng_ml, 'tBID_obs',
                                                   alpha=0.1, color='k', label='10ng/mL')
sim_res_param_best_median = calc.simulation_results_quantile(sim_res_param_best.opt2q_dataframe, 0.5)
sim_res_param_best_10_median = calc.simulation_results_quantile(sim_res_param_best_10ng_ml.opt2q_dataframe, 0.5)
plot.plot_simulation_results(ax, sim_res_param_best_median, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_best_10_median, 'tBID_obs', alpha=0.3, color='k')

prior_sim_res_low_quantile, prior_sim_res_high_quantile = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble, [0.05, 0.95])
prior_sim_res_low_quantile_10ng_ml, prior_sim_res_high_quantile_10ng_ml = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_10ng_ml, [0.05, 0.95])

plot.plot_simulation_results(ax, prior_sim_res_low_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior 50ng/mL')

plot.plot_simulation_results(ax, prior_sim_res_low_quantile_10ng_ml, 'tBID_obs',
                             alpha=0.7, color='k', linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile_10ng_ml, 'tBID_obs',
                             alpha=0.7, color='k', linestyle='--', label='prior 10ng/mL')
plt.legend()
plt.show()

# tBID 90% Credible Interval Calculation
fig4, ax = plt.subplots()
ax.set_title('tBID Credible Interval using Random Sample from Posterior \n of Model Trained to '
             'Time at Half-Max tBID Data')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_10ng_ml, 'tBID_obs',
                                                   alpha=0.1, color='k', label='10ng/mL')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble.opt2q_dataframe, 0.5)
sim_res_param_ensemble_10_median = calc.simulation_results_quantile(sim_res_param_ensemble_10ng_ml.opt2q_dataframe, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_ensemble_10_median, 'tBID_obs', alpha=0.7, color='k')
plot.plot_simulation_results(ax, prior_sim_res_low_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior 50ng/mL')

plot.plot_simulation_results(ax, prior_sim_res_low_quantile_10ng_ml, 'tBID_obs',
                             alpha=0.7, color='k', linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile_10ng_ml, 'tBID_obs',
                             alpha=0.7, color='k', linestyle='--', label='prior 10ng/mL')

plt.legend()
plt.show()


# Normalized tBID 90% Credible Interval Calculation
fig4_0, ax = plt.subplots()
ax.set_title('Normalized tBID Credible Interval using Random Sample from Posterior \n '
             'of Model Trained to Time at Half-Max tBID Data')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_10ng_ml_normed, 'tBID_obs',
                                                   alpha=0.1, color='k', label='10ng/mL')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
sim_res_param_ensemble_10_median = calc.simulation_results_quantile(sim_res_param_ensemble_10ng_ml_normed, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_ensemble_10_median, 'tBID_obs', alpha=0.7, color='k')

prior_sim_res_low_quantile_n, prior_sim_res_high_quantile_n = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_normed, [0.05, 0.95])
prior_sim_res_low_quantile_10ng_ml_n, prior_sim_res_high_quantile_10ng_ml_n = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_10ng_ml_normed, [0.05, 0.95])

plot.plot_simulation_results(ax, prior_sim_res_low_quantile_n, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile_n, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior 50ng/mL')

plot.plot_simulation_results(ax, prior_sim_res_low_quantile_10ng_ml_n, 'tBID_obs',
                             alpha=0.7, color='k', linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile_10ng_ml_n, 'tBID_obs',
                             alpha=0.7, color='k', linestyle='--', label='prior 10ng/mL')

plt.legend()
plt.show()

