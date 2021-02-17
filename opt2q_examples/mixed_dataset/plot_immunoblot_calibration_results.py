import os
import pickle
import numpy as np
import pandas as pd
from opt2q.measurement.base import ScaleToMinMax
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib import colors
from opt2q_examples.mixed_dataset.calibration_opt2q_measurement_model import immunoblot


# ================ UPDATE THIS PART ==================
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'mixed_data_calibration_results'
calibration_date = '20201212'  # calibration file name contains date string
calibration_tag = 'opt2q'

# ====================================================

# ====================================================
# Load data
cd_folder = os.path.join(os.path.dirname(script_dir), 'cell_death_data_calibration')
cell_death_dataset = utils.load_cell_death_data(cd_folder, 'synthetic_tbid_dependent_apoptosis_data_large.csv')

with open(f'synthetic_IC_DISC_localization_blot_dataset_2020_10_18.pkl', 'rb') as data_input:
    immunoblot_dataset = pickle.load(data_input)

cal_args = (script_dir, calibration_folder, calibration_date, calibration_tag)

gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)

# Parameter names
model_param_names = utils.get_model_param_names(model, include_extra_reactions=True)
ib_measurement_param_names = utils.get_measurement_param_names('immunoblot_disc')

# Starting params
model_param_start = utils.get_model_param_start(model, include_extra_reactions=True)

# True params
model_param_true = utils.get_model_param_true(include_extra_reactions=True)
ib_measurement_model_param_start = utils.get_measurement_model_true_params('immunoblot_disc')

# Priors
model_param_priors = utils.get_model_param_priors(include_extra_reactions=True)

# Post-Burn-in Parameters
burn_in = 120000
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

# Sample of Max Posterior Parameters
sample_size = 100
best_parameter_sample, best_log_p_sample, best_indices = utils.get_max_posterior_parameters(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# # Random Sample from Posterior
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)

# Sample Measurement Model Params
calibration_tag = 'opt2q'
if 'opt2q' in calibration_tag:
    ib_best_measurement_model_parameters = best_parameter_sample[:, -4:]
    ib_random_post_measurement_model_parameters = parameter_sample[:, -4:]
else:
    ib_best_measurement_model_parameters = []
    ib_random_post_measurement_model_parameters = []

# =====================================================
# Run Simulations
sim = utils.set_up_simulator('immunoblot', model)

# Simulate Starting Params
sim.param_values = pd.DataFrame([model_param_start], columns=model_param_names)
sim_res_param_start = sim.run()

# Simulate True Params
sim.param_values = pd.DataFrame([10**model_param_true], columns=model_param_names)
sim_res_param_true = sim.run()
sim_res_param_true.opt2q_dataframe = sim_res_param_true.opt2q_dataframe.reset_index()
sim_res_param_true_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                          do_fit_transform=True).transform(sim_res_param_true.opt2q_dataframe)

model_param_true_10ng_ml = pd.DataFrame([10**model_param_true], columns=model_param_names)
model_param_true_10ng_ml['L_0'] = 600

sim.param_values = model_param_true_10ng_ml
sim_res_param_true_10ng_mL = sim.run()
sim_res_param_true_10ng_mL.opt2q_dataframe = sim_res_param_true_10ng_mL.opt2q_dataframe.reset_index()
sim_res_param_true_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                                  do_fit_transform=True)\
    .transform(sim_res_param_true_10ng_mL.opt2q_dataframe)

# Simulate Best Params (wo extrinsic noise)
best_parameters = pd.DataFrame(10**best_parameter_sample[:, :len(model_param_names)], columns=model_param_names)
best_parameters.reset_index(inplace=True)
best_parameters = best_parameters.rename(columns={'index': 'simulation'})

sim.param_values = best_parameters
sim_res_param_best = sim.run()
sim_res_param_best.opt2q_dataframe = sim_res_param_best.opt2q_dataframe.reset_index()
sim_res_param_best_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                          groupby=['simulation'], do_fit_transform=True)\
    .transform(sim_res_param_best.opt2q_dataframe)

best_parameters_10ng_ml = pd.DataFrame(10**best_parameter_sample[:, :len(model_param_names)], columns=model_param_names)
best_parameters_10ng_ml.reset_index(inplace=True)
best_parameters_10ng_ml = best_parameters_10ng_ml.rename(columns={'index': 'simulation'})
best_parameters_10ng_ml['L_0'] = 600

sim.param_values = best_parameters_10ng_ml
sim_res_param_best_10ng_ml = sim.run()
sim_res_param_best_10ng_ml.opt2q_dataframe = sim_res_param_best_10ng_ml.opt2q_dataframe.reset_index()
sim_res_param_best_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                                  groupby=['simulation'], do_fit_transform=True)\
    .transform(sim_res_param_best_10ng_ml.opt2q_dataframe)

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :len(model_param_names)], columns=model_param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()
sim_res_param_ensemble.opt2q_dataframe = sim_res_param_ensemble.opt2q_dataframe.reset_index()
sim_res_param_ensemble_normed = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                              groupby=['simulation'], do_fit_transform=True).transform(sim_res_param_ensemble.opt2q_dataframe)

ensemble_parameters_10ng_ml = pd.DataFrame(10**parameter_sample[:, :len(model_param_names)], columns=model_param_names)
ensemble_parameters_10ng_ml.reset_index(inplace=True)
ensemble_parameters_10ng_ml = ensemble_parameters_10ng_ml.rename(columns={'index': 'simulation'})
ensemble_parameters_10ng_ml['L_0'] = 600

sim.param_values = ensemble_parameters_10ng_ml
sim_res_param_ensemble_10ng_ml = sim.run()
sim_res_param_ensemble_10ng_ml.opt2q_dataframe = sim_res_param_ensemble_10ng_ml.opt2q_dataframe.reset_index()
sim_res_param_ensemble_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1),
                                                      columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                                      groupby=['simulation'], do_fit_transform=True)\
    .transform(sim_res_param_ensemble_10ng_ml.opt2q_dataframe)

# Simulate Random Ensemble of Prior Parameters (wo extrinsic noise)
prior_ensemble_parameters = pd.DataFrame(10**prior_parameter_sample[:, :34], columns=model_param_names)
prior_ensemble_parameters.reset_index(inplace=True)
prior_ensemble_parameters = prior_ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = prior_ensemble_parameters
prior_sim_res_param_ensemble = sim.run()
prior_sim_res_param_ensemble.opt2q_dataframe = prior_sim_res_param_ensemble.opt2q_dataframe.reset_index()
prior_sim_res_param_ensemble_normed = ScaleToMinMax(feature_range=(0, 1),
                                                    columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                                    groupby=['simulation'],
                                                    do_fit_transform=True)\
    .transform(prior_sim_res_param_ensemble.opt2q_dataframe)

prior_ensemble_parameters_10ng_ml = pd.DataFrame(10**prior_parameter_sample[:, :34], columns=model_param_names)
prior_ensemble_parameters_10ng_ml.reset_index(inplace=True)
prior_ensemble_parameters_10ng_ml = prior_ensemble_parameters_10ng_ml.rename(columns={'index': 'simulation'})
prior_ensemble_parameters_10ng_ml['L_0'] = 600

sim.param_values = prior_ensemble_parameters_10ng_ml
prior_sim_res_param_ensemble_10ng_ml = sim.run()
prior_sim_res_param_ensemble_10ng_ml.opt2q_dataframe = prior_sim_res_param_ensemble_10ng_ml.opt2q_dataframe.reset_index()
prior_sim_res_param_ensemble_10ng_ml_normed = ScaleToMinMax(feature_range=(0, 1),
                                                            columns=['tBID_obs', 'C8_DISC_recruitment_obs'],
                                                            groupby=['simulation'], do_fit_transform=True)\
    .transform(prior_sim_res_param_ensemble_10ng_ml.opt2q_dataframe)

# =====================================================
# ======================= Plots =======================

cm = plt.get_cmap('tab10')


# Normalized DISC 90% Credible Interval Calculation and Measurement Model
fig00, ax1 = plt.subplots()
ax1.set_title('DISC-IC Immunoblot Data')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Ordinal Measurement of DISC-Caspase Concentration')

ax1.scatter(
    x=immunoblot_dataset.data['time'],
    y=immunoblot_dataset.data['IC_DISC_localization'].values / immunoblot_dataset.data['IC_DISC_localization'].max(),
    s=10, color=cm.colors[1], label=f'DISC-Caspase ordinal data', alpha=0.5)

ax1.legend()
plt.show()
quit()

# Plot GR and Parameter Value Estimates
fig = plt.figure(1)
gs = gridspec.GridSpec(1, 5)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:])

plot.gelman_rubin_values(ax1, model_param_names, gr_values[:len(model_param_names)])
plot.model_param_estimates_relative_to_prior(ax2, parameter_traces, model_param_priors)

ax2.set_title('Posterior Estimates Relative to Prior')
ax2.axes.get_yaxis().set_visible(False)
plt.suptitle('Convergence and Parameter Values for \n '
             'Mixed Cell Death and DISC Immunoblot Dataset (Opt2Q measurement Model)')
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.93])
plt.show()

# population parameter and classifier coefficient
fig4, ax = plt.subplots()
plot.gelman_rubin_values(ax, ib_measurement_param_names,
                         gr_values[len(model_param_names):])
ax.set_title('Gelman-Ruben Diagnostic for population parameter \n and Opt2Q Measurement Model Parameters')
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.93])
plt.show()

# Plot parameter traces and histograms Model Parameters
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
             'DISC Immunoblot Dataset (Opt2Q measurement Model)', y=0.95, fontsize=16)
plt.show()

fig01 = plt.figure(1, figsize=(10.5, 5.5))
gs = gridspec.GridSpec(len(ib_measurement_param_names), 4, hspace=0.1)
ax_trace_list = []
ax_hist_list = []
for i, param in enumerate(ib_measurement_param_names):
    ax_trace_list.append(fig01.add_subplot(gs[i, :3]))
    ax_hist_list.append(fig01.add_subplot(gs[i, 3]))
    plot.parameter_traces(ax_trace_list[i], parameter_traces_burn_in,
                          burnin=burn_in, param_idx=i+len(model_param_names), labels=False)
    plot.parameter_traces_histogram(ax_hist_list[i], parameter_traces_burn_in,
                                    param_idx=i+len(model_param_names), labels=False)
    ax_trace_list[i].set_yticks([])
    ax_trace_list[i].set_ylabel(ib_measurement_param_names[i], rotation=0, ha='right')
    ax_hist_list[i].axes.get_yaxis().set_visible(False)
ax_trace_list[0].set_title('Parameter Traces')
ax_trace_list[-1].set_xlabel('Iteration')
ax_hist_list[0].set_title('Parameter Histogram')
ax_hist_list[-1].set_xlabel('Parameter Value')
fig01.subplots_adjust(top=0.8, left=0.25)
plt.suptitle('Parameter Value Traces and Histograms \n '
             'DISC Immunoblot (Opt2Q Measurement Model)', y=0.97, fontsize=16)
plt.show()

# Plot Dynamics
fig, ax = plt.subplots()
ax.set_title('Parameter Value Traces and Histograms \n '
             'DISC Immunoblot Dataset \n (Opt2Q measurement Model)')
plot.plot_simulation_results(ax, sim_res_param_ensemble, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_best_10ng_ml, 'tBID_obs', alpha=0.3, color=cm.colors[7])
plt.show()

# tBID 90% Credible Interval Calculation
fig3, ax = plt.subplots()
ax.set_title('tBID Credible Interval using Maximum Posterior Sample \n of Model Trained to '
             'DISC Immunoblot Data \n (Opt2Q Measurement Model)')
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
             'DISC Immunoblot Data \n (Opt2Q Measurement Model)')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_10ng_ml, 'tBID_obs',
                                                   alpha=0.1, color='k', label='10ng/mL')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble.opt2q_dataframe, 0.5)
sim_res_param_ensemble_10_median = calc.simulation_results_quantile(sim_res_param_ensemble_10ng_ml.opt2q_dataframe, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_ensemble_10_median, 'tBID_obs', alpha=0.7, color='k')

plot.plot_simulation_results(ax, sim_res_param_true, 'tBID_obs', linestyle=':', alpha=1.0, color=cm.colors[1],
                             label='"true" 50ng/mL ')

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
ax.set_title('Normalized tBID Credible Interval using Random Sample from Posterior \n o'
             'f Model Trained to DISC Immunoblot Data \n (Opt2Q Measurement Model)')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_10ng_ml_normed, 'tBID_obs',
                                                   alpha=0.1, color='k', label='10ng/mL')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
sim_res_param_ensemble_10_median = calc.simulation_results_quantile(sim_res_param_ensemble_10ng_ml_normed, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, sim_res_param_ensemble_10_median, 'tBID_obs', alpha=0.7, color='k')

plot.plot_simulation_results(ax, sim_res_param_true_normed, 'tBID_obs', linestyle=':', alpha=1.0, color=cm.colors[1],
                             label='"true" 50ng/mL ')
plot.plot_simulation_results(ax, sim_res_param_true_10ng_ml_normed, 'tBID_obs', linestyle=':', alpha=1.0,
                             color=cm.colors[7], label='"true" 10ng/mL ')

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


# Normalized DISC-Caspase 8 Credible Interval
fig5, ax = plt.subplots()
ax.set_title('Normalized DISC Credible Interval using Random Sample from Posterior \n o'
             'f Model Trained to DISC Immunoblot Data \n (Opt2Q Measurement Model)')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'C8_DISC_recruitment_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'C8_DISC_recruitment_obs', alpha=1.0, color=cm.colors[1])

plot.plot_simulation_results(ax, sim_res_param_true_normed, 'C8_DISC_recruitment_obs', linestyle=':', alpha=1.0, color=cm.colors[1],
                             label='"true" 50ng/mL ')

prior_sim_res_low_quantile_n, prior_sim_res_high_quantile_n = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_normed, [0.05, 0.95])

plot.plot_simulation_results(ax, prior_sim_res_low_quantile_n, 'C8_DISC_recruitment_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile_n, 'C8_DISC_recruitment_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior 50ng/mL')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Concentration of DISC-Caspase Complex')

plt.legend()
plt.show()

# Normalized DISC 90% Credible Interval Calculation and Measurement Model
fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey='all', gridspec_kw={'width_ratios': [2, 1]})
ax1.set_title('Normalized DISC Credible Interval using Random Sample from Posterior \n o'
              'f Model Trained to DISC Immunoblot Data \n (Opt2Q Measurement Model)')
plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'C8_DISC_recruitment_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')

sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)

plot.plot_simulation_results(ax1, sim_res_param_ensemble_median, 'C8_DISC_recruitment_obs',
                             alpha=1.0, color=cm.colors[1])

plot.plot_simulation_results(ax1, sim_res_param_true_normed, 'C8_DISC_recruitment_obs', linestyle=':', alpha=1.0,
                             color=cm.colors[1],
                             label='"true" 50ng/mL ')

prior_sim_res_low_quantile_n, prior_sim_res_high_quantile_n = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_normed, [0.05, 0.95])

plot.plot_simulation_results(ax1, prior_sim_res_low_quantile_n, 'C8_DISC_recruitment_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax1, prior_sim_res_high_quantile_n, 'C8_DISC_recruitment_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior 50ng/mL')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized Concentration of DISC-Caspase Concentration')

ax1.scatter(
    x=immunoblot_dataset.data['time'],
    y=immunoblot_dataset.data['IC_DISC_localization'].values / immunoblot_dataset.data['IC_DISC_localization'].max(),
    s=10, color=cm.colors[1], label=f'DISC-Caspase ordinal data', alpha=0.5)

ax1.legend()

# measurement model
wb = immunoblot

plot_domain = pd.DataFrame({'C8_DISC_recruitment_obs': np.linspace(0, 1, 100)})

if 'opt2q' in calibration_tag:
    lc_params = ib_random_post_measurement_model_parameters
    for row_id in range(len(lc_params)):
        row = lc_params[row_id]
        c0 = row[0]
        t1 = row[1]
        t2 = t1 + row[2]
        t3 = t2 + row[3]

        wb.process.get_step('classifier').set_params(
            **{'coefficients__IC_DISC_localization__coef_': np.array([c0]),
               'coefficients__IC_DISC_localization__theta_': np.array([t1, t2, t3]) * c0})

        lc_results = wb.process.get_step('classifier').transform(plot_domain)
        disc_results = lc_results.filter(regex='IC_DISC_localization')
        ci = 0
        for col in sorted(list(disc_results.columns)):
            ax2.plot(disc_results[col].values, np.linspace(0, 1, 100), alpha=0.1, color=cm.colors[ci])
            ci += 1
    ax2.set_title('Opt2Q Calibration of Measurement Model')
    ax2.set_xlabel('Probability of Class Membership')

    classifier_params = ib_measurement_model_param_start
    c0 = classifier_params[0]
    t1 = classifier_params[1]
    t2 = t1 + classifier_params[2]
    t3 = t2 + classifier_params[3]

    wb.process.get_step('classifier').set_params(
        **{'coefficients__IC_DISC_localization__coef_': np.array([c0]),
           'coefficients__IC_DISC_localization__theta_': np.array([t1, t2, t3]) * c0})

    lc_results = wb.process.get_step('classifier').transform(plot_domain)
    disc_results = lc_results.filter(regex='IC_DISC_localization')
    ci = 0
    for col in sorted(list(disc_results.columns)):
        ax2.plot(disc_results[col].values, np.linspace(0, 1, 100), alpha=1, color=cm.colors[ci])
        ci += 1

plt.show()

