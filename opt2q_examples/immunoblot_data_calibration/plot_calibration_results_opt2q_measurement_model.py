import os
import pandas as pd
import numpy as np
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from opt2q_examples.immunoblot_data_calibration.generate_synthetic_immunoblot_dataset import synthetic_immunoblot_data
from opt2q_examples.immunoblot_data_calibration.immunoblot_data_calibration_fixed_measurement_model import wb
import pickle
from opt2q.measurement import Fluorescence


# ================ UPDATE THIS PART ==================
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'immunoblot_calibration_results'
calibration_date = '2020118'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot'

# ====================================================
# Load data
with open(f'synthetic_WB_dataset_300s_2020_12_3.pkl', 'rb') as data_input:
    dataset = pickle.load(data_input)

cal_args = (script_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)

# Parameter names
model_param_names = utils.get_model_param_names(model)
population_param_name = utils.get_population_param('immunoblot')
measurement_param_names = utils.get_measurement_param_names('immunoblot')

# Starting params
model_param_start = utils.get_model_param_start(model)
population_param_start = utils.get_population_param_start('immunoblot')
measurement_model_param_start = utils.get_measurement_model_true_params('immunoblot')

# True params
model_param_true = utils.get_model_param_true()
population_param_true = population_param_start
measurement_model_param_true = measurement_model_param_start

# Priors
model_param_priors = utils.get_model_param_priors()

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

calibration_tag = 'opt2q'
if 'opt2q' in calibration_tag:
    best_measurement_model_parameters = best_parameter_sample[:, -10:]
    random_post_measurement_model_parameters = parameter_sample[:, -10:]
else:
    best_measurement_model_parameters = []
    random_post_measurement_model_parameters = []

# =====================================================
# Run Simulations
sim = utils.set_up_simulator('immunoblot', model)

# Simulate Starting Params
sim.param_values = pd.DataFrame([model_param_start], columns=model_param_names)
sim_res_param_start = sim.run()

# Simulate True Params
sim.param_values = pd.DataFrame([10**model_param_true], columns=model_param_names)
sim_res_param_true = sim.run()

# Simulate Best Params (wo extrinsic noise)
best_parameters = pd.DataFrame(10**best_parameter_sample[:, :len(model_param_names)], columns=model_param_names)
best_parameters.reset_index(inplace=True)
best_parameters = best_parameters.rename(columns={'index': 'simulation'})

sim.param_values = best_parameters
sim_res_param_best = sim.run()

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :len(model_param_names)], columns=model_param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()
sim_res_param_ensemble_normed = Fluorescence(sim_res_param_ensemble, observables=['tBID_obs', 'cPARP_obs']).run()

# Simulate Random Ensemble of Prior Parameters (wo extrinsic noise)
prior_ensemble_parameters = pd.DataFrame(10**prior_parameter_sample[:, :len(model_param_names)],
                                         columns=model_param_names)
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
plt.suptitle('Convergence and Parameter Values for Ordinal Data \n '
             'Calibration (Opt2Q Measurement Model)')
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
             'Ordinal Data Calibration (Opt2Q Measurement Model)', y=0.95, fontsize=16)
plt.show()

fig, ax = plt.subplots()
ax.set_title('tBID Simulation using Random Sample from Posterior \n of Model Trained to Ordinal Data '
             '(Opt2Q Measurement Model)')
plot.plot_simulation_results(ax, sim_res_param_ensemble, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plt.show()

fig2, ax = plt.subplots()
ax.set_title('tBID Simulation using Maximum Posterior Sample \n of Model Trained to Ordinal Data '
             '(Opt2Q Measurement Model)')
plot.plot_simulation_results(ax, sim_res_param_best, 'tBID_obs', alpha=0.4, color=cm.colors[1])
plt.show()

# Plot the Data
fig3, ax1 = plt.subplots()
ax1.set_title('Dataset with Synthetic Ordinal Measurements of tBID Concentration')
ax1.scatter(x=synthetic_immunoblot_data.data['time'],
            y=synthetic_immunoblot_data.data['tBID_blot'].values,
            s=10, color=cm.colors[1], label=f'tBID ordinal data', alpha=0.5)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Ordinal Categories of tBID')
ax1.legend()
plt.show()

# tBID 90% Credible Interval Calculation
prior_sim_res_low_quantile, prior_sim_res_high_quantile = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble, [0.05, 0.95])

fig4, ax = plt.subplots()
ax.set_title('tBID Credible Interval using Random Sample from Posterior \n of Model Trained to Cell Death Data '
             '(Opt2Q Measurement Model)')
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='50ng/mL')


sim_res_param_ensemble_median = calc.simulation_results_quantile(sim_res_param_ensemble.opt2q_dataframe, 0.5)

plot.plot_simulation_results(ax, sim_res_param_ensemble_median, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax, prior_sim_res_low_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax, prior_sim_res_high_quantile, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior 50ng/mL')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized tBID Concentration')
plt.legend()
plt.show()

# Normalized tBID 90% Credible Interval Calculation and Measurement Model
prior_sim_res_low_quantile_normed, prior_sim_res_high_quantile_normed = calc.simulation_results_quantiles_list(
    prior_sim_res_param_ensemble_normed, [0.05, 0.95])

measurement_model = wb

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey='all', gridspec_kw={'width_ratios': [2, 1]})
ax1.set_title('Normalized tBID 90% Credible Interval using Random Sample from Posterior '
              '\n of Model Trained to Fluorescence Data')

plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='posterior')

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)

plot.plot_simulation_results(ax1, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax1, prior_sim_res_low_quantile_normed, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax1, prior_sim_res_high_quantile_normed, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized tBID Concentration')

ax1.scatter(x=synthetic_immunoblot_data.data['time'],
            y=synthetic_immunoblot_data.data['tBID_blot'].values / synthetic_immunoblot_data.data['tBID_blot'].max(),
            s=10, color=cm.colors[1], label=f'tBID ordinal data', alpha=0.5)

ax1.legend()

plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})

if 'opt2q' in calibration_tag:
    lc_params = random_post_measurement_model_parameters
    for row_id in range(len(lc_params)):
        row = lc_params[row_id]
        c0 = row[0]
        t1 = row[1]
        t2 = t1 + row[2]
        t3 = t2 + row[3]
        t4 = t3 + row[4]

        c5 = row[5]
        t6 = row[6]
        t7 = t6 + row[7]
        t8 = t7 + row[8]

        wb.process.get_step('classifier').set_params(
            **{'coefficients__tBID_blot__coef_': np.array([c0]),
               'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
               'coefficients__cPARP_blot__coef_': np.array([c5]),
               'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})

        lc_results = wb.process.get_step('classifier').transform(plot_domain)
        tBID_results = lc_results.filter(regex='tBID_blot')
        ci = 0
        for col in sorted(list(tBID_results.columns)):
            ax2.plot(tBID_results[col].values, np.linspace(0, 1, 100), alpha=0.1, color=cm.colors[ci])
            ci += 1
    ax2.set_title('Opt2Q Calibration of Measurement Model')
    ax2.set_xlabel('Probability of Class Membership')

    classifier_params = measurement_model_param_true
    c0 = classifier_params[0]
    t1 = classifier_params[1]
    t2 = t1 + classifier_params[2]
    t3 = t2 + classifier_params[3]
    t4 = t3 + classifier_params[4]

    c5 = classifier_params[5]
    t6 = classifier_params[6]
    t7 = t6 + classifier_params[7]
    t8 = t7 + classifier_params[8]

    wb.process.get_step('classifier').set_params(
        **{'coefficients__tBID_blot__coef_': np.array([c0]),
           'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
           'coefficients__cPARP_blot__coef_': np.array([c5]),
           'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})

    lc_results = wb.process.get_step('classifier').transform(plot_domain)
    cPARP_results = lc_results.filter(regex='cPARP_blot')
    tBID_results = lc_results.filter(regex='tBID_blot')
    ax2.set_title('Opt2Q Calibration of Measurement Model')

    c_id = 0
    for col in sorted(list(tBID_results.columns)):
        ax2.plot(tBID_results[col].values, np.linspace(0, 1, 100), alpha=1, color=cm.colors[c_id])
        c_id += 1

    ax2.legend(loc='lower center')
    plt.show()

    # classifier_params = all_true_params[len(true_params):]
    # c0 = classifier_params[0]
    # t1 = classifier_params[1]
    # t2 = t1 + classifier_params[2]
    # t3 = t2 + classifier_params[3]
    # t4 = t3 + classifier_params[4]
    #
    # c5 = classifier_params[5]
    # t6 = classifier_params[6]
    # t7 = t6 + classifier_params[7]
    # t8 = t7 + classifier_params[8]
    #
    # wb.process.get_step('classifier').set_params(
    #     **{'coefficients__tBID_blot__coef_': np.array([c0]),
    #        'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
    #        'coefficients__cPARP_blot__coef_': np.array([c5]),
    #        'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})
    #
    # lc_results = wb.process.get_step('classifier').transform(plot_domain)
    # cPARP_results = lc_results.filter(regex='cPARP_blot')
    # tBID_results = lc_results.filter(regex='tBID_blot')

fig6, ax1 = plt.subplots()
ax1.set_title('Normalized tBID 90% Credible Interval using Random Sample from Posterior '
              '\n of Model Trained to Ordinal Data, Using Incorrect Opt2Q Measurement Model')

plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'tBID_obs',
                                                   alpha=0.2, color=cm.colors[1], label='posterior')

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)

plot.plot_simulation_results(ax1, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=1.0, color=cm.colors[1])
plot.plot_simulation_results(ax1, prior_sim_res_low_quantile_normed, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--')
plot.plot_simulation_results(ax1, prior_sim_res_high_quantile_normed, 'tBID_obs',
                             alpha=1.0, color=cm.colors[1], linestyle='--', label='prior')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized tBID Concentration')

ax1.legend()
plt.show()
