# MW Irvin -- Lopez Lab -- 2021-03-04
import os
from matplotlib import pyplot as plt
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import synth_data, extrinsic_noise_params
import numpy as np
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
import matplotlib.gridspec as gridspec
import pandas as pd
import random
from opt2q.measurement.base import ScaleToMinMax
from matplotlib import colors
from matplotlib.lines import Line2D

# Calibration Methods
# ===================
# We calibrated aEARM to cell death vs. survival outcomes for cells that had known starting concentrations of ligand
# and other observables (e.g. DISC, MOMP, and Unrelated species). The cell death and survival outcomes were modeled
# using a preset classifier, which models the probability of cell death as a function of variables extracted for "ground
# truth" tBID dynamics.
#
# All cell death vs survival outcomes were considered to be independent (i.e. the log-likelihood of the dataset is the
# sum of the individual log-likelihoods).
#
# aEARM priors were log-norm distributed located at the "ground truth" and scaled +/- 1.5 orders of magnitude.
# Measurement model priors were laplace priors as shown below:

from pydream.parameters import SampledParam
from scipy.stats import norm, laplace, invgamma
true_params = utils.get_model_param_true(include_extra_reactions=True)

# Priors
nu = 100
noisy_param_stdev = 0.20

alpha = int(np.ceil(nu/2.0))
beta = alpha/noisy_param_stdev**2

model_param_priors = [SampledParam(norm, loc=p, scale=1.5) for p in true_params] + \
                   [SampledParam(invgamma, *[alpha], scale=beta),
                    SampledParam(laplace, loc=0.0, scale=1.0),  # slope   float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # intercept  float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # "Unrelated_Signal" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # "tBID_obs" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # "time" coef  float
                    ]  # coef are assigned in order by their column names' ASCII values

# The calibration files were saved as:
# 'apoptosis_model_tbid_cell_death_data_calibration_opt2q_...'

# =====================================
# ======== File Details ===========
# Update this part with the new log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

# ==================================================================
# Plot settings
cm = plt.get_cmap('tab10')
cm2 = plt.get_cmap('Set2')

tick_labels_x = 16
tick_labels_y = 17
line_width = 2

# ====================================================
# ====================================================
# Plot Cell-Death vs. Survival Data
fig00, ax = plt.subplots(figsize=(6, 3.75))

condition1 = (synth_data['apoptosis'] == 1) & (synth_data['TRAIL_conc'] == '10ng/mL')
condition2 = (synth_data['apoptosis'] == 1) & (synth_data['TRAIL_conc'] == '50ng/mL')
condition3 = (synth_data['apoptosis'] == 0) & (synth_data['TRAIL_conc'] == '10ng/mL')
condition4 = (synth_data['apoptosis'] == 0) & (synth_data['TRAIL_conc'] == '50ng/mL')

ax.scatter(x=extrinsic_noise_params[condition1]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition1]['apoptosis'].shape[0]), marker='x', color=cm.colors[1],
           label='Apoptosis at 10ng/mL TRAIL', alpha=0.5)

ax.scatter(x=extrinsic_noise_params[condition3]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition3]['apoptosis'].shape[0])+0.5, marker='o', color=cm.colors[1],
           label='Survival at 10ng/mL TRAIL', alpha=0.5)

ax.scatter(x=extrinsic_noise_params[condition2]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition2]['apoptosis'].shape[0])+1.0, marker='x', color=cm.colors[7],
           label='Apoptosis at 50ng/mL TRAIL', alpha=0.5)

ax.scatter(x=extrinsic_noise_params[condition4]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition4]['apoptosis'].shape[0])+1.5, marker='o', color=cm.colors[7],
           label='Survival at 50ng/mL TRAIL', alpha=0.5)
ax.set_ylim(-0.25, 2.0)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('DISC formation rate coefficient X 10^5 [1/s]')
# ax.legend(loc='upper right')
plt.savefig('Fig3b__Synthetic_Cell_Death_vs_Survival_Dataset.pdf')
plt.show()

# Plot Cell-Death vs. Survival Data
fig01, ax = plt.subplots(figsize=(6, 3.75))

condition1 = (synth_data['apoptosis'] == 1) & (synth_data['TRAIL_conc'] == '10ng/mL')
condition2 = (synth_data['apoptosis'] == 1) & (synth_data['TRAIL_conc'] == '50ng/mL')
condition3 = (synth_data['apoptosis'] == 0) & (synth_data['TRAIL_conc'] == '10ng/mL')
condition4 = (synth_data['apoptosis'] == 0) & (synth_data['TRAIL_conc'] == '50ng/mL')

ax.scatter(x=extrinsic_noise_params[condition1]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition1]['apoptosis'].shape[0]), marker='x', color=cm.colors[1],
           label='Apoptosis at 10ng/mL TRAIL', alpha=0.)


ax.scatter(x=extrinsic_noise_params[condition3]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition3]['apoptosis'].shape[0])+0.5, marker='o', color=cm.colors[1],
           label='Survival at 10ng/mL TRAIL', alpha=0.)

ax.scatter(x=extrinsic_noise_params[condition2]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition2]['apoptosis'].shape[0])+1.0, marker='x', color=cm.colors[7],
           label='Apoptosis at 50ng/mL TRAIL', alpha=0.)

ax.scatter(x=extrinsic_noise_params[condition4]['kc0']*1e5,
           y=0.25*np.random.random(synth_data[condition4]['apoptosis'].shape[0])+1.5, marker='o', color=cm.colors[7],
           label='Survival at 50ng/mL TRAIL', alpha=0.)
ax.set_ylim(-0.25, 2.0)
ax.axes.get_yaxis().set_visible(False)
ax.set_xlabel('DISC formation rate coefficient X 10^5 [1/s]')
ax.legend(loc='upper right')
plt.savefig('Fig3b__Synthetic_Cell_Death_vs_Survival_Dataset_Legend_Only.pdf')
plt.show()

# ====================================================
# ======== File Details ==============================

calibration_folder = 'cell_death_data_calibration_results'
calibration_date = '2020911'  # calibration file name contains date string
calibration_tag = 'apoptosis_model_tbid_cell_death_data_calibration_opt2q_'


cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.50*len(parameter_traces[0]))
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model, include_extra_reactions=True) + \
              utils.get_population_param('cell_death_data') + \
              utils.get_measurement_param_names('cell_death_data')
print('Cell Death Model')
for v in gr_values:
    print(v)
# =======================================================================
# ============== Plot parameter traces and histograms ===================
# Trace of value vs. iteration for parameters in the model for each chain in the PyDREAM algorithm.
# The second column is the histogram of these values for each chain.
# Third column is gelman-rubin metric for each parameter.

n_params_subset = 12
local_tick_labels_x = tick_labels_x-4
for param_subset_idx in range(int(np.ceil(len(param_names)/n_params_subset))):
    param_subset = param_names[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))]
    param_traces_subset = [p[:, 12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))] for p in
                           parameter_traces_burn_in]

    fig = plt.figure(1, figsize=(9, 11*len(param_subset)/12.0))
    gs = gridspec.GridSpec(len(param_subset), 5, hspace=0.6)
    ax_trace_list = []
    ax_hist_list = []

    for i, param in enumerate(param_subset):
        ax_trace_list.append(fig.add_subplot(gs[i, :3]))
        ax_hist_list.append(fig.add_subplot(gs[i, 3]))
        plot.parameter_traces(ax_trace_list[i], param_traces_subset,
                              burnin=burn_in, param_idx=i, labels=False)
        plot.parameter_traces_histogram(ax_hist_list[i], param_traces_subset,
                                        param_idx=i, labels=False)
        ax_trace_list[i].set_yticks([])
        ax_trace_list[i].set_ylabel(param_subset[i], rotation=0, labelpad=50)
        ax_trace_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    plt.savefig(f'Supplemental__Log_Posterior_Traces_and_Hist_for_Cell_Death_Data_Calibration_Plot{param_subset_idx}.pdf')
    ax_trace_list[-1].set_ylabel(param_subset[-1], rotation=0, labelpad=40)
    gs.tight_layout(fig, rect=[0.2, 0.0, 1, 0.93])
    plt.show()

# =======================================================================
# =======================================================================

# True Parameter Values
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))

# Random Sample from Posterior
random.seed(100)
sample_size = 1000
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)

# =======================================================================
# =======================================================================
# Comparing KDE of posterior, prior and "true" for parameters in measurement model.
# Plots KDE of the prior (blue), posterior (orange), and true (vertical dotted line)
log_model_param_true = utils.get_model_param_true(include_extra_reactions=True)
population_param_true = utils.get_population_param_start('cell_death_data')

# The measurement model formula is m*(x+y+...) where m, x, y, ... can be positive or negative.
# Therefore an equivalent measurement model is -m*(-x-y-...). The calibration converged on the latter formula.
# We account for this by multiplying the "ground truth" by -1.
measurement_param_true = [-1.0*p for p in utils.get_measurement_model_true_params('cell_death_data')]
params_true = np.hstack((log_model_param_true, population_param_true, measurement_param_true))

n = 24
gs_columns = 3

for ps in range(int(np.ceil(len(param_names)/n_params_subset))):
    param_subset = param_names[n*ps:min(n*(ps+1), len(param_names))]

    # priors_subset = prior_parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
    priors_subset = model_param_priors[n*ps:min(n*(ps+1), len(param_names))]
    posteriors_subset = parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
    true_subset = params_true[n*ps:min(n*(ps+1), len(param_names))]

    gs_rows = int(np.ceil(len(param_subset)/gs_columns))

    fig = plt.figure(1, figsize=(9, 11*gs_rows/8.0))
    gs = gridspec.GridSpec(gs_rows, gs_columns, hspace=0.1)

    for i, param in enumerate(param_subset):
        r = int(i / gs_columns)
        c = i % gs_columns
        ax = fig.add_subplot(gs[r, c])
        ax.set_yticks([])
        ax.set_title(param)
        # Prior
        # plot.kde_of_parameter(ax, priors_subset[:, i], color=cm.colors[0], alpha=0.6)
        x = np.linspace(*priors_subset[i].dist.interval(0.99), 100)
        plt.plot(x, priors_subset[i].dist.pdf(x))

        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Supplemental__KDE_Parameter_Priors_Posteriors_Cell_Death_Data_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Fluorescence Data
sim = utils.set_up_simulator('cell_death_data', model)

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample, columns=param_names)
ensemble_parameters.reset_index(inplace=True)
ensemble_parameters = ensemble_parameters.rename(columns={'index': 'simulation'})

sim.param_values = ensemble_parameters
sim_res_param_ensemble = sim.run()
results_param_ensemble = sim_res_param_ensemble.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
sim_res_param_ensemble_normed = ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'], groupby='simulation').\
    transform(results_param_ensemble[['time', 'tBID_obs', 'cPARP_obs', 'simulation']])

# Simulate True Params
sim.param_values = model_param_true
sim_res_param_true = sim.run()
results_param_true = sim_res_param_true.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
sim_res_param_true_normed = ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs']).\
    transform(results_param_true[['time', 'tBID_obs', 'cPARP_obs']])

# Posterior Prediction of the Measurement Model
plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})


# Plot tBID Dynamics ==================================================
sim_res_param_ensemble_lower_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.025)
sim_res_param_ensemble_upper_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.975)

y_lower = sim_res_param_ensemble_lower_normed['tBID_obs']
y_upper = sim_res_param_ensemble_upper_normed['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the 95% CI is ', area)

fig, ax1 = plt.subplots()

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'tBID_obs', color='k', alpha=0.2,
                                                   upper_quantile=0.975, lower_quantile=0.025, label='posterior',
                                                   linewidth=line_width)
plot.plot_simulation_results(ax1, sim_res_param_ensemble_median_normed, 'tBID_obs', color='k', alpha=0.4)
plot.plot_simulation_results(ax1, sim_res_param_true_normed, 'tBID_obs', color='k', alpha=0.4, label='true',
                             linewidth=line_width, linestyle=':')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized tBID Concentration')
ax1.legend()
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)

plt.savefig('Fig3b__Calibration_to_cell_death_data_tBID_Plot.pdf')
plt.show()

# =====================================================
# Simulate extrinsic noise
parameter_sample_size = 100

try:
    features_random_post_populations = pd.read_csv(
        f'features_random_post_populations_manuscript_{calibration_tag}.csv', index_col=0)

except FileNotFoundError:
    # Simulate populations based on random sample of posterior
    # Takes about 24hrs to run on my laptop
    parameter_sample_en, log_p_sample_en = utils.get_parameter_sample(
        parameter_traces_burn_in, log_p_traces_burn_in, sample_size=parameter_sample_size)
    param_populations = calc.simulate_population_multi_params(parameter_sample_en)

    sim.param_values = param_populations
    sim_res_random_post_populations = sim.run()
    features_random_post_populations = calc.pre_process_simulation(sim_res_random_post_populations)
    features_random_post_populations.to_csv(f'features_random_post_populations_manuscript_{calibration_tag}.csv')

cols = features_random_post_populations.columns

data_and_features_random_post_populations = pd.DataFrame(
    np.column_stack([features_random_post_populations[cols].values,
                     np.tile(synth_data['apoptosis'].values, parameter_sample_size)]),
    columns=cols.append(pd.Index(['apoptosis'])))

# =====================================================
# Plot Predicted Features
fig18, ax = plt.subplots(figsize=(7, 7))
ax.scatter(data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 0]
           .iloc[::80]['tBID_obs'],  # take only
           data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 0]
           .iloc[::80]['time'], marker='o', color='k', alpha=0.2)
ax.scatter(data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 1]
           .iloc[::80]['tBID_obs'],
           data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 1]
           .iloc[::80]['time'], marker='x', color='k', alpha=0.2)
plot.measurement_model_quantile_fill_between(ax, parameter_sample[:, -5:], 'tBID_obs', 'time', 0.5,
                                             np.linspace(-4, 4, 100), color='k', alpha=0.15)
plot.measurement_model_quantile(ax, parameter_sample[:, -5:], 'tBID_obs', 'time', 0.5,
                                np.linspace(-4, 4, 100), color='k', alpha=1)
plot.measurement_model_sample(ax, np.array([[4.00, -0.25, 0.00, 0.25, -1.00]]), 'tBID_obs', 'time', 0.5,
                              np.linspace(-4, 4, 100), color='b', alpha=0.5)
plot.population_kde_of_features(ax, features_random_post_populations[features_random_post_populations.TRAIL_conc
                                                                     == '10ng/mL'],
                                'tBID_obs', 'time', levels=[0.05],
                                cmap=colors.ListedColormap([cm.colors[1]]), alpha=0.2)
plot.population_kde_of_features(ax, features_random_post_populations[features_random_post_populations.TRAIL_conc
                                                                     == '50ng/mL'],
                                'tBID_obs', 'time', levels=[0.05],
                                cmap=colors.ListedColormap([cm.colors[7]]), alpha=0.2)
# plt.title('Measurement Model Predicted in the Calibration')
plt.xlabel('max Bid truncation rate')
plt.ylabel('time at max Bid truncation rate')
plt.xlim(-3.8, 3.8)
plt.ylim(-3.8, 3.8)
plt.savefig('Fig6__Calibration_to_cell_death_data_Measurement_Model_Plot.pdf')
plt.show()

fig19, ax0 = plt.subplots(figsize=(7, 7))
legend_elements = [Line2D([0], [0], color=cm.colors[1], alpha=0.5, label='10ng/mL TRAIL 95% contour'),
                   Line2D([0], [0], color=cm.colors[7], alpha=0.5, label='50ng/mL TRAIL 95% contour'),
                   Line2D([0], [0], color=cm.colors[7], marker='o', alpha=0.5, label='Surviving Cells'),
                   Line2D([0], [0], color=cm.colors[7], marker='x', alpha=0.5, label='Dead Cells'),
                   Line2D([0], [0], color='k', alpha=1, label='expected 50% probability line'),
                   Line2D([0], [0], color='b', alpha=0.5, label='preset 50% probability line')
                   ]
plt.legend(handles=legend_elements)
plt.savefig('Fig6__Calibration_to_cell_death_data_Measurement_Model_Legend.pdf')
plt.show()

fig20, ax = plt.subplots(figsize=(7, 7))
ax.scatter(data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 0]
           .iloc[::80]['tBID_obs'],
           data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 0]
           .iloc[::80]['Unrelated_Signal'], marker='o', color='k', alpha=0.2)
ax.scatter(data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 1]
           .iloc[::80]['tBID_obs'],
           data_and_features_random_post_populations[data_and_features_random_post_populations.apoptosis == 1]
           .iloc[::80]['Unrelated_Signal'], marker='x', color='k', alpha=0.2)
plot.measurement_model_quantile_fill_between(ax, parameter_sample[:, -5:], 'tBID_obs', 'Unrelated_Signal', 0.5,
                                             np.linspace(-4, 4, 100), color='k', alpha=0.15)
plot.measurement_model_quantile(ax, parameter_sample[:, -5:], 'tBID_obs', 'Unrelated_Signal', 0.5,
                                np.linspace(-4, 4, 100), color='k', alpha=1)
plot.measurement_model_sample(ax, np.array([[4.00, -0.25, 0.00, 0.25, -1.00]]), 'tBID_obs', 'Unrelated_Signal', 0.5,
                              np.linspace(-4, 4, 100), color='b', alpha=0.5)
plot.population_kde_of_features(ax, features_random_post_populations[features_random_post_populations.TRAIL_conc
                                                                 == '10ng/mL'],
                                'tBID_obs', 'Unrelated_Signal', levels=[0.05],
                                cmap=colors.ListedColormap([cm.colors[1]]), alpha=0.2)
plot.population_kde_of_features(ax, features_random_post_populations[features_random_post_populations.TRAIL_conc
                                                                     == '50ng/mL'],
                                'tBID_obs', 'Unrelated_Signal', levels=[0.05],
                                cmap=colors.ListedColormap([cm.colors[7]]), alpha=0.2)
plt.xlabel('max Bid truncation rate')
plt.ylabel('Unrelated Signal')
plt.xlim(-3.8, 3.8)
plt.ylim(-3.8, 3.8)
plt.savefig('Fig6__Calibration_to_cell_death_data_Measurement_Model_Plot_Unrelated_Species.pdf')
plt.show()


fig22, ax = plt.subplots()
plot.kde_of_parameter(ax, calc.feature_values(parameter_sample, 'Unrelated_Signal'),
                      color=cm.colors[0], label='Unrelated Signal')
plot.kde_of_parameter(ax, calc.feature_values(parameter_sample, 'time'),
                      color=cm.colors[1], label='Time at Maximum BID truncation')
plot.kde_of_parameter(ax, calc.feature_values(parameter_sample, 'tBID_obs'),
                      color=cm.colors[2], label='Maximum BID truncation')
plt.savefig('Fig6__Posterior_Estimates_of_Weight_Coefficients_of_Cell_Death_Predictors.pdf')
plt.show()
