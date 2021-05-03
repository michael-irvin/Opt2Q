# MW Irvin -- Lopez Lab -- 2018-10-10
import os
import pandas as pd
import numpy as np
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from opt2q.measurement.base import ScaleToMinMax, Interpolate, LogisticClassifier
import random
import pickle


# Calibration Methods
# ===================
# We calibrate the aEARM to the dataset that contains ordinal values of tBID and cPARP at every 60s time-point.
# The dataset is stored here: 'synthetic_WB_dataset_60s_2020_12_7.pkl'
#
# The values of tBID and cPARP are all independent (i.e. the log-likelihood of the dataset is the sum of the individual
# log-likelihoods).
#
# aEARM priors were lon-norm distributions located at the "ground truth" and scaled +/- 1.5 orders of magnitude.
# We calibrate the measurement model parameters using different parameter sets.


from pydream.parameters import SampledParam
from scipy.stats import norm, uniform, cauchy
true_params = utils.get_model_param_true(include_extra_reactions=False)

model_param_priors_uniform = [SampledParam(norm, loc=p, scale=1.5) for p in true_params] + \
                             [SampledParam(uniform, loc=0.0, scale=100.0),  # coefficients__tBID_blot__coef_    float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__tBID_blot__theta_1  float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__tBID_blot__theta_2  float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__tBID_blot__theta_3  float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__tBID_blot__theta_4  float
                              SampledParam(uniform, loc=0.0, scale=100.0),  # coefficients__cPARP_blot__coef_   float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__cPARP_blot__theta_1 float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__cPARP_blot__theta_2 float
                              SampledParam(uniform, loc=0.0, scale=1.0),  # coefficients__cPARP_blot__theta_3 float
                              ]

model_param_priors_cauchy = [SampledParam(norm, loc=p, scale=1.5) for p in true_params] + \
                            [SampledParam(cauchy, loc=50.0, scale=10.0),  # coefficients__tBID_blot__coef_    float
                             SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_1  float
                             SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_2  float
                             SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_3  float
                             SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_4  float
                             SampledParam(cauchy, loc=50.0, scale=10.0),  # coefficients__cPARP_blot__coef_   float
                             SampledParam(cauchy, loc=0.25, scale=0.05),  # coefficients__cPARP_blot__theta_1 float
                             SampledParam(cauchy, loc=0.25, scale=0.05),  # coefficients__cPARP_blot__theta_2 float
                             SampledParam(cauchy, loc=0.25, scale=0.05),  # coefficients__cPARP_blot__theta_3 float
                             ]

model_param_priors_cauchy2 = [SampledParam(norm, loc=p, scale=1.5) for p in true_params] + \
                             [SampledParam(cauchy, loc=50.0, scale=1.0),  # coefficients__tBID_blot__coef_    float
                              SampledParam(cauchy, loc=0.2, scale=0.005),  # coefficients__tBID_blot__theta_1  float
                              SampledParam(cauchy, loc=0.2, scale=0.005),  # coefficients__tBID_blot__theta_2  float
                              SampledParam(cauchy, loc=0.2, scale=0.005),  # coefficients__tBID_blot__theta_3  float
                              SampledParam(cauchy, loc=0.2, scale=0.005),  # coefficients__tBID_blot__theta_4  float
                              SampledParam(cauchy, loc=50.0, scale=1.0),  # coefficients__cPARP_blot__coef_   float
                              SampledParam(cauchy, loc=0.25, scale=0.005),  # coefficients__cPARP_blot__theta_1 float
                              SampledParam(cauchy, loc=0.25, scale=0.005),  # coefficients__cPARP_blot__theta_2 float
                              SampledParam(cauchy, loc=0.25, scale=0.005),  # coefficients__cPARP_blot__theta_3 float
                              ]
#
# The calibration files were saved as:
# 'apoptosis_params_and_immunoblot_classifier_calibration_uniform_priors_'
# 'apoptosis_params_and_immunoblot_classifier_calibration_cauchy_05_'
# 'apoptosis_params_and_immunoblot_classifier_calibration_cauchy_005_'

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
# Load and Plot Synthetic Ordinal tBID vs. time-series data (60s)
with open(f'synthetic_WB_dataset_60s_2020_12_7.pkl', 'rb') as data_input:
    dataset_60 = pickle.load(data_input)

# ================================================== #
# ================================================== #
#                                                    #
#     Plot the Uniform Priors Measurement Models     #
#     Ordinal Data Calibration                       #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '2021210'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration_uniform_priors_'


cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.50*len(parameter_traces[0]))
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]
print('Uniform')
print(gr_values)
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
        ax_trace_list[i].set_ylabel(param_subset[i], rotation=0, labelpad=90)
        ax_trace_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)
        ax_trace_list[i].set_xticks([500000, 750000, 1000000])
        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    plt.savefig(f'Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_Uniform_Priors'
                f'_Calibration_Plot{param_subset_idx}.pdf')
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
model_param_priors = model_param_priors_uniform
# prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
# for p in model_param_priors[1:]:  # Add measurement model priors
#     prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

# =======================================================================
# =======================================================================
# Comparing KDE of posterior, prior and "true" for parameters in measurement model.
# Plots KDE of the prior (blue), posterior (orange), and true (vertical dotted line)
log_model_param_true = utils.get_model_param_true()
measurement_param_true = utils.get_measurement_model_true_params('immunoblot')
params_true = np.hstack((log_model_param_true, measurement_param_true))

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
        x = np.linspace(*priors_subset[i].dist.interval(0.95), 100)
        plt.plot(x, priors_subset[i].dist.pdf(x))

        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Supplemental__KDE_Parameter_Priors_Posteriors_Uniform_Priors_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Fluorescence Data
sim = utils.set_up_simulator('fluorescence', model)

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

# set up classifier
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset_60.data['time']).transform(sim_res_param_true_normed)
lc = LogisticClassifier(dataset_60, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')
lc.set_up(x_int)
lc.do_fit_transform = False

lc_results_list = []
p_dom = pd.DataFrame(np.linspace(0, 1, 100), columns=['plot_domain'])
for row in parameter_sample[:, -9:]:
    lc.set_params(**utils.get_classifier_params(row, measurement_type='immunoblot'))
    lc_results = pd.concat([lc.transform(plot_domain), p_dom], axis=1)
    lc_results_list.append(lc_results)

lc_results_df = pd.concat(lc_results_list, ignore_index=True)
lc_results_df.drop(columns=['tBID_blot', 'cPARP_blot'], inplace=True)

upper_lc = calc.simulation_results_quantile(lc_results_df, 0.975, groupby='plot_domain')
median_lc = calc.simulation_results_quantile(lc_results_df, 0.50, groupby='plot_domain')
lower_lc = calc.simulation_results_quantile(lc_results_df, 0.025, groupby='plot_domain')

# Plot tBID Dynamics ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

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

# Prediction of the Measurement Model
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset_60.data['time']).transform(sim_res_param_true_normed)
lc = LogisticClassifier(dataset_60, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')

for level in dataset_60.data['tBID_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['tBID_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['tBID_blot'].values/(dataset_60.data['tBID_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

# Plot measurement model prediction
tBID_results = upper_lc.filter(regex='tBID_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['tBID_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['tBID_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Fig4__Calibration_of_Ordinal_Uniform_Priors_tBID_Plot.pdf')
plt.show()

# Plot cPARP Dynamics ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'cPARP_obs', color='k', alpha=0.2,
                                                   upper_quantile=0.975, lower_quantile=0.025, label='posterior',
                                                   linewidth=line_width)
plot.plot_simulation_results(ax1, sim_res_param_ensemble_median_normed, 'cPARP_obs', color='k', alpha=0.4)
plot.plot_simulation_results(ax1, sim_res_param_true_normed, 'cPARP_obs', color='k', alpha=0.4, label='true',
                             linewidth=line_width, linestyle=':')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized cPARP Concentration')
ax1.legend()
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)

for level in dataset_60.data['cPARP_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['cPARP_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['cPARP_blot'].values/(dataset_60.data['cPARP_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

# Prediction of the Measurement Model (See above)

# Plot measurement model prediction
tBID_results = upper_lc.filter(regex='cPARP_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['cPARP_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['cPARP_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Supplemental__Calibration_of_Ordinal_Uniform_Priors_cPARP_Plot.pdf')
plt.show()

# ================================================== #
# ================================================== #
#                                                    #
#     Plot the Cauchy Priors v1 Measurement Models   #
#     Ordinal Data Calibration                       #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '2021210'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration_cauchy_05_'

cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.50*len(parameter_traces[0]))
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]

print('Cauchy 05')
print(gr_values)
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
        ax_trace_list[i].set_ylabel(param_subset[i], rotation=0, labelpad=90)
        ax_trace_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)
        ax_trace_list[i].set_xticks([650000, 975000, 1300000])
        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    plt.savefig(f'Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_Cauchy_05_Priors'
                f'_Calibration_Plot{param_subset_idx}.pdf')
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
model_param_priors = model_param_priors_cauchy
# prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
# for p in model_param_priors[1:]:  # Add measurement model priors
#     prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

# =======================================================================
# =======================================================================
# Comparing KDE of posterior, prior and "true" for parameters in measurement model.
# Plots KDE of the prior (blue), posterior (orange), and true (vertical dotted line)
log_model_param_true = utils.get_model_param_true()
measurement_param_true = utils.get_measurement_model_true_params('immunoblot')
params_true = np.hstack((log_model_param_true, measurement_param_true))

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
        x = np.linspace(*priors_subset[i].dist.interval(0.95), 100)
        plt.plot(x, priors_subset[i].dist.pdf(x))

        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Supplemental__KDE_Parameter_Priors_Posteriors_Cauchy_05_Priors_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Fluorescence Data
sim = utils.set_up_simulator('immunoblot', model)

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

# set up classifier
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset_60.data['time']).transform(sim_res_param_true_normed)
lc = LogisticClassifier(dataset_60, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')
lc.set_up(x_int)
lc.do_fit_transform = False

lc_results_list = []
p_dom = pd.DataFrame(np.linspace(0, 1, 100), columns=['plot_domain'])
for row in parameter_sample[:, -9:]:
    lc.set_params(**utils.get_classifier_params(row, measurement_type='immunoblot'))
    lc_results = pd.concat([lc.transform(plot_domain), p_dom], axis=1)
    lc_results_list.append(lc_results)

lc_results_df = pd.concat(lc_results_list, ignore_index=True)
lc_results_df.drop(columns=['tBID_blot', 'cPARP_blot'], inplace=True)

upper_lc = calc.simulation_results_quantile(lc_results_df, 0.975, groupby='plot_domain')
median_lc = calc.simulation_results_quantile(lc_results_df, 0.50, groupby='plot_domain')
lower_lc = calc.simulation_results_quantile(lc_results_df, 0.025, groupby='plot_domain')

# Plot tBID Dynamics ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

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

# Prediction of the Measurement Model
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset_60.data['time']).transform(sim_res_param_true_normed)
lc = LogisticClassifier(dataset_60, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')

for level in dataset_60.data['tBID_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['tBID_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['tBID_blot'].values/(dataset_60.data['tBID_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

# Plot measurement model prediction
tBID_results = upper_lc.filter(regex='tBID_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['tBID_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['tBID_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Fig4__Calibration_of_Ordinal_Cauchy_05_Priors_tBID_Plot.pdf')
plt.show()

# Plot cPARP Dynamics ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'cPARP_obs', color='k', alpha=0.2,
                                                   upper_quantile=0.975, lower_quantile=0.025, label='posterior',
                                                   linewidth=line_width)
plot.plot_simulation_results(ax1, sim_res_param_ensemble_median_normed, 'cPARP_obs', color='k', alpha=0.4)
plot.plot_simulation_results(ax1, sim_res_param_true_normed, 'cPARP_obs', color='k', alpha=0.4, label='true',
                             linewidth=line_width, linestyle=':')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized cPARP Concentration')
ax1.legend()
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)

for level in dataset_60.data['cPARP_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['cPARP_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['cPARP_blot'].values/(dataset_60.data['cPARP_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

# Prediction of the Measurement Model (See above)

# Plot measurement model prediction
tBID_results = upper_lc.filter(regex='cPARP_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['cPARP_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['cPARP_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Supplemental__Calibration_of_Ordinal_Cauchy_05_Priors_cPARP_Plot.pdf')
plt.show()


# ================================================== #
# ================================================== #
#                                                    #
#     Plot the Cauchy Priors v2 Measurement Models   #
#     Ordinal Data Calibration                       #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '2021210'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration_cauchy_005_'


cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.50*len(parameter_traces[0]))
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]
print('Cauchy 005')
print(gr_values)


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
        ax_trace_list[i].set_ylabel(param_subset[i], rotation=0, labelpad=90)
        ax_trace_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)
        ax_trace_list[i].set_xticks([850000, 1275000, 1700000])
        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    plt.savefig(f'Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_Cauchy_005_Priors'
                f'_Calibration_Plot{param_subset_idx}.pdf')
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
model_param_priors = model_param_priors_cauchy2
# prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
# for p in model_param_priors[1:]:  # Add measurement model priors
#     prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

# =======================================================================
# =======================================================================
# Comparing KDE of posterior, prior and "true" for parameters in measurement model.
# Plots KDE of the prior (blue), posterior (orange), and true (vertical dotted line)
log_model_param_true = utils.get_model_param_true()
measurement_param_true = utils.get_measurement_model_true_params('immunoblot')
params_true = np.hstack((log_model_param_true, measurement_param_true))

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
        interval_pdf = priors_subset[i].dist.interval(0.95)
        x = np.linspace(*priors_subset[i].dist.interval(0.95), 100)
        plt.plot(x, priors_subset[i].dist.pdf(x))

        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Supplemental__KDE_Parameter_Priors_Posteriors_Cauchy_005_Priors_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Fluorescence Data
sim = utils.set_up_simulator('immunoblot', model)

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

# set up classifier
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset_60.data['time']).transform(sim_res_param_true_normed)
lc = LogisticClassifier(dataset_60, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')
lc.set_up(x_int)
lc.do_fit_transform = False

lc_results_list = []
p_dom = pd.DataFrame(np.linspace(0, 1, 100), columns=['plot_domain'])
for row in parameter_sample[:, -9:]:
    lc.set_params(**utils.get_classifier_params(row, measurement_type='immunoblot'))
    lc_results = pd.concat([lc.transform(plot_domain), p_dom], axis=1)
    lc_results_list.append(lc_results)

lc_results_df = pd.concat(lc_results_list, ignore_index=True)
lc_results_df.drop(columns=['tBID_blot', 'cPARP_blot'], inplace=True)

upper_lc = calc.simulation_results_quantile(lc_results_df, 0.975, groupby='plot_domain')
median_lc = calc.simulation_results_quantile(lc_results_df, 0.50, groupby='plot_domain')
lower_lc = calc.simulation_results_quantile(lc_results_df, 0.025, groupby='plot_domain')

# Plot tBID Dynamics ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

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

# Prediction of the Measurement Model
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset_60.data['time']).transform(sim_res_param_true_normed)
lc = LogisticClassifier(dataset_60, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')

for level in dataset_60.data['tBID_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['tBID_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['tBID_blot'].values/(dataset_60.data['tBID_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

# Plot measurement model prediction
tBID_results = upper_lc.filter(regex='tBID_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['tBID_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['tBID_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Fig4__Calibration_of_Ordinal_Cauchy_005_Priors_tBID_Plot.pdf')
plt.show()

# Plot cPARP Dynamics ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax1, sim_res_param_ensemble_normed, 'cPARP_obs', color='k',
                                                   alpha=0.2, upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax1, sim_res_param_ensemble_median_normed, 'cPARP_obs', color='k', alpha=0.4)
plot.plot_simulation_results(ax1, sim_res_param_true_normed, 'cPARP_obs', color='k', alpha=0.4, label='true',
                             linewidth=line_width, linestyle=':')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized cPARP Concentration')
ax1.legend()
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)

for level in dataset_60.data['cPARP_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['cPARP_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['cPARP_blot'].values/(dataset_60.data['cPARP_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

# Prediction of the Measurement Model (See above)

# Plot measurement model prediction
tBID_results = upper_lc.filter(regex='cPARP_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['cPARP_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['cPARP_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Supplemental__Calibration_of_Ordinal_Cauchy_005_Priors_cPARP_Plot.pdf')
plt.show()
