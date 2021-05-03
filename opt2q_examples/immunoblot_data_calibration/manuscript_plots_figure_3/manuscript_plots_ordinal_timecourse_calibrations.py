# MW Irvin -- Lopez Lab -- 2018-10-10
import os
import pandas as pd
import numpy as np
from opt2q_examples.plot_tools import utils, plot, calc
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from opt2q.measurement.base import ScaleToMinMax, Interpolate, LogisticClassifier
from matplotlib.lines import Line2D
import random
from pydream.convergence import Gelman_Rubin


# Calibration Methods
# ===================
# We calibrated aEARM to fluorescence data by Sorger et al. [REF] and different sized ordinal time-course datasets
# Using a "ground-truth" pre-parameterized aEARM and ordinal measurement model, we synthesized different sized ordinal
# datasets; i.e. containing measurements at every 60, 300, and 1500s time-point.
#
# The values of tBID and cPARP are all independent (i.e. the log-likelihood of the dataset is the sum of the individual
# log-likelihoods).
#
# aEARM priors were lon-norm distributions located at the "ground truth" and scaled +/- 1.5 orders of magnetude.
# Priors for the ordinal measurement model parameters were exponential distributions (see below)
#

from pydream.parameters import SampledParam
from scipy.stats import norm, expon
true_params = utils.get_model_param_true(include_extra_reactions=False)

model_param_priors = [SampledParam(norm, loc=true_params, scale=1.5),     # rate parameters floats
                      SampledParam(expon, loc=0.0, scale=100.0),          # coefficients__tBID_blot__coef_    float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_1  float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_2  float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_3  float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_4  float
                      SampledParam(expon, loc=0.0, scale=100.0),          # coefficients__cPARP_blot__coef_   float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_1 float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_2 float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_3 float
                      SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_4 float
                      ]
#
# The calibration files were saved as:
# 'fluorescence_calibration_2020113...'
# 'apoptosis_params_and_immunoblot_classifier_calibration_2020118...'
# 'apoptosis_params_and_immunoblot_classifier_calibration_300s_2021225...'
# 'apoptosis_params_and_immunoblot_classifier_calibration_180s_2021319...'
# 'apoptosis_params_and_immunoblot_classifier_calibration_1500s_20201212...'

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
# Load Fluorescence Data
opt2q_example_folder = os.path.dirname(parent_dir)
fluorescence_folder = os.path.join(opt2q_example_folder, 'fluorescence_data_calibration/')
fluorescence_data = utils.load_cell_death_data(fluorescence_folder, 'fluorescence_data.csv')

# ====================================================
# Plot of time-course IC-RP data from Sorger et al.
fig00, ax = plt.subplots(figsize=(6, 3.75))
ax.plot(fluorescence_data['# Time']*60, fluorescence_data['norm_IC-RP'], ':', label='tBID Data', color=cm.colors[1])
ax.fill_between(fluorescence_data['# Time']*60,
                fluorescence_data['norm_IC-RP'] - np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                fluorescence_data['norm_IC-RP'] + np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                color=cm.colors[1], alpha=0.3)
legend_elements = [Line2D([0], [0], color=cm.colors[1], linestyle=':', label='tBID Data', linewidth=line_width)]
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
ax.set_ylim((-0.15, 1.15))

plt.legend(handles=legend_elements)

plt.savefig('Fig2__Sorger_ICRP_Data.pdf')
plt.show()

# Plot of time-course EC-RP data from Sorger et al.
fig01, ax = plt.subplots(figsize=(6, 3.75))
ax.plot(fluorescence_data['# Time']*60, fluorescence_data['norm_EC-RP'], ':', label='cPARP Data', color=cm.colors[0])
ax.fill_between(fluorescence_data['# Time']*60,
                fluorescence_data['norm_EC-RP'] - np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                fluorescence_data['norm_EC-RP'] + np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                color=cm.colors[0], alpha=0.3)
legend_elements = [Line2D([0], [0], color=cm.colors[0], linestyle=':', label='cPARP Data', linewidth=line_width)]
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend(handles=legend_elements)

plt.savefig('Fig2_Supplemental__Sorger_ECRP_Data.pdf')
plt.show()

# ====================================================
# ====================================================
# Load and Plot Synthetic Ordinal tBID vs. time-series data (1500s)
with open(f'synthetic_WB_dataset_1500s_2020_12_3.pkl', 'rb') as data_input:
    dataset_1500 = pickle.load(data_input)

fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_1500.data['tBID_blot'].unique():  # Data
    d = dataset_1500.data[dataset_1500.data['tBID_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['tBID_blot'].values,
               s=500, color=cm.colors[level], alpha=0.5)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal tBID levels')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.savefig('Fig2__Synth_tBID_Ordinal_Data_1500.pdf')
plt.show()

# Plot representative immunoblot
fig02, ax = plt.subplots(figsize=(6, 1.5))
for level in dataset_1500.data['tBID_blot'].unique():  # Data
    d = dataset_1500.data[dataset_1500.data['tBID_blot'] == level]
    ax.scatter(x=d['time'],
               y=np.ones_like(d['time']),
               s=400+10*level, color='k', alpha=0.5+0.05*level, marker='_', linewidth=level+2)
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.set_xlabel('time [s]')
ax.axes.get_yaxis().set_visible(False)
plt.savefig('Fig2__Representative_tBID_blot_1500.pdf')
plt.show()

# Plot Synthetic Ordinal cPARP vs. time-series data (1500s)
fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_1500.data['cPARP_blot'].unique():  # Data
    d = dataset_1500.data[dataset_1500.data['cPARP_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['cPARP_blot'].values,
               s=500, color=cm2.colors[level], alpha=0.7)
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal cPARP levels')
plt.savefig('Fig2_Supplemental__Synth_cPARP_Ordinal_Data_1500.pdf')
plt.show()


# ====================================================
# ====================================================
# Load and Plot Synthetic Ordinal tBID vs. time-series data (300s)
with open(f'synthetic_WB_dataset_300s_2020_12_3.pkl', 'rb') as data_input:
    dataset_300 = pickle.load(data_input)

fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_300.data['tBID_blot'].unique():  # Data
    d = dataset_300.data[dataset_300.data['tBID_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['tBID_blot'].values,
               s=20, color=cm.colors[level], alpha=0.5)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal tBID levels')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.savefig('Fig2__Synth_tBID_Ordinal_Data_300.pdf')
plt.show()

# Plot Synthetic Ordinal cPARP vs. time-series data (180s)
fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_300.data['cPARP_blot'].unique():  # Data
    d = dataset_300.data[dataset_300.data['cPARP_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['cPARP_blot'].values,
               s=20, color=cm2.colors[level], alpha=0.7)
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal cPARP levels')
plt.savefig('Fig2_Supplemental__Synth_cPARP_Ordinal_Data_300.pdf')
plt.show()

# ====================================================
# ====================================================
# Load and Plot Synthetic Ordinal tBID vs. time-series data (180s)
with open(f'synthetic_WB_dataset_180s_2021_3_11.pkl', 'rb') as data_input:
    dataset_180 = pickle.load(data_input)

fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_180.data['tBID_blot'].unique():  # Data
    d = dataset_180.data[dataset_180.data['tBID_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['tBID_blot'].values,
               s=20, color=cm.colors[level], alpha=0.5)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal tBID levels')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.savefig('Fig2__Synth_tBID_Ordinal_Data_180.pdf')
plt.show()

# Plot Synthetic Ordinal cPARP vs. time-series data (180s)
fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_180.data['cPARP_blot'].unique():  # Data
    d = dataset_180.data[dataset_180.data['cPARP_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['cPARP_blot'].values,
               s=20, color=cm2.colors[level], alpha=0.7)
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal cPARP levels')
plt.savefig('Fig2_Supplemental__Synth_cPARP_Ordinal_Data_180.pdf')
plt.show()

# ====================================================
# ====================================================
# Load and Plot Synthetic Ordinal tBID vs. time-series data (60s)
with open(f'synthetic_WB_dataset_60s_2020_12_7.pkl', 'rb') as data_input:
    dataset_60 = pickle.load(data_input)

fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_60.data['tBID_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['tBID_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['tBID_blot'].values,
               s=10, color=cm.colors[level], alpha=0.5)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal tBID levels')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.savefig('Fig2__Synth_tBID_Ordinal_Data_60.pdf')
plt.show()

# Plot Synthetic Ordinal cPARP vs. time-series data (1500s)
fig02, ax = plt.subplots(figsize=(6, 3.75))
for level in dataset_60.data['cPARP_blot'].unique():  # Data
    d = dataset_60.data[dataset_60.data['cPARP_blot'] == level]
    ax.scatter(x=d['time'],
               y=d['cPARP_blot'].values,
               s=10, color=cm2.colors[level], alpha=0.7)
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
ax.set_xlabel('time [s]')
ax.set_ylabel('Ordinal cPARP levels')
plt.savefig('Fig2_Supplemental__Synth_cPARP_Ordinal_Data_60.pdf')
plt.show()

# ================================================== #
# ================================================== #
#                                                    #
#     Plot the Calibration to Fluorescence Data      #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'fluorescence_calibration_results'
calibration_date = '2020113'  # calibration file name contains date string
calibration_tag = 'fluorescence'

cal_args = (fluorescence_folder, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=False)
burn_in = int(0.50*len(parameter_traces[0]))

parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model)

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

    plt.savefig(f'Fig2_Supplemental__Log_Posterior_Traces_and_Hist_for_Fluorescence_Calibration_Plot{param_subset_idx}.pdf')
    ax_trace_list[-1].set_ylabel(param_subset[-1], rotation=0, labelpad=40)
    gs.tight_layout(fig, rect=[0.2, 0.0, 1, 0.93])
    plt.show()

# =======================================================================
# =======================================================================

# True Parameter Values
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))

# Random Sample from Posterior
random.seed(0)
sample_size = 1000
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)

# =======================================================================
# =======================================================================
# Comparing KDE of posterior, prior and "true" for parameters in measurement model.
# Plots KDE of the prior (blue), posterior (orange), and true (vertical dotted line)
log_model_param_true = utils.get_model_param_true()

n = 24
gs_columns = 3

for ps in range(int(np.ceil(len(param_names)/n_params_subset))):
    param_subset = param_names[n*ps:min(n*(ps+1), len(param_names))]

    priors_subset = prior_parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
    posteriors_subset = parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
    true_subset = log_model_param_true[n*ps:min(n*(ps+1), len(param_names))]

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
        plot.kde_of_parameter(ax, priors_subset[:, i], color=cm.colors[0], alpha=0.6)
        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Fig2_Supplemental__KDE_Parameter_Priors_Posteriors_Fluorescence_Plot{ps}.pdf')
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

# =======================================================================
# =======================================================================
# Plot posterior tBID dynamics predictions (95% credible region) of aEARM trained to Fluorescence IC-RP and EC-RP data.
sim_res_param_ensemble_lower_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.025)
sim_res_param_ensemble_upper_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.975)

y_lower = sim_res_param_ensemble_lower_normed['tBID_obs']
y_upper = sim_res_param_ensemble_upper_normed['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the Fluorescence 95% CI is ', area)

fig5, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs', color='k', alpha=0.2,
                                                   upper_quantile=0.975, lower_quantile=0.025, label='posterior',
                                                   linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'tBID_obs', color='k', alpha=0.4)
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'tBID_obs', color='k', alpha=0.4, label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
ax.set_ylim((-0.15, 1.15))

ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2__Posterior_Prediction_of_tBID_Fluorescence_Calibration.pdf')
plt.show()

fig6, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'cPARP_obs', alpha=0.2,
                                                   color=cm.colors[0], upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0])
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0], label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized Fluorescence Units')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2_Supplemental__Posterior_Prediction_of_cPARP_Fluorescence_Calibration.pdf')
plt.show()

# ================================================== #
# ================================================== #
#                                                    #
#     Plot the Calibration to Ordinal Data (60s)     #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '2020118'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration'

cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)
burn_in = int(0.50*len(parameter_traces[0]))

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]

print('Ordinal 60s')
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

        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    ax_trace_list[-1].set_ylabel(param_subset[-1], rotation=0, labelpad=90)
    gs.tight_layout(fig, rect=[0.2, 0.0, 1, 0.93])
    plt.savefig(f'Fig2_Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_60s_Calibration_Plot{param_subset_idx}.pdf')
    plt.show()

# =======================================================================
# =======================================================================

# True Parameter Values
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))

# Random Sample from Posterior
random.seed(0)
sample_size = 1000
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
for p in model_param_priors[1:]:  # Add measurement model priors
    prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

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

    priors_subset = prior_parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
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
        plot.kde_of_parameter(ax, priors_subset[:, i], color=cm.colors[0], alpha=0.6)
        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Fig2_Supplemental__KDE_Parameter_Priors_Posteriors_Ordinal_60s_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Ordinal 60s Data
sim = utils.set_up_simulator('fluorescence', model)

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :28], columns=param_names[:28])
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

# =======================================================================
# =======================================================================
# Plot posterior tBID dynamics predictions (95% credible region) of aEARM trained to Fluorescence IC-RP and EC-RP data.
sim_res_param_ensemble_lower_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.025)
sim_res_param_ensemble_upper_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.975)

y_lower = sim_res_param_ensemble_lower_normed['tBID_obs']
y_upper = sim_res_param_ensemble_upper_normed['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the Ordinal 60s 95% CI is ', area)

fig5, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs', alpha=0.2,
                                                   color='k', upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=0.4, color='k')
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'tBID_obs', alpha=0.4, color='k', label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized tBID Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2__Posterior_Prediction_of_tBID_Ordinal_60s_Calibration.pdf')
plt.show()

fig6, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'cPARP_obs', alpha=0.2,
                                                   color=cm.colors[0], upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0])
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0], label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized cPARP Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2_Supplemental__Posterior_Prediction_of_cPARP_Ordinal_60s_Calibration.pdf')
plt.show()

# ================================================== #
# ================================================== #
#                                                    #
#    Plot the Calibration to Ordinal Data (180s)     #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '2021319'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration_180s'

cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
# gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
parameter_traces = parameter_traces[1:]
log_p_traces = log_p_traces[1:]
gr_values = Gelman_Rubin(parameter_traces)

burn_in = int(0.50*len(parameter_traces[0]))

parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]

print('Ordinal 180s')
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

        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    ax_trace_list[-1].set_ylabel(param_subset[-1], rotation=0, labelpad=90)
    gs.tight_layout(fig, rect=[0.1, 0.0, 1, 0.93])
    plt.savefig(f'Fig2_Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_180s_Calibration_Plot{param_subset_idx}.pdf')
    plt.show()

# =======================================================================
# =======================================================================

# True Parameter Values
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))

# Random Sample from Posterior
random.seed(0)
sample_size = 1000
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
for p in model_param_priors[1:]:  # Add measurement model priors
    prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

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

    priors_subset = prior_parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
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
        plot.kde_of_parameter(ax, priors_subset[:, i], color=cm.colors[0], alpha=0.6)
        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Fig2_Supplemental__KDE_Parameter_Priors_Posteriors_Ordinal_180s_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Ordinal 180s Data
sim = utils.set_up_simulator('immunoblot', model)

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :28], columns=param_names[:28])
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

# =======================================================================
# =======================================================================
# Plot posterior tBID dynamics predictions (95% credible region) of aEARM trained to Ordinal 180s time-point data.
sim_res_param_ensemble_lower_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.025)
sim_res_param_ensemble_upper_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.975)

y_lower = sim_res_param_ensemble_lower_normed['tBID_obs']
y_upper = sim_res_param_ensemble_upper_normed['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the Ordinal 180s 95% CI is ', area)

fig5, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs', alpha=0.2,
                                                   color='k', upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=0.4, color='k')
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'tBID_obs', alpha=0.4, color='k', label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized tBID Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2__Posterior_Prediction_of_tBID_Ordinal_180s_Calibration.pdf')
plt.show()

fig6, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'cPARP_obs', alpha=0.2,
                                                   color=cm.colors[0], upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0])
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0], label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized cPARP Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2_Supplemental__Posterior_Prediction_of_cPARP_Ordinal_180s_Calibration.pdf')
plt.show()

# ================================================== #
# ================================================== #
#                                                    #
#    Plot the Calibration to Ordinal Data (300s)     #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '202133'  # calibration file name contains date string
# calibration_date = '2021225'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration_300s'

cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.50*len(parameter_traces[0]))

parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]

print('Ordinal 300s')
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

        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    ax_trace_list[-1].set_ylabel(param_subset[-1], rotation=0, labelpad=90)
    gs.tight_layout(fig, rect=[0.1, 0.0, 1, 0.93])
    plt.savefig(f'Fig2_Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_300s_Calibration_Plot{param_subset_idx}.pdf')
    plt.show()

# =======================================================================
# =======================================================================

# True Parameter Values
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))

# Random Sample from Posterior
random.seed(0)
sample_size = 1000
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
for p in model_param_priors[1:]:  # Add measurement model priors
    prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

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

    priors_subset = prior_parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
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
        plot.kde_of_parameter(ax, priors_subset[:, i], color=cm.colors[0], alpha=0.6)
        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Fig2_Supplemental__KDE_Parameter_Priors_Posteriors_Ordinal_300s_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Ordinal 300s Data
sim = utils.set_up_simulator('immunoblot', model)
sim.run()

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :28], columns=param_names[:28])
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

# =======================================================================
# =======================================================================
# Plot posterior tBID dynamics predictions (95% credible region) of aEARM trained to Ordinal 300s time-point data.
sim_res_param_ensemble_lower_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.025)
sim_res_param_ensemble_upper_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.975)

y_lower = sim_res_param_ensemble_lower_normed['tBID_obs']
y_upper = sim_res_param_ensemble_upper_normed['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the 300s 95% CI is ', area)

fig5, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs', alpha=0.2,
                                                   color='k', upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=0.4, color='k')
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'tBID_obs', alpha=0.4, color='k', label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized tBID Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2__Posterior_Prediction_of_tBID_Ordinal_300s_Calibration.pdf')
plt.show()

fig6, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'cPARP_obs', alpha=0.2,
                                                   color=cm.colors[0], upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0])
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0], label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized cPARP Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2_Supplemental__Posterior_Prediction_of_cPARP_Ordinal_300s_Calibration.pdf')
plt.show()

# ================================================== #
# ================================================== #
#                                                    #
#    Plot the Calibration to Ordinal Data (1500s)    #
#                                                    #
# ================================================== #
# ================================================== #

# ====================================================
# ======== File Details ==============================

calibration_folder = 'immunoblot_calibration_results'
calibration_date = '20201212'  # calibration file name contains date string
calibration_tag = 'apoptosis_params_and_immunoblot_classifier_calibration_1500s'

cal_args = (parent_dir, calibration_folder, calibration_date, calibration_tag)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.50*len(parameter_traces[0]))
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)

param_names = utils.get_model_param_names(model) + utils.get_measurement_param_names('immunoblot')[:-1]

print('Ordinal 1500s')
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

        ax_hist_list[i].axes.get_yaxis().set_visible(False)
        ax_hist_list[i].tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    gr_ax = fig.add_subplot(gs[:, 4])
    plot.gelman_rubin_values(
        gr_ax, param_subset, gr_values[12*param_subset_idx:min(12*(param_subset_idx+1), len(param_names))],
        labels=False)
    gr_ax.set_yticks([])
    gr_ax.tick_params(axis='x', which='major', labelsize=local_tick_labels_x)

    ax_trace_list[-1].set_ylabel(param_subset[-1], rotation=0, labelpad=90)
    gs.tight_layout(fig, rect=[0.1, 0.0, 1, 0.93])
    plt.savefig(f'Fig2_Supplemental__Log_Posterior_Traces_and_Hist_for_Ordinal_1500s_Calibration_Plot{param_subset_idx}.pdf')
    plt.show()

# =======================================================================
# =======================================================================

# True Parameter Values
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))

# Random Sample from Posterior
random.seed(0)
sample_size = 1000
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# Random Sample from Prior
prior_parameter_sample = utils.sample_model_param_priors(model_param_priors, sample_size)
for p in model_param_priors[1:]:  # Add measurement model priors
    prior_parameter_sample = np.column_stack([prior_parameter_sample, p.dist.rvs(sample_size).T])

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

    priors_subset = prior_parameter_sample[:, n*ps:min(n*(ps+1), len(param_names))]
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
        plot.kde_of_parameter(ax, priors_subset[:, i], color=cm.colors[0], alpha=0.6)
        # Posterior
        plot.kde_of_parameter(ax, posteriors_subset[:, i], color=cm.colors[1], alpha=0.6)
        # True
        ax.axvline(true_subset[i], color='k', alpha=0.4, linestyle='--')

    gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
    plt.savefig(f'Fig2_Supplemental__KDE_Parameter_Priors_Posteriors_Ordinal_1500s_Plot{ps}.pdf')
    plt.show()

# =======================================================================
# =======================================================================
# Simulate Ordinal 1500s Data
sim = utils.set_up_simulator('immunoblot', model)

# Simulate Random Ensemble of Parameters (wo extrinsic noise)
ensemble_parameters = pd.DataFrame(10**parameter_sample[:, :28], columns=param_names[:28])
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

# =======================================================================
# =======================================================================
# Plot posterior tBID dynamics predictions (95% credible region) of aEARM trained to Ordinal 1500s time-point data.
sim_res_param_ensemble_lower_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.025)
sim_res_param_ensemble_upper_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.975)

y_lower = sim_res_param_ensemble_lower_normed['tBID_obs']
y_upper = sim_res_param_ensemble_upper_normed['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the 95% Ordinal 1500s CI is ', area)

fig5, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'tBID_obs', alpha=0.2,
                                                   color='k', upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'tBID_obs', alpha=0.4, color='k')
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'tBID_obs', alpha=0.4, color='k', label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized tBID Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2__Posterior_Prediction_of_tBID_Ordinal_1500s_Calibration.pdf')
plt.show()

fig6, ax = plt.subplots(figsize=(6, 3.75))
sim_res_param_ensemble_median_normed = calc.simulation_results_quantile(sim_res_param_ensemble_normed, 0.5)
plot.plot_simulation_results_quantile_fill_between(ax, sim_res_param_ensemble_normed, 'cPARP_obs', alpha=0.2,
                                                   color=cm.colors[0], upper_quantile=0.975, lower_quantile=0.025,
                                                   label='posterior', linewidth=line_width)
plot.plot_simulation_results(ax, sim_res_param_ensemble_median_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0])
plot.plot_simulation_results(ax, sim_res_param_true_normed, 'cPARP_obs', alpha=1.0, color=cm.colors[0], label='true',
                             linewidth=line_width, linestyle=':')
ax.set_xlabel('time [s]')
ax.set_ylabel('Normalized cPARP Concentration')
ax.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.legend()
plt.savefig('Fig2_Supplemental__Posterior_Prediction_of_cPARP_Ordinal_1500s_Calibration.pdf')
plt.show()
