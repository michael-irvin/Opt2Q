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
import random


# Calibration Methods
# ===================
# The mechanistic model, aEARM, was pre-parameterized with "ground-truth" parameters.
# A synthetic dataset containing one ordinal measurement per every 60s was generated using "ground-truth" aEARM
# predictions of tBID and cPARP vs time, and a pre-parameterized measurement model.
#
# We trained only the measurement models. They maps the "ground-truth" aEARM predictions of tBID and cPARP vs. time
# (targets) to synthetic ordinal values of tBID and cPARP vs. time.
#
# The values of tBID and cPARP are all independent (i.e. the log-likelihood of the dataset is the sum of the individual
# log-likelihoods.
#
# The priors are uniform distributions:
from pydream.parameters import SampledParam
from scipy.stats import uniform
model_param_priors = [SampledParam(uniform, loc=0.0, scale=100.0),         # coefficients__tBID_blot__coef_    float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_1  float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_2  float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_3  float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_4  float
                      SampledParam(uniform, loc=0.0, scale=100.0),         # coefficients__cPARP_blot__coef_   float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__cPARP_blot__theta_1 float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__cPARP_blot__theta_2 float
                      SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__cPARP_blot__theta_3 float
                      ]
#
# The calibration files were saved as:
# 'immunoblot_classifier_calibration_202129 ...'

# =====================================
# ======== File Details ===========
# Update this part with the new log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
calibration_folder = 'immunoblot_calibration_results'
calibration_name = 'immunoblot_classifier_calibration_202129'
calibration_date = '202129'

plot_title = 'Immunoblot Calibration'

# ====================================================
# ====================================================
# Load data
with open(os.path.join(parent_dir, 'synthetic_WB_dataset_60s_2020_12_7.pkl'), 'rb') as data_input:
    dataset = pickle.load(data_input)

cal_args = (parent_dir, calibration_folder, calibration_date, calibration_name)

# Chain Statistics
gr_values = utils.load_gelman_rubin_values(*cal_args)
chain_history = utils.load_chain_history(*cal_args)
parameter_traces, log_p_traces = utils.load_parameter_and_log_p_traces(*cal_args, include_extra_reactions=True)
burn_in = int(0.80*len(parameter_traces[0]))
parameter_traces_burn_in, log_p_traces_burn_in = utils.thin_traces(parameter_traces, log_p_traces, 1, burn_in)
param_names = utils.get_measurement_param_names('immunoblot')[:-1]


# ==================================================================
# Plot settings
cm = plt.get_cmap('tab10')

tick_labels_x = 16
tick_labels_y = 17
line_width = 2

# =======================================================================
# ============== Plot parameter traces and histograms ==============
# Trace of value vs. iteration for parameters in the measurement model for each chain in the PyDREAM algorithm.
# The second column is the histogram of these values for each chain.
# Third column is gelman-rubin metric for each parameter.

fig = plt.figure(1, figsize=(9, 8.25))
gs = gridspec.GridSpec(len(param_names), 5, hspace=0.5)
ax_trace_list = []
ax_hist_list = []

for i, param in enumerate(param_names):
    ax_trace_list.append(fig.add_subplot(gs[i, :3]))
    ax_hist_list.append(fig.add_subplot(gs[i, 3]))
    plot.parameter_traces(ax_trace_list[i], parameter_traces_burn_in,
                          burnin=burn_in, param_idx=i, labels=False)
    plot.parameter_traces_histogram(ax_hist_list[i], parameter_traces_burn_in,
                                    param_idx=i, labels=False)
    ax_trace_list[i].set_yticks([])
    ax_trace_list[i].set_ylabel(param_names[i], rotation=0, labelpad=90)
    ax_hist_list[i].axes.get_yaxis().set_visible(False)

gr_ax = fig.add_subplot(gs[:, 4])
plot.gelman_rubin_values(gr_ax, param_names, gr_values[:len(param_names)], labels=False)
gr_ax.set_yticks([])

# plt.savefig('Supplemental__Log_Posterior_Traces_and_Hist_for_Fig2.pdf')
ax_trace_list[-1].set_ylabel(param_names[-1], rotation=0, labelpad=90)
gs.tight_layout(fig, rect=[0.06, 0.0, 1, 0.93])
plt.show()
# =======================================================================
# =======================================================================

# True Parameter Values
measurement_model_param_true = utils.get_measurement_model_true_params('immunoblot')
model_param_true = pd.DataFrame([10**utils.get_model_param_true()], columns=utils.get_model_param_names(model))


# Random Sample from Posterior
random.seed(100)
sample_size = 400
parameter_sample, log_p_sample = utils.get_parameter_sample(
    parameter_traces_burn_in, log_p_traces_burn_in, sample_size=sample_size)

# =======================================================================
# =======================================================================
# Comparing KDE of posterior, prior and "true" for parameters in measurement model.
# Plots KDE of the prior (blue), posterior (orange), and true (vertical dotted line)

gs_columns = 3
gs_rows = int(np.ceil(len(param_names)/gs_columns))

fig = plt.figure(1, figsize=(9, 11*gs_rows/8.0))
gs = gridspec.GridSpec(gs_rows, gs_columns, hspace=0.1)

for i, param in enumerate(param_names):
    r = int(i / gs_columns)
    c = i % gs_columns
    ax = fig.add_subplot(gs[r, c])
    ax.set_yticks([])
    ax.set_title(param)
    # Prior
    p = model_param_priors[i].dist
    px = np.linspace(*p.interval(1), 100)
    py = p.pdf(px)
    ax.plot(px, py, color=cm.colors[0], alpha=0.6)
    # Posterior
    plot.kde_of_parameter(ax, parameter_sample[:, i], color=cm.colors[1], alpha=0.6)
    # True
    ax.axvline(measurement_model_param_true[i], color='k', alpha=0.4, linestyle='--')

gs.tight_layout(fig, rect=[0.0, 0.0, 1, 1.0])
plt.savefig('Supplemental__KDE_Parameter_Priors_Posteriors.pdf')
plt.show()

# ==================================================================
# ==================================================================
# Figure 2: Data-driven probabilistic measurement model.
# Left plot: Grey curve is the simulated tBID trajectory from aEARM pre-parameterized with ground-truth parameters
# Left plot: Dots are ordinal values of the intensity of the measurement model. They a color-coded to match the ordinal
# categories modeled in the measurement model (Right Plot)
# Right plot: The x-axis is the probability of class membership. The y axis is the normalized tBID value.
# A second plot of the cPARP measurement model is also generated below.

# Simulate Ground Truth Dynamics
sim = utils.set_up_simulator('immunoblot', model)
sim.param_values = model_param_true
sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

x_scaled = ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs']).transform(results[['time', 'tBID_obs', 'cPARP_obs']])

# set up classifier
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], dataset.data['time']).transform(x_scaled)
lc = LogisticClassifier(dataset, column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True, classifier_type='ordinal_eoc')
lc.set_up(x_int)
lc.do_fit_transform = False

# Posterior Prediction of the Measurement Model
plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})

lc_results_list = []
p_dom = pd.DataFrame(np.linspace(0, 1, 100), columns=['plot_domain'])
for row in parameter_sample:
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

ax1.plot(x_scaled['time'].values, x_scaled['tBID_obs'],  # Dynamics
         label=f'Simulated Normed tBID', color='k', alpha=0.4, linewidth=line_width)
ax1.legend()
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)

for level in dataset.data['tBID_blot'].unique():  # Data
    d = dataset.data[dataset.data['tBID_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['tBID_blot'].values/(dataset.data['tBID_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

tBID_results = upper_lc.filter(regex='tBID_blot')
for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.fill_betweenx(plot_domain['tBID_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['tBID_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Fig2__Calibration_of_Measurement_Model_Only_tBID_Plot.pdf')
plt.show()

# Plot cPARP Dynamics =====================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

ax1.plot(x_scaled['time'].values, x_scaled['cPARP_obs'],  # Dynamics
         label=f'Simulated Normed cPARP', color='k', alpha=0.4, linewidth=line_width)
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)
# ax1.legend()

for level in dataset.data['cPARP_blot'].unique():  # Data
    d = dataset.data[dataset.data['cPARP_blot'] == level]

    ax1.scatter(x=d['time'],
                y=d['cPARP_blot'].values/(dataset.data['cPARP_blot'].max()),
                s=10, color=cm.colors[level], alpha=0.5)

cPARP_results = upper_lc.filter(regex='cPARP_blot')
for n, col in enumerate(sorted(list(cPARP_results.columns))):
    ax2.fill_betweenx(plot_domain['cPARP_obs'], upper_lc[col], lower_lc[col], alpha=0.4, color=cm.colors[n])
    ax2.plot(median_lc[col], plot_domain['cPARP_obs'], alpha=0.7, color=cm.colors[n])

ax2.tick_params(axis='x', which='major', labelsize=tick_labels_x)
plt.savefig('Supplemental__Calibration_of_Measurement_Model_Only_cPARP_Plot.pdf')
plt.show()

