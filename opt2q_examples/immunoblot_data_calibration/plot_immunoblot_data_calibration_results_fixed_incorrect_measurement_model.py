# MW Irvin -- Lopez Lab -- 2019-11-21

# Plotting results of apoptosis model calibration to immunoblot data using fixed incorrect measurement model

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations
from matplotlib import pyplot as plt
from opt2q_examples.apoptosis_model import model
from opt2q_examples.fluorescence_data_calibration.fluorescence_data_calibration import dataset
from opt2q_examples.immunoblot_data_calibration.immunoblot_data_calibration_fixed_incorrect_measurement_model import \
    synthetic_immunoblot_data, sim, wb, burn_in_len, n_chains, \
    n_iterations, sampled_params_0
from opt2q_examples.immunoblot_data_calibration.immunoblot_data_calibration_fixed_measurement_model import wb \
    as wb_true


# ======== UPDATE THIS PART ===========
# Update this part with the new log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
calibration_folder = 'immunoblot_calibration_results'
plot_title = 'Immunoblot Calibration (fixed erroneous measurement model)'

burn_in_len = 1000000
number_of_traces = n_chains

chain_history_file = [os.path.join(script_dir, calibration_folder, f) for f in
                      os.listdir(os.path.join(script_dir, calibration_folder))
                      if '2020131' in f and 'inc' in f and 'chain_history' in f][0]

log_p_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                            os.listdir(os.path.join(script_dir, calibration_folder))
                            if '2020131' in f and 'inc' in f and 'log_p' in f])

parameter_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                                os.listdir(os.path.join(script_dir, calibration_folder))
                                if '2020131' in f and 'inc' in f and 'parameters' in f])

# reorder traces to be in numerical order (i.e. 1000 before 10000).
file_order = [str(n_iterations*n) for n in range(1, int(len(log_p_file_paths_)/number_of_traces)+1)]
log_p_file_paths = []
parameter_file_paths = []
for file_num in file_order:
    log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
    parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]


gr_file = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                  os.listdir(os.path.join(script_dir, calibration_folder))
                  if '2020131' in f and file_order[-1] in f and '.txt' in f and 'inc' in f])[0]

gr = np.loadtxt(gr_file)

param_names = [p.name for p in model.parameters_rules()][:-6]

true_params = np.load('true_params.npy')[:len(param_names)]

param_names = [p.name for p in model.parameters_rules()][:-6]

num_params = len(param_names)

params_of_interest = range(num_params)
thin = 1000

# =====================================
# convergence
# plt.figure(figsize=(10, 3))
plt.bar(x=param_names[:17], height=gr[:17])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.title('Gelman Ruben convergence metric for model parameters')
plt.show()

plt.bar(x=param_names[17:], height=gr[17:])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.title('Gelman Ruben convergence metric for model parameters')
plt.show()

# chain history
chain_history = np.load(chain_history_file)

# get parameters
log_p_traces = []
parameter_samples = []
for trace_num in range(number_of_traces):
    log_p_trace = np.concatenate(np.roll([np.load(os.path.join(script_dir, calibration_folder, lp))
                                          for lp in log_p_file_paths if f'_{trace_num}_' in lp], 0))
    log_p_traces.append(log_p_trace)

    parameter_sample = np.concatenate(np.roll([np.load(os.path.join(script_dir, calibration_folder, pp))
                                               for pp in parameter_file_paths if f'_{trace_num}_' in pp], 0))
    parameter_samples.append(parameter_sample)


# ------- Plots -------
if __name__ == '__main__':
    # plot calibration metrics
    for trace in log_p_traces:
        plt.plot(trace)

    plt.title(f'{plot_title} Posterior Trace')
    plt.ylabel('log(Posterior)')
    plt.xlabel('iteration')
    plt.show()

    for j in params_of_interest:
        s = pd.DataFrame({f'trace {n}': parameter_samples[n][burn_in_len:, j][::thin]
                          for n in range(len(parameter_samples))})
        ax = s.plot.hist(alpha=0.5, bins=20)
        s.plot(kind='kde', ax=ax, secondary_y=True)

        plt.title(param_names[j])
        plt.show()

    for j, k in combinations(params_of_interest, 2):
        param_cor = []
        for param_array in parameter_samples:
            param_cor.append(abs(np.corrcoef(param_array[burn_in_len:, j], param_array[burn_in_len:, k])[0][1]))
        if any(x > 0.9 for x in param_cor):
            for param_array in parameter_samples:
                plt.scatter(x=param_array[burn_in_len:, j][::thin], y=param_array[burn_in_len:, k][::thin], alpha=0.5)
            plt.title('Parameters of Each Trace')
            plt.xlabel(param_names[j])
            plt.ylabel(param_names[k])
            plt.show()

# plot parameter values compared to prior and true
params = np.concatenate([p[burn_in_len:, :] for p in parameter_samples])

if __name__ == '__main__':
    # plot parameter distributions
    cm = plt.get_cmap('tab10')
    prior_params = sampled_params_0[0].dist.kwds

    for idx in params_of_interest:
        s = pd.DataFrame({f'{param_names[idx]}': params[:, idx][::thin]})
        ax1 = s.plot.hist(alpha=0.4, bins=20, color='k')
        s.plot(kind='kde', ax=ax1, color=cm.colors[1], secondary_y=True)

        x_range = np.linspace(1.1*s.min(), 1.1*s.max(), 1000)
        prior = norm(loc=prior_params['loc'][idx], scale=prior_params['scale'])

        ax2 = ax1.twinx()
        ax2.spines["right"].set_position(("axes", 1.2))
        ax2.plot(x_range, prior.pdf(x_range), '--', color='k', label=f'{param_names[0]}, prior')
        ax2.axvline(x=true_params[idx], color='k', alpha=0.5, label='true mean')
        ax2.set_ylabel('Density (prior)')

        handles = ax1.get_legend().legendHandles
        preset_label = handles[0].get_label()
        handles[0].set_label(f'{preset_label} hist (left)')
        handles[1].set_label(f'{preset_label} kde (right)')
        handles.append(ax2.get_legend_handles_labels()[0][0])
        handles.append(ax2.get_legend_handles_labels()[0][1])
        ax1.get_legend().remove()

        plt.legend(handles=handles, bbox_to_anchor=(0.5, -0.6, 0.5, 0.5))
        plt.title(f'Calibrated {param_names[idx]} Value')
        plt.show()

# plot best performing parameters
log_ps = np.concatenate([lpt for lpt in log_p_traces])
params = np.concatenate([p for p in parameter_samples])

# idx = np.argmax(log_ps)
ids = np.argsort(log_ps[:, 0])[-5000::thin]

sim_parameters = pd.DataFrame([[10**p for p in params[idx, :num_params]] for idx in ids], columns=param_names)

parameters = pd.DataFrame([[p.value for p in model.parameters_rules()]],
                          columns=[p.name for p in model.parameters_rules()])

# ------- Dynamics -------
sim.param_values = parameters
sim_results_0 = sim.run()
results_0 = sim_results_0.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

sim.param_values = sim_parameters
sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

sim.param_values = pd.DataFrame([[10**p for p in true_params]], columns=param_names)
sim_results_true = sim.run()
results_true = sim_results_true.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

IC_RP__n_cats = synthetic_immunoblot_data.data['tBID_blot'].max()
EC_RP__n_cats = synthetic_immunoblot_data.data['cPARP_blot'].max()

# set up classifier
plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})
lc_results = wb.process.get_step('classifier').transform(plot_domain)
cPARP_results = lc_results.filter(regex='cPARP_blot')
tBID_results = lc_results.filter(regex='tBID_blot')

lc_results_true = wb_true.process.get_step('classifier').transform(plot_domain)
cPARP_results_true = lc_results_true.filter(regex='cPARP_blot')
tBID_results_true = lc_results_true.filter(regex='tBID_blot')

# ------- Plots -------
if __name__ == '__main__':
    cm = plt.get_cmap('tab10')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1, 1]})

    ax1.plot(results_0['time'].values, results_0['tBID_obs'].values / max(results_0['tBID_obs'].values), '--',
             label=f'tBID simulations initial fit', color=cm.colors[1])

    ax1.fill_between(dataset.data['time'],
                     dataset.data['norm_IC-RP'] - np.sqrt(dataset.measurement_error_df['norm_IC-RP__error']),
                     dataset.data['norm_IC-RP'] + np.sqrt(dataset.measurement_error_df['norm_IC-RP__error']),
                     color=cm.colors[1], alpha=0.2, label='IC-RP Data')

    for name, df in results.groupby('simulation'):
        ax1.plot(df['time'].values, df['tBID_obs'].values / max(df['tBID_obs'].values),
                 label=f'tBID simulations best fit', color=cm.colors[1], alpha=0.20)

    ax1.scatter(x=synthetic_immunoblot_data.data['time'],
                y=synthetic_immunoblot_data.data['tBID_blot'].values / IC_RP__n_cats,
                s=10, color=cm.colors[1], label=f'tBID blot data', alpha=0.5)

    ax1.plot(results_true['time'].values, results_true['tBID_obs'].values / max(results_true['tBID_obs'].values),
             label=f'tBID simulations true params', color=cm.colors[7])

    ax1.legend()
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[0:2]+handles[-3:], labels=labels[0:2]+labels[-3:])
    ax1.set_title('Predicted BID truncation dynamics')

    c_id = 0
    for col in sorted(list(tBID_results.columns)):
        ax2.plot(tBID_results[col].values, np.linspace(0, 1, 100), color=cm.colors[c_id], label=col)
        c_id += 1
    ax2.legend(loc='lower center')
    ax2.set_title('Assumed measurement model')

    c_id = 0
    for col in sorted(list(tBID_results.columns)):
        ax3.plot(tBID_results_true[col].values, np.linspace(0, 1, 100), color=cm.colors[c_id],
                 label=col)
        c_id += 1

    ax3.set_title('True measurement model')
    ax3.legend(loc='lower center')
    plt.show()

# ------- Plots -------
if __name__ == '__main__':
    cm = plt.get_cmap('tab10')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1, 1]})

    ax1.plot(results_0['time'].values, results_0['cPARP_obs'].values / max(results_0['cPARP_obs'].values), '--',
             label=f'cPARP simulations initial fit', color=cm.colors[0])

    ax1.fill_between(dataset.data['time'],
                     dataset.data['norm_EC-RP'] - np.sqrt(dataset.measurement_error_df['norm_EC-RP__error']),
                     dataset.data['norm_EC-RP'] + np.sqrt(dataset.measurement_error_df['norm_EC-RP__error']),
                     color=cm.colors[0], alpha=0.2, label='EC-RP Data')

    for name, df in results.groupby('simulation'):
        ax1.plot(df['time'].values, df['cPARP_obs'].values / max(df['cPARP_obs'].values),
                 label=f'cPARP simulations best fit', color=cm.colors[0], alpha=0.20)

    ax1.scatter(x=synthetic_immunoblot_data.data['time'],
                y=synthetic_immunoblot_data.data['cPARP_blot'].values / EC_RP__n_cats,
                s=10, color=cm.colors[0], label=f'cPARP blot data', alpha=0.5)

    ax1.plot(results_true['time'].values, results_true['cPARP_obs'].values / max(results_true['cPARP_obs'].values),
             label=f'cPARP simulations true params', color=cm.colors[7])

    ax1.legend()
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[0:2]+handles[-3:], labels=labels[0:2]+labels[-3:])

    c_id = 0
    for col in sorted(list(cPARP_results.columns)):
        ax2.plot(cPARP_results[col].values, np.linspace(0, 1, 100), color=cm.colors[c_id], label=col)
        c_id += 1
    ax2.legend(loc='lower center')
    ax2.set_title('Assumed measurement model')

    c_id = 0
    for col in sorted(list(cPARP_results.columns)):
        ax3.plot(cPARP_results_true[col].values, np.linspace(0, 1, 100), color=cm.colors[c_id], label=col)
        c_id += 1

    ax3.set_title('True measurement model')
    ax3.legend(loc='lower center')

    # c_id = 0
    # for col in sorted(list(cPARP_results.columns)):
    #     ax2.plot(cPARP_results[col].values, np.linspace(0, 1, 100), color=cm.colors[c_id], label=col)
    #     ax2.plot(cPARP_results_true[col].values, np.linspace(0, 1, 100), color=cm.colors[c_id], label=f'{col} true')
    #     c_id += 1
    #
    # ax2.legend()

    plt.show()

