# MW Irvin -- Lopez Lab -- 2019-09-16

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations
from matplotlib import pyplot as plt
from opt2q_examples.apoptosis_model import model
from opt2q_examples.fluorescence_data_calibration.fluorescence_data_calibration import dataset, sim, fl, burn_in_len, \
    n_chains, n_iterations, sampled_params_0

# ======== UPDATE THIS PART ===========
# Update this part with the new log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'fluorescence_calibration_results'
plot_title = 'Fluorescence Calibration'

burn_in_len = burn_in_len
number_of_traces = n_chains

chain_history_file = [os.path.join(script_dir, calibration_folder, f) for f in
                      os.listdir(os.path.join(script_dir, calibration_folder))
                      if '2020113' in f and 'chain_history' in f][0]

log_p_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                            os.listdir(os.path.join(script_dir, calibration_folder))
                            if '2020113' in f and 'log_p' in f])

parameter_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                                os.listdir(os.path.join(script_dir, calibration_folder))
                                if '2020113' in f and 'parameters' in f])

# reorder traces to be in numerical order (i.e. 1000 before 10000).
n_iterations = 100000
burn_in_len = 100000

file_order = [str(n_iterations*n) for n in range(1, int(len(log_p_file_paths_)/number_of_traces)+1)]
log_p_file_paths = []
parameter_file_paths = []
for file_num in file_order:
    log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
    parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]

gr_file = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                  os.listdir(os.path.join(script_dir, calibration_folder))
                  if '2020113' in f and file_order[-1] in f and '.txt' in f])[0]

gr = np.loadtxt(gr_file)

param_names = [p.name for p in model.parameters_rules()]
true_params = np.load('true_params.npy')[:len(param_names)]

# params_of_interest = [2, 5, 13, 23]
# params_of_interest = list(range(num_params))
params_of_interest = list(range(len(param_names)))

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

num_params = len(param_names)

# ------- Plots -------
thin = 100

if __name__ == '__main__':
    # plot calibration metrics
    for trace in log_p_traces:
        plt.plot(trace)

    plt.title('Fluorescence Calibration Posterior Trace')
    plt.ylabel('log(Posterior)')
    plt.xlabel('iteration')
    plt.show()

    for j in params_of_interest:
        s = pd.DataFrame({f'trace {n}': parameter_samples[n][burn_in_len:, j][::thin]
                          for n in range(len(parameter_samples))})
        ax = s.plot.hist(alpha=0.5, bins=20)
        s.plot(kind='kde', ax=ax, secondary_y=True)

        # plt.hist(param_array[burn_in_len:, j], normed=True)
        plt.title(param_names[j])
        plt.show()

    for j, k in combinations(params_of_interest, 2):
        param_cor = []
        for param_array in parameter_samples:
            param_cor.append(abs(np.corrcoef(param_array[burn_in_len:, j][::thin],
                                             param_array[burn_in_len:, k][::thin])[0][1]))
        if any(x > 0.7 for x in param_cor):
            for param_array in parameter_samples:
                plt.scatter(x=param_array[burn_in_len:, j][::thin],
                            y=param_array[burn_in_len:, k][::thin], alpha=0.5)
            plt.title('Parameters of Each Trace')
            plt.xlabel(param_names[j])
            plt.ylabel(param_names[k])
            plt.show()

    # for j, k in combinations(params_of_interest, 2):
    #     for param_array in parameter_samples:
    #         plt.scatter(x=param_array[burn_in_len:, j], y=param_array[burn_in_len:, k], alpha=0.5)
    #     plt.title('Parameters of Each Trace')
    #     plt.xlabel(param_names[j])
    #     plt.ylabel(param_names[k])
    #     plt.show()

# plot parameter values compared to prior and true
params = np.concatenate([p[burn_in_len:, :] for p in parameter_samples])

if __name__ == '__main__':
    # plot parameter distributions
    cm = plt.get_cmap('tab10')
    prior_params = sampled_params_0[0].dist.kwds

    for idx in params_of_interest:
        s = pd.DataFrame({f'{param_names[idx]}': params[::thin, idx]})
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

idx = np.argmax(log_ps)
# idx = ids[-50][0]  # second best

sim_parameters = pd.DataFrame([[10**p for p in params[idx, :num_params]]], columns=param_names)

parameters = pd.DataFrame([[p.value for p in model.parameters_rules()]],
                          columns=[p.name for p in model.parameters_rules()])

# ------- Dynamics -------
sim.param_values = parameters
sim_results_0 = sim.run()

# ------- Measurement -------
fl.update_simulation_result(sim_results_0)
results_0 = fl.run()

sim.param_values = sim_parameters
sim_results = sim.run()

fl.update_simulation_result(sim_results)
results = fl.run()

# ------- Plots -------
if __name__ == '__main__':
    cm = plt.get_cmap('tab10')
    plt.plot(results['time'], results_0['cPARP_obs'], '--', label='cPARP_obs initial', color=cm.colors[0])
    plt.plot(results['time'], results['cPARP_obs'], label='cPARP_obs best log-p', color=cm.colors[0])
    plt.plot(dataset.data['time'], dataset.data['norm_EC-RP'], ':', label='norm EC-RP Data', color=cm.colors[0])
    plt.fill_between(dataset.data['time'],
                     dataset.data['norm_EC-RP'] - np.sqrt(dataset.measurement_error_df['norm_EC-RP__error']),
                     dataset.data['norm_EC-RP'] + np.sqrt(dataset.measurement_error_df['norm_EC-RP__error']),
                     color=cm.colors[0], alpha=0.2)

    plt.legend()
    plt.show()

    cm = plt.get_cmap('tab10')
    plt.plot(results['time'], results_0['tBID_obs'], '--', label='tBID_obs initial', color=cm.colors[1])
    plt.plot(results['time'], results['tBID_obs'], label='tBID_obs best log-p', color=cm.colors[1])
    plt.plot(dataset.data['time'], dataset.data['norm_IC-RP'], ':', label='norm IC-RP Data', color=cm.colors[1])
    plt.fill_between(dataset.data['time'],
                     dataset.data['norm_IC-RP'] - np.sqrt(dataset.measurement_error_df['norm_IC-RP__error']),
                     dataset.data['norm_IC-RP'] + np.sqrt(dataset.measurement_error_df['norm_IC-RP__error']),
                     color=cm.colors[1], alpha=0.2)

    plt.legend()
    plt.show()

    # plt.plot(sim_results_0.opt2q_dataframe.index, sim_results_0.opt2q_dataframe['MOMP_signal'], '--',
    #          color=cm.colors[0], label='MOMP signal initial')
    plt.plot(sim_results.opt2q_dataframe.index, sim_results.opt2q_dataframe['MOMP_signal'],
             color=cm.colors[0], label='MOMP signal best lop-p')
    plt.legend()
    plt.show()


# idx = np.argmax(log_ps)

print(params[idx, :num_params])

# np.save("true_params.npy", params[idx, :num_params])

