# MW Irvin -- Lopez Lab -- 2019-09-16

import os
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import pyplot as plt
from opt2q_examples.apoptosis_model import model
from opt2q_examples.fluorescence_data_calibration.fluorescence_data_calibration import dataset, sim, fl, burn_in_len, \
    n_chains, n_iterations

# ======== UPDATE THIS PART ===========
# Update this part with the new log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'fluorescence_calibration_results'
plot_title = 'Fluorescence Calibration'

burn_in_len = burn_in_len
number_of_traces = n_chains

chain_history_file = [os.path.join(script_dir, calibration_folder, f) for f in
                      os.listdir(os.path.join(script_dir, calibration_folder))
                      if '2019114' in f and 'chain_history' in f][0]

log_p_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                            os.listdir(os.path.join(script_dir, calibration_folder))
                            if '2019114' in f and 'log_p' in f])

parameter_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                                os.listdir(os.path.join(script_dir, calibration_folder))
                                if '2019114' in f and 'parameters' in f])

# reorder traces to be in numerical order (i.e. 1000 before 10000).
n_iterations = 10000
burn_in_len = 9000

file_order = [str(n_iterations*n) for n in range(1, int(len(log_p_file_paths_)/number_of_traces)+1)]
log_p_file_paths = []
parameter_file_paths = []
for file_num in file_order:
    log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
    parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]

param_names = [p.name for p in model.parameters_rules()]

# =====================================
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
if __name__ == '__main__':
    # plot calibration metrics
    for trace in log_p_traces:
        plt.plot(trace)

    plt.title('Fluorescence Calibration Posterior Trace')
    plt.ylabel('log(Posterior)')
    plt.xlabel('iteration')
    plt.show()

    params_of_interest = [2, 5, 13, 23]
    # params_of_interest = list(range(num_params))
    # params_of_interest = list(range(0, 10))

    # for j in params_of_interest:
    #     s = pd.DataFrame({f'trace {n}': parameter_samples[n][burn_in_len:, j] for n in range(len(parameter_samples))})
    #     ax = s.plot.hist(alpha=0.5, bins=20)
    #     s.plot(kind='kde', ax=ax, secondary_y=True)
    #
    #     # plt.hist(param_array[burn_in_len:, j], normed=True)
    #     plt.title(param_names[j])
    #     plt.show()

    for j, k in combinations(params_of_interest, 2):
        for param_array in parameter_samples:
            plt.scatter(x=param_array[burn_in_len:, j], y=param_array[burn_in_len:, k], alpha=0.5)
        plt.title('Parameters of Each Trace')
        plt.xlabel(param_names[j])
        plt.ylabel(param_names[k])
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

np.save("true_params.npy", params[idx, :num_params])

