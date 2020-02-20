# MW Irvin -- Lopez Lab -- 2020-02-08

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations
from matplotlib import pyplot as plt
from opt2q_examples.apoptosis_model import model
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_tbid_dependent_opt2q_measurement_model \
    import sampled_params_0, params_df, simulate_heterogeneous_population, sim, extrinsic_noise_params

# ======== UPDATE THIS PART ===========
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'cell_death_data_calibration_results'
plot_title = 'tBID Dependent Apoptosis Calibration'

cal_date = '2020110'  # calibration file name contains date string

# Update this part with calibration settings corresponding to the above calibration files.
n_iterations = 100000
burn_in_len = 50000
thin = 100
number_of_traces = 4

# =====================================
# Load files
chain_history_file = [os.path.join(script_dir, calibration_folder, f) for f in
                      os.listdir(os.path.join(script_dir, calibration_folder))
                      if cal_date in f and 'chain_history' in f and 'fmm' not in f][0]

log_p_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                            os.listdir(os.path.join(script_dir, calibration_folder))
                            if cal_date in f and 'log_p' in f and 'fmm' not in f])

parameter_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                                os.listdir(os.path.join(script_dir, calibration_folder))
                                if cal_date in f and 'parameters' in f and 'fmm' not in f])

# reorder traces to be in numerical order (i.e. 1000 before 10000).
file_order = [str(n_iterations*n) for n in range(1, int(len(log_p_file_paths_)/number_of_traces)+1)]
log_p_file_paths = []
parameter_file_paths = []
for file_num in file_order:
    log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
    parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]

gr_file = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                  os.listdir(os.path.join(script_dir, calibration_folder))
                  if '2020110' in f and file_order[-1] in f and '.txt' in f and 'fmm' not in f])[0]

gr = np.loadtxt(gr_file)

# Parameters
num_rxn_model_params = len([p for p in model.parameters_rules()])
param_names = [p.name for p in model.parameters_rules()] + \
              ['Population Covariance Term', 'slope', 'intercept',
               '"Unrelated_Signal" coef', '"tBID_obs" coef', '"time" coef']

true_params = np.concatenate((np.load('true_params.npy'),
                              [sampled_params_0[1].dist.stats(moments='m'), 4, -0.25, 0.0, 0.25, -1.0]))

# Convergence
# plt.figure(figsize=(10, 3))
plt.bar(x=param_names[:17], height=gr[:17])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.title('Gelman-Rubin convergence metric for model parameters')
plt.show()

plt.bar(x=param_names[17:-7], height=gr[17:-7])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.title('Gelman-Rubin convergence metric for model parameters')
plt.show()

plt.bar(x=param_names[-7:], height=gr[-7:])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.xticks(rotation='vertical')
plt.title('Gelman-Rubin convergence metric for model parameters')
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

params_of_interest = range(len(param_names))

if __name__ == '__main__':
    # plot calibration metrics
    for trace in log_p_traces:
        plt.plot(trace)

    plt.title(f'{plot_title} \n Posterior Trace')
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
        if any(x > 0.9 for x in param_cor):
            for param_array in parameter_samples:
                plt.scatter(x=param_array[burn_in_len:, j][::thin],
                            y=param_array[burn_in_len:, k][::thin], alpha=0.5)
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

    for idx in params_of_interest[:num_rxn_model_params]:
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

    for idx in params_of_interest[num_rxn_model_params:]:
        s = pd.DataFrame({f'{param_names[idx]}': params[:, idx][::thin]})
        ax1 = s.plot.hist(alpha=0.4, bins=20, color='k')
        s.plot(kind='kde', ax=ax1, color=cm.colors[1], secondary_y=True)

        x_range = np.linspace(1.1*s.min(), 1.1*s.max(), 1000)
        prior = sampled_params_0[idx-num_rxn_model_params+1]

        ax2 = ax1.twinx()
        ax2.spines["right"].set_position(("axes", 1.2))
        ax2.plot(x_range, prior.dist.pdf(x_range), '--', color='k', label=f'{param_names[idx]}, prior')
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

# ------- Dynamics -------
# plot best performing parameters
log_ps = np.concatenate([lpt for lpt in log_p_traces])
params = np.concatenate([p for p in parameter_samples])

idx = np.argmax(log_ps)

# Adjust heterogeneous population based on new parameter values and coefficient of variation
new_rate_params = pd.DataFrame([10 ** np.array(params[idx, :num_rxn_model_params])],
                               columns=param_names[:num_rxn_model_params]).iloc[
    np.repeat(0, 100)].reset_index(drop=True)
cv_term = abs(params[idx, -1])**-0.5

noisy_param_names = ['MOMP_sig_0', 'USM1_0', 'USM2_0', 'USM3_0', 'kc0']
model_presets = pd.DataFrame({p.name: [p.value] for p in model.parameters if p.name in noisy_param_names})
model_presets.update(new_rate_params.iloc[0:1])

noisy_params = simulate_heterogeneous_population(model_presets, cv=cv_term)

params_df.update(new_rate_params)
params_df.update(pd.DataFrame(noisy_params, columns=noisy_param_names))

sim_parameters = params_df
parameters = pd.DataFrame([[p.value for p in model.parameters_rules()]],
                          columns=[p.name for p in model.parameters_rules()])

sim.param_values = parameters
sim_results_0 = sim.run()
results_0 = sim_results_0.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

sim.param_values = sim_parameters
sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

sim.param_values = pd.DataFrame([[10**p for p in true_params]], columns=param_names)
sim_results_true = sim.run()
results_true = sim_results_true.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

if __name__ == '__main__':
    cm = plt.get_cmap('tab10')
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    ax1.plot(results_0['time'].values, results_0['tBID_obs'].values / max(results_0['tBID_obs'].values), '--',
             label=f'tBID simulations initial fit', color=cm.colors[1])

    for name, df in results.groupby('simulation'):
        ax1.plot(df['time'].values, df['tBID_obs'].values / max(df['tBID_obs'].values),
                 label=f'tBID simulations best fit', color=cm.colors[1], alpha=0.15)

    ax1.plot(results_true['time'].values, results_true['tBID_obs'].values / max(results_true['tBID_obs'].values),
             label=f'tBID simulations true params', color=cm.colors[7])

    ax1.legend()
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[0:1]+handles[-2:], labels=labels[0:1]+labels[-2:])
    ax1.set_title('Predicted BID truncation dynamics')

    plt.show()


