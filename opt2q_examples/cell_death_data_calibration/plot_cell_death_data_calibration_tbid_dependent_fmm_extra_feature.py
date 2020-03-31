# MW Irvin -- Lopez Lab -- 2020-03-30

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations
from matplotlib import pyplot as plt
import seaborn as sns
from opt2q_examples.apoptosis_model import model
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_tbid_dependent_fmm_extra_feature \
    import sampled_params_0, params_df, simulate_heterogeneous_population, sim, extrinsic_noise_params, \
    pre_processing, synth_data, tbid_classifier
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_tbid_dependent_fixed_measurement_model \
    import tbid_classifier as tbid_classifier_true


# ======== UPDATE THIS PART ===========
# Update this part with the new file name/location info for log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'cell_death_data_calibration_results'
plot_title = 'tBID Dependent Apoptosis Calibration \n (Fixed Measurement Model with Extra Feature)'

cal_date = '2020220'  # calibration file name contains date string

# Update this part with calibration settings corresponding to the above calibration files.
n_iterations = 100000
burn_in_len = 50000
thin = 100
number_of_traces = 4

# =====================================
# Load files
chain_history_file = [os.path.join(script_dir, calibration_folder, f) for f in
                      os.listdir(os.path.join(script_dir, calibration_folder))
                      if cal_date in f and 'chain_history' in f and 'fmm_extra' in f][0]

log_p_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                            os.listdir(os.path.join(script_dir, calibration_folder))
                            if cal_date in f and 'log_p' in f and 'fmm_extra' in f])

parameter_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                                os.listdir(os.path.join(script_dir, calibration_folder))
                                if cal_date in f and 'parameters' in f and 'fmm_extra' in f])

# reorder traces to be in numerical order (i.e. 1000 before 10000).
file_order = [str(n_iterations*n) for n in range(1, int(len(log_p_file_paths_)/number_of_traces)+1)]
log_p_file_paths = []
parameter_file_paths = []
for file_num in file_order:
    log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
    parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]

gr_file = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                  os.listdir(os.path.join(script_dir, calibration_folder))
                  if cal_date in f and file_order[-1] in f and '.txt' in f and 'fmm_extra' in f])[0]

gr = np.loadtxt(gr_file)

# Parameters
param_names = [p.name for p in model.parameters_rules()] + ['Population Covariance Term']
true_params = np.concatenate((np.load('true_params.npy'), [sampled_params_0[-1].dist.stats(moments='m')]))

# Convergence
# plt.figure(figsize=(10, 3))
plt.bar(x=param_names[:17], height=gr[:17])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.title('Gelman-Rubin convergence metric for model parameters \n (fixed measurement model with extra feature)')
plt.xlabel('Parameters ("PCT" is population covariance term)')
plt.rcParams.update({'font.size': 9})
plt.show()

plt.bar(x=param_names[17:-1]+['PCT'], height=gr[17:])
plt.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
plt.title('Gelman-Rubin convergence metric for model parameters \n (fixed measurement model with extra feature)')
plt.xlabel('Parameters ("PCT" is population covariance term)')
plt.rcParams.update({'font.size': 9})
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
num_rxn_model_params = len([p for p in model.parameters_rules()])

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
# Initial Parameters
parameters = pd.concat([
    pd.DataFrame([[p.value for p in model.parameters_rules()] + [3000, '50ng/mL']],
                 columns=[p.name for p in model.parameters_rules()] + ['L_0', 'TRAIL_conc']),
    pd.DataFrame([[p.value for p in model.parameters_rules()] + [600, '10ng/mL']],
                 columns=[p.name for p in model.parameters_rules()] + ['L_0', 'TRAIL_conc'])]
    , ignore_index=True)

sim.param_values = parameters
sim_results_0 = sim.run()
results_0 = sim_results_0.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# "True" Parameters
sim.param_values = pd.concat([
    pd.DataFrame([[10**p for p in true_params] + [3000, '50ng/mL']], columns=param_names + ['L_0', 'TRAIL_conc']),
    pd.DataFrame([[10**p for p in true_params] + [600, '10ng/mL']], columns=param_names + ['L_0', 'TRAIL_conc'])],
    ignore_index=True)

sim_results_true = sim.run()
results_true = sim_results_true.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# Best Performing
# plot best performing parameters
log_ps = np.concatenate([lpt for lpt in log_p_traces])
params = np.concatenate([p for p in parameter_samples])

# thin = 2500
# ids = np.argsort(log_ps[:, 0])[-5000::thin]
ids = [np.argmax(log_ps[:, 0])]


def get_parameters(ix):
    # Adjust heterogeneous population based on new parameter values and coefficient of variation
    new_rate_params = pd.DataFrame([10 ** np.array(params[ix, :num_rxn_model_params])],
                                   columns=param_names[:num_rxn_model_params]).iloc[
        np.repeat(0, 100)].reset_index(drop=True)
    cv_term = abs(params[ix, -1])**-0.5

    noisy_param_names = ['MOMP_sig_0', 'USM1_0', 'USM2_0', 'USM3_0', 'kc0']
    model_presets = pd.DataFrame({p.name: [p.value] for p in model.parameters if p.name in noisy_param_names})
    model_presets.update(new_rate_params.iloc[0:1])

    noisy_params = simulate_heterogeneous_population(model_presets, cv=cv_term)

    params_df.update(new_rate_params)
    params_df.update(pd.DataFrame(noisy_params, columns=noisy_param_names))
    return params_df


best_parameters_dfs = pd.concat([get_parameters(idx) for idx in ids], ignore_index=True)
best_parameters_dfs = best_parameters_dfs.reset_index(drop=True)
best_parameters_dfs.drop(columns='simulation', inplace=True)
best_parameters_dfs['simulation'] = range(len(best_parameters_dfs))

# sim = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver='cupsoda',
#                 integrator_options={'vol': 4.0e-15})
sim.param_values = best_parameters_dfs
sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

if __name__ == '__main__':
    cm = plt.get_cmap('tab10')
    for i, ob in enumerate(['cPARP_obs', 'tBID_obs', 'Unrelated_Signal']):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        ax1.plot(results_0[results_0.TRAIL_conc == '50ng/mL']['time'].values,
                 results_0[results_0.TRAIL_conc == '50ng/mL'][ob].values /
                 max(results_0[results_0.TRAIL_conc == '50ng/mL'][ob].values),
                 '--', label=f'{ob.strip("_obs")} simulations initial fit, 50ng/mL TRAIL', color=cm.colors[i])
        ax1.plot(results_0[results_0.TRAIL_conc == '10ng/mL']['time'].values,
                 results_0[results_0.TRAIL_conc == '10ng/mL'][ob].values /
                 max(results_0[results_0.TRAIL_conc == '10ng/mL'][ob].values),
                 '--', label=f'{ob.strip("_obs")} simulations initial fit, 10ng/mL TRAIL', color=cm.colors[i+3])

        for name, df in results[results.TRAIL_conc == '50ng/mL'].groupby('simulation'):
            ax1.plot(df['time'].values, df[ob].values / max(df[ob].values),
                     label=f'{ob.strip("_obs")} simulations best fit 50ng/mL', color=cm.colors[i], alpha=0.15)

        for name, df in results[results.TRAIL_conc == '10ng/mL'].groupby('simulation'):
            ax1.plot(df['time'].values, df[ob].values / max(df[ob].values),
                     label=f'{ob.strip("_obs")} simulations best fit 10ng/mL', color=cm.colors[7], alpha=0.15)

        ax1.plot(results_true[results_true.TRAIL_conc == '50ng/mL']['time'].values,
                 results_true[results_true.TRAIL_conc == '50ng/mL'][ob].values /
                 max(results_true[results_true.TRAIL_conc == '50ng/mL'][ob].values),
                 '--', label=f'{ob.strip("_obs")} simulations true fit, 50ng/mL TRAIL', color=cm.colors[7])
        ax1.plot(results_true[results_true.TRAIL_conc == '10ng/mL']['time'].values,
                 results_true[results_true.TRAIL_conc == '10ng/mL'][ob].values /
                 max(results_true[results_true.TRAIL_conc == '10ng/mL'][ob].values),
                 '--', label=f'{ob.strip("_obs")} simulations true fit, 10ng/mL TRAIL', color=cm.colors[8])

        ax1.legend()
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles[0:3]+handles[-3:], labels=labels[0:3]+labels[-3:])
        ax1.set_title(f'Predicted {ob.strip("_obs")} dynamics. Solver: {sim.sim.integrator}')
        plt.savefig(f'{ob}_v2.png')
        plt.show()
