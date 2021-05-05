# Calibrating the apoptosis model to cell death data using a fixed measurement model
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import LogisticClassifier, Scale, ScaleGroups, Standardize
from opt2q.measurement.base.functions import derivative, where_max
from opt2q_examples.apoptosis_model import model
from opt2q.utils import _list_the_errors as list_items

# ------- Synthetic Data ----
script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, 'synthetic_tbid_dependent_apoptosis_data_large.csv')
synth_data = pd.read_csv(file_path)
synth_data['simulation'] = range(len(synth_data))

file_path = os.path.join(script_dir, 'true_params_extrinsic_noise_large.csv')
extrinsic_noise_params = pd.read_csv(file_path)
extrinsic_noise_params['simulation'] = range(len(extrinsic_noise_params))

# ------- Starting Point ----
param_names = [p.name for p in model.parameters_rules()]
true_params = np.load(os.path.join(script_dir, 'true_params.npy'))

# ============ Simulate Heterogeneous Population =============
noisy_param_names = ['MOMP_sig_0', 'USM1_0', 'USM2_0', 'USM3_0', 'kc0']

# divide extrinsic noise columns by the 'true_value' or model preset value (m0) to get the population for each column
model_presets = pd.DataFrame({p.name: [p.value] for p in model.parameters if p.name in noisy_param_names})
starting_params = pd.DataFrame([10**true_params], columns=param_names)
model_presets.update(starting_params)  # m0

# scale the extrinsic noise to a population centered at 0 (the scale is 1).
standard_population = (extrinsic_noise_params[noisy_param_names].values - model_presets[noisy_param_names].values) \
                      / (model_presets[noisy_param_names].values * 0.20)  # coef variation is %20


def simulate_heterogeneous_population(m, cv, population_0=standard_population):
    # rescale the extrinsic noise from (0, 1) to (m, m*cv).
    population = cv * m.values * population_0 + m.values
    return population


params_df = extrinsic_noise_params.copy()


def shift_and_scale_heterogeneous_population_to_new_params(x_):
    new_rate_params = pd.DataFrame([10 ** np.array(x_[:len(param_names)])], columns=param_names).iloc[
        np.repeat(0, len(params_df))].reset_index(drop=True)

    cv_term = abs(x_[len(param_names)]) ** -0.5
    model_presets.update(new_rate_params.iloc[0:1])
    noisy_params = simulate_heterogeneous_population(model_presets, cv=cv_term)

    params_df.update(new_rate_params)
    params_df.update(pd.DataFrame(noisy_params, columns=noisy_param_names))
    return params_df


# ------- Simulations -------
# fluorescence data as reference
file_path = os.path.join(script_dir, '../fluorescence_data_calibration/fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
time_axis = np.linspace(0, raw_fluorescence_data['# Time'].max()*60, 100)

sim = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver='cupsoda',
                integrator_options={'vol': 4.0e-15, 'max_steps': 2**20})


def set_up_simulator(solver_name):
    # 'cupsoda' and 'scipydoe' are valid solver names
    if solver_name == 'cupsoda':
        integrator_options = {'vol': 4.0e-15, 'max_steps': 2**20}
        solver_options = dict()
        if 'timeout' in Simulator.supported_solvers['cupsoda']._integrator_options_allowed:
            solver_options.update({'timeout': 60})
    elif solver_name == 'scipyode':
        solver_options = {'integrator': 'lsoda'}
        integrator_options = {'mxstep': 2**20}
    else:
        solver_options = {'atol': 1e-12}
        integrator_options = {}
    sim_ = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver=solver_name,
                     solver_options=solver_options, integrator_options=integrator_options)
    sim_.run()

    return sim_


sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})


# Measurement model attributes
# ============ Create tBID dynamics etc. features ============
def pre_processing(sim_res):
    obs = 'tBID_obs'

    ddx = ScaleGroups(columns=[obs], groupby='simulation', scale_fn=derivative) \
        .transform(sim_res[[obs, 'Unrelated_Signal', 'cPARP_obs', 'time', 'simulation', 'TRAIL_conc']])
    t_at_max = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var': obs}).transform(ddx)

    if t_at_max['time'].max() > 0.95 * sim_res['time'].max():
        # Enforce BID truncation rate maximization to occur within 5.6 hours.
        # This is consistent with our knowledge of the apoptosis system.
        return None

    log_max_ddx = Scale(columns='tBID_obs', scale_fn='log10').transform(t_at_max)
    standardized_features = Standardize(columns=['tBID_obs', 'time', 'Unrelated_Signal']).transform(log_max_ddx)
    return standardized_features


std_tbid_features = pre_processing(results)


# ============ Classify tBID into survival and death cell-fates =======
# Setup supervised ML classifier using a small proxy dataset
def set_up_classifier():
    tbid_0s_1s = pd.DataFrame({'apoptosis': [0, 1, 0, 1],
                               'TRAIL_conc': ['50ng/mL', '50ng/mL', '10ng/mL', '10ng/mL'],
                               'simulation': [48, 49, 50, 51]})
    tbid_classifier = LogisticClassifier(tbid_0s_1s,
                                         column_groups={'apoptosis': ['tBID_obs', 'time', 'Unrelated_Signal']},
                                         classifier_type='nominal')
    tbid_classifier.transform(std_tbid_features.iloc[48:52].reset_index(drop=True)
                              [['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

    # Simulate Cell death outcomes based on tBID features
    tbid_classifier.set_params(**{'do_fit_transform': False})  # Manually set ML parameters
    tbid_classifier.transform(std_tbid_features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])
    return tbid_classifier


# Plot Dataset Simulation and Pre-processing
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    # plot heterogeneous population of dataset
    cm = plt.get_cmap('tab10')
    x = extrinsic_noise_params[(extrinsic_noise_params.TRAIL_conc == '10ng/mL') & (synth_data.apoptosis == 0)]['kc0']
    y = np.random.random(len(x))
    plt.scatter(x * 1e5, y, color=cm.colors[7], marker='o', alpha=0.5, label='surviving cells 10ng/mL TRAIL')

    x = extrinsic_noise_params[(extrinsic_noise_params.TRAIL_conc == '10ng/mL') & (synth_data.apoptosis == 1)]['kc0']
    y = np.random.random(len(x))
    plt.scatter(x * 1e5, y, color=cm.colors[7], marker='x', alpha=0.5, label='apoptotic cells 10ng/mL TRAIL')

    x = extrinsic_noise_params[(extrinsic_noise_params.TRAIL_conc == '50ng/mL') & (synth_data.apoptosis == 0)]['kc0']
    y = np.random.random(len(x))
    plt.scatter(x * 1e5, y, color=cm.colors[1], marker='o', alpha=0.5, label='surviving cell 50ng/mL TRAIL')

    x = extrinsic_noise_params[(extrinsic_noise_params.TRAIL_conc == '50ng/mL') & (synth_data.apoptosis == 1)]['kc0']
    y = np.random.random(len(x))
    plt.scatter(x * 1e5, y, color=cm.colors[1], marker='x', alpha=0.5, label='apoptotic cells 50ng/mL TRAIL')
    # plt.xlim(0.95 * min(x * 1e5), 1.05 * max(x * 1e5))
    plt.title('Dataset relating Initial Conditions to Apoptosis vs Survival Outcome')
    plt.ylim(0.85 * min(y), 1.15 * max(y))
    plt.xlabel('DISC Initial Conditions')
    plt.legend()
    plt.show()

    cm = plt.get_cmap('tab10')

    # plot extrinsic noise parameters
    for p in noisy_param_names:
        m_data = np.average(extrinsic_noise_params[p])
        s_data = np.std(extrinsic_noise_params[p])
        mu_data = model_presets[p].values[0]
        sig_data = model_presets[p].values[0] * 0.235

        plt.hist(extrinsic_noise_params[p], density=True)
        plt.axvline(m_data, linestyle='--', color=cm.colors[5], label='mean of simulated population')
        plt.axvline(mu_data, linestyle='-', color='k', label='underlying value')

        plt.axvline(m_data + s_data, linestyle='--', color=cm.colors[5])
        plt.axvline(mu_data + sig_data, linestyle='--', color='k')

        plt.axvline(m_data - s_data, linestyle='--', color=cm.colors[5])
        plt.axvline(mu_data - sig_data, linestyle='--', color='k')
        plt.title(f'Simulated extrinsic noise on {p}')
        plt.show()

    # plot standard population
    for i, p in enumerate(noisy_param_names):
        plt.hist(standard_population[:, i], density=True)
        plt.plot(np.linspace(-2, 2, 100), norm.pdf(np.linspace(-2, 2, 100)))
        plt.title(f'STD {p}')
        plt.show()

    new_params = pd.DataFrame(simulate_heterogeneous_population(2*model_presets, 0.1),
                              columns=noisy_param_names)

    for p in noisy_param_names:
        plt.hist(extrinsic_noise_params[p], density=True, label='Initial', alpha=0.5)
        plt.hist(new_params[p], density=True, label='shifted', alpha=0.5)
        plt.title(f'Simulated extrinsic noise on {p} shifted and scaled \n to new mean and standard deviation')
        plt.show()

    # Set up simulation
    from opt2q_examples.generate_synthetic_cell_death_dataset import results_lg, labels

    for k, v in labels.items():
        idx, obs = v
        for name, df in results_lg.groupby(['simulation', 'TRAIL_conc']):
            if name[1] == '10ng/mL':
                plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[7])
            else:
                plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[idx])

        plt.legend([Line2D([0], [0], color=cm.colors[idx]),
                    Line2D([0], [0], alpha=0.6, color='k')],
                   [f'{k} 50ng/mL TRAIL', f'{k} 10ng/mL TRAIL'])
        plt.title(f'simulations based on "true parameters"  and 20% variation in \n '
                  f'{list_items(noisy_param_names)} large-dataset')
        plt.xlabel('time [seconds]')
        plt.ylabel('copies per cell')
        plt.show()

    for k, v in labels.items():
        idx, obs = v
        for name, df in results.groupby(['simulation', 'TRAIL_conc']):
            if name[1] == '10ng/mL':
                plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[7])
            else:
                plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[idx])

        plt.legend([Line2D([0], [0], color=cm.colors[idx]),
                    Line2D([0], [0], alpha=0.6, color='k')],
                   [f'{k} 50ng/mL TRAIL', f'{k} 10ng/mL TRAIL'])
        plt.title(f'simulations based on "true parameters"  and 20% variation in \n '
                  f'{list_items(noisy_param_names)} large-dataset \n CALIBRATION SETUP')
        plt.xlabel('time [seconds]')
        plt.ylabel('copies per cell')
        plt.show()

    # plot preprocessing
    cm = plt.get_cmap('tab10')
    k = 'tBID'
    idx, obs = labels[k]

    ddx_ = ScaleGroups(columns=[obs], groupby='simulation', scale_fn=derivative) \
        .transform(results_lg[[obs, 'Unrelated_Signal', 'cPARP_obs', 'time', 'simulation', 'TRAIL_conc']])
    t_at_max_ = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var': obs}).transform(ddx_)
    log_max_ddx_ = Scale(columns='tBID_obs', scale_fn='log10').transform(t_at_max_)
    std_tbid_features = Standardize(columns=['tBID_obs', 'time', 'Unrelated_Signal']).transform(log_max_ddx_)

    for name, df in ddx_.groupby(['simulation', 'TRAIL_conc']):
        if name[1] == '10ng/mL':
            plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[7])
        else:
            plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[idx])

    plt.legend([Line2D([0], [0], color=cm.colors[idx]),
                Line2D([0], [0], alpha=0.6, color='k')],
               [f'd{k}/dx 50ng/mL TRAIL', f'd{k}/dx 10ng/mL TRAIL'])
    plt.title(f'simulations based on "true parameters"  and 20% variation in \n '
              f'{list_items(noisy_param_names)} large-dataset')
    plt.xlabel('time [seconds]')
    plt.ylabel('copies per cell')
    plt.show()

    ddx_cs = ScaleGroups(columns=[obs], groupby='simulation', scale_fn=derivative) \
        .transform(results_lg[[obs, 'Unrelated_Signal', 'cPARP_obs', 'time', 'simulation', 'TRAIL_conc']])
    t_at_max_cs = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var': obs}).transform(ddx_cs)
    log_max_ddx_cs = Scale(columns='tBID_obs', scale_fn='log10').transform(t_at_max_cs)
    std_tbid_features = Standardize(columns=['tBID_obs', 'time', 'Unrelated_Signal']).transform(log_max_ddx_cs)

    for name, df in ddx_.groupby(['simulation', 'TRAIL_conc']):
        if name[1] == '10ng/mL':
            plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[7])
        else:
            plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[idx])

    plt.legend([Line2D([0], [0], color=cm.colors[idx]),
                Line2D([0], [0], alpha=0.6, color='k')],
               [f'd{k}/dx 50ng/mL TRAIL', f'd{k}/dx 10ng/mL TRAIL'])
    plt.title(f'simulations based on "true parameters"  and 20% variation in \n '
              f'{list_items(noisy_param_names)} large-dataset\n CALIBRATION SETUP')
    plt.xlabel('time [seconds]')
    plt.ylabel('copies per cell')
    plt.show()
