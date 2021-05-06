# MW Irvin -- Lopez Lab -- 2019-10-08

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from opt2q.simulator import Simulator
from opt2q.data import DataSet
from opt2q.measurement.base import LogisticClassifier, Interpolate, ScaleToMinMax, Pipeline
from opt2q_examples.apoptosis_model import model


save_dataset = True
# ------- Data -------
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

file_path = os.path.join(script_dir, '../fluorescence_data_calibration/fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative',
                                      'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error',
                    'nrm_var_EC-RP': 'norm_EC-RP__error'})  # DataSet expects error columns to have "__error" suffix

# ------- Starting Parameters -------
param_names = [p.name for p in model.parameters_rules()][:-6]

true_params = np.load('true_params.npy')[:len(param_names)]

parameters = pd.DataFrame([[10**p for p in true_params]], columns=param_names)


# ------- Simulations -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
sim_results = sim.run(np.linspace(0, fluorescence_data.time.max(), 100))

results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

cm = plt.get_cmap('tab10')
if __name__ == '__main__':
    plt.plot(results['time'], results['cPARP_obs'], label=f'cPARP_obs', alpha=0.8, color=cm.colors[0])
    plt.plot(results['time'], results['tBID_obs'], label=f'tBID_obs', alpha=0.8, color=cm.colors[1])
    plt.legend()
    plt.title('simulations based on "true parameters"')
    plt.xlabel('time [seconds]')
    plt.ylabel('copies per cell')
    plt.show()

# ------- Fluorescence -------
# The "true parameters" are based on best fit to these data.
measurement_model = Pipeline(
    steps=[('interpolate', Interpolate('time', ['cPARP_obs', 'tBID_obs'], dataset.data['time'])),
           ('normalize', ScaleToMinMax(feature_range=(0, 1), columns=['cPARP_obs', 'tBID_obs']))
           ])

measurement_results = measurement_model.transform(results[['tBID_obs', 'cPARP_obs', 'time']])

if __name__ == '__main__':
    plt.plot(measurement_results['time'], measurement_results['cPARP_obs'], label=f'simulated PARP cleavage')
    plt.plot(fluorescence_data['time'], fluorescence_data['norm_EC-RP'], '--', label=f'norm_EC-RP data', color=cm.colors[0])
    plt.fill_between(fluorescence_data['time'],
                     fluorescence_data['norm_EC-RP']-np.sqrt(dataset.measurement_error_df['norm_EC-RP__error']),
                     fluorescence_data['norm_EC-RP']+np.sqrt(dataset.measurement_error_df['norm_EC-RP__error']),
                     color=cm.colors[0], alpha=0.2)
    plt.title('"True Parameters" Compared w/ cPARP Fluorescence Data')
    plt.xlabel('time [seconds]')
    plt.ylabel('fluorescence [AU]')
    plt.legend()
    plt.show()

    plt.plot(measurement_results['time'], measurement_results['tBID_obs'],
             label=f'simulated Bid truncation', color=cm.colors[1])
    plt.plot(fluorescence_data['time'], fluorescence_data['norm_IC-RP'], '--',
             label=f'norm_IC-RP data', color=cm.colors[1])
    plt.fill_between(fluorescence_data['time'],
                     fluorescence_data['norm_IC-RP']-np.sqrt(dataset.measurement_error_df['norm_IC-RP__error']),
                     fluorescence_data['norm_IC-RP']+np.sqrt(dataset.measurement_error_df['norm_IC-RP__error']),
                     color=cm.colors[1], alpha=0.2)
    plt.title('"True Parameters" compared w/ tBID Fluorescence Data')
    plt.xlabel('time [seconds]')
    plt.ylabel('fluorescence [AU]')
    plt.legend()
    plt.show()


# ------- Immunoblot -------
def immunoblot_number_of_categories(variances, expected_misclassification_rate=0.05, data_range=1):
    # Effective Number of Bits in Fluorescence Data
    # ref -- https://en.wikipedia.org/wiki/Effective_number_of_bits
    # Fluorescence data was normalized to 0-1 :. data_range=1.
    data_rms = np.sqrt(variances).mean()
    z_stat = norm.ppf(1 - expected_misclassification_rate)
    peak_noise = z_stat*data_rms
    signal_to_noise_ratio = 20*np.log10(peak_noise/data_range)
    effective_number_of_bits = -(signal_to_noise_ratio+1.76)/6.02

    return int(np.floor(0.70*(2**effective_number_of_bits)))  # No. of categories: 70% of fluorescence data bit capacity


IC_RP__n_cats = immunoblot_number_of_categories(dataset.measurement_error_df['norm_IC-RP__error'])
EC_RP__n_cats = immunoblot_number_of_categories(dataset.measurement_error_df['norm_EC-RP__error'])

# ------- Immunoblot Data Set -------
ordinal_dataset_size = 14  # 28, 16, 14, 7 divide evenly into the total 112 rows.
len_fl_data = len(fluorescence_data)

# immunoblot_data_0 is necessary to setup the classifier
immunoblot_data_0 = fluorescence_data[['time']].iloc[1::int(len_fl_data / ordinal_dataset_size)]
immunoblot_data_0['tBID_blot'] = np.tile(range(IC_RP__n_cats), int(np.ceil(ordinal_dataset_size/IC_RP__n_cats)))[:ordinal_dataset_size]
immunoblot_data_0['cPARP_blot'] = np.tile(range(EC_RP__n_cats), int(np.ceil(ordinal_dataset_size/EC_RP__n_cats)))[:ordinal_dataset_size]

immunoblot_dataset = DataSet(immunoblot_data_0, measured_variables={'tBID_blot': 'ordinal', 'cPARP_blot': 'ordinal'})

# set up classifier
x_scaled = ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'])\
            .transform(results[['time', 'tBID_obs', 'cPARP_obs']])
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], immunoblot_data_0['time'])\
            .transform(x_scaled)
lc = LogisticClassifier(immunoblot_dataset,
                        column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True,
                        classifier_type='ordinal_eoc')
lc.set_up(x_int)

# ------- Define Classifier Parameters-------
a = 50
lc.set_params(** {'coefficients__cPARP_blot__coef_': np.array([a]),
                  'coefficients__cPARP_blot__theta_': np.array([0.03,  0.20, 0.97])*a,
                  'coefficients__tBID_blot__coef_': np.array([a]),
                  'coefficients__tBID_blot__theta_': np.array([0.03,  0.4,  0.82, 0.97])*a})


# plot classifier
lc.do_fit_transform = False
plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})
lc_results = lc.transform(plot_domain)
cPARP_results = lc_results.filter(regex='cPARP_blot')
tBID_results = lc_results.filter(regex='tBID_blot')

# ------- Synthetic Immunoblot Data -------
n = 180
time_span = list(range(fluorescence_data['time'].max()))[::n]  # ::30 = one measurement per 30s; 6x fluorescence data

x_scaled = ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'])\
            .transform(results[['time', 'tBID_obs', 'cPARP_obs']])
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], time_span)\
            .transform(x_scaled)

lc_results = lc.transform(x_int)

tBID_blot_cols = lc_results.filter(regex='tBID_blot__').columns
cPARP_blot_cols = lc_results.filter(regex='cPARP_blot__').columns

lc_results['tBID_blot'] = lc_results.apply(lambda x: np.random.choice(
    [int(c.split('__')[1]) for c in tBID_blot_cols],
    p=[x[c] for c in tBID_blot_cols]), axis=1)

lc_results['cPARP_blot'] = lc_results.apply(lambda x: np.random.choice(
    [int(c.split('__')[1]) for c in cPARP_blot_cols],
    p=[x[c] for c in cPARP_blot_cols]), axis=1)

immunoblot_data = lc_results[['time', 'tBID_blot', 'cPARP_blot']]
synthetic_immunoblot_data = DataSet(immunoblot_data,
                                    measured_variables={'tBID_blot': 'ordinal', 'cPARP_blot': 'ordinal'})

if __name__ == '__main__' and save_dataset:
    import pickle
    import datetime as dt

    now = dt.datetime.now()

    with open(f'synthetic_WB_dataset_{n}s_{now.year}_{now.month}_{now.day}.pkl', 'wb') as output:
        pickle.dump(synthetic_immunoblot_data, output, pickle.HIGHEST_PROTOCOL)

    with open(f'synthetic_WB_dataset_{n}s_{now.year}_{now.month}_{now.day}.pkl', 'rb') as data_input:
        loaded_dataset = pickle.load(data_input)

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1]})
    ax1.scatter(x=synthetic_immunoblot_data.data['time'],
                y=synthetic_immunoblot_data.data['cPARP_blot'].values / (EC_RP__n_cats-1),
                s=10, color=cm.colors[0], label=f'cPARP blot data', alpha=0.5)
    ax1.plot(x_scaled['time'], x_scaled['cPARP_obs'], color=cm.colors[0], label=f'simulated cPARP')
    ax1.legend()

    for col in sorted(list(cPARP_results.columns)):
        ax2.plot(cPARP_results[col].values, np.linspace(0, 1, 100), label=col)

    ax1.set_title('Classification of Simulated cPARP')
    ax1.set_xlabel('time [seconds]')
    ax1.set_ylabel('fluorescence [AU]')
    ax2.set_xlabel('category probability')
    ax2.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(x_scaled['time'], x_scaled['tBID_obs'], color=cm.colors[1], label=f'simulated tBID')
    ax1.scatter(x=synthetic_immunoblot_data.data['time'],
                y=synthetic_immunoblot_data.data['tBID_blot'].values / (IC_RP__n_cats-1),
                s=10, color=cm.colors[1], label=f'tBID blot data', alpha=0.5)
    ax1.legend()

    for col in sorted(list(tBID_results.columns)):
        ax2.plot(tBID_results[col].values, np.linspace(0, 1, 100), label=col)

    ax1.set_title('Classification of Simulated tBID')
    ax1.set_xlabel('time [seconds]')
    ax1.set_ylabel('fluorescence [AU]')
    ax2.set_xlabel('category probability')
    ax2.legend()
    plt.show()
