# MW Irvin -- Lopez Lab -- 2019-10-15

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import ScaleGroups, Scale, Standardize, LogisticClassifier
from opt2q.measurement.base.functions import derivative, where_max
from opt2q.utils import _list_the_errors as list_items
from opt2q_examples.apoptosis_model import model


# ------- Starting Parameters -------
overwrite_synthetic_datasets = False
overwrite_large_synthetic_datasets = False
large_dataset_size = 200  # trajectories per experiment

script_dir = os.path.dirname(__file__)
true_params = np.load(script_dir + '/true_params.npy')
true_params = true_params[:-6]  # the last 6 parameters pertain to reactions not related to apoptosis


param_names = [p.name for p in model.parameters_rules()][:-6]

params = pd.DataFrame({'value': 2*([10**p for p in true_params] + [3000]),
                       'param': 2*(param_names + ['L_0']),
                       'TRAIL_conc': ['50ng/mL' for p in range(len(true_params)+1)] +
                                     ['10ng/mL' for p in range(len(true_params)+1)]})

params_lg = pd.DataFrame({'value': 2*([10**p for p in true_params] + [3000]),
                          'param': 2*(param_names + ['L_0']),
                          'TRAIL_conc': ['50ng/mL' for p in range(len(true_params)+1)] +
                                        ['10ng/mL' for p in range(len(true_params)+1)],
                          'num_sims': 2*(len(true_params)+1)*[large_dataset_size]})


# add extrinsic noise to MOMP reactions
noisy_param_names = ['MOMP_sig_0', 'USM1_0', 'USM2_0', 'USM3_0', 'kc0']
params_cv = pd.DataFrame()
for noisy_param_name in noisy_param_names:
    if noisy_param_name in param_names:
        noisy_param_val = params[params.param.str.contains(noisy_param_name)]['value'].values[0]
    else:
        noisy_param_val = [p.value for p in model.parameters if p.name == noisy_param_name][0]
    noisy_param_var = (0.2*noisy_param_val)**2  # variance if coef variation is %20
    params_cv = pd.concat([params_cv,
                           pd.DataFrame([[noisy_param_name, noisy_param_name, noisy_param_var]],
                                        columns=['param_i', 'param_j', 'value'])])

ligands = pd.DataFrame([['L_0', 3000, '50ng/mL'],
                        ['L_0',  600, '10ng/mL']],
                       columns=['param', 'value', 'TRAIL_conc'])
nm_0 = NoiseModel(model=model, param_mean=params)  # parameters without extrinsic noise
nm_0.update_values(param_mean=ligands)
nm_lg = NoiseModel(model=model, param_mean=params_lg, param_covariance=params_cv)
nm_lg.update_values(param_mean=ligands)

sim_parameters = nm_0.run()
parameters_lg = nm_lg.run()

# save heterogeneous parameter population for use in calibrations
# save synthetic tbid dependent apoptosis data
if __name__ == '__main__':
    if overwrite_large_synthetic_datasets:
        parameters_lg.to_csv(r'true_params_extrinsic_noise_large.csv', index=None, header=True)
        print("Overwriting true_params_extrinsic_noise_large.csv")

cm = plt.get_cmap('tab10')
if __name__ == '__main__':
    for noisy_param_name in noisy_param_names:
        parameters_lg[noisy_param_name].hist()
        plt.title(noisy_param_name)
        plt.show()

# ------- Simulations -------
# fluorescence data as reference
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data_calibration', 'fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

# simulate dynamics
sim_0 = Simulator(model=model, param_values=sim_parameters, solver='cupsoda')
sim_lg = Simulator(model=model, param_values=parameters_lg, solver='cupsoda')


sim_results_lg = sim_lg.run(np.linspace(0, fluorescence_data.time.max(), 100))
sim_results_0 = sim_0.run(np.linspace(0, fluorescence_data.time.max(), 100))

results_lg = sim_results_lg.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
results_0 = sim_results_0.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})


# plot settings
labels = {f'cPARP': (0, 'cPARP_obs'),
          f'tBID': (1, 'tBID_obs'),
          f'MOMP dependent signal': (2, 'MOMP_signal'),
          'TRAIL-Receptor Complex': (3, 'TRAIL_receptor_obs'),
          f'Caspase 3 active': (4, 'C3_active_obs'),
          f'Caspase 8 active': (5, 'C8_active_obs'),
          f'Unrelated Signal': (6, 'Unrelated_Signal')}

if __name__ == '__main__':
    for k, v in labels.items():
        idx, obs = v
        for name, df in results_lg.groupby(['simulation', 'TRAIL_conc']):
            if name[1] == '10ng/mL':
                plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[7])
            else:
                plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[idx])

        plt.plot(results_0[results_0.TRAIL_conc == '10ng/mL']['time'],
                 results_0[results_0.TRAIL_conc == '10ng/mL'][obs],
                 linewidth=3, alpha=0.6, color='k')
        plt.plot(results_0[results_0.TRAIL_conc == '50ng/mL']['time'],
                 results_0[results_0.TRAIL_conc == '50ng/mL'][obs],
                 linewidth=3, alpha=1, color=cm.colors[idx])

        plt.legend([Line2D([0], [0], color=cm.colors[idx]),
                    Line2D([0], [0], alpha=0.6, color='k')],
                   [f'{k} 50ng/mL TRAIL', f'{k} 10ng/mL TRAIL'])
        plt.title(f'simulations based on "true parameters"  and 20% variation in \n '
                  f'{list_items(noisy_param_names)} large-dataset')
        plt.xlabel('time [seconds]')
        plt.ylabel('copies per cell')
        plt.show()


# ========== BID truncation dynamics features (Large Dataset) ==========
k = f'tBID'
idx, obs = labels[k]

ddx = ScaleGroups(columns=[obs], groupby='simulation', scale_fn=derivative) \
    .transform(results_lg[[obs, 'Unrelated_Signal', 'cPARP_obs', 'time', 'simulation', 'TRAIL_conc']])
t_at_max = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var': obs}).transform(ddx)
log_max_ddx = Scale(columns='tBID_obs', scale_fn='log10').transform(t_at_max)
std_tbid_features = Standardize(columns=['tBID_obs', 'time', 'Unrelated_Signal']).transform(log_max_ddx)

# Assume any trace that exceeds 10% PARP cleavage before 9900s undergoes apoptosis
t = 9900
thr = 0.10 * ddx.cPARP_obs.max()
apo_simulations = np.unique(ddx[(ddx.time < t) & (ddx.cPARP_obs > thr)].simulation.values)
apo_cells = std_tbid_features[std_tbid_features.simulation.isin(apo_simulations)]
live_cells = std_tbid_features[~std_tbid_features.simulation.isin(apo_simulations)]

# Synthetic dataset
tbid_0s_1s = pd.DataFrame({'apoptosis': [0, 1, 0, 1],
                           'TRAIL_conc': ['50ng/mL', '50ng/mL', '10ng/mL', '10ng/mL'],
                           'simulation': [48, 49, 50, 51]})
tbid_classifier = LogisticClassifier(tbid_0s_1s,
                                     column_groups={'apoptosis': ['tBID_obs', 'time', 'Unrelated_Signal']},
                                     classifier_type='nominal')
tbid_classifier.transform(std_tbid_features.iloc[48:52].reset_index(drop=True)
                          [['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

# Classifier Parameters that Generate Similar Cell Death Dynamics
a = 4
tbid_classifier.set_params(**{'coefficients__apoptosis__coef_': np.array([[0.0, 0.25, -1.0]]) * a,
                              'coefficients__apoptosis__intercept_': np.array([-0.25]) * a,
                              'do_fit_transform': False})

# Synthetic data based on tBID features
tbid_predictions = tbid_classifier.transform(
    std_tbid_features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

std_tbid_features['apoptosis'] = tbid_predictions.apply(
    lambda xx: np.random.choice([0, 1], p=xx[['apoptosis__0', 'apoptosis__1']]), axis=1)

synthetic_tbid_dependent_apoptosis_data = std_tbid_features[['TRAIL_conc', 'simulation', 'apoptosis']]

# save synthetic tbid dependent apoptosis data
if __name__ == '__main__':
    if overwrite_large_synthetic_datasets:
        synthetic_tbid_dependent_apoptosis_data.to_csv(r'synthetic_tbid_dependent_apoptosis_data_large.csv',
                                                       index=None, header=True)
        print("Overwriting synthetic_tbid_dependent_apoptosis_data_large.csv")

if __name__ == '__main__':
    # plot derivative of tBID vs. time
    for name, df in ddx.groupby(['simulation', 'TRAIL_conc']):
        if name[1] == '10ng/mL':
            plt.plot(df['time'].values, df[obs].values, alpha=0.2, color=cm.colors[7])
        else:
            plt.plot(df['time'], df[obs], alpha=0.2, color=cm.colors[idx])

    plt.legend([Line2D([0], [0], color=cm.colors[idx]),
                Line2D([0], [0], alpha=0.6, color='k')],
               [f'{k} 50ng/mL TRAIL', f'{k} 10ng/mL TRAIL'])
    plt.title(f'd/dx based on "true parameters" and 20% variation in \n {list_items(noisy_param_names)}')
    plt.xlabel('time [seconds]')
    plt.ylabel('d/dx copies per cell')
    plt.show()

    cols = {'10ng/mL': cm.colors[7],
            '50ng/mL': cm.colors[idx]}

    # plot tBID related apoptosis predictors
    for name, df in log_max_ddx.groupby('TRAIL_conc'):
        plt.scatter(df['tBID_obs'], df['time'], alpha=0.5, color=cols[name])
    plt.title(f'max d {k}/dx based on "true parameters" and 20% variation in \n {list_items(noisy_param_names)}')
    plt.xlabel(f'log-max d{k}/dx copies per cell')
    plt.ylabel('time [seconds]')
    plt.show()

    # Plot Joint Apoptosis Distributions
    bid_features = log_max_ddx[[obs, 'time', 'Unrelated_Signal', 'TRAIL_conc']]. \
        rename(columns={obs: f'log-max d {k}/dx copies per cell',
                        'Unrelated_Signal': 'Unrelated Signal',
                        'time': 'time @ max Bid truncation rate'})

    g = sns.pairplot(bid_features, diag_kind="kde", markers="o", diag_kws=dict(shade=True),
                     hue="TRAIL_conc", palette={'10ng/mL': cm.colors[7], '50ng/mL': cm.colors[1]})
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    g.fig.suptitle('Comparing Bid truncation and apoptosis-unrelated signal feature \n '
                   'covariance with TRAIL concentration')
    plt.show()

    # plot tBID related apoptosis predictors corresponding to 10% cPARP before 165min
    # (this criteria was a guide for deciding "true" parameters of the measurement model)
    fig = plt.subplots(figsize=(6, 6))
    for name, df in apo_cells.groupby('TRAIL_conc'):
        plt.scatter(df['tBID_obs'], df['time'], marker='x', alpha=0.5, color=cols[name])
    for name, df in live_cells.groupby('TRAIL_conc'):
        plt.scatter(df['tBID_obs'], df['time'], marker='o', alpha=0.5, color=cols[name])
    plt.plot([-2, 2], [-0.5, 0.5], color='k', alpha=0.4)

    live_10 = Line2D([], [], color=cm.colors[7], marker='o', linestyle='None', label='10ng/mL TRAIL Surviving')
    apo_10 = Line2D([], [], color=cm.colors[7], marker='x', linestyle='None', label='10ng/mL TRAIL Apoptotic')
    live_50 = Line2D([], [], color=cm.colors[idx], marker='o', linestyle='None', label='50ng/mL TRAIL Surviving')
    apo_50 = Line2D([], [], color=cm.colors[idx], marker='x', linestyle='None', label='50ng/mL TRAIL Apoptotic')

    plt.legend(handles=[live_10, apo_10, live_50, apo_50], bbox_to_anchor=(0.5, -0.2))
    plt.title(f'Apoptosis threshold based on "true parameters" and tBID features\n '
              f'corresponding PARP cleavage exceeding 10% within 165 min')
    plt.xlabel(f'standardized log-max d{k}/dx copies per cell')
    plt.ylabel('standardized time [seconds]')

    # plot tBID related apoptosis classifiers
    x_ = np.linspace(-2.5, 2.5, 21)
    y_ = np.linspace(-2.5, 2.5, 21)
    z_ = np.linspace(-2.5, 2.5, 21)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    grid = np.c_[x.ravel(), y.ravel(), z.ravel()]
    tbid_grid_cols = ['tBID_obs', 'time', 'Unrelated_Signal']
    tbid_grid = pd.DataFrame(grid, columns=tbid_grid_cols)
    tbid_grid['TRAIL_conc'] = '10ng/mL'
    tbid_grid['simulation'] = 0
    x_prob = tbid_classifier.transform(tbid_grid)

    std_tbid_features['apoptosis_plot'] = std_tbid_features. \
        apply(lambda xx: f'{xx["TRAIL_conc"]} TRAIL {["Surviving", "Apoptotic"][xx["apoptosis"]]}', axis=1)

    g = sns.pairplot(std_tbid_features[['tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc', 'apoptosis_plot']],
                     vars=['tBID_obs', 'time', 'Unrelated_Signal'], diag_kind="kde", diag_kws=dict(alpha=0.0),
                     hue='apoptosis_plot',
                     # palette={'10ng/mL TRAIL Surviving': cm.colors[7], '10ng/mL TRAIL Apoptotic': cm.colors[7],
                     #          '50ng/mL TRAIL Surviving': cm.colors[idx], '50ng/mL TRAIL Apoptotic': cm.colors[idx]},
                     palette=OrderedDict([('10ng/mL TRAIL Surviving', cm.colors[7]),
                                          ('10ng/mL TRAIL Apoptotic', cm.colors[7]),
                                          ('50ng/mL TRAIL Surviving', cm.colors[idx]),
                                          ('50ng/mL TRAIL Apoptotic', cm.colors[idx])]),
                     markers=["o", "x", "o", "x"])

    g._hue_var = 'TRAIL_conc'
    g.hue_names = ['50ng/mL', '10ng/mL']
    g.hue_vals = std_tbid_features['TRAIL_conc']
    g.hue_kws = {}
    g.palette = g.palette[::2]
    g.map_diag(sns.kdeplot, **dict(shade=True))

    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

        x_non_feature = tbid_grid_cols[[a for a in [0, 1, 2] if a not in [i, j]][0]]
        x_grid = tbid_grid[tbid_grid[x_non_feature] == 0]
        x_shape = x[0].shape

        cs = g.axes[j, i].contour(z[0],
                                  y[0],
                                  x_prob.iloc[x_grid.index]['apoptosis__1'].values.reshape(x_shape).T,
                                  colors=['black'], alpha=0.5, levels=np.linspace(0.1, 0.9, 3))

        g.axes[j, i].clabel(cs, inline=1, fontsize=10)

    g._legend_data = {k.split: v for k, v in g._legend_data.items()}
    g.fig.suptitle('Bid truncation and apoptosis-unrelated signal feature \n '
                   'and apoptosis probability contours and predictions (Large Dataset)')
    plt.show()