# MW Irvin -- Lopez Lab -- 2018-12-20

"""
The pipeline that converts simulation results into relevant features for predicting cell
death or survival:

``log_k``: The log of the caspase activity
``tau``: Time at maximum caspase activity
"""

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

from opt2q.examples.apoptosis_model import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement.base import Scale, ScaleGroups
from opt2q.measurement.base.functions import derivative, where_max, polynomial_features
from opt2q.measurement import FractionalKilling

from opt2q.data import DataSet

# ------- Data -------
cell_viability = pd.read_csv('Cell_Viability_Data.csv')
experimental_conditions = cell_viability[['TRAIL_conc']]
len_ec = experimental_conditions.shape[0]
data = DataSet(data=cell_viability[['TRAIL_conc', 'viability']],
               measured_variables={'viability': 'nominal'})
data.measurement_error_df = cell_viability[['stdev']]

# Parameter values (formatted for noise model)
kc3 = np.array([['kc3', 1.0, True]])
kc4 = np.array([['kc4', 1.0, True]])
kc3_mean_df = pd.DataFrame(np.repeat(kc3, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])
kc4_mean_df = pd.DataFrame(np.repeat(kc4, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])

ligand = pd.DataFrame(cell_viability['TRAIL_conc'].values, columns=['value'])
ligand['param'] = 'L_0'
ligand['apply_noise'] = False

param_m = pd.concat([kc3_mean_df, kc4_mean_df, ligand], sort=False, ignore_index=True)
param_m['TRAIL_conc'] = np.tile(cell_viability['TRAIL_conc'].values, 3)
param_cov = pd.DataFrame([['kc3', 'kc3', 0.009],
                          ['kc4', 'kc4', 0.009],
                          ['kc4', 'kc3', 0.001]],
                         columns=['param_i', 'param_j', 'value'])

# ------- Noise Model -------
NoiseModel.default_sample_size = 5
noise = NoiseModel(param_mean=param_m, param_covariance=param_cov)
parameters = noise.run()

# ------- Simulate dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))
results_df = results.opt2q_dataframe
results_df.reset_index(inplace=True)
results_df = results_df[['cPARP_obs', 'time', 'TRAIL_conc','simulation']]

# Plot Dynamics
cm = plt.get_cmap('Accent')
fig, ax = plt.subplots()
legend_handles = []
for i, (label, df) in enumerate(results_df.groupby('TRAIL_conc')):
    label = '{} ng/mL'.format(label)
    legend_handles.append(mpatches.Patch(color=cm.colors[i], label=label))
    for name, group in df.groupby(by='simulation'):
        group.plot.line(x='time', y='cPARP_obs', ax=ax, color=cm.colors[i], alpha=0.1, legend=False)
ax.set_xlabel('time')
ax.set_ylabel('cPARP')
ax.legend(handles=legend_handles)
plt.show()

# ======= Pipeline =======
# ------- Getting Cell Death Predictors -------
# k = d/dt Caspase Activty, where Caspase Activity is indicated by a fluorescent caspase substrate
# In our model, d/dt cPARP functions as a similar marker of caspase activity.

# k = d/dx cPARP
ddx = Scale(columns='cPARP_obs', scale_fn=derivative)
k = ddx.transform(results_df)
k = k.assign(cPARP_obs=lambda x: np.clip(x['cPARP_obs'], 0.0, np.inf))

# Plot k
cm = plt.get_cmap('Accent')
fig, ax = plt.subplots()
legend_handles = []
for i, (label, df) in enumerate(k.groupby('TRAIL_conc')):
    label = '{} ng/mL'.format(label)
    legend_handles.append(mpatches.Patch(color=cm.colors[i], label=label))
    for name, group in df.groupby(by='simulation'):
        group.plot.line(x='time', y='cPARP_obs', ax=ax, color=cm.colors[i], alpha=0.1, legend=False)
ax.set_xlabel('time')
ax.set_ylabel('cPARP')
ax.legend(handles=legend_handles)
plt.show()

# Max k and tau = t @ max_k are given by the 'where_max' function .
# (it returns the row where a variable is max).
max_k_and_tau = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var':'cPARP_obs'}).transform(k)
max_log_k_and_tau = Scale(columns='cPARP_obs', scale_fn='log10').transform(max_k_and_tau, name='try_me')
# Plot Feature Space
groups = max_log_k_and_tau.groupby('TRAIL_conc')
fig, ax = plt.subplots()
for i, group_ in enumerate(groups):
    name, group = group_
    ax.plot(group['cPARP_obs'], group['time'],
            marker='o', linestyle='', ms=10, alpha=0.7,  label=f'{name} ng/mL', color=cm.colors[i%8])

ax.set_xlabel('log10(k), k=rate of change in Caspase indicator')
ax.set_ylabel('Time of Max Caspase activity (tau)')
ax.set_title('Apoptosis in cells treated with TRAIL')
ax.legend()
plt.show()