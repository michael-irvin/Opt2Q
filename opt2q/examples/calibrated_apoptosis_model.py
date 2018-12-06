"""
========================
Apoptosis Model Analysis
========================
Plot the results of a calibration of an apoptosis model.
"""

import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.calibrating_apoptosis_model import likelihood_fn
from opt2q.data import DataSet


parameters = [-2.63779273,  0.34718117,  0.29324646,  0.65493748,  0.28320008,
              0.94277704,  -8.84685471,  0.93414489,   1.3133719, -0.87390272,
              0.80429802,   0.69954572,  0.77205767]
likelihood_fn(parameters)  # update the calibrator with current best param values
likelihood_fn.noise_model.update_values(param_mean=pd.DataFrame([['-', 500], ['+', 500]], columns=['inhibitor', 'num_sims']))
print(likelihood_fn.noise_model.param_mean)

simulator_parameters = likelihood_fn.noise_model.run()
likelihood_fn.simulator.param_values = simulator_parameters
simulation_results = likelihood_fn.simulator.run(np.linspace(0, 5000, 100)).opt2q_dataframe

simulation_results.reset_index(inplace=True)

legend_handles = []
cm = plt.get_cmap('tab10')
colors = [cm.colors[1], cm.colors[0]]

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1]})
for i, (label, df) in enumerate(simulation_results.groupby('inhibitor')):
    label = '{} zVAD'.format(label)
    legend_handles.append(mpatches.Patch(color=colors[i], label=label))
    for name, group in df.groupby(by='simulation'):
        group.plot.line(x='time', y='cPARP_obs', ax=ax, color=colors[i], alpha=0.1, legend=False)

ax.legend(handles=legend_handles)
ax.set_xlabel('time')
ax.set_ylabel('cPARP')
ax.set_ylim(0, 1500)

# plot results
wb = likelihood_fn.measurement_model
wb.process.set_params(classifier__do_fit_transform=False)
measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(1, 1500, 100), 'cPARP_obs': np.linspace(1, 1500, 100)})
print(likelihood_fn.measurement_model.process.steps[-1][1].coefficients)
measurement_model_y = wb.process.transform(measurement_model_x)

cPARP_results = measurement_model_y.filter(regex='cPARP')
cPARP_results['cPARP'] = measurement_model_x['cPARP_obs']

for col in sorted(list(set(cPARP_results.columns)-{'cPARP'})):
    cPARP_results.plot(y='cPARP', x=col, ax=ax1, label=col)
ax1.set_title('Probability')
ax1.set_ylim(0, 1500)

plt.show()

western_blot = pd.read_csv('Albeck_Sorger_WB.csv')
western_blot['time'] = western_blot['time'].apply(lambda x: x*500)
western_blot['inhibitor'] = '-'

dataset = DataSet(data=western_blot, measured_variables=['cPARP', 'PARP'])

ec = pd.DataFrame(['-', '+'], columns=['inhibitor'])

western_blot_results = wb.run(use_dataset=False)  # runs dataset first to get coefficients, then predicts the rest.
