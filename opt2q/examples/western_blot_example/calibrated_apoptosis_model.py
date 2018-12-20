"""
========================
Apoptosis Model Analysis
========================
Plot the results of a calibration of an apoptosis model.

.. note: Before running this comment out the differential evolution step, and change num_sims to 10
"""

# Todo: REWRITE THIS FILE!! It no longer runs the calibrated results! Like there were tons of problems

import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.western_blot_example.calibrating_apoptosis_model import likelihood_fn
from opt2q.data import DataSet

parameters = np.array([-8.30180790e-01,  1.91663286e-01,  1.55065997e-01,  3.89427632e-01,
                        8.88879880e-03,  1.41117873e+00, -9.75730589e+00,  9.97027992e-01,
                        1.53601101e+00, -4.80892763e+00,  9.91605082e-01,  9.97653591e-01,
                        9.98322812e-01])

likelihood_fn(parameters)  # update the calibrator with current best param values
likelihood_fn.noise_model.update_values(param_mean=pd.DataFrame([['-', 50], ['+', 50]], columns=['inhibitor', 'num_sims']))

simulator_parameters = likelihood_fn.noise_model.run()
likelihood_fn.simulator.param_values = simulator_parameters
simulation_results = likelihood_fn.simulator.run(np.linspace(0, 5000, 100)).opt2q_dataframe

simulation_results.reset_index(inplace=True)


quit()

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
# ax.set_ylim(0, 10000)

# plot results
wb = likelihood_fn.measurement_model
wb.process.set_params(classifier__do_fit_transform=False)

# Todo: Out-of-Sample Predictions should not require experimental conditions.
measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(1, 150000, 100), 'cPARP_obs': np.linspace(1, 150000, 100),
                                    'inhibitor':np.full((100), '-'), 'time':np.full((100), 1)})
measurement_model_y = wb.process.transform(measurement_model_x)

cPARP_results = measurement_model_y.filter(regex='cPARP')
cPARP_results['cPARP'] = measurement_model_x['cPARP_obs']

for col in sorted(list(set(cPARP_results.columns)-{'cPARP'})):
    cPARP_results.plot(y='cPARP', x=col, ax=ax1, label=col)
ax1.set_title('Probability')
ax1.set_ylim(0, 150000)

plt.show()

western_blot = pd.read_csv('Albeck_Sorger_WB.csv')
western_blot['time'] = western_blot['time'].apply(lambda x: x*500)
western_blot['inhibitor'] = '-'

dataset = DataSet(data=western_blot, measured_variables=['cPARP', 'PARP'])

ec = pd.DataFrame(['-', '+'], columns=['inhibitor'])

western_blot_results = wb.run(use_dataset=False)  # runs dataset first to get coefficients, then predicts the rest.
plt.show()

