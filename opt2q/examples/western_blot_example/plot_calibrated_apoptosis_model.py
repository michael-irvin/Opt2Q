# MW Irvin -- Lopez Lab -- 2018-12-19

"""
==========================
Calibrated Apoptosis Model
==========================

Plot the results of the calibration of an apoptosis model
"""

import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.western_blot_example.apoptosis_model_likelihood_fn import likelihood_fn
from opt2q.data import DataSet

num_sims = 500
calibration_results = np.array([-8.30180790e-01,  1.91663286e-01,  1.55065997e-01,  3.89427632e-01,
                                8.88879880e-03,  1.41117873e+00, -9.75730589e+00,  9.97027992e-01,
                                1.53601101e+00, -4.80892763e+00,  9.91605082e-01,  9.97653591e-01,
                                9.98322812e-01])

likelihood_fn(calibration_results)  # update the calibrator with current best param values
likelihood_fn.noise_model.update_values(param_mean=pd.DataFrame([['-', num_sims],
                                                                 ['+', num_sims]],
                                                                columns=['inhibitor', 'num_sims']))

sim_params = likelihood_fn.noise_model.run()

# plot noise model results
cm = plt.get_cmap('tab10')
colors = [cm.colors[1], cm.colors[0]]
fig, ax = plt.subplots(figsize=(8, 6))
for i, (label, df) in enumerate(sim_params.groupby('inhibitor')):
    df.plot.scatter(x='kc3', y='kc4', ax=ax, label='{} zVAD'.format(label), color=colors[i], alpha=0.5)
plt.legend()
plt.show()

# plot simulated western blot
sim_wb = likelihood_fn.measurement_model.run(use_dataset=False)

sim_wb['loc1'] = 0.3
sim_wb['loc2'] = 0.7
sim_wb = sim_wb.assign(time_hrs=lambda x:x['time']/500)

result_obs = {'cPARP': 'loc1', 'PARP':'loc2'}
result_size = {'cPARP': [0, 200, 400, 600, 800], 'PARP': [200, 500, 800]}
fig, ax = plt.subplots(figsize=(8, 3))
for label, df in sim_wb.groupby('time_hrs'):
    for col in [i for i in df.columns if '__' in i]:
        obs, level = tuple([k_rvs[::-1] for k_rvs in col[::-1].split('__')][::-1])
        df.plot.scatter(x='time_hrs', y=result_obs[obs], ax=ax, s=result_size[obs][int(level)],
                        alpha=0.5*np.mean(df[col])**2)
plt.ylim((0, 1))
plt.xlim(2, 10)
plt.xlabel('time [hrs]')
plt.savefig('simulated_wb.png')
plt.show()

# plot dynamics
likelihood_fn.simulator.param_values = sim_params
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

# plot measurement model
wb = likelihood_fn.measurement_model
wb.process.set_params(classifier__do_fit_transform=False)
wb.process.remove_step('sample_average')  # The response curve between x and y does not require an averaging step

measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(1, 300000, 100), 'cPARP_obs': np.linspace(1, 300000, 100)})
measurement_model_y = wb.process.transform(measurement_model_x)

cPARP_results = measurement_model_y.filter(regex='cPARP')
cPARP_results['cPARP'] = measurement_model_x['cPARP_obs']

for col in sorted(list(set(cPARP_results.columns)-{'cPARP'})):
    cPARP_results.plot(y='cPARP', x=col, ax=ax1, label=col)
ax1.set_title('Probability')
ax1.set_ylim(0, 300000)

plt.show()
