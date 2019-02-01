# MW Irvin -- Lopez Lab -- 2019-01-18

"""
==========================
Calibrated Apoptosis Model
==========================

Plot the results of the calibration of an apoptosis model
"""

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from opt2q.measurement.base import Scale, ScaleGroups
from opt2q.measurement.base.functions import derivative, where_max
from opt2q.examples.cell_viability_example.cell_viability_likelihood_fn import likelihood_fn

calibration_results = np.array([-0.70636527,   0.64134076,   1.00516322,   0.97326623,   0.60482481,
                                 0.12083115,   -70.75661173, 90.5471205,  46.22491245,
                                 66.16385524,  3.58463852, 0.64228147])  # -121

# calibration_results = np.array([ -0.88802359,   0.71834574,   1.00998571,   0.99977224,   0.58950459,
#                                   0.24862495,  33.97529859,  86.94037295,  48.95156694, -89.54725684,
#                                   79.7659045,   -0.11688959])


# ------- Likelihood Function -------
# Todo: make a better way of updating num_sims
params_for_update = likelihood_fn.noise_model.param_mean[['TRAIL_conc']].drop_duplicates().reset_index(drop=True)
params_for_update['num_sims'] = 1000
likelihood_fn.noise_model.update_values(param_mean=params_for_update)

likelihood_fn(calibration_results)
params = likelihood_fn.noise_model.run()

# -------plot calibrated param values --------
cm = plt.get_cmap('tab10')
params[['kc3', 'kc4']].plot.scatter(x='kc3', y='kc4', alpha=0.05, color=cm.colors[0])
plt.show()


# ------- Simulate dynamics -------
results = likelihood_fn.simulator.run(tspan=np.linspace(0, 5000, 100), param_values=params)
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
        group.plot.line(x='time', y='cPARP_obs', ax=ax, color=cm.colors[i], alpha=0.05, legend=False)
ax.set_xlabel('time')
ax.set_ylabel('cPARP')
ax.legend(handles=legend_handles)
plt.show()

# Max k and tau = t @ max_k are given by the 'where_max' function .
# (it returns the row where a variable is max).
max_k_and_tau = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var':'cPARP_obs'}).transform(k)
max_log_k_and_tau = Scale(columns='cPARP_obs', scale_fn='log10').transform(max_k_and_tau)

# Plot Feature Space for the Classifier
groups = max_log_k_and_tau.groupby('TRAIL_conc')
fig, ax = plt.subplots()
for i, group_ in enumerate(groups):
    name, group = group_
    ax.plot(group['cPARP_obs'], group['time'],
            marker='o', linestyle='', ms=10, alpha=0.1,  label=f'{name} ng/mL', color=cm.colors[i%8])

# classifier contour lines
x1 = np.linspace(4.7, 5.1, 1000)
x2 = np.linspace(1500, 4500, 1000)
nx1, nx2 = np.meshgrid(x1, x2)

print(likelihood_fn.measurement_model.process.steps)  # Todo: These steps are mislabeled. Fix!

grid = np.c_[nx1.ravel(), nx2.ravel()]
x = pd.DataFrame(grid, columns=['cPARP_obs', 'time'])
standardize_ = likelihood_fn.measurement_model.process.steps[3][1]
standardize_.set_params(**{'do_fit_transform': False})
x_std = standardize_.transform(x)
x_poly = likelihood_fn.measurement_model.process.steps[4][1].transform(x_std)  # polynomial features
x_prob = likelihood_fn.measurement_model.process.steps[5][1].transform(x_poly)  # probability of cell death

cs = ax.contour(nx1, nx2, x_prob['viability__1'].values.reshape(nx1.shape), colors=['black'], alpha=0.75, levels=np.linspace(0.1, 0.9, 5))
ax.clabel(cs, inline=1, fontsize=10)

ax.set_xlabel('log10(k), k=rate of change in Caspase indicator')
ax.set_ylabel('Time of Max Caspase activity (tau)')
ax.set_title('Apoptosis in cells treated with TRAIL')
ax.set_xlim(4.7, 5.1)
ax.legend()
plt.show()

# ------- plot predicted cell viability ------
likelihood_fn.measurement_model.update_simulation_result(results)
likelihood_fn.measurement_model.run().plot.bar(x='TRAIL_conc', y='viability__1', rot=0, color='k', alpha=0.5)
plt.show()
