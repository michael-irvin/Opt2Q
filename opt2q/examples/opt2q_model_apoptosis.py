# MW Irvin -- Lopez Lab -- 2018-10-01
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from opt2q.examples.apoptosis_model import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.data import DataSet
from opt2q.measurement import WesternBlot
from opt2q.utils import profile

# ------- simulate extrinsic noise -------
params_m = pd.DataFrame([['kc3', 1.0, '-', True],
                         ['kc3', 0.3, '+', True],
                         ['kc4', 1.0, '-', True],
                         ['kc4', 0.3, '+', True]],
                        columns=['param', 'value', 'inhibitor', 'apply_noise'])

param_cov = pd.DataFrame([['kc4', 'kc3', 0.01,   '-'],
                          ['kc3', 'kc3', 0.009,  '+'],
                          ['kc4', 'kc4', 0.009,  '+'],
                          ['kc4', 'kc3', 0.001,  '+']],
                         columns=['param_i', 'param_j', 'value', 'inhibitor'])

NoiseModel.default_sample_size = 1000
noise = NoiseModel(param_mean=params_m, param_covariance=param_cov)
parameters = noise.run()

# plot noise model results
cm = plt.get_cmap('tab10')
colors = [cm.colors[1], cm.colors[0]]
fig, ax = plt.subplots(figsize=(8, 6))
for i, (label, df) in enumerate(parameters.groupby('inhibitor')):
    df.plot.scatter(x='kc3', y='kc4', ax=ax, label='{} zVAD'.format(label), color=colors[i], alpha=0.5)
plt.legend()
plt.show()

# ------- simulate dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))
results_df = results.opt2q_dataframe
results_df = results_df.reset_index()
results_df['time_axis'] = results_df['time'].apply(lambda x: x/500.)

# ------- cell lysis step of pipeline -------
from opt2q.measurement.base import SampleAverage, Interpolate
interpolate = Interpolate('time',
                          ['cPARP_obs', 'PARP_obs'],
                          [1500, 2000, 2500, 3500, 4500],
                          groupby='simulation')
res = interpolate.transform(results_df[['cPARP_obs', 'PARP_obs','time','simulation', 'inhibitor']])
sa = SampleAverage(columns=['cPARP_obs', 'PARP_obs'], drop_columns='simulation',
                   groupby=['time', 'inhibitor'], apply_noise=True, variances=10000)
avr = sa.transform(res)

# plot simulation results
cm = plt.get_cmap('tab10')
fig, axes = plt.subplots()

axes.violinplot(dataset=[avr[(avr.inhibitor == '-')&(avr.time == 1500)]["cPARP_obs"],
                         avr[(avr.inhibitor == '-')&(avr.time == 2000)]["cPARP_obs"],
                         avr[(avr.inhibitor == '-')&(avr.time == 2500)]["cPARP_obs"],
                         avr[(avr.inhibitor == '-')&(avr.time == 3500)]["cPARP_obs"],
                         avr[(avr.inhibitor == '-')&(avr.time == 4500)]["cPARP_obs"]])

axes.violinplot(dataset=[avr[(avr.inhibitor == '+')&(avr.time == 1500)]["cPARP_obs"],
                         avr[(avr.inhibitor == '+')&(avr.time == 2000)]["cPARP_obs"],
                         avr[(avr.inhibitor == '+')&(avr.time == 2500)]["cPARP_obs"],
                         avr[(avr.inhibitor == '+')&(avr.time == 3500)]["cPARP_obs"],
                         avr[(avr.inhibitor == '+')&(avr.time == 4500)]["cPARP_obs"]])
axes.set_xticks([3, 4, 5, 7, 9])
axes.set_xlabel('time [hrs]')
axes.set_ylabel('cPARP')
axes.legend(handles=[
    mpatches.Patch(color=colors[1], label='- zVAD'),
    mpatches.Patch(color=colors[0], label='+ zVAD')])
plt.show()

# ------ plot simulation results -------
fig2, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey='all', gridspec_kw={'width_ratios':[2, 1]})

legend_handles = []
for i, (label, df) in enumerate(results_df.groupby('inhibitor')):
    label = '{} zVAD'.format(label)
    legend_handles.append(mpatches.Patch(color=colors[i], label=label))
    for name, group in df.groupby(by='simulation'):
        group.plot.line(x='time_axis', y='cPARP_obs', ax=ax, color=colors[i], alpha=0.05, legend=False)
ax.set_xlabel('time')
ax.set_ylabel('cPARP')
ax.legend(handles=legend_handles)

# simulate measurement
# data
western_blot = pd.read_csv('Albeck_Sorger_WB.csv')
western_blot['time'] = western_blot['time'].apply(lambda x: x*500)
western_blot['inhibitor'] = '-'

dataset = DataSet(data=western_blot, measured_variables=['cPARP', 'PARP'])

ec = pd.DataFrame(['-', '+'], columns=['inhibitor'])
wb = WesternBlot(simulation_result=results,
                 dataset=dataset,
                 measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                 observables=['PARP_obs', 'cPARP_obs'],
                 experimental_conditions=pd.DataFrame(['-', '+'], columns=['inhibitor']),
                 time_points=[1500, 2000, 2500, 3500, 4500])

western_blot_results = wb.run(use_dataset=False)  # runs dataset first to get coefficients, then predicts the rest.

# plot results
wb.process.set_params(classifier__do_fit_transform=False)
wb.process.remove_step('sample_average')
measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(0, 1000000, 100), 'cPARP_obs': np.linspace(0, 1000000, 100)})
measurement_model_y = wb.process.transform(measurement_model_x)

cPARP_results = measurement_model_y.filter(regex='cPARP')
cPARP_results['cPARP'] = measurement_model_x['cPARP_obs']

for col in sorted(list(set(cPARP_results.columns)-{'cPARP'})):
    cPARP_results.plot(y='cPARP', x=col, ax=ax1, label=col)
ax1.set_title('Probability')
plt.savefig('wb_measurement_model')

# -------- plot blot ---------
western_blot_results['time_axis'] = western_blot_results['time'].apply(lambda x: x/500.0)
western_blot_results['loc1'] = 0.3
western_blot_results['loc2'] = 0.7

result_obs = {'cPARP': 'loc1', 'PARP':'loc2'}
result_size = {'cPARP': [100, 200, 400, 600, 1000], 'PARP': [200, 500, 1000]}
fig, ax = plt.subplots(figsize=(8, 3))
for label, df in western_blot_results[western_blot_results['inhibitor'] == '-'].groupby('time'):
    for col in [i for i in df.columns if '__' in i]:
        obs, level = tuple([k_rvs[::-1] for k_rvs in col[::-1].split('__')][::-1])
        df.plot.scatter(x='time_axis', y=result_obs[obs], ax=ax, s=result_size[obs][int(level)],
                        alpha=0.05*np.mean(df[col])**2)
plt.ylim((0, 1))
plt.savefig('simulated_wb.png')
print("Finished file")
# western_blot_results['loc1'] = 0.3
# western_blot_results['loc2'] = 0.7
#
# result_obs = {'cPARP': 'loc1', 'PARP':'loc2'}
# result_size = {'cPARP': [100, 200, 400, 600, 1000], 'PARP': [200, 500, 1000]}
# fig, ax = plt.subplots(figsize=(8, 3))
# for label, df in western_blot_results[western_blot_results['inhibitor'] == '+'].groupby('time'):
#     for col in [i for i in df.columns if '__' in i]:
#         obs, level = tuple([k_rvs[::-1] for k_rvs in col[::-1].split('__')][::-1])
#         df.plot.scatter(x='time_axis', y=result_obs[obs], ax=ax, s=result_size[obs][int(level)],
#                         alpha=0.05*np.mean(df[col])**2, color=colors[0])
# plt.ylim((0, 1))
# plt.show()
