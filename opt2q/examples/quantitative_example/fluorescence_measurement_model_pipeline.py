# MW Irvin -- Lopez Lab -- 2019-01-27
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.apoptosis_model import model
from opt2q.simulator import Simulator
from opt2q.measurement import Fluorescence
from opt2q.data import DataSet

# ------- Calibrated Parameters --------
x = [2.38859837, 0.99997108, 2.99631024]
kc3 = 10 ** x[0]                                                        # float [(-3, 3),
kc4 = 10 ** x[1]                                                        # float  (-3, 3),
l_0 = 10 ** x[2]  # value of corresponding to the 1 ng/ml TRAIL         # float  (1, 3)

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace

# convert time scale to that of simulation
fluorescence_data['time'] = fluorescence_data['time'].apply(lambda x: x*500.0/3600)

cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.errorbar(fluorescence_data['time'], fluorescence_data['norm_IC-RP'],
             yerr=np.sqrt(fluorescence_data['nrm_var_IC-RP']),
             color=cm.colors[0], alpha=0.5, label='IC-RP')
plt.errorbar(fluorescence_data['time'], fluorescence_data['norm_EC-RP'],
             yerr=np.sqrt(fluorescence_data['nrm_var_EC-RP']), color=cm.colors[1], alpha=0.5, label='EC-RP')
plt.legend()
fig.show()

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative', 'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error', 'nrm_var_EC-RP': 'norm_EC-RP__error'})

# ------- Parameters --------
parameters = pd.DataFrame([[kc3, kc4, l_0]], columns=['kc3', 'kc4', 'L_0'])

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))
results_df = results.opt2q_dataframe
results_df = results_df.reset_index()

cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.plot(results_df['time'], results_df['Caspase_obs'], color=cm.colors[0], alpha=0.5, label='Active Caspase')
plt.plot(results_df['time'], results_df['cPARP_obs'], color=cm.colors[1], alpha=0.5, label='cPARP')
plt.legend()
fig.show()

# ------- Measurement -------
fl = Fluorescence(results,
                  dataset=dataset,
                  measured_values={'norm_IC-RP': ['Caspase_obs'],
                                   'norm_EC-RP': ['cPARP_obs']},
                  observables=['Caspase_obs', 'cPARP_obs'])
measurement_results = fl.run()

# ------- Plot Results ------
cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.fill_between(fluorescence_data['time'],
                 fluorescence_data['norm_IC-RP']+np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                 fluorescence_data['norm_IC-RP']-np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                 color=cm.colors[0], alpha=0.25)
plt.plot(fluorescence_data['time'], fluorescence_data['norm_IC-RP'], label='IC-RP', color=cm.colors[0], alpha=0.5)

plt.fill_between(fluorescence_data['time'],
                 fluorescence_data['norm_EC-RP']+np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                 fluorescence_data['norm_EC-RP']-np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                 color=cm.colors[1], alpha=0.25)
plt.plot(fluorescence_data['time'], fluorescence_data['norm_EC-RP'], label='EC-RP', color=cm.colors[1], alpha=0.5)

plt.plot(measurement_results['time'], measurement_results['Caspase_obs'], '--',
         color=cm.colors[0], alpha=0.5, label='Active Caspase')
plt.plot(measurement_results['time'], measurement_results['cPARP_obs'], '--',
         color=cm.colors[1], alpha=0.5, label='cPARP')
plt.legend()
fig.show()

print(fl.likelihood())