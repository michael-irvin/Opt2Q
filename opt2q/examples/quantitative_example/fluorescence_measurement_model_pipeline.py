# MW Irvin -- Lopez Lab -- 2019-01-27
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.apoptosis_model import model
from opt2q.simulator import Simulator
from opt2q.measurement import Fluorescence
from opt2q.data import DataSet

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace

fluorescence_data_time_hrs = fluorescence_data['time'].apply(lambda x: x/3600)  # convert to hrs for plot.

cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.errorbar(fluorescence_data_time_hrs, fluorescence_data['norm_IC-RP'],
             yerr=np.sqrt(fluorescence_data['nrm_var_IC-RP']),
             color=cm.colors[0], alpha=0.5, label='IC-RP')
plt.errorbar(fluorescence_data_time_hrs, fluorescence_data['norm_EC-RP'],
             yerr=np.sqrt(fluorescence_data['nrm_var_EC-RP']), color=cm.colors[1], alpha=0.5, label='EC-RP')
plt.legend()
plt.xlabel('time [hrs]')
plt.ylabel('fluorescence')
plt.title("Initiator and Effector Caspase Activity Reporter Fluorescence")
fig.show()

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative', 'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error', 'nrm_var_EC-RP': 'norm_EC-RP__error'})

# ------- Dynamics -------
sim = Simulator(model=model, solver='scipyode')
results = sim.run(np.linspace(0, 21600, 100))
results_df = results.opt2q_dataframe
results_df = results_df.reset_index()
results_df_time_hrs = results_df['time'].apply(lambda x: x/3600)  # convert to hrs for plot

cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.plot(results_df_time_hrs, results_df['Caspase_obs'], color=cm.colors[0], alpha=0.5, label='Active Caspase')
plt.plot(results_df_time_hrs, results_df['cPARP_obs'], color=cm.colors[1], alpha=0.5, label='cPARP')
plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results')
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

plt.fill_between(fluorescence_data_time_hrs,
                 fluorescence_data['norm_IC-RP']+np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                 fluorescence_data['norm_IC-RP']-np.sqrt(fluorescence_data['nrm_var_IC-RP']),
                 color=cm.colors[0], alpha=0.25)
plt.plot(fluorescence_data_time_hrs, fluorescence_data['norm_IC-RP'], label='IC-RP', color=cm.colors[0], alpha=0.5)

plt.fill_between(fluorescence_data_time_hrs,
                 fluorescence_data['norm_EC-RP']+np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                 fluorescence_data['norm_EC-RP']-np.sqrt(fluorescence_data['nrm_var_EC-RP']),
                 color=cm.colors[1], alpha=0.25)
plt.plot(fluorescence_data_time_hrs, fluorescence_data['norm_EC-RP'], label='EC-RP', color=cm.colors[1], alpha=0.5)


measurement_results_time_hrs = measurement_results['time'].apply(lambda x:x/3600)

plt.plot(measurement_results_time_hrs, measurement_results['Caspase_obs'], '--',
         color=cm.colors[0], alpha=0.5, label='Active Caspase')
plt.plot(measurement_results_time_hrs, measurement_results['cPARP_obs'], '--',
         color=cm.colors[1], alpha=0.5, label='cPARP')
plt.xlabel('time [hrs]')
plt.ylabel('modeled fluorescence')
plt.title('Measurement Model')
plt.legend()
fig.show()

print(fl.likelihood())