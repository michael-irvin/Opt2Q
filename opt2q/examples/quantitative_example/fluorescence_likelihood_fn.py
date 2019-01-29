# MW Irvin -- Lopez Lab -- 2019-01-27
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.apoptosis_model import model
from opt2q.simulator import Simulator

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace
fluorescence_data['time'] = fluorescence_data['time'].apply(lambda x: x/3600)  # convert to hours

cm = plt.get_cmap('tab10')
fig = plt.figure()
plt.errorbar(fluorescence_data['time'], fluorescence_data['norm_IC-RP'],
             yerr=np.sqrt(fluorescence_data['nrm_var_IC-RP']),
             color=cm.colors[0], alpha=0.5, label='IC-RP')
plt.errorbar(fluorescence_data['time'], fluorescence_data['norm_EC-RP'],
             yerr=np.sqrt(fluorescence_data['nrm_var_EC-RP']), color=cm.colors[1], alpha=0.5, label='EC-RP')
plt.legend()
fig.show()

# ------- Parameters --------
parameters = pd.DataFrame([[1.0, 1.0, 'fluorescence_data']], columns=['kc3', 'kc4', 'experiment'])

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


