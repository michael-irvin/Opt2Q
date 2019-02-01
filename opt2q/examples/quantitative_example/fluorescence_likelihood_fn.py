# MW Irvin -- Lopez Lab -- 2019-01-27
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt2q.examples.apoptosis_model import model
from opt2q.simulator import Simulator
from opt2q.measurement import Fluorescence
from opt2q.data import DataSet
from opt2q.calibrator import objective_function

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace

# convert time scale to that of simulation
fluorescence_data['time'] = fluorescence_data['time'].apply(lambda x: x*500.0/3600)

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative', 'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error', 'nrm_var_EC-RP': 'norm_EC-RP__error'})

# ------- Parameters --------
parameters = pd.DataFrame([[1.0, 1.0]], columns=['kc3', 'kc4'])

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='scipyode')
results = sim.run(np.linspace(0, 5000, 100))
results_df = results.opt2q_dataframe
results_df = results_df.reset_index()

# ------- Measurement -------
fl = Fluorescence(results,
                  dataset=dataset,
                  measured_values={'norm_IC-RP': ['Caspase_obs'],
                                   'norm_EC-RP': ['cPARP_obs']},
                  observables=['Caspase_obs', 'cPARP_obs'])
measurement_results = fl.run()


@objective_function(simulator=sim, measurement_model=fl, return_results=False, evals=0)
def likelihood_fn(x):
    kc3 = 10 ** x[0]                                                        # float [(-3, 3),
    kc4 = 10 ** x[1]                                                        # float  (-3, 3),
    l_0 = 10 ** x[2]  # value of corresponding to the 1 ng/ml TRAIL         # float  (1, 3),

    params = pd.DataFrame([[kc3, kc4, l_0*50]], columns=['kc3', 'kc4', 'L_0'])
    likelihood_fn.simulator.param_values = params

    # dynamics
    sim_results = likelihood_fn.simulator.run(np.linspace(0, 5000, 100))

    # measurement
    likelihood_fn.measurement_model.update_simulation_result(sim_results)
    l = likelihood_fn.measurement_model.likelihood()

    likelihood_fn.evals += 1

    print(likelihood_fn.evals)
    print(x)
    print(l)
    return l