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
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace in column name

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative',
                                      'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error',
                    'nrm_var_EC-RP': 'norm_EC-RP__error'})  # DataSet expects error columns to have "__error" suffix

# ------- Parameters --------
# The model is sensitive to these parameters
parameters = pd.DataFrame([[1e4, 1e-2, 1e-2, 1e-6, 1e-6]], columns=['C_0', 'kc3', 'kc4', 'kf3', 'kf4'])

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='scipyode')
results = sim.run(np.linspace(0, 21600, 100))

# ------- Measurement -------
fl = Fluorescence(results,
                  dataset=dataset,
                  measured_values={'norm_IC-RP': ['Caspase_obs'],
                                   'norm_EC-RP': ['cPARP_obs']},
                  observables=['Caspase_obs', 'cPARP_obs'])
measurement_results = fl.run()


@objective_function(simulator=sim, measurement_model=fl, return_results=False, evals=0)
def likelihood_fn(x):
    C_0 = 10 ** x[0]        # :  [( 2, 6),   # float  C_0
    kc3 = 10 ** x[1]        # :   (-2, 4),   # float  kc3
    kc4 = 10 ** x[2]        # :   (-2, 4),   # float  kc4
    kf3 = 10 ** x[3]        # :   (-8,-4),   # float  kf3
    kf4 = 10 ** x[4]        # :   (-8,-4)],  # float  kf4

    params = pd.DataFrame([[C_0, kc3, kc4, kf3, kf4]], columns=['C_0', 'kc3', 'kc4', 'kf3', 'kf4'])
    likelihood_fn.simulator.param_values = params

    # dynamics
    sim_results = likelihood_fn.simulator.run()
    if sim_results.dataframe.isna().any(axis=None):
        return 100000000  # if integration fails return high number to reject

    # measurement
    likelihood_fn.measurement_model.update_simulation_result(sim_results)
    l = likelihood_fn.measurement_model.likelihood()

    likelihood_fn.evals += 1

    print(likelihood_fn.evals)
    print(x)
    print(l)
    return l

