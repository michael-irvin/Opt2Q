# MW Irvin -- Lopez Lab -- 2018-12-22

import pandas as pd
import numpy as np
import os

from opt2q.examples.apoptosis_model import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement.base import Scale, ScaleGroups, SampleAverage
from opt2q.measurement.base.functions import derivative, where_max, polynomial_features
from opt2q.measurement import FractionalKilling
from opt2q.calibrator import objective_function
from opt2q.data import DataSet

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Cell_Viability_Data.csv')

cell_viability = pd.read_csv(file_path)
experimental_conditions = cell_viability[['TRAIL_conc']]
len_ec = experimental_conditions.shape[0]
data = DataSet(data=cell_viability[['TRAIL_conc', 'viability']],
               measured_variables={'viability': 'nominal'})
data.measurement_error_df = cell_viability[['TRAIL_conc', 'stdev']]

# Parameter values (formatted for noise model)
kc3 = np.array([['kc3', 1.0, True]])
kc4 = np.array([['kc4', 1.0, True]])
kc3_mean_df = pd.DataFrame(np.repeat(kc3, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])
kc4_mean_df = pd.DataFrame(np.repeat(kc4, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])

ligand = pd.DataFrame(cell_viability['TRAIL_conc'].values, columns=['value'])
ligand['param'] = 'L_0'
ligand['apply_noise'] = False

param_m = pd.concat([kc3_mean_df, kc4_mean_df, ligand], sort=False, ignore_index=True)
param_m['TRAIL_conc'] = np.tile(cell_viability['TRAIL_conc'].values, 3)
param_cov = pd.DataFrame([['kc3', 'kc3', 0.009],
                          ['kc4', 'kc4', 0.009],
                          ['kc4', 'kc3', 0.001]],
                         columns=['param_i', 'param_j', 'value'])

# ------- Noise Model -------
NoiseModel.default_sample_size = 200
noise = NoiseModel(param_mean=param_m, param_covariance=param_cov)
parameters = noise.run()

# ------- Simulate dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))

# ------- Measurement model -------
fk = FractionalKilling(simulation_result=results,
                       dataset=data,
                       measured_values={'viability':['cPARP_obs', 'time']},
                       observables=['cPARP_obs'],
                       experimental_conditions=cell_viability[['TRAIL_conc']],
                       time_dependent=False)

fk.process.remove_step('log_scale')
fk.process.add_step(('ddx', Scale(columns='cPARP_obs', scale_fn=derivative)),0)
fk.process.add_step(('at_max_t', ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var':'cPARP_obs'})),1)
fk.process.add_step(('log10',Scale(columns='cPARP_obs', scale_fn='log10')), 2)
fk.process.add_step(('polynomial',Scale(columns=['cPARP_obs', 'time'], scale_fn=polynomial_features, **{'degree':2})),
                    'standardize')  # add after the 'standardize' step
fk.run()


# -------- likelihood function -----------
@objective_function(noise_model=noise, simulator=sim, measurement_model=fk, return_results=False, evals=0)
def likelihood_fn(x):
    kc3 = 10**x[0]                                                  # float [(-3, 3),
    kc4 = 10**x[1]                                                  # float  (-3, 3),
    l_0 = 10**x[2]  # value of corresponding to the 1 ng/ml TRAIL   # float  (1, 3),

    kc3_var = (kc3*x[3])**2                                         # float  (0, 1),
    kc4_var = (kc4*x[4])**2                                         # float  (0, 1),
    kc_cov = kc3_var*x[5]                                           # float  (0, 1),

    viability_coef = np.array([[x[6],                               # float  (-100, 100),
                                x[7],                               # float  (-100, 100),
                                x[8],                               # float  (-100, 100),
                                x[9],                               # float  (-100, 100),
                                x[10]]])                            # float  (-100, 100),
    viability_intercept = np.array([x[11]])                         # float  (-10, 10)]

    kc3_val = np.array([['kc3', kc3, True]])
    kc4_val = np.array([['kc4', kc4, True]])
    kc3_mean_df = pd.DataFrame(np.repeat(kc3_val, len_ec, axis=0), columns=['param', 'value', 'apply_noise'])
    kc4_mean_df = pd.DataFrame(np.repeat(kc4_val, len_ec, axis=0), columns=['param', 'value', 'apply_noise'])

    ligand = pd.DataFrame(cell_viability['TRAIL_conc'].values*l_0, columns=['value'])
    ligand['param'] = 'L_0'
    ligand['apply_noise'] = False

    param_m = pd.concat([kc3_mean_df, kc4_mean_df, ligand], sort=False, ignore_index=True)
    param_m['TRAIL_conc'] = np.tile(cell_viability['TRAIL_conc'].values, 3)
    param_m['value'] = param_m['value'].astype(float)
    param_cov = pd.DataFrame([['kc3', 'kc3', kc3_var],
                              ['kc4', 'kc4', kc4_var],
                              ['kc4', 'kc3', kc_cov]],
                             columns=['param_i', 'param_j', 'value'])

    measurement_model_params = {'classifier__coefficients__viability__coef_':viability_coef,
                                'classifier__coefficients__viability__intercept_': viability_intercept}

    likelihood_fn.noise_model.update_values(param_mean=param_m,
                                            param_covariance=param_cov)

    simulator_parameters = likelihood_fn.noise_model.run()
    likelihood_fn.simulator.param_values = simulator_parameters

    sim_results = likelihood_fn.simulator.run(np.linspace(0, 5000, 100))
    likelihood_fn.measurement_model.update_simulation_result(sim_results)
    likelihood_fn.measurement_model.process.set_params(**measurement_model_params)

    likelihood_fn.evals += 1
    l = likelihood_fn.measurement_model.likelihood()

    print(likelihood_fn.evals)
    print(x)
    print(l)
    return l

