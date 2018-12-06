# MW Irvin -- Lopez Lab -- 2018-10-01
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import time
from opt2q.examples.apoptosis_model import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.data import DataSet
from opt2q.measurement import WesternBlot
from opt2q.calibrator import objective_function

from scipy.optimize import differential_evolution

# ------- simulate extrinsic noise -------
params_m = pd.DataFrame([['kc3', 1.0, '-', True],
                         ['kc3', 0.3, '+', True],
                         ['kc4', 1.0, '-', True],
                         ['kc4', 0.3, '+', True]],
                        columns=['param', 'value', 'inhibitor', 'apply_noise'])

param_cov = pd.DataFrame([['kc3', 'kc3', 0.009,  '+'],
                          ['kc4', 'kc4', 0.009,  '+'],
                          ['kc4', 'kc3', 0.01,   '-'],
                          ['kc4', 'kc3', 0.001,  '+']],
                         columns=['param_i', 'param_j', 'value', 'inhibitor'])

NoiseModel.default_sample_size = 1000
noise = NoiseModel(param_mean=params_m, param_covariance=param_cov)
parameters = noise.run()

# ------- simulate dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')

results = sim.run(np.linspace(0, 5000, 100))

# ------- simulate measurement -------
# data
western_blot = pd.read_csv('Albeck_Sorger_WB.csv')
western_blot['time'] = western_blot['time'].apply(lambda x: x*500)
western_blot['inhibitor'] = '-'

dataset = DataSet(data=western_blot, measured_variables={'cPARP': 'ordinal', 'PARP': 'ordinal'})

ec = pd.DataFrame(['-', '+'], columns=['inhibitor'])
wb = WesternBlot(simulation_result=results,
                 dataset=dataset,
                 measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                 observables=['PARP_obs', 'cPARP_obs'],
                 experimental_conditions=pd.DataFrame(['-', '+'], columns=['inhibitor']),
                 time_points=[1500, 2000, 2500, 3500, 4500])
wb.process.set_params(**{'sample_average__sample_size': 200})
results0 = wb.run(use_dataset=False)

# -------- calibrate model -----------
@objective_function(noise_model=noise, simulator=sim, measurement_model=wb, return_results=False, evals=0)
def likelihood_fn(x):
    wt_activity = 10**x[0]                                               # x0 = float [(-3, 3),
    inh_activity = wt_activity*x[1]                                      # x1 = float  (0, 1),
    variance_activity = (wt_activity*x[2])**2                            # x2 = float  (0, 1),
    wt_cov_activity = (variance_activity*x[3])                           # x3 = float  (0, 1),
    ko_cov_activity = (x[3]*x[4]*variance_activity)**2                   # x4 = float  (0, 1),

    PARP_coef_ = np.array([10**x[5]])                                    # x5 = float  (-3, 3),
    PARP_theta_ = np.array([x[6],                                        # x6 = float  (-100, 100),
                            x[6] + 10**x[7]])                            # x7 = float  (-1, 1),
    cPARP_coef_ = np.array([10**x[8]])                                   # x8 = float  (-3, 3),
    cPARP_theta_ = np.array([x[9],                                       # x9 = float  (-100, 100),
                             x[9] + 10**x[10],                           # x10 = float (-1, 1),
                             x[9] + 10**x[10] + 10**x[11],               # x11 = float (-1, 1),
                             x[9] + 10**x[10] + 10**x[11] + 10**x[12]])  # x12 = float (-1, 1)]

    noise_model_param_means = pd.DataFrame([['kc3', wt_activity,  '-'],
                                            ['kc3', inh_activity, '+'],
                                            ['kc4', wt_activity,  '-'],
                                            ['kc4', inh_activity, '+']],
                                           columns=['param', 'value', 'inhibitor'])
    noise_model_param_cov = pd.DataFrame([['kc3', 'kc3', variance_activity, '+'],
                                          ['kc4', 'kc4', variance_activity, '+'],
                                          ['kc4', 'kc3', wt_cov_activity,   '-'],
                                          ['kc4', 'kc3', ko_cov_activity,   '+']],
                                         columns=['param_i', 'param_j', 'value', 'inhibitor'])

    measurement_model_params = {'classifier__coefficients__PARP__coef_': PARP_coef_,
                                'classifier__coefficients__PARP__theta_': PARP_theta_,
                                'classifier__coefficients__cPARP__coef_': cPARP_coef_,
                                'classifier__coefficients__cPARP__theta_': cPARP_theta_
                                }

    likelihood_fn.noise_model.update_values(param_mean=noise_model_param_means,
                                            param_covariance=noise_model_param_cov)

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

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
    likelihood_fn,
    bounds=[(-3, 3),        # x0
            (0, 1),         # x1
            (0, 1),         # x2
            (0, 1),         # x3
            (0, 1),         # x4
            (-3, 3),        # x5
            (-100, 100),    # x6
            (-1, 1),        # x7
            (-3, 3),        # x8
            (-100, 100),    # x9
            (-1, 1),        # x10
            (-1, 1),        # x11
            (-1, 1)])       # x12

print(x)
np.save('calibrated_params_scipy_diff_evolution.npy', x)
