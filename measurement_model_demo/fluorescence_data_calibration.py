import os
import pandas as pd
import numpy as np
import datetime as dt
from opt2q.simulator import Simulator
from opt2q.measurement import Fluorescence
from opt2q.data import DataSet
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from measurement_model_demo.apoptosis_model import model

from scipy.stats import norm


# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative',
                                      'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error',
                    'nrm_var_EC-RP': 'norm_EC-RP__error'})  # DataSet expects error columns to have "__error" suffix

# ------- Parameters --------
# The model is sensitive to these parameters
parameters = pd.DataFrame([[1.0e-5,    # x0  float  kc0 -- 95% bounded in (-7,  -3)
                            1.0e-2,    # x1  float  kc2 -- 95% bounded in (-5,   1)
                            3.0e-8,    # x2  float  kf3 -- 95% bounded in (-11, -6)
                            1.0e-2,    # x3  float  kc3 -- 95% bounded in (-5,   1)
                            1.0e-7,    # x4  float  kf4 -- 95% bounded in (-10, -4)
                            1.0e-2,    # x5  float  kr7 -- 95% bounded in (-8,   4)
                            1.0e-6]],  # x6  float  kc8 -- 95% bounded in (-9,  -3)
                          columns=['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc8']
                          )

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, fluorescence_data.time.max(), 100))

# ------- Measurement -------
fl = Fluorescence(results,
                  dataset=dataset,
                  measured_values={'norm_IC-RP': ['tBID_obs'],
                                   'norm_EC-RP': ['cPARP_obs']},
                  observables=['tBID_obs', 'cPARP_obs'])
measurement_results = fl.run()


# ------- Likelihood Function ------
@objective_function(simulator=sim, measurement_model=fl, return_results=False, evals=0)
def likelihood_fn(x):
    kc0 = 10 ** x[0]   # x0  float  kc0 -- 95% bounded in (-7,  -3)
    kc2 = 10 ** x[1]   # x1  float  kc2 -- 95% bounded in (-5,   1)
    kf3 = 10 ** x[2]   # x2  float  kf3 -- 95% bounded in (-11, -6)
    kc3 = 10 ** x[3]   # x3  float  kc3 -- 95% bounded in (-5,   1)
    kf4 = 10 ** x[4]   # x4  float  kf4 -- 95% bounded in (-10, -4)
    kr7 = 10 ** x[5]   # x5  float  kr7 -- 95% bounded in (-8,   4)
    kc8 = 10 ** x[6]   # x6  float  kc8 -- 95% bounded in (-9,  -3)

    kf0 = 10 ** x[7]   # x7  float  kf0 -- 95% bounded in (-9,  -4)
    kr0 = 10 ** x[8]   # x8  float  kr0 -- 95% bounded in (-5,  -1)
    kf1 = 10 ** x[9]   # x9  float  kf1 -- 95% bounded in (-10, -4)
    kr1 = 10 ** x[10]  # x10 float  kr1 -- 95% bounded in ( -5, -1)
    kc1 = 10 ** x[11]  # x11 float  kc1 -- 95% bounded in ( -2,  2)
    kf2 = 10 ** x[12]  # x12 float  kf2 -- 95% bounded in (-12, -8)
    kr2 = 10 ** x[13]  # x13 float  kr2 -- 95% bounded in ( -5, -1)
    kr3 = 10 ** x[14]  # x14 float  kr3 -- 95% bounded in ( -5, -1)
    kr4 = 10 ** x[15]  # x15 float  kr4 -- 95% bounded in ( -5, -1)
    kc4 = 10 ** x[16]  # x16 float  kc4 -- 95% bounded in ( -2,  2)
    kf5 = 10 ** x[17]  # x17 float  kf5 -- 95% bounded in ( -8, -4)
    kr5 = 10 ** x[18]  # x18 float  kr5 -- 95% bounded in ( -5, -1)
    kc5 = 10 ** x[19]  # x19 float  kc5 -- 95% bounded in ( -7, -3)
    kf6 = 10 ** x[20]  # x20 float  kf6 -- 95% bounded in ( -8, -4)
    kr6 = 10 ** x[21]  # x21 float  kr6 -- 95% bounded in ( -5, -1)
    kc6 = 10 ** x[22]  # x22 float  kc6 -- 95% bounded in ( -2,  2)
    kf7 = 10 ** x[23]  # x23 float  kf7 -- 95% bounded in ( -8, -4)
    kc7 = 10 ** x[24]  # x24 float  kc7 -- 95% bounded in ( -2,  2)

    params = pd.DataFrame([[kc0,    # x0  float  kc0 -- 95% bounded in (-7,  -3)
                            kc2,    # x1  float  kc2 -- 95% bounded in (-5,   1)
                            kf3,    # x2  float  kf3 -- 95% bounded in (-11, -6)
                            kc3,    # x3  float  kc3 -- 95% bounded in (-5,   1)
                            kf4,    # x4  float  kf4 -- 95% bounded in (-10, -4)
                            kr7,    # x5  float  kr7 -- 95% bounded in (-8,   4)
                            kc8,    # x6  float  kc8 -- 95% bounded in (-9,  -3)

                            kf0,    # x7  float  kf0 -- 95% bounded in (-9,  -4)
                            kr0,    # x8  float  kr0 -- 95% bounded in (-5,  -1)
                            kf1,    # x9  float  kf1 -- 95% bounded in (-10, -4)
                            kr1,    # x10 float  kr1 -- 95% bounded in ( -5, -1)
                            kc1,    # x11 float  kc1 -- 95% bounded in ( -2,  2)
                            kf2,    # x12 float  kf2 -- 95% bounded in (-12, -8)
                            kr2,    # x13 float  kr2 -- 95% bounded in ( -5, -1)
                            kr3,    # x14 float  kr3 -- 95% bounded in ( -5, -1)
                            kr4,    # x15 float  kr4 -- 95% bounded in ( -5, -1)
                            kc4,    # x16 float  kc4 -- 95% bounded in ( -2,  2)
                            kf5,    # x17 float  kf5 -- 95% bounded in ( -8, -4)
                            kr5,    # x18 float  kr5 -- 95% bounded in ( -5, -1)
                            kc5,    # x19 float  kc5 -- 95% bounded in ( -7, -3)
                            kf6,    # x20 float  kf6 -- 95% bounded in ( -8, -4)
                            kr6,    # x21 float  kr6 -- 95% bounded in ( -5, -1)
                            kc6,    # x22 float  kc6 -- 95% bounded in ( -2,  2)
                            kf7,    # x23 float  kf7 -- 95% bounded in ( -8, -4)
                            kc7,    # x24 float  kc7 -- 95% bounded in ( -2,  2)
                            ]],
                          columns=['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc8',
                                   'kf0', 'kr0', 'kf1', 'kr1', 'kc1', 'kf2', 'kr2', 'kr3', 'kr4', 'kc4', 'kf5', 'kr5',
                                   'kc5', 'kf6', 'kr6', 'kc6', 'kf7', 'kc7']
                          )
    likelihood_fn.simulator.param_values = params

    # dynamics
    sim_results = likelihood_fn.simulator.run()

    # measurement
    likelihood_fn.measurement_model.update_simulation_result(sim_results)
    likelihood_fn.evals += 1

    try:
        ll = -likelihood_fn.measurement_model.likelihood()
    except (ValueError, ZeroDivisionError):
        return -1e10

    if np.isnan(ll):
        return -1e10
    else:
        print(likelihood_fn.evals)
        print(x)
        print(ll)
        return ll


# -------- Calibration -------
# Model Inference via PyDREAM
sampled_params_0 = [SampledParam(norm, loc=-5, scale=1.0),           # x0  float  kc0 -- 95% bounded in (-7,  -3)
                    SampledParam(norm, loc=-2, scale=1.5),           # x1  float  kc2 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-8.5, scale=1.25),        # x2  float  kf3 -- 95% bounded in (-11, -6)
                    SampledParam(norm, loc=-2, scale=1.5),           # x3  float  kc3 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-7, scale=1.5),           # x4  float  kf4 -- 95% bounded in (-10, -4)
                    SampledParam(norm, loc=-2, scale=3.0),           # x5  float  kr7 -- 95% bounded in (-8,   4)
                    SampledParam(norm, loc=-6, scale=1.5),           # x6  float  kc8 -- 95% bounded in (-9,  -3)

                    SampledParam(norm, loc=-7, scale=1.0),           # x7  float  kf0 -- 95% bounded in (-9,  -4)
                    SampledParam(norm, loc=-3, scale=1.0),           # x8  float  kr0 -- 95% bounded in (-5,  -1)
                    SampledParam(norm, loc=-8, scale=1.0),           # x9  float  kf1 -- 95% bounded in (-10, -4)
                    SampledParam(norm, loc=-3, scale=1.0),           # x10 float  kr1 -- 95% bounded in ( -5, -1)
                    SampledParam(norm, loc=-0, scale=1.0),           # x11 float  kc1 -- 95% bounded in ( -2,  2)
                    SampledParam(norm, loc=-10, scale=1.0),          # x12 float  kf2 -- 95% bounded in (-12, -8)
                    SampledParam(norm, loc=-3, scale=1.0),           # x13 float  kr2 -- 95% bounded in ( -5, -1)
                    SampledParam(norm, loc=-3, scale=1.0),           # x14 float  kr3 -- 95% bounded in ( -5, -1)
                    SampledParam(norm, loc=-3, scale=1.0),           # x15 float  kr4 -- 95% bounded in ( -5, -1)
                    SampledParam(norm, loc=-0, scale=1.0),           # x16 float  kc4 -- 95% bounded in ( -2,  2)
                    SampledParam(norm, loc=-6, scale=1.0),           # x17 float  kf5 -- 95% bounded in ( -8, -4)
                    SampledParam(norm, loc=-3, scale=1.0),           # x18 float  kr5 -- 95% bounded in ( -5, -1)
                    SampledParam(norm, loc=-5, scale=1.0),           # x19 float  kc5 -- 95% bounded in ( -7, -3)
                    SampledParam(norm, loc=-6, scale=1.0),           # x20 float  kf6 -- 95% bounded in ( -8, -4)
                    SampledParam(norm, loc=-3, scale=1.0),           # x21 float  kr6 -- 95% bounded in ( -5, -1)
                    SampledParam(norm, loc=-0, scale=1.0),           # x22 float  kc6 -- 95% bounded in ( -2,  2)
                    SampledParam(norm, loc=-6, scale=1.0),           # x23 float  kf7 -- 95% bounded in ( -8, -4)
                    SampledParam(norm, loc=-0, scale=1.0),           # x24 float  kc7 -- 95% bounded in ( -2,  2)
                    ]

n_chains = 4
n_iterations = 1000
max_iterations = 10000
now = dt.datetime.now()
model_name = f'fluorescence_data_calibration_{now.year}{now.month}{now.day}'

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood_fn,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       gamma_levels=4,
                                       adapt_gamma=True,
                                       history_thin=1,
                                       model_name=model_name,
                                       verbose=True,
                                       crossover_burnin=1000)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters', sampled_params[chain])
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

    GR = Gelman_Rubin(sampled_params)

    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(model_name + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.isnan(GR).any() or np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

        # append sample with a re-run of the pyDream algorithm
        while not converged or (total_iterations < max_iterations):
            total_iterations += n_iterations
            sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                               likelihood=likelihood_fn,
                                               niterations=n_iterations,
                                               nchains=n_chains,
                                               multitry=False,
                                               gamma_levels=4,
                                               adapt_gamma=True,
                                               history_thin=1,
                                               model_name=model_name,
                                               verbose=True,
                                               restart=True,  # restart at the last sampled position
                                               start=starts,
                                               crossover_burnin=0)

            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters',
                        sampled_params[chain])
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(n_chains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt(model_name + str(total_iterations) + '.txt', GR)

            if np.all(GR < 1.2):
                converged = True

