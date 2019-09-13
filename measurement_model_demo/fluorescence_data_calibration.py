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
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace in column name

dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'norm_EC-RP']],
                  measured_variables={'norm_IC-RP': 'semi-quantitative',
                                      'norm_EC-RP': 'semi-quantitative'})
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error',
                    'nrm_var_EC-RP': 'norm_EC-RP__error'})  # DataSet expects error columns to have "__error" suffix

# ------- Parameters --------
# The model is sensitive to these parameters
parameters = pd.DataFrame([[1.0e-5,   # x0  float  kc0 -- 95% bounded in (-7,  -3)
                            1.0e-2,   # x1  float  kc2 -- 95% bounded in (-5,   1)
                            3.0e-8,   # x2  float  kf3 -- 95% bounded in (-11, -6)
                            1.0e-2,   # x3  float  kc3 -- 95% bounded in (-5,   1)
                            1.0e-7,   # x4  float  kf4 -- 95% bounded in (-10, -4)
                            1.0e-2,   # x5  float  kr7 -- 95% bounded in (-8,   4)
                            1.0e-6]],  # x6  float  kc8 -- 95% bounded in (-9,  -3)
                          columns=['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc8']
                          )

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 21600, 100))

# ------- Measurement -------
fl = Fluorescence(results,
                  dataset=dataset,
                  measured_values={'norm_IC-RP': ['BID_obs'],
                                   'norm_EC-RP': ['cPARP_obs']},
                  observables=['BID_obs', 'cPARP_obs'])
measurement_results = fl.run()


# ------- Likelihood Function ------
@objective_function(simulator=sim, measurement_model=fl, return_results=False, evals=0)
def likelihood_fn(x):
    kc0 = x[0]  # x0  float  kc0 -- 95% bounded in (-7,  -3)
    kc2 = x[1]  # x1  float  kc2 -- 95% bounded in (-5,   1)
    kf3 = x[2]  # x2  float  kf3 -- 95% bounded in (-11, -6)
    kc3 = x[3]  # x3  float  kc3 -- 95% bounded in (-5,   1)
    kf4 = x[4]  # x4  float  kf4 -- 95% bounded in (-10, -4)
    kr7 = x[5]  # x5  float  kr7 -- 95% bounded in (-8,   4)
    kc8 = x[6]  # x6  float  kc8 -- 95% bounded in (-9,  -3)

    params = pd.DataFrame([[kc0,    # x0  float  kc0 -- 95% bounded in (-7,  -3)
                            kc2,    # x1  float  kc2 -- 95% bounded in (-5,   1)
                            kf3,    # x2  float  kf3 -- 95% bounded in (-11, -6)
                            kc3,    # x3  float  kc3 -- 95% bounded in (-5,   1)
                            kf4,    # x4  float  kf4 -- 95% bounded in (-10, -4)
                            kr7,    # x5  float  kr7 -- 95% bounded in (-8,   4)
                            kc8]],  # x6  float  kc8 -- 95% bounded in (-9,  -3)
                          columns=['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc8']
                          )
    likelihood_fn.simulator.param_values = params

    # dynamics
    sim_results = likelihood_fn.simulator.run()
    if sim_results.dataframe.isna().any(axis=None):
        return 100000000  # if integration fails return high number to reject

    # measurement
    likelihood_fn.measurement_model.update_simulation_result(sim_results)
    ll = likelihood_fn.measurement_model.likelihood()

    likelihood_fn.evals += 1

    print(likelihood_fn.evals)
    print(x)
    print(ll)
    return -ll


# -------- Calibration -------
# Model Inference via PyDREAM
sampled_params_0 = [SampledParam(norm, loc=-5, scale=1.0),           # x0  float  kc0 -- 95% bounded in (-7,  -3)
                    SampledParam(norm, loc=-2, scale=1.5),           # x1  float  kc2 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-8.5, scale=1.25),        # x2  float  kf3 -- 95% bounded in (-11, -6)
                    SampledParam(norm, loc=-2, scale=1.5),           # x3  float  kc3 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-7, scale=1.5),           # x4  float  kf4 -- 95% bounded in (-10, -4)
                    SampledParam(norm, loc=-2, scale=3.0),           # x5  float  kr7 -- 95% bounded in (-8,   4)
                    SampledParam(norm, loc=-6, scale=1.5),           # x6  float  kc8 -- 95% bounded in (-9,  -3)
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
                                       verbose=True)

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
            print("Saved Results")
            print(total_iterations)
            print(not converged and (total_iterations < max_iterations))

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
                                               start=starts)

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

