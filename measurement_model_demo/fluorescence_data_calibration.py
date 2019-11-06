# MW Irvin -- Lopez Lab -- 2018-10-09
import os
import pandas as pd
import numpy as np
import datetime as dt
from opt2q.noise import NoiseModel
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
# parameters = pd.DataFrame([[p.value for p in model.parameters_rules()]],
#                           columns=[p.name for p in model.parameters_rules()])

true_params = [-6.9370612,  -4.98273231, -4.95067084, -5.71425451, -2.76587087, -0.48233495,
               -7.82720074, -1.1626445,  -1.96149179, -5.41782659, -4.85626718, -1.52969156,
               -4.79153366, -1.76493099, -2.46972459, -9.06088874, -2.9369674,  -5.07077935,
               -6.11397529, -3.13346599, -2.2852632 , -4.70290624, -3.73949354, -2.40889128,
               -4.75438846, -2.87013358, -0.77520986, -7.08167013,  # Rate parameters for apoptosis related reactions
               # Rate params for unrelated reactions
               -3.98704917, -2.41849717,  -4.28322569, -7.40096803, -2.27460174,  0.31984082]

param_names = ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1', 'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3',
               'kf4', 'kr4', 'kc4', 'kf5', 'kr5', 'kc5', 'kf6', 'kr6', 'kc6', 'kr7', 'kf7', 'kc7',
               'kf8', 'kr8', 'kc8', 'kc9', 'kf10', 'kr10', 'kc10', 'kf11', 'kr11', 'kc11']

params = pd.DataFrame({'value': [10**p for p in true_params], 'param': param_names})
parameters = NoiseModel(params).run()

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
    params = pd.DataFrame([[10**p for p in x]],
                          columns=[p.name for p in model.parameters_rules()])
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
# sampled_params_0 = [SampledParam(norm, loc=[np.log10(p.value) for p in model.parameters_rules()], scale=1.5)]

# Use recent calibration as starting point
sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5)]

n_chains = 4
n_iterations = 10000  # iterations per file-save
burn_in_len = 5000   # number of iterations during burn-in
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
                                       nCR=10,
                                       gamma_levels=4,
                                       adapt_gamma=True,
                                       history_thin=1,
                                       model_name=model_name,
                                       verbose=True,
                                       crossover_burnin=min(n_iterations, burn_in_len),
                                       )

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters', sampled_params[chain])
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    burn_in_len = max(burn_in_len-n_iterations, 0)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')

    np.savetxt(model_name + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.isnan(GR).any() or np.any(GR > 1.2):
        # append sample with a re-run of the pyDream algorithm
        while not converged or (total_iterations < max_iterations):
            starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

            total_iterations += n_iterations
            sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                               likelihood=likelihood_fn,
                                               niterations=n_iterations,
                                               nchains=n_chains,
                                               multitry=False,
                                               nCR=10,
                                               gamma_levels=4,
                                               adapt_gamma=True,
                                               history_thin=1,
                                               model_name=model_name,
                                               verbose=True,
                                               restart=True,  # restart at the last sampled position
                                               start=starts,
                                               crossover_burnin=min(n_iterations, burn_in_len))

            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters',
                        sampled_params[chain])
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(n_chains)]
            GR = Gelman_Rubin(old_samples)
            burn_in_len = max(burn_in_len - n_iterations, 0)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')

            np.savetxt(model_name + str(total_iterations) + '.txt', GR)

            if np.all(GR < 1.2):
                converged = True

