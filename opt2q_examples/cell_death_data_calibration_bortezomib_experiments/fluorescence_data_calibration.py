import os
import pandas as pd
import numpy as np
from opt2q_examples.apoptosis_model import model
from opt2q.simulator import Simulator
from opt2q.data import DataSet
from opt2q.measurement import Fluorescence
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from scipy.stats import norm
import datetime as dt
from pydream.convergence import Gelman_Rubin
from pydream.core import run_dream


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
# error is variance
dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error',
                    'nrm_var_EC-RP': 'norm_EC-RP__error'})  # DataSet expects error columns to have "__error" suffix

# ------- Parameters --------
param_start = [-6.,  -3., -4.5,  # DISC formation
               -6., -3., 1.,
               -6., -3.,  1.,
               -6., -3., 1.,
               -6., -3., 1.,
               -6., -3., -4.5,  # MOMP formation I
               -6., -3., -4.5,  # MOMP formation II
               -6., -3.,  1.,   # effector caspase activation
               -6., -3., -1., -1.0,  # PARP cleavage and IC degradation reactions
               # Rate params for unrelated reactions
               -3.98704917, -2.41849717,  -4.28322569, -7.40096803, -2.27460174,  0.31984082]

parameters = pd.DataFrame([[10**p for p in param_start]], columns=[p.name for p in model.parameters_rules()])

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='scipyode',
                solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 1000000})
sim_res = sim.run(tspan=np.linspace(0, 24000, 100))

# ------- Measurement -------
fl = Fluorescence(sim_res,
                  dataset=dataset,
                  measured_values={'norm_IC-RP': ['tBID_obs'],
                                   'norm_EC-RP': ['cPARP_obs']},
                  observables=['tBID_obs', 'cPARP_obs'])
measurement_results = fl.run()

# ------- Priors --------
sampled_params_0 = [SampledParam(norm, loc=param_start[:-7], scale=1.5)]  # Exclude IC degradation and subsequent rxn


# ------- Likelihood Function ------
@objective_function(simulator=sim, measurement_model=fl, return_results=False, evals=0)
def likelihood_fn(x):
    new_params = pd.DataFrame([[10**p for p in x]+param_start[-7:]],
                              columns=[p.name for p in model.parameters_rules()])
    likelihood_fn.simulator.param_values = new_params

    # dynamics
    if hasattr(likelihood_fn.simulator.sim, 'gpu'):
        # process_id = current_process().ident % 4
        # likelihood.simulator.sim.gpu = [process_id]
        likelihood_fn.simulator.sim.gpu = 0
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


# ------- PyDREAM Parameters -------
n_chains = 4
n_iterations = 40000  # iterations per file-save
burn_in_len = 160000   # number of iterations during burn-in
max_iterations = 200000

now = dt.datetime.now()
model_name = f'fluorescence_data_calibration_{now.year}{now.month}{now.day}'

# -------- Calibration -------
# Model Inference via PyDREAM
if __name__ == '__main__':
    ncr = 25
    gamma_levels = 8
    p_gamma_unity = 0.1
    print(ncr, gamma_levels, p_gamma_unity)

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood_fn,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       nCR=ncr,
                                       gamma_levels=gamma_levels,
                                       adapt_gamma=True,
                                       p_gamma_unity=p_gamma_unity,
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
    burn_in_len = max(burn_in_len - n_iterations, 0)
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
                                               nCR=ncr,
                                               gamma_levels=gamma_levels,
                                               adapt_gamma=True,
                                               p_gamma_unity=p_gamma_unity,
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

