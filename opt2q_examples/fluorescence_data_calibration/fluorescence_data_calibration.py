# MW Irvin -- Lopez Lab -- 2018-10-09
import os
import pandas as pd
import numpy as np
import datetime as dt
from opt2q.simulator import Simulator
from opt2q.measurement.base.likelihood import normal_pdf_empirical_var_likelihood as likelihood
from opt2q.measurement.base.transforms import Pipeline, Interpolate, ScaleToMinMax
from opt2q.data import DataSet
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from opt2q_examples.apoptosis_model import model

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
param_names = [p.name for p in model.parameters_rules()]

# Starting Parameters
# parameters = pd.DataFrame([[p.value for p in model.parameters_rules()]],
#                           columns=[p.name for p in model.parameters_rules()])

# Ground truth parameters
true_params = np.load(os.path.join(script_dir, 'true_params.npy'))
parameters = pd.DataFrame([[10**p for p in true_params]], columns=[p.name for p in model.parameters_rules()])


# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='scipyode',
                solver_options={'integrator': 'lsoda'},
                integrator_options={'mxstep': 2**20})  # effort to speed-up solver

# sim = Simulator(model=model, param_values=parameters, solver='cupsoda', integrator_options={'vol': 4.0e-15})
sim_results = sim.run(np.linspace(0, fluorescence_data.time.max(), 100))
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# ------- Measurement -------
measurement_model = Pipeline(
    steps=[('interpolate', Interpolate('time', ['cPARP_obs', 'tBID_obs'], dataset.data['time'])),
           ('normalize', ScaleToMinMax(feature_range=(0, 1), columns=['cPARP_obs', 'tBID_obs']))
           ])

p = measurement_model.transform(results[['tBID_obs', 'cPARP_obs', 'time']])
print(likelihood(p, dataset, {'norm_IC-RP': ['tBID_obs'], 'norm_EC-RP': ['cPARP_obs']}))


# ------- Likelihood Function ------
@objective_function(simulator=sim, measurement=measurement_model, likelihood=likelihood, return_results=False, evals=0)
def likelihood_fn(x):
    new_params = pd.DataFrame([[10**p for p in x]],
                              columns=[p.name for p in model.parameters_rules()])
    likelihood_fn.simulator.param_values = new_params

    # dynamics
    if hasattr(likelihood_fn.simulator.sim, 'gpu'):
        # process_id = current_process().ident % 4
        # likelihood.simulator.sim.gpu = [process_id]
        likelihood_fn.simulator.sim.gpu = 0
    results_ = likelihood_fn.simulator.run().opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

    # measurement
    prediction = likelihood_fn.measurement.transform(results_[['tBID_obs', 'cPARP_obs', 'time']])
    likelihood_fn.evals += 1

    try:
        ll = -likelihood_fn.likelihood(prediction, dataset, measured_values={'norm_IC-RP': ['tBID_obs'],
                                                                             'norm_EC-RP': ['cPARP_obs']})
    except (ValueError, ZeroDivisionError):
        return -1e10

    if not np.isfinite(ll):
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
n_iterations = 100000   # iterations per file-save
burn_in_len = 80000   # number of iterations during burn-in
max_iterations = 120000
now = dt.datetime.now()
model_name = f'fluorescence_data_calibration_{now.year}{now.month}{now.day}'

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    ncr = 25
    gamma_levels = 8
    p_gamma_unity = 0.1
    print(ncr, gamma_levels, p_gamma_unity)

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
                                       crossover_burnin=min(n_iterations, burn_in_len))

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

