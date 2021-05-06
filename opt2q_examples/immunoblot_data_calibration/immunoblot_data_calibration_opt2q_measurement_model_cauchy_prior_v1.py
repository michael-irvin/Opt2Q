# MW Irvin -- Lopez Lab -- 2019-11-20

# Calibrating the apoptosis model to immunoblot data using variable measurement model

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm, cauchy
from opt2q.simulator import Simulator
from opt2q.measurement.base.likelihood import categorical_dist_likelihood as likelihood
from opt2q.measurement.base.transforms import Pipeline, ScaleToMinMax, Interpolate, LogisticClassifier
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from opt2q_examples.apoptosis_model import model
import pickle


with open(f'synthetic_WB_dataset_60s_2020_12_7.pkl', 'rb') as data_input:
    synthetic_immunoblot_data = pickle.load(data_input)

param_names = [p.name for p in model.parameters_rules()][:-6]
num_params = len(param_names)
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
true_params = np.load('true_params.npy')[:num_params]

parameters = pd.DataFrame([[10**p for p in true_params]], columns=param_names)


# ------- Simulations -------
# sim = Simulator(model=model, param_values=parameters, solver='cupsoda', integrator_options={'vol': 4.0e-15})
sim = Simulator(model=model, param_values=parameters, solver='scipyode', solver_options={'integrator': 'lsoda'},
                integrator_options={'mxstep': 2**20})  # effort to speed-up solver

sim_results = sim.run(np.linspace(0, synthetic_immunoblot_data.data.time.max(), 100))

results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# ------- Measurement -------
measurement_model = Pipeline(steps=[('x_scaled', ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'])),
                                    ('x_int', Interpolate('time', ['tBID_obs', 'cPARP_obs'],
                                                          synthetic_immunoblot_data.data['time'])),
                                    ('classifier', LogisticClassifier(
                                     synthetic_immunoblot_data,
                                     column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                                     do_fit_transform=False,
                                     classifier_type='ordinal_eoc'))])
processed_results = measurement_model.transform(results[['time', 'tBID_obs', 'cPARP_obs']])
print(likelihood(processed_results, synthetic_immunoblot_data))


# -------- Calibration -------
# Model Inference via PyDREAM
# Use recent calibration as starting point
sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),     # rate parameters floats
                    SampledParam(cauchy, loc=50.0, scale=10.0),          # coefficients__tBID_blot__coef_    float
                    SampledParam(cauchy, loc=0.2, scale=0.05),           # coefficients__tBID_blot__theta_1  float
                    SampledParam(cauchy, loc=0.2, scale=0.05),           # coefficients__tBID_blot__theta_2  float
                    SampledParam(cauchy, loc=0.2, scale=0.05),           # coefficients__tBID_blot__theta_3  float
                    SampledParam(cauchy, loc=0.2, scale=0.05),           # coefficients__tBID_blot__theta_4  float
                    SampledParam(cauchy, loc=50.0, scale=10.0),          # coefficients__cPARP_blot__coef_   float
                    SampledParam(cauchy, loc=0.25, scale=0.05),           # coefficients__cPARP_blot__theta_1 float
                    SampledParam(cauchy, loc=0.25, scale=0.05),           # coefficients__cPARP_blot__theta_2 float
                    SampledParam(cauchy, loc=0.25, scale=0.05),           # coefficients__cPARP_blot__theta_3 float
                    ]

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000
now = dt.datetime.now()
model_name = f'apoptosis_params_and_immunoblot_classifier_calibration_cauchy_05_{now.year}{now.month}{now.day}'


# ------- Likelihood Function ------
@objective_function(simulator=sim, measurement=measurement_model, likelihood=likelihood, return_results=False, evals=0)
def likelihood_fn(x):
    new_params = pd.DataFrame([[10 ** p for p in x[:num_params]]], columns=param_names)
    if any(xi <= 0 for xi in x[num_params:]):
        return -10000000.0

    # classifier
    c0 = x[num_params+0]
    t1 = x[num_params+1]
    t2 = t1 + x[num_params+2]
    t3 = t2 + x[num_params+3]
    t4 = t3 + x[num_params+4]

    c5 = x[num_params+5]
    t6 = x[num_params+6]
    t7 = t6 + x[num_params+7]
    t8 = t7 + x[num_params+8]

    likelihood_fn.simulator.param_values = new_params
    # dynamics
    if hasattr(likelihood_fn.simulator.sim, 'gpu'):
        likelihood_fn.simulator.sim.gpu = [0]

    # dynamics
    new_results = likelihood_fn.simulator.run().opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

    # measurement
    likelihood_fn.measurement.get_step('classifier').set_params(
        **{'coefficients__tBID_blot__coef_': np.array([c0]),
           'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
           'coefficients__cPARP_blot__coef_': np.array([c5]),
           'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})
    prediction = likelihood_fn.measurement.transform(new_results[['time', 'tBID_obs', 'cPARP_obs']])

    print(likelihood_fn.evals)
    print(x)
    likelihood_fn.evals += 1

    try:
        ll = -likelihood_fn.likelihood(prediction, synthetic_immunoblot_data)
    except (ValueError, ZeroDivisionError):
        return -1e10

    if np.isnan(ll):
        return -1e10
    else:
        print(ll)
        return ll


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


