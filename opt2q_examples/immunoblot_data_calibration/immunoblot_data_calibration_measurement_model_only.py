# MW Irvin -- Lopez Lab -- 2019-10-10

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import expon, uniform, cauchy
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from opt2q.calibrator import objective_function
from opt2q.simulator import Simulator
from opt2q.noise import NoiseModel
from opt2q.measurement import WesternBlot
from opt2q.measurement.base.transforms import Pipeline, ScaleToMinMax, Interpolate, LogisticClassifier
from opt2q_examples.apoptosis_model import model
from opt2q_examples.immunoblot_data_calibration.generate_synthetic_immunoblot_dataset import synthetic_immunoblot_data

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
true_params = np.load('true_params.npy')

param_names = ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1', 'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3',
               'kf4', 'kr4', 'kc4', 'kf5', 'kr5', 'kc5', 'kf6', 'kr6', 'kc6', 'kr7', 'kf7', 'kc7',
               'kf8', 'kr8', 'kc8', 'kc9']
param_names = [p.name for p in model.parameters_rules()]
params = pd.DataFrame({'value': [10**p for p in true_params], 'param': param_names})
parameters = NoiseModel(params).run()

# ------- Simulations -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
sim_results = sim.run(np.linspace(0, synthetic_immunoblot_data.data.time.max(), 100))

results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

wb = WesternBlot(simulation_result=sim_results,
                 dataset=synthetic_immunoblot_data,
                 measured_values={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                 observables=['tBID_obs', 'cPARP_obs'])

wb.process = Pipeline(steps=[('x_scaled',ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'])),
                             ('x_int', Interpolate(
                                 'time',
                                 ['tBID_obs', 'cPARP_obs'],
                                 synthetic_immunoblot_data.data['time'])),
                             ('classifier', LogisticClassifier(
                                 synthetic_immunoblot_data,
                                 column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                                 do_fit_transform=False,
                                 classifier_type='ordinal_eoc'))])

a = 50
wb.process.get_step('classifier'). \
    set_params(**{'coefficients__cPARP_blot__coef_': np.array([a]),
                  'coefficients__cPARP_blot__theta_': np.array([0.03, 0.20, 0.97]) * a,
                  'coefficients__tBID_blot__coef_': np.array([a]),
                  'coefficients__tBID_blot__theta_': np.array([0.03, 0.4, 0.82, 0.97]) * a})
wb.run()

# -------- Calibration -------
# Model Inference via PyDREAM
sampled_params_0 = [SampledParam(expon, loc=0.0, scale=100.0),      # coefficients__tBID_blot__coef_    float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__tBID_blot__theta_1  float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__tBID_blot__theta_2  float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__tBID_blot__theta_3  float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__tBID_blot__theta_4  float
                    SampledParam(expon, loc=0.0, scale=100.0),      # coefficients__cPARP_blot__coef_   float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__cPARP_blot__theta_1 float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__cPARP_blot__theta_2 float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__cPARP_blot__theta_3 float
                    SampledParam(expon, loc=0.0, scale=0.25),       # coefficients__cPARP_blot__theta_4 float
                    ]

# sampled_params_0 = [
#                     SampledParam(uniform, loc=0.0, scale=100.0),          # coefficients__tBID_blot__coef_    float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_1  float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_2  float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_3  float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__tBID_blot__theta_4  float
#                     SampledParam(uniform, loc=0.0, scale=100.0),          # coefficients__cPARP_blot__coef_   float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__cPARP_blot__theta_1 float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__cPARP_blot__theta_2 float
#                     SampledParam(uniform, loc=0.0, scale=1.0),           # coefficients__cPARP_blot__theta_3 float
#                     ]

sampled_params_0 = [
                    SampledParam(cauchy, loc=50.0, scale=1.0),          # coefficients__tBID_blot__coef_    float
                    SampledParam(cauchy, loc=0.2, scale=0.005),           # coefficients__tBID_blot__theta_1  float
                    SampledParam(cauchy, loc=0.2, scale=0.005),           # coefficients__tBID_blot__theta_2  float
                    SampledParam(cauchy, loc=0.2, scale=0.005),           # coefficients__tBID_blot__theta_3  float
                    SampledParam(cauchy, loc=0.2, scale=0.005),           # coefficients__tBID_blot__theta_4  float
                    SampledParam(cauchy, loc=50.0, scale=1.0),          # coefficients__cPARP_blot__coef_   float
                    SampledParam(cauchy, loc=0.25, scale=0.005),           # coefficients__cPARP_blot__theta_1 float
                    SampledParam(cauchy, loc=0.25, scale=0.005),           # coefficients__cPARP_blot__theta_2 float
                    SampledParam(cauchy, loc=0.25, scale=0.005),           # coefficients__cPARP_blot__theta_3 float
                    ]
n_chains = 4
n_iterations = 10000  # iterations per file-save
burn_in_len = 5000   # number of iterations during burn-in
max_iterations = 10000
now = dt.datetime.now()
model_name = f'immunoblot_classifier_calibration_005_cauchy_priors_{now.year}{now.month}{now.day}'


# ------- Likelihood Function ------
@objective_function(immunoblot_model=wb, return_results=False, evals=0)
def likelihood(x):
    if any(xi <= 0 for xi in x):
        return -10000000.0

    # x = abs(x)
    c0 = x[0]
    t1 = x[1]
    t2 = t1 + x[2]
    t3 = t2 + x[3]
    t4 = t3 + x[4]

    c5 = x[5]
    t6 = x[6]
    t7 = t6 + x[7]
    t8 = t7 + x[8]

    print(likelihood.evals)
    print(x)

    likelihood.immunoblot_model.process.get_step('classifier').set_params(
        **{'coefficients__tBID_blot__coef_': np.array([c0]),
           'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
           'coefficients__cPARP_blot__coef_': np.array([c5]),
           'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})
    likelihood.evals += 1
    ll = -likelihood.immunoblot_model.likelihood()
    print(ll)
    return ll


if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       nCR=2,
                                       gamma_levels=3,
                                       adapt_gamma=True,
                                       history_thin=1,
                                       model_name=model_name,
                                       verbose=True,
                                       crossover_burnin=min(n_iterations, burn_in_len)
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
    if np.isnan(GR).any() or np.any(GR > 1.2) or n_iterations <= burn_in_len:
        # append sample with a re-run of the pyDream algorithm
        while not converged or (total_iterations < max_iterations):
            starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

            total_iterations += n_iterations
            sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                               likelihood=likelihood,
                                               niterations=n_iterations,
                                               nchains=n_chains,
                                               multitry=False,
                                               nCR=2,
                                               gamma_levels=3,
                                               adapt_gamma=True,
                                               history_thin=1,
                                               model_name=model_name,
                                               verbose=True,
                                               restart=True,  # restart at the last sampled position
                                               start=starts,
                                               crossover_burnin=min(n_iterations, burn_in_len)
                                               )

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
