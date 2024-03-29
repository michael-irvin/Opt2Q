# MW Irvin -- Lopez Lab -- 2019-11-24

# Calibrating the apoptosis model to immunoblot data using fixed measurement model with incorrect parameters

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import WesternBlot
from opt2q.measurement.base.transforms import Pipeline, ScaleToMinMax, Interpolate, LogisticClassifier
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from opt2q_examples.immunoblot_data_calibration.generate_synthetic_immunoblot_dataset import synthetic_immunoblot_data
from opt2q_examples.apoptosis_model import model


param_names = [p.name for p in model.parameters_rules()][:-6]  # exclude parameters from unrelated reactions

script_dir = os.path.dirname(__file__)
true_params = np.load('true_params.npy')[:len(param_names)]

params = pd.DataFrame({'value': [10**p for p in true_params], 'param': param_names})
parameters = NoiseModel(params).run()  # No extrinsic noise applied

# ------- Simulations -------
# sim = Simulator(model=model, param_values=parameters, solver='cupsoda', integrator_options={'vol': 4.0e-15})
sim = Simulator(model=model, param_values=parameters, solver='scipyode', solver_options={'integrator': 'lsoda'})

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

wb.run()

a = 50
# wb.process.get_step('classifier')\
#     .set_params(** {'coefficients__cPARP_blot__coef_': np.array([a]),  # incorrect thresholds
#                     'coefficients__cPARP_blot__theta_': np.array([0.00,  0.5, 1.0])*a,
#                     'coefficients__tBID_blot__coef_': np.array([a]),
#                     'coefficients__tBID_blot__theta_': np.array([0.00,  0.37,  0.67, 1.0])*a,
#                     'do_fit_transform': False})

wb.process.get_step('classifier')\
    .set_params(** {'coefficients__cPARP_blot__coef_': np.array([a]),  # incorrect thresholds
                    'coefficients__cPARP_blot__theta_': np.array([0.25,  0.5, 0.75])*a,
                    'coefficients__tBID_blot__coef_': np.array([a]),
                    'coefficients__tBID_blot__theta_': np.array([0.20,  0.40,  0.60, 0.8])*a,
                    'do_fit_transform': False})

# -------- Calibration -------
# Model Inference via PyDREAM
# Use recent calibration as starting point
sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5)]


# ------- Likelihood Function ------
@objective_function(simulator=sim, measurement_model=wb, return_results=False, evals=0)
def likelihood_fn(x):
    new_params = pd.DataFrame([[10**p for p in x]], columns=param_names)
    likelihood_fn.simulator.param_values = new_params

    # dynamics
    if hasattr(likelihood_fn.simulator.sim, 'gpu'):
        # process_id = current_process().ident % 4
        # likelihood.simulator.sim.gpu = [process_id]
        likelihood_fn.simulator.sim.gpu = [1]
    new_results = likelihood_fn.simulator.run()

    # measurement
    likelihood_fn.measurement_model.update_simulation_result(new_results)
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


n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000
now = dt.datetime.now()
# model_name = f'immunoblot_data_calibration_fmm_inc_{now.year}{now.month}{now.day}'
model_name = f'immunoblot_data_calibration_fmm_inc_v2_{now.year}{now.month}{now.day}'

if __name__ == '__main__':
    ncr = 25
    gamma_levels = 8
    p_gamma_unity = 0.1
    print(ncr, gamma_levels, p_gamma_unity)
    print(wb.process.get_step('classifier').get_params())

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


