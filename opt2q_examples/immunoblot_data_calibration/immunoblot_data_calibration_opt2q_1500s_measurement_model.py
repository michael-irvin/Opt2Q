# MW Irvin -- Lopez Lab -- 2019-11-20

# Calibrating the apoptosis model to immunoblot data using variable measurement model

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm, expon
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import WesternBlot
from opt2q.measurement.base.transforms import Pipeline, ScaleToMinMax, Interpolate, LogisticClassifier
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
# from opt2q_examples.immunoblot_data_calibration.generate_synthetic_immunoblot_dataset import synthetic_immunoblot_data
from opt2q_examples.apoptosis_model import model
import pickle


with open(f'synthetic_WB_dataset_1500s_2020_12_3.pkl', 'rb') as data_input:
    synthetic_immunoblot_data = pickle.load(data_input)


param_names = [p.name for p in model.parameters_rules()][:-6]
num_params = len(param_names)
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
true_params = np.load('true_params.npy')[:num_params]

params = pd.DataFrame({'value': [10**p for p in true_params], 'param': param_names})
parameters = NoiseModel(params).run()  # No extrinsic noise applied

# ------- Simulations -------
# sim = Simulator(model=model, param_values=parameters, solver='cupsoda', integrator_options={'vol': 4.0e-15})
sim = Simulator(model=model, param_values=parameters, solver='scipyode', solver_options={'integrator': 'lsoda'},
                integrator_options={'rtol': 1e-3, 'atol': 1e-1})  # effort to speed-up solver

sim_results = sim.run(np.linspace(0, synthetic_immunoblot_data.data.time.max(), 100))

results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

wb = WesternBlot(simulation_result=sim_results,
                 dataset=synthetic_immunoblot_data,
                 measured_values={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                 observables=['tBID_obs', 'cPARP_obs'])

wb.process = Pipeline(steps=[('x_scaled', ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'])),
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


# -------- Calibration -------
# Model Inference via PyDREAM
# Use recent calibration as starting point
sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),     # rate parameters floats
                    SampledParam(expon, loc=0.0, scale=100.0),          # coefficients__tBID_blot__coef_    float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_1  float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_2  float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_3  float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__tBID_blot__theta_4  float
                    SampledParam(expon, loc=0.0, scale=100.0),          # coefficients__cPARP_blot__coef_   float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_1 float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_2 float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_3 float
                    SampledParam(expon, loc=0.0, scale=0.25),           # coefficients__cPARP_blot__theta_4 float
                    ]

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000
now = dt.datetime.now()
model_name = f'apoptosis_params_and_immunoblot_classifier_calibration_1500s_{now.year}{now.month}{now.day}'


# ------- Likelihood Function ------
@objective_function(simulator=sim, immunoblot_model=wb, return_results=False, evals=0)
def likelihood_fn(x):
    new_params = pd.DataFrame([[10 ** p for p in x[:num_params]]], columns=param_names)

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
    new_results = likelihood_fn.simulator.run()

    # measurement
    likelihood_fn.immunoblot_model.update_simulation_result(new_results)
    likelihood_fn.immunoblot_model.process.get_step('classifier').set_params(
        **{'coefficients__tBID_blot__coef_': np.array([c0]),
           'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
           'coefficients__cPARP_blot__coef_': np.array([c5]),
           'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})

    print(likelihood_fn.evals)
    print(x)
    likelihood_fn.evals += 1

    try:
        ll = -likelihood_fn.immunoblot_model.likelihood()
    except (ValueError, ZeroDivisionError):
        return -1e10

    if np.isnan(ll):
        return -1e10
    else:
        print(ll)
        return ll


# x_ = [-8.13475153e+00,  9.17708787e+00 , 4.16611420e+00 ,-1.75471230e+01 ,
#       -9.52928418e+00,  9.32671653e-01, -1.21824596e+01, -3.48615289e+01,
#       -1.51397095e+01, -3.20696741e+01,  1.08640759e+01,  1.10822178e+01,
#       -1.38664377e+01, -2.72909594e+01, -1.73126888e+01, -1.55647822e+01,
#       -1.04773552e+01, -1.05649037e+01, -4.53471226e+00, -1.24367160e+01,
#        9.76079853e-01, -1.36115104e+00, -2.14204777e+01, -1.28266823e+01,
#       -2.05758431e+01, -2.25599947e+01, -7.94466216e-01,  1.06393724e+01,
#        3.45066195e+02,  3.31410222e-01,  2.22110330e+00,  1.23686260e+00,
#        2.21589231e+00,  5.98761382e+02,  2.26288920e-02,  2.87476277e+00,
#        1.38313381e+00,  2.80363829e+00]
# likelihood_fn(x_)
# quit()

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


