# MW Irvin -- Lopez Lab -- 2019-11-20

# Calibrating the apoptosis model to immunoblot data using variable measurement model

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm, cauchy
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin

from opt2q.simulator import Simulator
from opt2q.measurement import WesternBlot
from opt2q.measurement.base.transforms import Pipeline, ScaleToMinMax, Interpolate, LogisticClassifier
from opt2q.calibrator import objective_function
from opt2q.data import DataSet
from opt2q_examples.apoptosis_model import model

param_names = [p.name for p in model.parameters_rules()][:-6]
param_values = [np.log10(p.value) for p in model.parameters_rules()][:-6]

parameters = pd.DataFrame([[10**p for p in param_values]], columns=param_names)
parameters['Initial_TRAIL'] = '50ng/mL'  # experimental Conditions column

# ------- Dataset -----------
blot_data = pd.DataFrame({'time': [0, 1800, 3600, 5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800],
                          'tBID_blot': [0, 1, 0, 0, 1, 1, 2, 4, 4, 4, 3, 4],  # Ordinal Categories from WB image
                         })

blot_data['Initial_TRAIL'] = '50ng/mL'  # Annotate the data with an experimental condition column
# Note: experimental conditions annotated in the data must also be in the simulation result. This is done by annotating
# the model parameters (see above).
# Also note: experimental conditions columns are not necessary, but can be useful for organizing multiple datasets.

blot_dataset = DataSet(blot_data,
                       measured_variables={'tBID_blot': 'ordinal'},  # dependent variable columns in your dataset.
                       measurement_error=0.1  # 10% misclassification rate
                       # additional columns in your dataset are allowed e.g., to annotate experimental conditions.
                       )

# ------- Simulations -------
# Opt2Q simulator can take and return dataframes, and can annotate experimental conditions.
sim = Simulator(model=model, param_values=parameters, solver='scipyode', tspan=np.linspace(0, 19800, 100),
                solver_options={'integrator': 'lsoda'},
                integrator_options={'mxstep':2**10})  # effort to speed-up solver
sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})


wb = WesternBlot(simulation_result=sim_results,
                 dataset=blot_dataset,
                 measured_values={'tBID_blot': ['tBID_obs']},  # Column in data that corresponds to observables in model
                 observables=['tBID_obs'])

# Add a pre-processing pipeline.
wb.process = Pipeline(steps=[('x_scaled', ScaleToMinMax(columns=['tBID_obs'])),  # scale output to 0-1
                             ('x_int', Interpolate('time', ['tBID_obs'],  # interpolate (time=IV, tBID_obs=DV)
                                                   blot_data['time'])),
                             ('classifier', LogisticClassifier(
                                 blot_dataset,
                                 column_groups={'tBID_blot': ['tBID_obs']},
                                 do_fit_transform=False,  # Don't fit-transform since we calibrate the classifier below
                                 classifier_type='ordinal_eoc'))  # empirical ordering constraint
                             ])

wb.run()
# print(wb.process.get_params())
# Notice the parameters that begin with 'classifier'. We will make these free-parameters that are calibrated along with
# the apoptosis model.
# The print-out should have something like this:
# 'classifier__coefficients__tBID_blot__coef_': array([1.94956948]),
# 'classifier__coefficients__tBID_blot__theta_': array([-0.71406479,  1.08190499,  2.43190302,  4.09242425]),
# coef_ is the slope parameter. Since we are using 'ordinal_eoc', this term must be greater than zero.
# theta_ lists the boundaries between adjacent categories. We have 5 categories and therefore 4 boundaries.
#   The boundaries should be monotonically increasing.

# -------- Calibration -------
# Model Inference via PyDREAM
# Use recent calibration as starting point
sampled_params_0 = [SampledParam(norm, loc=param_values, scale=1.5),     # rate parameters floats
                    # Place priors on the slope and boundaries parameters.
                    SampledParam(cauchy, loc=50.0, scale=10.0),  # coefficients__tBID_blot__coef_    float
                    SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_1  float
                    SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_2  float
                    SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_3  float
                    SampledParam(cauchy, loc=0.2, scale=0.05),  # coefficients__tBID_blot__theta_4  float
                    ]

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000
now = dt.datetime.now()
model_name = f'apoptosis_params_and_immunoblot_classifier_calibration_{now.year}{now.month}{now.day}'


# ------- Likelihood Function ------
n_params = len(param_values)


@objective_function(simulator=sim, immunoblot_model=wb, return_results=False, evals=0)
def likelihood_fn(x):
    # x is a concatenation of the apoptosis-model parameters and the classifier parameters.
    new_params = pd.DataFrame([[10 ** p for p in x[:n_params]]+['50ng/mL']], columns=param_names+['Initial_TRAIL'])

    # classifier
    if any(xi < 0 for xi in x[n_params:]):
        return -np.inf  # negative terms violate empirical order constraints

    c0 = x[n_params+0]
    t1 = x[n_params+1]
    t2 = t1 + x[n_params+2]  # Doing these successive sums helps me insure the boundaries are monotonic increasing
    t3 = t2 + x[n_params+3]
    t4 = t3 + x[n_params+4]

    likelihood_fn.simulator.param_values = new_params  # Set new parameters

    # dynamics
    new_results = likelihood_fn.simulator.run()

    # measurement
    likelihood_fn.immunoblot_model.update_simulation_result(new_results)  # Update the immunoblot model with this
    likelihood_fn.immunoblot_model.process.get_step('classifier').set_params(  # set the parameters of the classifier
        **{'coefficients__tBID_blot__coef_': np.array([c0]),
           'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0})
    #       the product of slope-boundary is a convenient parameterization scheme

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
#        2.21589231e+00]
#
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


