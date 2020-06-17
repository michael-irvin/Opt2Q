import numpy as np
import datetime as dt
from scipy.stats import norm, invgamma
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from multiprocessing import current_process
from opt2q.calibrator import objective_function
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import shift_and_scale_heterogeneous_population_to_new_params as sim_population
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import set_up_simulator, pre_processing, true_params, set_up_classifier, synth_data, \
    time_axis, handle_timeouts, TimeoutException
from pysb.simulator import ScipyOdeSimulator
from opt2q_examples.apoptosis_model import model
import time
import signal


# Model name
now = dt.datetime.now()
model_name = f'apoptosis_model_tbid_cell_death_data_calibration_fmm_{now.year}{now.month}{now.day}'

# Priors
nu = 100
noisy_param_stdev = 0.20

alpha = int(np.ceil(nu/2.0))
beta = alpha/noisy_param_stdev**2

sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),
                    SampledParam(invgamma, *[alpha], scale=beta)]

n_chains = 4
n_iterations = 10000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000

# Simulator
# opt2q_solver doesn't run on Power9, but has useful methods for handling simulation results
opt2q_solver = set_up_simulator('cupsoda')
delattr(opt2q_solver, 'sim')
delattr(opt2q_solver, 'solver')

solver = ScipyOdeSimulator(model, tspan=time_axis, **{'integrator': 'lsoda', 'integrator_options': {'mxstep': 2**20}})

# Measurement Model
slope = 4
intercept = slope * -0.25  # Intercept (-0.25)
unr_coef = slope * 0.00  # "Unrelated_Signal" coef (0.00)
tbid_coef = slope * 0.25  # "tBID_obs" coef  (0.25)
time_coef = slope * -1.00  # "time" coef  (-1.00)

classifier = set_up_classifier()
classifier.set_params(**{'coefficients__apoptosis__coef_': np.array([[unr_coef, tbid_coef, time_coef]]),
                         'coefficients__apoptosis__intercept_': np.array([intercept]),
                         'do_fit_transform': False})

# Register the signal function handler
signal.signal(signal.SIGALRM, handle_timeouts)


# likelihood function
def likelihood(x):
    params_df = sim_population(x)  # simulate heterogeneous population around new param values
    opt2q_solver.param_values = params_df
    opt2q_solver.params_df = params_df

    # Add scipyodesolver using parameter values from Opt2Q solver
    params_array = opt2q_solver._param_values_run

    start_time = time.time()
    try:
        signal.alarm(90)
        results = solver.run(param_values=params_array, num_processors=2)  # run model
        signal.alarm(0)

        new_results = opt2q_solver.opt2q_dataframe(results.dataframe).reset_index()

        features = pre_processing(new_results)

        # run fixed classifier
        prediction = classifier.transform(
            features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

        # calculate likelihood
        ll = sum(np.log(prediction[synth_data.apoptosis == 1]['apoptosis__1']))
        ll += sum(np.log(prediction[synth_data.apoptosis == 0]['apoptosis__0']))

        elapsed_time = time.time() - start_time
        print("Elapsed time: ", elapsed_time)
        print(x[:len(true_params)])
        print(ll)

        return ll

    except (ValueError, ZeroDivisionError, TypeError, TimeoutException):
        elapsed_time = time.time() - start_time
        print("Elapsed time: ", elapsed_time)
        print(x[:len(true_params)])
        return -1e10


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
                                       likelihood=likelihood,
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
                                               likelihood=likelihood,
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


