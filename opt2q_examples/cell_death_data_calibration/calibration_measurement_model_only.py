import numpy as np
import datetime as dt
from scipy.stats import laplace, cauchy
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from opt2q.calibrator import objective_function
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import sim_results, pre_processing, true_params, set_up_classifier, synth_data
import time


use_cauchy = False

# Priors
nu = 100
noisy_param_stdev = 0.20

alpha = int(np.ceil(nu/2.0))
beta = alpha/noisy_param_stdev**2

sampled_params_0 = [SampledParam(laplace, loc=0.0, scale=1.0),  # slope   float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # intercept  float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # "Unrelated_Signal" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # "tBID_obs" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),  # "time" coef  float
                    ]  # coef are assigned in order by their column names' ASCII values

s = 2.0*np.exp(1)/(4.0*np.pi)  # convert above Laplace Priors to Cauchy priors with the same Entropy
cauchy_priors = [SampledParam(cauchy, loc=0.0, scale=1.0*s),      # slope   float
                 SampledParam(cauchy, loc=0.0, scale=0.1*s),      # intercept  float
                 SampledParam(cauchy, loc=0.0, scale=0.1*s),      # "Unrelated_Signal" coef  float
                 SampledParam(cauchy, loc=0.0, scale=0.1*s),      # "tBID_obs" coef  float
                 SampledParam(cauchy, loc=0.0, scale=0.1*s),      # "time" coef  float
                 ]  # coef are assigned in order by their column names' ASCII values

# Model name
now = dt.datetime.now()

if use_cauchy:
    model_name = f'apoptosis_model_tbid_cell_death_data_calibration_measurement_model_only_cauchy{now.year}{now.month}{now.day}'
    param_priors = cauchy_priors
else:
    model_name = f'apoptosis_model_tbid_cell_death_data_calibration_measurement_model_only_{now.year}{now.month}{now.day}'
    param_priors = sampled_params_0

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000

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


# likelihood function
@objective_function(sim_results=sim_results, pre_processing=pre_processing, classifier=classifier,
                    target=synth_data, return_results=False, evals=0)
def likelihood(x):
    try:
        start_time = time.time()
        new_results = likelihood.sim_results.opt2q_dataframe.reset_index()

        # run pre-processing
        features = likelihood.pre_processing(new_results)

        # update and run classifier
        n = -1
        slope_ = x[n + 1]
        intercept_ = x[n + 2] * slope_
        unr_coef_ = x[n + 3] * slope_
        tbid_coef_ = x[n + 4] * slope_
        time_coef_ = x[n + 5] * slope_

        likelihood.classifier.set_params(**{'coefficients__apoptosis__coef_': np.array([[unr_coef_, tbid_coef_, time_coef_]]),
                                            'coefficients__apoptosis__intercept_': np.array([intercept_]),
                                            'do_fit_transform': False})

        # run fixed classifier
        prediction = likelihood.classifier.transform(
            features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

        # calculate likelihood
        ll = sum(np.log(prediction[likelihood.target.apoptosis == 1]['apoptosis__1']))
        ll += sum(np.log(prediction[likelihood.target.apoptosis == 0]['apoptosis__0']))

        elapsed_time = time.time() - start_time
        print("Elapsed time: ", elapsed_time)

        print(x[:len(true_params)])
        print(likelihood.evals)
        print(ll)

        likelihood.evals += 1
        return ll
    except (ValueError, ZeroDivisionError, TypeError):
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


