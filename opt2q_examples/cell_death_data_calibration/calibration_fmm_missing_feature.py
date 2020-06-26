import numpy as np
import datetime as dt
from scipy.stats import norm, invgamma
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from multiprocessing import current_process
from multiprocess.context import TimeoutError
from multiprocess.pool import Pool
from opt2q.calibrator import objective_function
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import shift_and_scale_heterogeneous_population_to_new_params as sim_population
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import set_up_simulator, pre_processing, true_params, set_up_classifier, synth_data
import time

# Model name
now = dt.datetime.now()
model_name = f'apoptosis_model_tbid_cell_death_data_calibration_fmm_missing_feature_{now.year}{now.month}{now.day}'

# Priors
nu = 100
noisy_param_stdev = 0.20

alpha = int(np.ceil(nu/2.0))
beta = alpha/noisy_param_stdev**2

sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),
                    SampledParam(invgamma, *[alpha], scale=beta)]

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000

# Simulator
sim = set_up_simulator('cupsoda')

# Measurement Model
slope = 4
intercept = slope * -0.25  # Intercept (-0.25)
unr_coef = slope * 0.00  # "Unrelated_Signal" coef (0.00)
tbid_coef = slope * 0.25  # "tBID_obs" coef  (0.25)
time_coef = slope * 0.00  # "time" coef  (-1.00 --> 0.00) missing feature

classifier = set_up_classifier()
classifier.set_params(**{'coefficients__apoptosis__coef_': np.array([[unr_coef, tbid_coef, time_coef]]),
                         'coefficients__apoptosis__intercept_': np.array([intercept]),
                         'do_fit_transform': False})


# likelihood function
@objective_function(gen_param_df=sim_population, sim=sim, pre_processing=pre_processing, classifier=classifier,
                    target=synth_data, return_results=False, evals=0)
def likelihood(x):
    params_df = likelihood.gen_param_df(x)  # simulate heterogeneous population around new param values
    likelihood.sim.param_values = params_df

    try:
        if hasattr(likelihood.sim.sim, 'gpu'):
            process_id = current_process().ident % 4
            likelihood.sim.sim.gpu = [process_id]

            # likelihood.sim.sim.gpu = [1]
        new_results = likelihood.sim.run().opt2q_dataframe.reset_index()

        # run pre-processing
        features = likelihood.pre_processing(new_results)

        # run fixed classifier
        prediction = likelihood.classifier.transform(
            features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])
        likelihood.prediction = prediction

        # calculate likelihood
        ll = sum(np.log(prediction[likelihood.target.apoptosis == 1]['apoptosis__1']))
        ll += sum(np.log(prediction[likelihood.target.apoptosis == 0]['apoptosis__0']))

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


