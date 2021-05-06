import numpy as np
import datetime as dt
import pandas as pd
from pydream.parameters import SampledParam
from scipy.stats import norm, expon
from opt2q_examples.mixed_dataset.calibration_setup import set_up_simulator as immunoblot_set_up_sim, set_up_immunoblot
from opt2q_examples.apoptosis_model import model
from opt2q.calibrator import objective_function
from opt2q.measurement.base.likelihood import categorical_dist_likelihood as likelihood_f
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
import pickle

# Model name
now = dt.datetime.now()
model_name = f'apoptosis_model_disc_immunoblot_opt2q_{now.year}{now.month}{now.day}'

# Priors
true_params = np.load('true_params.npy')

sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),  # model priors
                    SampledParam(expon, loc=0.0, scale=100.0),  # coefficients__IC_DISC_localization__coef_    float
                    SampledParam(expon, loc=0.0, scale=0.25),   # coefficients__IC_DISC_localization__theta_1  float
                    SampledParam(expon, loc=0.0, scale=0.25),   # coefficients__IC_DISC_localization__theta_2  float
                    SampledParam(expon, loc=0.0, scale=0.25),   # coefficients__IC_DISC_localization__theta_3  float
                    ]

# PyDREAM Settings
n_chains = 4
n_iterations = 20000  # iterations per file-save
burn_in_len = 140000   # number of iterations during burn-in
max_iterations = 180000

# Simulator
sim = immunoblot_set_up_sim(model)

# Measurement Model
with open(f'synthetic_IC_DISC_localization_blot_dataset_2020_10_18.pkl', 'rb') as data_input:
    immunoblot_dataset = pickle.load(data_input)

immunoblot = set_up_immunoblot(sim, immunoblot_dataset)


# likelihood function
param_names = [p.name for p in model.parameters_rules()][:-6]
num_params = len(param_names)


@objective_function(sim_wb=sim, immunoblot_model=immunoblot, return_results=False, evals=0)
def likelihood(x):
    ll = 0
    try:
        # run immunoblot data
        new_params = pd.DataFrame([[10 ** p for p in x[:num_params]]], columns=param_names)

        # classifier
        c0 = x[-4]
        t1 = x[-3]
        t2 = t1 + x[-2]
        t3 = t2 + x[-1]

        likelihood.sim_wb.param_values = new_params
        # dynamics
        if hasattr(likelihood.sim_wb.sim, 'gpu'):
            likelihood.sim_wb.sim.gpu = [0]

        # dynamics
        new_results = likelihood.sim_wb.run().opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
        likelihood.new_results_immuno = new_results
        # measurement
        likelihood.immunoblot_model.get_step('classifier').set_params(
            **{'coefficients__IC_DISC_localization__coef_': np.array([c0]),
               'coefficients__IC_DISC_localization__theta_': np.array([t1, t2, t3]) * c0})
        prediction_im = likelihood.immunoblot_model.transform(new_results[['time', 'C8_DISC_recruitment_obs']])

        print(likelihood.evals)
        print(x)
        likelihood.evals += 1

        ll += -likelihood_f(prediction_im, immunoblot_dataset)

    except (ValueError, ZeroDivisionError, TypeError):
        return -1e10

    if not np.isfinite(ll):
        return -1e10

    else:
        print(ll)
        return ll


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
