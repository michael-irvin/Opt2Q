# MW Irvin -- Lopez Lab -- 2019-12-06

# Calibrating the apoptosis model to time elapsed before death using a fixed but incorrect measurement model

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import norm, invgamma, truncnorm
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import ScaleToMinMax, Interpolate
from opt2q.calibrator import objective_function
from measurement_model_demo.apoptosis_model import model
from measurement_model_demo.generate_synthetic_cell_death_dataset import noisy_param_names


# ------- Starting Point ----
param_names = [p.name for p in model.parameters_rules()][:-6]  # exclude parameters from unrelated reactions
true_params = np.load('true_params.npy')[:len(param_names)]

# ------- Synthetic Data ----
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'synthetic_cPARP_dependent_apoptosis_data_noisy_threshold_model.csv')
synth_data = pd.read_csv(file_path)

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'true_params_extrinsic_noise.csv')
extrinsic_noise_params = pd.read_csv(file_path)

# ============ Simulate Heterogeneous Population =============
# divide extrinsic noise columns by the 'true_value' or model preset value (m0) to get the population for each column
model_presets = pd.DataFrame({p.name: [p.value] for p in model.parameters if p.name in noisy_param_names})
starting_params = pd.DataFrame([10**true_params], columns=param_names)
model_presets.update(starting_params)  # m0

standard_population = extrinsic_noise_params[noisy_param_names].values/model_presets[noisy_param_names].values


def simulate_heterogeneous_population(m, cv, population_0=standard_population):
    # scale the extrinsic noise to a population centered at 1 (the scale is 20%).
    population = population_0 ** cv/0.2  # scale population from 20% to cv
    population *= m.values  # shift population from m0 to m
    return population


# ------- Simulations -------
# fluorescence data as reference
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

time_axis = np.linspace(0, fluorescence_data.time.max(), 100)
sim = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver='cupsoda',
                integrator_options={'vol': 4.0e-15})

sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# ============ PARP cleavage dynamics feature ============
norm_x = ScaleToMinMax(feature_range=(0, 1), columns=['cPARP_obs'], groupby=['TRAIL_conc'])\
    .transform(results[['cPARP_obs', 'time', 'TRAIL_conc', 'simulation']])
lethal_cPARP_amt = Interpolate('time', 'cPARP_obs', synth_data)\
    .transform(norm_x)  # place PCD threshold at 10%

# ============ Truncated Normal Distribution of PARP cleavage lethality thresholds =============
threshold_mean = 0.5
threshold_stdev = 0.02
a, b = (0.0 - threshold_mean) / threshold_stdev, (1.0 - threshold_mean) / threshold_stdev

# ------- Priors -----------
# Heterogeneous population standard deviation term.
nu = 100  # Sample size
noisy_param_stdev = 0.235  # Extrinsic variability in protein expression is 15-30%

alpha = int(np.ceil(nu/2.0))
beta = alpha/noisy_param_stdev**2

sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),
                    SampledParam(invgamma, *[alpha], scale=beta)
                    ]


# ------- Likelihood Function ------
@objective_function(simulator=sim, a=a, b=b, mu=threshold_mean, sig=threshold_stdev, evals=0)
def likelihood(x):
    # Adjust heterogeneous population based on new parameter values and coefficient of variation
    new_rate_params = pd.DataFrame([10 ** np.array(x[:len(param_names)])], columns=param_names).iloc[
        np.repeat(0, 100)].reset_index(drop=True)
    n = len(param_names)
    cv_term = abs(x[n]) ** -0.5

    likelihood.model_presets.update(new_rate_params.iloc[0:1])
    noisy_params = likelihood.sim_population(likelihood.model_presets, cv=cv_term)

    likelihood.params_df.update(new_rate_params)
    likelihood.params_df.update(pd.DataFrame(noisy_params, columns=noisy_param_names))

    # dynamics
    if hasattr(likelihood.simulator.sim, 'gpu'):
        # process_id = current_process().ident % 4
        # likelihood.simulator.sim.gpu = [process_id]
        likelihood.simulator.sim.gpu = [0]
    sim_res = likelihood.simulator.run()
    res = sim_res.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

    normed_x = ScaleToMinMax(feature_range=(0, 1), columns=['cPARP_obs'], groupby=['TRAIL_conc']) \
        .transform(res[['cPARP_obs', 'time', 'TRAIL_conc', 'simulation']])
    obs_amt = Interpolate('time', 'cPARP_obs', synth_data).transform(normed_x)

    ll = sum(truncnorm.logpdf(obs_amt.cPARP_obs, likelihood.a, likelihood.b, loc=likelihood.mu, scale=likelihood.sig))

    likelihood.evals += 1

    if np.isnan(ll):
        return -1e10
    else:
        print(likelihood.evals)
        print(x)
        print(ll)
        return ll


likelihood.params_df = extrinsic_noise_params.copy()
likelihood.model_presets = model_presets
likelihood.sim_population = simulate_heterogeneous_population

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000
now = dt.datetime.now()
model_name = f'time_elapsed_before_apoptosis_calibration_fmm_inc{now.year}{now.month}{now.day}'

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


