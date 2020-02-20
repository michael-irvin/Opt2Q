# MW Irvin -- Lopez Lab -- 2019-11-27

# Calibrating the apoptosis model to cell death data using Opt2Q measurement model

import os
import numpy as np
import pandas as pd
import datetime as dt
from multiprocessing import current_process
from scipy.stats import norm, invgamma, laplace
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import LogisticClassifier, Scale, ScaleGroups, Standardize
from opt2q.measurement.base.functions import derivative, where_max
from opt2q.calibrator import objective_function
from opt2q_examples.apoptosis_model import model
from opt2q_examples.generate_synthetic_cell_death_dataset import noisy_param_names


# ------- Synthetic Data ----
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(script_dir, 'synthetic_tbid_dependent_apoptosis_data_large.csv')
synth_data = pd.read_csv(file_path)

file_path = os.path.join(script_dir, 'true_params_extrinsic_noise_large.csv')
extrinsic_noise_params = pd.read_csv(file_path)

# ------- Starting Point ----
param_names = [p.name for p in model.parameters_rules()]
true_params = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'true_params.npy'))

# ============ Simulate Heterogeneous Population =============
# divide extrinsic noise columns by the 'true_value' or model preset value (m0) to get the population for each column
model_presets = pd.DataFrame({p.name: [p.value] for p in model.parameters if p.name in noisy_param_names})
starting_params = pd.DataFrame([10**true_params], columns=param_names)
model_presets.update(starting_params)  # m0

standard_population = (extrinsic_noise_params[noisy_param_names].values - model_presets[noisy_param_names].values) \
                      / (model_presets[noisy_param_names].values * 0.23263813098020095)


def simulate_heterogeneous_population(m, cv, population_0=standard_population):
    # scale the extrinsic noise to a population centered at 0 (the scale is 1).
    population = cv * m.values * population_0 + m.values
    return population


# ------- Simulations -------
# fluorescence data as reference
file_path = os.path.join(script_dir, '../fluorescence_data_calibration/fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

time_axis = np.linspace(0, fluorescence_data.time.max(), 100)
sim = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver='cupsoda',
                integrator_options={'vol': 4.0e-15})

sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})


# ============ Create tBID dynamics etc. features ============
def pre_processing(sim_res):
    obs = 'tBID_obs'

    ddx = ScaleGroups(columns=[obs], groupby='simulation', scale_fn=derivative) \
        .transform(sim_res[[obs, 'Unrelated_Signal', 'cPARP_obs', 'time', 'simulation', 'TRAIL_conc']])
    t_at_max = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var': obs}).transform(ddx)
    log_max_ddx = Scale(columns='tBID_obs', scale_fn='log10').transform(t_at_max)
    standardized_features = Standardize(columns=['tBID_obs', 'time', 'Unrelated_Signal']).transform(log_max_ddx)
    return standardized_features


std_tbid_features = pre_processing(results)

# ============ Classify tBID into survival and death cell-fates =======
# Synthetic dataset
tbid_0s_1s = pd.DataFrame({'apoptosis': [0, 1, 0, 1],
                           'TRAIL_conc': ['50ng/mL', '50ng/mL', '10ng/mL', '10ng/mL'],
                           'simulation': [48, 49, 50, 51]})  # minimum dataset for setting up supervised ML classifier

tbid_classifier = LogisticClassifier(tbid_0s_1s,
                                     column_groups={'apoptosis': ['tBID_obs', 'time', 'Unrelated_Signal']},
                                     classifier_type='nominal')
tbid_classifier.transform(std_tbid_features.iloc[48:52].reset_index(drop=True)
                          [['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])
a = 4
tbid_classifier.set_params(**{'coefficients__apoptosis__coef_': np.array([[0.0, 0.25, -1.0]])*a,
                              'coefficients__apoptosis__intercept_': np.array([-0.25])*a,
                              'do_fit_transform': False})

# Synthetic data based on tBID features
tbid_predictions = tbid_classifier.transform(
    std_tbid_features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

# -------- Calibration -------
# Model Inference via PyDREAM
nu = 100
noisy_param_stdev = 0.235

alpha = int(np.ceil(nu/2.0))
beta = alpha/noisy_param_stdev**2

# d = invgamma(alpha, scale=beta)

sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),
                    SampledParam(invgamma, *[alpha], scale=beta),
                    SampledParam(laplace, loc=0.0, scale=1.0),      # slope   float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # intercept  float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # "Unrelated_Signal" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # "tBID_obs" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # "time" coef  float
                    ]  # coef are assigned in order by their column names' ASCII values

n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 50000   # number of iterations during burn-in
max_iterations = 100000
now = dt.datetime.now()
model_name = f'apoptosis_model_tbid_cell_death_data_calibration_{now.year}{now.month}{now.day}'


# ------- Likelihood Function ------#
@objective_function(target=synth_data, sim=sim, pre_processing=pre_processing, lc=tbid_classifier,
                    return_results=False, evals=0)
def likelihood(x):
    # Adjust heterogeneous population based on new parameter values and coefficient of variation
    new_rate_params = pd.DataFrame([10 ** np.array(x[:len(param_names)])], columns=param_names).iloc[
        np.repeat(0, 100)].reset_index(drop=True)
    n = len(param_names)
    cv_term = abs(x[n])**-0.5

    likelihood.model_presets.update(new_rate_params.iloc[0:1])
    noisy_params = likelihood.sim_population(likelihood.model_presets, cv=cv_term)

    params_df.update(new_rate_params)
    params_df.update(pd.DataFrame(noisy_params, columns=noisy_param_names))

    try:
        # simulate dynamics using params_df as parameters
        likelihood.sim.param_values = params_df
        if hasattr(likelihood.sim.solver, 'gpu') or True:
            # process_id = current_process().ident % 4
            # likelihood.sim.sim.gpu = [process_id]
            likelihood.sim.sim.gpu = [0]

        new_results = likelihood.sim.run().opt2q_dataframe.reset_index()

        # run pre-processing
        features = pre_processing(new_results)

        # update and run classifier
        slope = x[n+1]
        intercept = x[n+2] * slope
        unr_coef = x[n+3] * slope
        tbid_coef = x[n+4] * slope
        time_coef = x[n+5] * slope

        likelihood.lc.set_params(**{'coefficients__apoptosis__coef_': np.array([[unr_coef, tbid_coef, time_coef]]),
                                    'coefficients__apoptosis__intercept_': np.array([intercept]),
                                    'do_fit_transform': False})

        prediction = likelihood.lc.transform(features[['simulation', 'tBID_obs', 'time', 'Unrelated_Signal', 'TRAIL_conc']])

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


# Additional attributes of the objective function
params_df = extrinsic_noise_params.copy()
likelihood.model_presets = model_presets
likelihood.sim_population = simulate_heterogeneous_population

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


