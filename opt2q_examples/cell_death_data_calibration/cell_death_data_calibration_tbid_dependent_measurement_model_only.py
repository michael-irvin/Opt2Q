# MW Irvin -- Lopez Lab -- 2019-11-19

# Calibrate tBID dependent Apoptosis vs. Survival measurement model.

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import laplace
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import LogisticClassifier, Scale, ScaleGroups, Standardize
from opt2q.measurement.base.functions import derivative, where_max
from opt2q.calibrator import objective_function
from opt2q_examples.apoptosis_model import model


# ------- Synthetic Data ----
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

file_path = os.path.join(script_dir, 'synthetic_tbid_dependent_apoptosis_data.csv')
synth_data = pd.read_csv(file_path)

file_path = os.path.join(script_dir, 'true_params_extrinsic_noise.csv')
extrinsic_noise_params = pd.read_csv(file_path)

# ------- Simulations -------
# fluorescence data as reference
file_path = os.path.join(parent_dir, 'fluorescence_data_calibration', 'fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

sim = Simulator(model=model, param_values=extrinsic_noise_params, solver='cupsoda')
sim_results = sim.run(np.linspace(0, fluorescence_data.time.max(), 100))
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# ============ Create tBID dynamics etc. features ============
obs = 'tBID_obs'

ddx = ScaleGroups(columns=[obs], groupby='simulation', scale_fn=derivative) \
    .transform(results[[obs, 'Unrelated_Signal', 'cPARP_obs', 'time', 'simulation', 'TRAIL_conc']])
t_at_max = ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var': obs}).transform(ddx)
log_max_ddx = Scale(columns='tBID_obs', scale_fn='log10').transform(t_at_max)
std_tbid_features = Standardize(columns=['tBID_obs', 'time', 'Unrelated_Signal']).transform(log_max_ddx)

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
sampled_params_0 = [SampledParam(laplace, loc=0.0, scale=1.0),      # slope   float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # intercept  float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # "Unrelated_Signal" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # "tBID_obs" coef  float
                    SampledParam(laplace, loc=0.0, scale=0.1),      # "time" coef  float
                    ]  # coef are assigned in order by their column names' ASCII values

n_chains = 4
n_iterations = 10000  # iterations per file-save
burn_in_len = 5000   # number of iterations during burn-in
max_iterations = 10000
now = dt.datetime.now()
model_name = f'apoptosis_classifier_calibration_{now.year}{now.month}{now.day}'


# ------- Likelihood Function ------
@objective_function(lc=tbid_classifier, features=std_tbid_features, target=synth_data,
                    return_results=False, evals=0)
def likelihood(x):
    slope = x[0]
    intercept = x[1] * slope
    unr_coef = x[2] * slope
    tbid_coef = x[3] * slope
    time_coef = x[4] * slope

    likelihood.lc.set_params(**{'coefficients__apoptosis__coef_': np.array([[unr_coef, tbid_coef, time_coef]]),
                                'coefficients__apoptosis__intercept_': np.array([intercept]),
                                'do_fit_transform': False})

    prediction = likelihood.lc.transform(likelihood.features)

    ll = sum(np.log(prediction[likelihood.target.apoptosis == 1]['apoptosis__1']))
    ll += sum(np.log(prediction[likelihood.target.apoptosis == 0]['apoptosis__0']))

    print(x)
    print(likelihood.evals)
    print(ll)

    likelihood.evals += 1
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









