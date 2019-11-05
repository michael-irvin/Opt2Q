# MW Irvin -- Lopez Lab -- 2018-10-23
import os
import pandas as pd
import numpy as np
import datetime as dt
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import Fluorescence
from opt2q.measurement.base.transforms import ScaleToMinMax
from opt2q.data import DataSet
from opt2q.calibrator import objective_function
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from apoptosis.kochen_ma_2019_apoptosis_model import model
from scipy.stats import norm, beta


# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'EC-RP_IMS-RP_IC-RP_data_for_models.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[
    ['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']
].rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace in column name

cPARP_dataset = DataSet(fluorescence_data[['time', 'norm_EC-RP']],
                        measured_variables={'norm_EC-RP': 'semi-quantitative'})
cPARP_dataset.measurement_error_df = fluorescence_data[['nrm_var_EC-RP']].\
    rename(columns={'nrm_var_EC-RP': 'norm_EC-RP__error'})

tBID_dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP']].dropna(),
                       measured_variables={'norm_IC-RP': 'semi-quantitative'})
tBID_dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error'})

# ------- Parameters --------
num_params = len(model.parameters)
param_values = pd.DataFrame({'value': [p.value for p in model.parameters],
                             'param': [p.name for p in model.parameters]})
param_model = NoiseModel(param_values)
parameters = param_model.run()
# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, fluorescence_data.time.max(), 1000))

# ------- Measurement -------
fl_cPARP = Fluorescence(results, dataset=cPARP_dataset, measured_values={'norm_EC-RP': ['ParpC_obs']},
                        observables=['ParpC_obs'])
fl_cPARP.process.add_step(('normalize', ScaleToMinMax(feature_range=(0, 1), columns=['ParpC_obs'], groupby=None,
                                                      do_fit_transform=True)))  # Scale all conditions jointly.

fl_tBid = Fluorescence(results, dataset=tBID_dataset, measured_values={'norm_IC-RP': ['BidT_obs']},
                       observables=['BidT_obs'])

measurement_results_cPARP = fl_cPARP.run()
measurement_results_tBid = fl_tBid.run()


# ------- Likelihood Function ------
@objective_function(pm=param_model, simulator=sim, measurements=[fl_cPARP, fl_tBid], return_results=False, evals=0)
def likelihood_fn(x):
    likelihood_fn.pm.update_values(pd.DataFrame({'value': [10**p for p in x[:62]],
                                                 'param': [p.name for p in model.parameters_rules()]}))

    params = likelihood_fn.pm.run()
    likelihood_fn.simulator.param_values = params

    # dynamics
    sim_results = likelihood_fn.simulator.run()
    # measurement
    likelihood_fn.evals += 1
    ll = 0.0
    try:
        for measurement_model in likelihood_fn.measurements:
            measurement_model.update_simulation_result(sim_results)
            ll -= measurement_model.likelihood()
    except (ValueError, ZeroDivisionError):
        return -1e10
    if np.isnan(ll):
        return -1e10
    else:
        print(likelihood_fn.evals)
        print(x)
        print(ll)
        return ll


# -------- Calibration -------
# Model Inference via PyDREAM
sampled_params_0 = [SampledParam(norm, loc=[np.log10(p.value) for p in model.parameters_rules()], scale=1.0)]

n_chains = 4
n_iterations = 20000  # iterations per file-save
burn_in_len = 10000    # number of iterations during burn-in
max_iterations = 20000
now = dt.datetime.now()
model_name = f'ECRP_ICRP_calibration_{now.year}{now.month}{now.day}'

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    dream_kwargs = {'parameters': sampled_params_0,
                    'likelihood': likelihood_fn,
                    'niterations': n_iterations,
                    'nchains': n_chains,
                    'multitry': False,
                    'nCR': 15,
                    'gamma_levels': 8,
                    'adapt_gamma': True,
                    'history_thin': 1,
                    'model_name': model_name,
                    'verbose': True,
                    'crossover_burnin': min(n_iterations, burn_in_len)}

    sampled_params, log_ps = run_dream(**dream_kwargs)

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
            sampled_params, log_ps = run_dream(restart=True,  # restart at the last sampled position
                                               start=starts,
                                               **dream_kwargs)

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
