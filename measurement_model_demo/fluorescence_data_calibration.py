# MW Irvin -- Lopez Lab -- 2018-10-09
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
from measurement_model_demo.apoptosis_model import model
from matplotlib import pyplot as plt
from scipy.stats import norm, beta, uniform


# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')

raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[
    ['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP', 'Condition']
].rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

cPARP_dataset = DataSet(fluorescence_data[['time', 'norm_EC-RP', 'Condition']],
                        measured_variables={'norm_EC-RP': 'semi-quantitative'})
cPARP_dataset.measurement_error_df = fluorescence_data[['nrm_var_EC-RP', 'Condition']].\
    rename(columns={'nrm_var_EC-RP': 'norm_EC-RP__error'})

tBID_dataset = DataSet(fluorescence_data[['time', 'norm_IC-RP', 'Condition']].dropna(),
                       measured_variables={'norm_IC-RP': 'semi-quantitative'})
tBID_dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'Condition']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error'})

# ------- Parameters --------
num_params = len(model.parameters)
param_values = pd.DataFrame({'value': [p.value for p in model.parameters]*2,
                             'param': [p.name for p in model.parameters]*2,
                             'Condition': ['Control']*num_params + ['BidKD']*num_params})
param_model = NoiseModel(param_values)
param_model.update_values(pd.DataFrame([[1.2e4, 'Bid_0', 'BidKD']], columns=['value', 'param', 'Condition']))
parameters = param_model.run()

# ------- Dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, fluorescence_data.time.max(), 100))

# ------- Measurement -------
fl_cPARP = Fluorescence(results, dataset=cPARP_dataset, measured_values={'norm_EC-RP': ['cPARP_obs']},
                        observables=['cPARP_obs'])
fl_cPARP.process.add_step(('normalize', ScaleToMinMax(feature_range=(0, 1), columns=['cPARP_obs'], groupby=None,
                                                      do_fit_transform=True))) # Scale all conditions jointly.

fl_tBid = Fluorescence(results, dataset=tBID_dataset, measured_values={'norm_IC-RP': ['tBID_obs']},
                       observables=['tBID_obs'])

measurement_results_cPARP = fl_cPARP.run()
measurement_results_tBid = fl_tBid.run()


# ------- Likelihood Function ------
@objective_function(pm=param_model, simulator=sim, measurements=[fl_cPARP, fl_tBid], return_results=False, evals=0)
def likelihood_fn(x):
    kf0 = 10 ** x[0]         # x0  float  kf0 -- 95% bounded in  (-8,  -4)
    kr0 = 10 ** x[1]         # x1  float  kr0 -- 95% bounded in  (-5,  -1)
    kc0 = 10 ** x[2]         # x2  float  kc0 -- 95% bounded in  (-7,  -1)  DISC formation

    kf1 = 10 ** x[3]         # x3  float  kf1 -- 95% bounded in  ( -8, -4)
    kr1 = 10 ** x[4]         # x4  float  kr1 -- 95% bounded in  ( -5, -1)
    kc1 = 10 ** x[5]         # x5  float  kc1 -- 95% bounded in  ( -2,  2)

    kf2 = 10 ** x[6]         # x6  float  kf2 -- 95% bounded in  ( -8, -4)
    kr2 = 10 ** x[7]         # x7  float  kr2 -- 95% bounded in  ( -5, -1)
    kc2 = 10 ** x[8]         # x8  float  kc2 -- 95% bounded in  ( -5,  1)

    kf3 = 10 ** x[9]         # x9  float  kf3 -- 95% bounded in  ( -8, -4)
    kr3 = 10 ** x[10]        # x10 float  kc3 -- 95% bounded in  ( -5, -1)
    kc3 = 10 ** x[11]        # x11 float  kr3 -- 95% bounded in  ( -2,  2)

    kf4 = 10 ** x[12]        # x12 float  kf4 -- 95% bounded in  ( -8, -4)
    kr4 = 10 ** x[13]        # x13 float  kr4 -- 95% bounded in  ( -5, -1)
    kc4 = 10 ** x[14]        # x14 float  kc4 -- 95% bounded in  ( -2,  2)

    kf5 = 10 ** x[15]        # x15 float  kf5 -- 95% bounded in  ( -8, -4)
    kr5 = 10 ** x[16]        # x16 float  kr5 -- 95% bounded in  ( -5, -1)
    kc5 = 10 ** x[17]        # x17 float  kc5 -- 95% bounded in  ( -7, -1)  MOMP Signal 1

    kf6 = 10 ** x[18]        # x18 float  kf6 -- 95% bounded in  ( -8, -4)
    kr6 = 10 ** x[19]        # x19 float  kr6 -- 95% bounded in  ( -5, -1)
    kc6 = 10 ** x[20]        # x20 float  kc6 -- 95% bounded in  ( -7,  1)  MOMP Signal 2

    kf7 = 10 ** x[21]        # x21 float  kf7 -- 95% bounded in  ( -8, -4)
    kr7 = 10 ** x[22]        # x22 float  kr7 -- 95% bounded in  ( -5, -1)
    kc7 = 10 ** x[23]        # x23 float  kc7 -- 95% bounded in  ( -2,  2)

    kf8 = 10 ** x[24]        # x24 float  kf8 -- 95% bounded in  ( -8, -4)
    kr8 = 10 ** x[25]        # x25 float  kr8 -- 95% bounded in  ( -5, -1)
    kc8 = 10 ** x[26]        # x26 float  kc8 -- 95% bounded in  ( -2,  2)  MOMP dependent effector Caspase activation

    kc9 = 10 ** x[27]        # x27 float  kc8 -- 95% bounded in  (-9,  -3)  effector Caspase Degradation
    kd_bid_0 = 4e4 * x[28]   # x28 float  kd_bid_0 -- bounded in ( 0, 0.5)

    # update parameters
    likelihood_fn.pm.update_values(
        pd.DataFrame({'value': [kf0, kr0, kc0, kf1, kr1, kc1, kf2, kr2, kc2, kf3, kr3, kc3, kf4, kr4, kc4, kf5, kr5,
                                kc5, kf6, kr6, kc6, kf7, kr7, kc7, kf8, kr8, kc8, kc9],
                     'param': ['kf0', 'kr0', 'kc0', 'kf1', 'kr1', 'kc1', 'kf2', 'kr2', 'kc2', 'kf3', 'kr3', 'kc3',
                               'kf4', 'kr4', 'kc4', 'kf5', 'kr5', 'kc5', 'kf6', 'kr6', 'kc6', 'kf7', 'kr7', 'kc7',
                               'kf8', 'kr8', 'kc8', 'kc9']}))
    likelihood_fn.pm.update_values(pd.DataFrame([[kd_bid_0, 'Bid_0', 'BidKD']],
                                                columns=['value', 'param', 'Condition']))
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
sampled_params_0 = [
    SampledParam(norm, loc=-6, scale=1.0),        # x0  float  kf0 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x1  float  kr0 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-4, scale=1.5),        # x2  float  kc0 -- 95% bounded in ( -7, -1) DISC formation

    SampledParam(norm, loc=-6, scale=1.0),        # x3  float  kf1 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x4  float  kr1 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-0, scale=1.0),        # x5  float  kc1 -- 95% bounded in ( -2,  2)

    SampledParam(norm, loc=-6, scale=1.0),        # x6  float  kf2 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x7  float  kr2 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-0, scale=1.0),        # x8  float  kc2 -- 95% bounded in ( -5,  1)

    SampledParam(norm, loc=-6, scale=1.0),        # x9  float  kf3 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x10 float  kr3 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-0, scale=1.0),        # x11 float  kc3 -- 95% bounded in ( -2,  2)

    SampledParam(norm, loc=-6, scale=1.0),        # x12 float  kf4 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x13 float  kr4 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-0, scale=1.0),        # x14 float  kc4 -- 95% bounded in ( -2,  2)

    SampledParam(norm, loc=-6, scale=1.0),        # x15 float  kf5 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x16 float  kr5 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-4, scale=1.5),        # x17 float  kc5 -- 95% bounded in ( -7, -1) MOMP Signal 1

    SampledParam(norm, loc=-6, scale=1.0),        # x18 float  kf6 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x19 float  kr6 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-4, scale=1.5),        # x20 float  kc6 -- 95% bounded in ( -7,  1) MOMP Signal 2

    SampledParam(norm, loc=-6, scale=1.0),        # x21 float  kf7 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x22 float  kr7 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-0, scale=1.0),        # x23 float  kc7 -- 95% bounded in ( -2,  2)

    SampledParam(norm, loc=-6, scale=1.0),        # x24 float  kf8 -- 95% bounded in ( -8, -4)
    SampledParam(norm, loc=-3, scale=1.0),        # x25 float  kr8 -- 95% bounded in ( -5, -1)
    SampledParam(norm, loc=-0, scale=1.0),        # x26 float  kc8 -- 95% bounded in ( -2,  2)

    SampledParam(norm, loc=-6, scale=1.0),        # x27 float  kc8 -- 95% bounded in (-9,  -3) Caspase-3 deg
    SampledParam(beta, a=2.5, b=5.83)             # x28 float  KD_Bid_0   bounded by ( 0, 0.5)
                    ]

n_chains = 4
n_iterations = 15000  # iterations per file-save
burn_in_len = 8000    # number of iterations during burn-in
max_iterations = 15000
now = dt.datetime.now()
model_name = f'fluorescence_data_calibration_{now.year}{now.month}{now.day}'

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood_fn,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       nCR=10,
                                       gamma_levels=4,
                                       adapt_gamma=True,
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
                                               nCR=10,
                                               gamma_levels=4,
                                               adapt_gamma=True,
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

