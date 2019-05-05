# MW Irvin -- Lopez Lab -- 2018-02-03
import os
import pandas as pd
import numpy as np

from opt2q.data import DataSet
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import WesternBlot
from opt2q.examples.apoptosis_model_ import model
from opt2q.calibrator import objective_function
import time

# ======= Measurement Model Pipeline ========

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')
western_blot = pd.read_csv(file_path)
western_blot['time'] = western_blot['time'].apply(lambda x: x*3600)  # Convert to [s] (as is in the PySB model).

# ------- Noise Model --------
n = 300
noise_model_sample_size = n

# Params
ligand = pd.DataFrame([['L_0',   600,  10, False],      # 'TRAIL_conc' column annotates experimental treatments
                       ['L_0',  3000,  50, False],      # 600 copies per cell corresponds to 10 ng/mL TRAIL
                       ['L_0', 15000, 250, False]],
                      columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

kc0, kc2, kf3, kc3, kf4, kr7 = (1.0e-05, 1.0e-02, 3.0e-08, 1.0e-02, 1.0e-06, 1.0e-02)
k_values = pd.DataFrame([['kc0', kc0, True],
                         ['kc2', kc2, True],   # co-vary with kc3
                         ['kf3', kf3, False],
                         ['kc3', kc3, True],
                         ['kf4', kf4, False],
                         ['kr7', kr7, False]],
                        columns=['param', 'value', 'apply_noise'])\
    .iloc[np.repeat(range(6), 3)]                       # Repeat for each of the 3 experimental treatments
k_values['TRAIL_conc'] = np.tile([10, 50, 250], 6)      # Repeat for each of the 5 parameter
param_means = pd.concat([ligand, k_values], sort=False)

kc2_cv, kc3_cv, kc2_kc3_cor = (0.2, 0.2, 0.25)
kc2_var, kc3_var, kc2_kc3_cov = ((kc2 * kc2_cv) ** 2, (kc3 * kc3_cv) ** 2, kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor)
param_variances = pd.DataFrame([['kc2', 'kc2', kc2_var],
                                ['kc3', 'kc3', kc3_var],
                                ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance between 'kf3' and kf4'
                               columns=['param_i', 'param_j', 'value'])

NoiseModel.default_coefficient_of_variation = 0.25      # 'kc_0' takes default variability of 25%
NoiseModel.default_sample_size = noise_model_sample_size

# Noise Model
noise = NoiseModel(param_mean=param_means, param_covariance=param_variances)
parameters = noise.run()

# ------- Dynamical Model -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda',
                integrator_options={'n_blocks': 64, 'memory_usage': 'global', 'vol': 4e-15})

results = sim.run(np.linspace(0, 32400, 100))

# ------- Measurement Model -------
# A separate western blot was done each experimental condition
experimental_conditions = [10, 50, 250]
dataset = {ec: DataSet(data=western_blot[western_blot['TRAIL_conc'] == ec],
                       measured_variables={'cPARP': 'ordinal', 'PARP': 'ordinal'})  # ordinal is treated differently.
           for ec in experimental_conditions}

western_blot_models = {ec: WesternBlot(simulation_result=results,
                                       dataset=dataset[ec],
                                       measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                                       observables=['PARP_obs', 'cPARP_obs'],
                                       experimental_conditions=pd.DataFrame([ec], columns=['TRAIL_conc']),
                                       time_points=[0, 3600, 7200, 10800, 14400, 18000, 25200, 32400])
                       for ec in experimental_conditions}

for ec in experimental_conditions:
    western_blot_models[ec].process.set_params(sample_average__sample_size=n, sample_average__noise_term=1000)
    western_blot_models[ec].process.get_step('classifier').do_fit_transform = True

sim_wb_results = {ec: western_blot_models[ec].run() for ec in experimental_conditions}


# -------- likelihood function -----------
@objective_function(noise_model=noise, simulator=sim, measurement_models=western_blot_models, return_results=False, evals=0)
def likelihood_fn(x):

    kc0 = 10 ** x[0]    # :  [(-7,  -3),   # float  kc0
    kc2 = 10 ** x[1]    # :   (-5,   1),   # float  kc2
    kf3 = 10 ** x[2]    # :   (-11, -6),   # float  kf3
    kc3 = 10 ** x[3]    # :   (-5,   1),   # float  kc3
    kf4 = 10 ** x[4]    # :   (-10, -4),   # float  kf4
    kr7 = 10 ** x[5]    # :   (-8,   4),   # float  kr7

    kc2_cv = x[6]       # :   (0, 1),      # float  kc2_cv
    kc3_cv = x[7]       # :   (0, 1),      # float  kc3_cv
    kc2_kc3_cor = x[8]  # :   (-1, 1)])    # float  kc2_kc3_cor

    kc2_var = (kc2 * kc2_cv) ** 2
    kc3_var = (kc3 * kc3_cv) ** 2
    kc2_kc3_covariance = kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor

    # noise model
    # start_time = time.time()
    ligand = pd.DataFrame([['L_0', 600, 10, False],  # 'TRAIL_conc' column annotates experimental treatments
                           ['L_0', 3000, 50, False],  # 600 copies per cell corresponds to 10 ng/mL TRAIL
                           ['L_0', 15000, 250, False]],
                          columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

    k_values = pd.DataFrame([['kc0', kc0, True],
                             ['kc2', kc2, True],  # co-vary with kc3
                             ['kf3', kf3, False],
                             ['kc3', kc3, True],
                             ['kf4', kf4, False],
                             ['kr7', kr7, False]],
                            columns=['param', 'value', 'apply_noise']) \
        .iloc[np.repeat(range(6), 3)]  # Repeat for each of the 3 experimental treatments
    k_values['TRAIL_conc'] = np.tile([10, 50, 250], 6)  # Repeat for each of the 5 parameter

    new_params = pd.concat([ligand, k_values], sort=False)

    new_params_cov = pd.DataFrame([['kc2', 'kc2', kc2_var],
                                   ['kc3', 'kc3', kc3_var],
                                   ['kc2', 'kc3', kc2_kc3_covariance]],  # Covariance between 'kc2' and kc3'
                                  columns=['param_i', 'param_j', 'value'])

    likelihood_fn.noise_model.update_values(param_mean=new_params,
                                            param_covariance=new_params_cov)

    simulator_parameters = likelihood_fn.noise_model.run()
    # end_time = time.time()
    # print("--- noise model %s seconds ---" % (end_time - start_time))

    # dynamical model
    start_time = time.time()
    likelihood_fn.simulator.param_values = simulator_parameters
    sim_results = likelihood_fn.simulator.run(np.linspace(0, 32400, 100))
    end_time = time.time()
    print("--- dynamical model %s seconds ---" % (end_time - start_time))

    # measurement model
    # start_time = time.time()
    l = 0
    for key, measurement in likelihood_fn.measurement_models.items():
        measurement.update_simulation_result(sim_results)
        l += measurement.likelihood()

    likelihood_fn.evals += 1
    # end_time = time.time()
    # print("--- measurement model %s seconds ---" % (end_time - start_time))
    print(likelihood_fn.evals)
    print(f"likelihood: {l}")
    print(x)

    return l

