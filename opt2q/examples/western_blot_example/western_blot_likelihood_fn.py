# MW Irvin -- Lopez Lab -- 2018-02-03
import os
import pandas as pd
import numpy as np

from opt2q.data import DataSet
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import WesternBlot
from opt2q.examples.apoptosis_model import model
from opt2q.calibrator import objective_function

# ======= Measurement Model Pipeline ========

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')
western_blot = pd.read_csv(file_path)
western_blot['time'] = western_blot['time'].apply(lambda x: x*3600)  # Convert to [s] (as is in the PySB model).

# ------- Noise Model --------
n =500
noise_model_sample_size = n

# Params
ligand = pd.DataFrame([['L_0',   600,  10, False],      # 'TRAIL_conc' column annotates experimental treatments
                       ['L_0',  3000,  50, False],      # 600 copies per cell corresponds to 10 ng/mL TRAIL
                       ['L_0', 15000, 250, False]],
                      columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

C_0, kc3, kc4, kf3, kf4 = (1e4, 1e-2, 1e-2, 1e-6, 1e-6)
k_values = pd.DataFrame([['kc3', kc3, False],
                         ['kc4', kc4, False],
                         ['kf3', kf3, True],
                         ['kf4', kf4, True],
                         ['C_0', C_0, True]],
                        columns=['param', 'value', 'apply_noise'])\
    .iloc[np.repeat(range(5), 3)]                       # Repeat for each of the 3 experimental treatments
k_values['TRAIL_conc'] = np.tile([10, 50, 250], 5)      # Repeat for each of the 5 parameter
param_means = pd.concat([ligand, k_values], sort=False)

kf3_cv, kf4_cv, kf3_kf4_cor = (0.2, 0.2, 0.25)
kf3_var, kf4_var, kf3_kf4_covariance = ((kf3 * kf3_cv) ** 2, (kf4 * kf4_cv) ** 2, kf3 * kf3_cv * kf4 * kf4_cv * kf3_kf4_cor)
param_variances = pd.DataFrame([['kf3', 'kf3', kf3_var],
                                ['kf4', 'kf4', kf4_var],
                                ['kf3', 'kf4', kf3_kf4_covariance]],  # Covariance between 'kf3' and kf4'
                               columns=['param_i', 'param_j', 'value'])

NoiseModel.default_coefficient_of_variation = 0.25      # 'C_0' takes default variability of 25%
NoiseModel.default_sample_size = noise_model_sample_size

# Noise Model
noise = NoiseModel(param_mean=param_means, param_covariance=param_variances)
parameters = noise.run()

# ------- Dynamical Model -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
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
    C_0 = 10 ** x[0]             # :  [( 2, 6),   # float  C_0
    kc3 = 10 ** x[1]             # :   (-2, 4),   # float  kc3
    kc4 = 10 ** x[2]             # :   (-2, 4),   # float  kc4
    kf3 = 10 ** x[3]             # :   (-8,-4),   # float  kf3
    kf4 = 10 ** x[4]             # :   (-8,-4),   # float  kf4

    kf3_cv = x[5]                # :   (0, 1),    # float  kf3_cv
    kf4_cv = x[6]                # :   (0, 1),    # float  kf4_cv
    kf3_kf4_cor = x[7]           # :   (0, 1)])   # float  kf3_kf4_vor

    kf3_var = (kf3 * kf3_cv) ** 2
    kf4_var = (kf4 * kf4_cv) ** 2
    kf3_kf4_covariance =kf3 * kf3_cv * kf4 * kf4_cv * kf3_kf4_cor

    # noise model
    ligand = pd.DataFrame([['L_0', 600, 10, False],  # 'TRAIL_conc' column annotates experimental treatments
                           ['L_0', 3000, 50, False],  # 600 copies per cell corresponds to 10 ng/mL TRAIL
                           ['L_0', 15000, 250, False]],
                          columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

    k_values = pd.DataFrame([['kc3', kc3, False],
                             ['kc4', kc4, False],
                             ['kf3', kf3, True],
                             ['kf4', kf4, True],
                             ['C_0', C_0, True]],
                            columns=['param', 'value', 'apply_noise']) \
        .iloc[np.repeat(range(5), 3)]  # Repeat for each of the 3 experimental treatments
    k_values['TRAIL_conc'] = np.tile([10, 50, 250], 5)  # Repeat for each of the 5 parameter
    new_params = pd.concat([ligand, k_values], sort=False)

    new_params_cov = pd.DataFrame([['kf3', 'kf3', kf3_var],
                                   ['kf4', 'kf4', kf4_var],
                                   ['kf3', 'kf4', kf3_kf4_covariance]],  # Covariance between 'kf3' and kf4'
                                  columns=['param_i', 'param_j', 'value'])

    likelihood_fn.noise_model.update_values(param_mean=new_params,
                                            param_covariance=new_params_cov)

    # dynamical model
    simulator_parameters = likelihood_fn.noise_model.run()
    likelihood_fn.simulator.param_values = simulator_parameters
    sim_results = likelihood_fn.simulator.run(np.linspace(0, 32400, 100))

    # measurement model
    l = 0
    for key, measurement in likelihood_fn.measurement_models.items():
        measurement.update_simulation_result(sim_results)
        l += measurement.likelihood()
    likelihood_fn.evals += 1

    print(likelihood_fn.evals)
    print(x)
    print(l)
    return l

