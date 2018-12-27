# MW Irvin -- Lopez Lab -- 2018-10-01
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os

from opt2q.examples.apoptosis_model import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.data import DataSet
from opt2q.measurement import WesternBlot
from opt2q.calibrator import objective_function

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')

western_blot = pd.read_csv(file_path)
western_blot['time'] = western_blot['time'].apply(lambda x: x*500)
experimental_conditions = western_blot[['TRAIL_conc']]
len_ec = experimental_conditions.drop_duplicates().shape[0]

data10 = DataSet(data=western_blot[western_blot.TRAIL_conc==10],
                 measured_variables={'cPARP': 'ordinal', 'PARP': 'ordinal'},
                 measurement_error=0.5)

data50 = DataSet(data=western_blot[western_blot.TRAIL_conc==50],
                 measured_variables={'cPARP': 'ordinal', 'PARP': 'ordinal'},
                 measurement_error=0.5)

data250 = DataSet(data=western_blot[western_blot.TRAIL_conc==250],
                  measured_variables={'cPARP': 'ordinal', 'PARP': 'ordinal'},
                  measurement_error=0.5)

# Parameter values (formatted for noise model)
kc3 = np.array([['kc3', 1.0, True]])
kc4 = np.array([['kc4', 1.0, True]])
kc3_mean_df = pd.DataFrame(np.repeat(kc3, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])
kc4_mean_df = pd.DataFrame(np.repeat(kc4, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])

ligand = pd.DataFrame(western_blot['TRAIL_conc'].drop_duplicates().values, columns=['value'])
ligand['param'] = 'L_0'
ligand['apply_noise'] = False

param_m = pd.concat([kc3_mean_df, kc4_mean_df, ligand], sort=False, ignore_index=True)
param_m['TRAIL_conc'] = np.tile(western_blot['TRAIL_conc'].drop_duplicates().values, 3)
param_cov = pd.DataFrame([['kc3', 'kc3', 0.009],
                          ['kc4', 'kc4', 0.009],
                          ['kc4', 'kc3', 0.001]],
                         columns=['param_i', 'param_j', 'value'])


# ------- simulate extrinsic noise -------
NoiseModel.default_sample_size = 10
noise = NoiseModel(param_mean=param_m, param_covariance=param_cov)
parameters = noise.run()

# ------- simulate dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))

# ------- simulate measurements -------
# A separate western blot was done for each TRAIL Concentration.
sample_sizes = 200
wb10 = WesternBlot(simulation_result=results,
                   dataset=data10,
                   measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                   observables=['PARP_obs', 'cPARP_obs'],
                   experimental_conditions=pd.DataFrame([10], columns=['TRAIL_conc']),
                   time_points=[1500, 2000, 2500, 3500, 4500])
wb10.process.set_params(**{'sample_average__sample_size': sample_sizes})
results10 = wb10.run(use_dataset=True)

wb50 = WesternBlot(simulation_result=results,
                   dataset=data50,
                   measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                   observables=['PARP_obs', 'cPARP_obs'],
                   experimental_conditions=pd.DataFrame([10], columns=['TRAIL_conc']),
                   time_points=[1500, 2000, 2500, 3500, 4500])
wb50.process.set_params(**{'sample_average__sample_size': sample_sizes})
results50 = wb50.run(use_dataset=True)

wb250 = WesternBlot(simulation_result=results,
                    dataset=data10,
                    measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                    observables=['PARP_obs', 'cPARP_obs'],
                    experimental_conditions=pd.DataFrame([10], columns=['TRAIL_conc']),
                    time_points=[1500, 2000, 2500, 3500, 4500])
wb250.process.set_params(**{'sample_average__sample_size': sample_sizes})
results250 = wb250.run(use_dataset=True)

# Use get_params() to see how the classifier structures its coefficients (they are calibrated too).
# print(wb10.process.get_params())
# print(wb50.process.get_params())
# print(wb250.process.get_params())

measurements = {"10 ng/mL": wb10, "50 ng/mL": wb50, "250 ng/mL": wb250}


# -------- likelihood function -----------
@objective_function(noise_model=noise, simulator=sim, measurement_models=measurements, return_results=False, evals=0)
def likelihood_fn(x):
    kc3 = 10 ** x[0]                                                        # float [(-3, 3),
    kc4 = 10 ** x[1]                                                        # float  (-3, 3),
    l_0 = 10 ** x[2]  # value of corresponding to the 1 ng/ml TRAIL         # float  (1, 3),

    kc3_var = (kc3 * x[3]) ** 2                                             # float  (0, 1),
    kc4_var = (kc4 * x[4]) ** 2                                             # float  (0, 1),
    kc_cov = kc3_var * x[5]                                                 # float  (0, 1),

    PARP_coef_ = {"10 ng/mL": np.array([10**x[6]]),                         # float  (-3, 3),
                  "50 ng/mL": np.array([10 ** x[7]]),                       # float  (-3, 3),
                  "250 ng/mL": np.array([10 ** x[8]])}                      # float  (-3, 3),

    cPARP_coef_ = {"10 ng/mL": np.array([10 ** x[9]]),                      # float  (-3, 3),
                   "50 ng/mL": np.array([10 ** x[10]]),                     # float  (-3, 3),
                   "250 ng/mL": np.array([10 ** x[11]])}                    # float  (-3, 3),

    PARP_theta_ = {"10 ng/mL": np.array([x[12],                             # float  (-100, 100),
                                         x[12] + 10 ** x[13]]),             # float  (-1, 1),
                   "50 ng/mL": np.array([x[14],                             # float  (-100, 100),
                                         x[14] + 10 ** x[15]]),             # float  (-1, 1),
                   "250 ng/mL": np.array([x[16],                            # float  (-100, 100),
                                          x[16] + 10 ** x[17]])}            # float  (-1, 1),

    cPARP_theta_ = {"10 ng/mL": np.array([x[18],                            # float  (-100, 100),
                                          x[18] + 10**x[19],                # float  (-1, 1),
                                          x[18] + 10**x[19] + 10**x[20],    # float  (-1, 1),
                                          x[18] + 10**x[19] + 10**x[20] +
                                          10**x[21]]),                      # float  (-1, 1),
                    "50 ng/mL": np.array([x[22],                            # float  (-100, 100),
                                          x[22] + 10**x[23]]),              # float  (-1, 1),
                    "250 ng/mL": np.array([x[24],                           # float  (-100, 100),
                                           x[24] + 10**x[25],               # float  (-1, 1),
                                           x[24] + 10**x[25] + 10**x[26],   # float  (-1, 1),
                                           x[24] + 10**x[25] + 10**x[26] +
                                           10**x[27]])}                     # float  (-1, 1)]

    # noise
    kc3_val = np.array([['kc3', kc3, True]])
    kc4_val = np.array([['kc4', kc4, True]])
    kc3_mean_df = pd.DataFrame(np.repeat(kc3_val, len_ec, axis=0), columns=['param', 'value', 'apply_noise'])
    kc4_mean_df = pd.DataFrame(np.repeat(kc4_val, len_ec, axis=0), columns=['param', 'value', 'apply_noise'])

    ligand = pd.DataFrame(western_blot['TRAIL_conc'].drop_duplicates().values, columns=['value'])
    ligand['param'] = 'L_0'
    ligand['apply_noise'] = False

    param_m = pd.concat([kc3_mean_df, kc4_mean_df, ligand], sort=False, ignore_index=True)
    param_m['TRAIL_conc'] = np.tile(western_blot['TRAIL_conc'].drop_duplicates().values, 3)
    param_m['value'] = param_m['value'].astype(float)
    param_cov = pd.DataFrame([['kc3', 'kc3', kc3_var],
                              ['kc4', 'kc4', kc4_var],
                              ['kc4', 'kc3', kc_cov]],
                             columns=['param_i', 'param_j', 'value'])

    likelihood_fn.noise_model.update_values(param_mean=param_m,
                                            param_covariance=param_cov)

    simulator_parameters = likelihood_fn.noise_model.run()
    likelihood_fn.simulator.param_values = simulator_parameters

    # dynamics
    sim_results = likelihood_fn.simulator.run(np.linspace(0, 5000, 100))

    # measurement
    l = 0
    for key, measurement in likelihood_fn.measurement_models.items():
        measurement.update_simulation_result(sim_results)
        measurement.process.set_params(**{'classifier__coefficients__PARP__coef_': PARP_coef_[key],
                                          'classifier__coefficients__PARP__theta_': PARP_theta_[key],
                                          'classifier__coefficients__cPARP__coef_': cPARP_coef_[key],
                                          'classifier__coefficients__cPARP__theta_': cPARP_theta_[key]
                                          })
        l += measurement.likelihood()

    likelihood_fn.evals += 1

    print(likelihood_fn.evals)
    print(x)
    print(l)
    return l