# MW Irvin -- Lopez Lab -- 2019-06-12

import os
import pandas as pd
import numpy as np
from multiprocessing import current_process
from opt2q.examples.apoptosis_model_ import model
import opt2q.examples.quantitative_example as fl_example
import opt2q.examples.western_blot_example as wb_example
import opt2q.examples.cell_viability_example as cv_example
from opt2q.data import DataSet
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import Fluorescence, WesternBlot, FractionalKilling
from opt2q.measurement.base import Scale, ScaleGroups
from opt2q.measurement.base.functions import derivative, where_max, polynomial_features
from opt2q.calibrator import objective_function

# ======= Kinetic Parameters ============
kc0, kc2, kf3, kc3, kf4, kr7 = (1.0e-05, 1.0e-02, 3.0e-08, 1.0e-02, 1.0e-06, 1.0e-02)

# ======= Data ===============
# ----- fluorescence data -------
script_dir = os.path.dirname(fl_example.__file__)
file_path = os.path.join(script_dir, 'fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time'})  # Remove unnecessary whitespace in column name
fluorescence_data['TRAIL_conc'] = 50
fluorescence_data['experiment'] = 'fluorescence_exp'
fluorescence_dataset = DataSet(fluorescence_data[['TRAIL_conc', 'experiment', 'time', 'norm_IC-RP', 'norm_EC-RP']],
                               measured_variables={'norm_IC-RP': 'semi-quantitative',
                                                   'norm_EC-RP': 'semi-quantitative'})
fluorescence_dataset.measurement_error_df = fluorescence_data[['nrm_var_IC-RP', 'nrm_var_EC-RP']].\
    rename(columns={'nrm_var_IC-RP': 'norm_IC-RP__error',
                    'nrm_var_EC-RP': 'norm_EC-RP__error'})  # DataSet expects error columns to have "__error" suffix


# ----- western blot data --------
script_dir = os.path.dirname(wb_example.__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')
western_blot = pd.read_csv(file_path)
western_blot['time'] = western_blot['time'].apply(lambda x: x*3600)  # Convert to [s] (as is in the PySB model).
western_blot['experiment'] = 'western_blot_exp'
# A separate western blot was done each experimental condition
western_blot_experimental_conditions = [10, 50, 250]
western_blot_dataset_dict = {ec: DataSet(data=western_blot[western_blot['TRAIL_conc'] == ec],
                                         measured_variables={'cPARP': 'ordinal', 'PARP': 'ordinal'})
                             for ec in western_blot_experimental_conditions}

# ----- cell viability -------
script_dir = os.path.dirname(cv_example.__file__)
file_path = os.path.join(script_dir, 'Cell_Viability_Data.csv')

cell_viability = pd.read_csv(file_path)
cell_viability['experiment'] = 'cell_viability_exp'
cell_viability_experimental_conditions = cell_viability[['TRAIL_conc']]
len_cv_ec = cell_viability_experimental_conditions.shape[0]
cell_viability_data = DataSet(data=cell_viability[['TRAIL_conc', 'viability', 'experiment']],
                              measured_variables={'viability': 'nominal'})
cell_viability_data.measurement_error_df = cell_viability[['TRAIL_conc', 'experiment', 'stdev']]

# ======= Parameters =========
# ------- fluorescence -------
fluorescence_params = pd.DataFrame([['L_0',  3000,   'fluorescence_exp',   50,    False,   1],  # 3000 is 50 ng/mL
                                    ['kc0',   kc0,   'fluorescence_exp',   50,    False,   1],
                                    ['kc2',   kc2,   'fluorescence_exp',   50,    False,   1],  # co-vary with kc3
                                    ['kf3',   kf3,   'fluorescence_exp',   50,    False,   1],
                                    ['kc3',   kc3,   'fluorescence_exp',   50,    False,   1],
                                    ['kf4',   kf4,   'fluorescence_exp',   50,    False,   1],
                                    ['kr7',   kr7,   'fluorescence_exp',   50,    False,   1]],
                                   columns=['param', 'value', 'experiment', 'TRAIL_conc', 'apply_noise', 'num_sims'])

# ------- western blot -------
wb_sample_size = 300
wb_ligands = pd.DataFrame([['L_0',   600,  'western_blot_exp',  10,   False,   wb_sample_size],
                           ['L_0',  3000,  'western_blot_exp',  50,   False,   wb_sample_size],
                           ['L_0', 15000,  'western_blot_exp', 250,   False,   wb_sample_size]],
                          columns=['param', 'value', 'experiment', 'TRAIL_conc', 'apply_noise', 'num_sims'])

wb_k_values = pd.DataFrame([['kc0', kc0,   'western_blot_exp',  True,    wb_sample_size],
                            ['kc2', kc2,   'western_blot_exp',  True,    wb_sample_size],  # co-vary with kc3
                            ['kf3', kf3,   'western_blot_exp',  False,   wb_sample_size],
                            ['kc3', kc3,   'western_blot_exp',  True,    wb_sample_size],
                            ['kf4', kf4,   'western_blot_exp',  False,   wb_sample_size],
                            ['kr7', kr7,   'western_blot_exp',  False,   wb_sample_size]],
                           columns=['param', 'value', 'experiment', 'apply_noise', 'num_sims'])\
    .iloc[np.repeat(range(6), 3)]                       # Repeat for each of the 3 experimental treatments
wb_k_values['TRAIL_conc'] = np.tile([10, 50, 250], 6)      # Repeat for each of the 5 parameter
wb_param_means = pd.concat([wb_ligands, wb_k_values], sort=False, ignore_index=True)

# -------- cell viability -----
cv_sample_size = 250
cv_k_values = pd.DataFrame([['kc0', kc0,   'cell_viability_exp',  True,    cv_sample_size],
                            ['kc2', kc2,   'cell_viability_exp',  True,    cv_sample_size],  # co-vary with kc3
                            ['kf3', kf3,   'cell_viability_exp',  False,   cv_sample_size],
                            ['kc3', kc3,   'cell_viability_exp',  True,    cv_sample_size],
                            ['kf4', kf4,   'cell_viability_exp',  False,   cv_sample_size],
                            ['kr7', kr7,   'cell_viability_exp',  False,   cv_sample_size]],
                           columns=['param', 'value', 'experiment', 'apply_noise', 'num_sims'])\
    .iloc[np.repeat(range(6), len_cv_ec)]  # Repeat for each experimental treatment in the cell viability experiment

cv_ligand = pd.DataFrame(cell_viability['TRAIL_conc'].values, columns=['value'])
cv_ligand['param'] = 'L_0'
cv_ligand['experiment'] = 'cell_viability_exp'
cv_ligand['apply_noise'] = False
cv_ligand['num_sims'] = cv_sample_size

cv_param_means = pd.concat([cv_k_values, cv_ligand], sort=False, ignore_index=True)
cv_param_means['TRAIL_conc'] = np.tile(cell_viability['TRAIL_conc'].values, 7)

# -------- combined -----------
kc2_cv, kc3_cv, kc2_kc3_cor = (0.2, 0.2, 0.25)
kc2_var, kc3_var, kc2_kc3_cov = ((kc2 * kc2_cv) ** 2, (kc3 * kc3_cv) ** 2, kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor)
parameter_variance = pd.DataFrame([['kc2', 'kc2', kc2_var],
                                   ['kc3', 'kc3', kc3_var],
                                   ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance between 'kf3' and kf4'
                                  columns=['param_i', 'param_j', 'value'])

parameter_means = pd.concat([fluorescence_params, wb_param_means, cv_param_means])


noise = NoiseModel(param_mean=parameter_means, param_covariance=parameter_variance)
parameters = noise.run()

# ======== Simulate Dynamics ========
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))
results_obs = [x.name for x in sim.model.observables]

# ========= Measurement Models ======
# --------- fluorescence ---------
fl_model = Fluorescence(results,
                        dataset=fluorescence_dataset,
                        measured_values={'norm_IC-RP': ['BID_obs'],
                                         'norm_EC-RP': ['cPARP_obs']},
                        observables=['BID_obs', 'cPARP_obs'],
                        experimental_conditions=pd.DataFrame([['fluorescence_exp',   50]], columns=['experiment', 'TRAIL_conc']))
fl_model.run()

# --------- western blot ----------
# A separate western blot was done each experimental condition
western_blot_models = {ec: WesternBlot(simulation_result=results,
                                       dataset=western_blot_dataset_dict[ec],
                                       measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                                       observables=['PARP_obs', 'cPARP_obs'],
                                       experimental_conditions=pd.DataFrame([[ec, 'western_blot_exp']],
                                                                            columns=['TRAIL_conc', 'experiment']),
                                       time_points=[0, 3600, 7200, 10800, 14400, 18000, 25200, 32400])
                       for ec in western_blot_experimental_conditions}

for ec in western_blot_experimental_conditions:
    western_blot_models[ec].process.set_params(sample_average__sample_size=wb_sample_size, sample_average__noise_term=1000)
    western_blot_models[ec].process.get_step('classifier').do_fit_transform = True
    western_blot_models[ec].run()

# --------- cell viability ----------
cell_viability_model = FractionalKilling(simulation_result=results,
                                         dataset=cell_viability_data,
                                         measured_values={'viability':['cPARP_obs', 'time']},
                                         observables=['cPARP_obs'],
                                         experimental_conditions=cell_viability[['TRAIL_conc', 'experiment']],
                                         time_dependent=False)

cell_viability_model.process.remove_step('log_scale')
cell_viability_model.process.add_step(('ddx', Scale(columns='cPARP_obs', scale_fn=derivative)), 0)
cell_viability_model.process.add_step(('at_max_t', ScaleGroups(groupby='simulation',
                                                               scale_fn=where_max, **{'var': 'cPARP_obs'})), 1)
cell_viability_model.process.add_step(('log10', Scale(columns='cPARP_obs', scale_fn='log10')), 2)
cell_viability_model.process.add_step(('polynomial', Scale(columns=['cPARP_obs', 'time'],
                                                           scale_fn=polynomial_features, **{'degree': 2})),
                                      'standardize')  # add after the 'standardize' step
cell_viability_model.process.get_step('classifier').n_jobs = 1  # set number of multiprocessing jobs
cell_viability_model.setup()  # Our IBM HPC stalls when cell_viability_model is run prior to use in objective_function


# ============= likelihood function ======================
@objective_function(noise_model=noise, parameter_means=parameter_means, simulator=sim,
                    simulator_observables=results_obs,
                    fluorescence_model=fl_model,
                    western_blot_models=western_blot_models,
                    cell_viability_model=cell_viability_model,
                    evals=0)
def likelihood_fn(x):
    kc0_ = 10 ** x[0]                           # :  [(-7,  -3),    float  kc0
    kc2_ = 10 ** x[1]                           # :   (-5,   1),    float  kc2
    kf3_ = 10 ** x[2]                           # :   (-11, -6),    float  kf3
    kc3_ = 10 ** x[3]                           # :   (-5,   1),    float  kc3
    kf4_ = 10 ** x[4]                           # :   (-10, -4),    float  kf4
    kr7_ = 10 ** x[5]                           # :   (-8,   4),    float  kr7

    kc2_cv_ = x[6]                              # :   (0, 1),       float  kc2_cv
    kc3_cv_ = x[7]                              # :   (0, 1),       float  kc3_cv
    kc2_kc3_cor_ = x[8]                         # :   (-1, 1)       float  kc2_kc3_cor

    viability_coef = np.array([[x[9],           # :  (-100, 100),   float
                                x[10],          # :  (-100, 100),   float
                                x[11],          # :  (-100, 100),   float
                                x[12],          # :  (-100, 100),   float
                                x[13]]])        # :  (-100, 100),   float
    viability_intercept = np.array([x[14]])     # :  (-10, 10)]     float

    likelihood_fn.parameter_means.loc[parameter_means['param'] == 'kc0', 'value'] = kc0_
    likelihood_fn.parameter_means.loc[parameter_means['param'] == 'kc2', 'value'] = kc2_
    likelihood_fn.parameter_means.loc[parameter_means['param'] == 'kf3', 'value'] = kf3_
    likelihood_fn.parameter_means.loc[parameter_means['param'] == 'kc3', 'value'] = kc3_
    likelihood_fn.parameter_means.loc[parameter_means['param'] == 'kf4', 'value'] = kf4_
    likelihood_fn.parameter_means.loc[parameter_means['param'] == 'kr7', 'value'] = kr7_

    kc2_var_, kc3_var_, kc2_kc3_cov_ = (
        (kc2_ * kc2_cv_) ** 2, (kc3_ * kc3_cv_) ** 2, kc2_ * kc2_cv_ * kc3_ * kc3_cv_ * kc2_kc3_cor_)
    param_cov = pd.DataFrame([['kc2', 'kc2', kc2_var_],
                              ['kc3', 'kc3', kc3_var_],
                              ['kc2', 'kc3', kc2_kc3_cov_]],  # Covariance between 'kc2' and kc3'
                             columns=['param_i', 'param_j', 'value'])

    likelihood_fn.noise_model.update_values(param_mean=likelihood_fn.parameter_means,
                                            param_covariance=param_cov)
    simulator_parameters = likelihood_fn.noise_model.run()
    likelihood_fn.simulator.param_values = simulator_parameters

    # Each process selects one of the 4 gpu
    process_id = current_process().ident % 4

    likelihood_fn.simulator.sim.gpu = [process_id]
    sim_results = likelihood_fn.simulator.run(np.linspace(0, 5000, 100))

    # if results.dataframe[likelihood_fn.simulator_observables].isna().any(axis=None):
    #     return 10000000000.0

    try:
        ll = 0.0
        likelihood_fn.fluorescence_model.update_simulation_result(sim_results)
        ll += likelihood_fn.fluorescence_model.likelihood()

        for key, measurement in likelihood_fn.western_blot_models.items():
            measurement.update_simulation_result(sim_results)
            ll += measurement.likelihood()

        measurement_model_params = {'classifier__coefficients__viability__coef_': viability_coef,
                                    'classifier__coefficients__viability__intercept_': viability_intercept}

        likelihood_fn.cell_viability_model.update_simulation_result(sim_results)
        likelihood_fn.cell_viability_model.process.set_params(**measurement_model_params)
        ll += likelihood_fn.cell_viability_model.likelihood()
    except ValueError:
        return 10000000000.0

    likelihood_fn.evals += 1

    print(likelihood_fn.evals)
    print(f"likelihood: {ll}")
    print(x)

    return ll


likelihood_fn([-1.2053062,   -2.50635065, -11.32608751,  -4.71373947, -10.74176349,
  -4.83712945,   0.19896594,   0.06511681,   0.05801066,   2.11147502,
  -1.36353197,   0.9637254,    1.32352659,   7.06380367,   2.21172402])
