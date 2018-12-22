# MW Irvin -- Lopez Lab -- 2018-12-22

import pandas as pd
import numpy as np

from opt2q.examples.apoptosis_model import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement.base import Scale, ScaleGroups, SampleAverage
from opt2q.measurement.base.functions import derivative, where_max, polynomial_features
from opt2q.measurement import FractionalKilling

from opt2q.data import DataSet

# ------- Data -------
cell_viability = pd.read_csv('Cell_Viability_Data.csv')
experimental_conditions = cell_viability[['TRAIL_conc']]
len_ec = experimental_conditions.shape[0]
data = DataSet(data=cell_viability[['TRAIL_conc', 'viability']],
               measured_variables={'viability': 'nominal'})
data.measurement_error_df = cell_viability[['stdev']]

# Parameter values (formatted for noise model)
kc3 = np.array([['kc3', 1.0, True]])
kc4 = np.array([['kc4', 1.0, True]])
kc3_mean_df = pd.DataFrame(np.repeat(kc3, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])
kc4_mean_df = pd.DataFrame(np.repeat(kc4, len_ec, axis=0),  columns=['param', 'value', 'apply_noise'])

ligand = pd.DataFrame(cell_viability['TRAIL_conc'].values, columns=['value'])
ligand['param'] = 'L_0'
ligand['apply_noise'] = False

param_m = pd.concat([kc3_mean_df, kc4_mean_df, ligand], sort=False, ignore_index=True)
param_m['TRAIL_conc'] = np.tile(cell_viability['TRAIL_conc'].values, 3)
param_cov = pd.DataFrame([['kc3', 'kc3', 0.009],
                          ['kc4', 'kc4', 0.009],
                          ['kc4', 'kc3', 0.001]],
                         columns=['param_i', 'param_j', 'value'])

# ------- Noise Model -------
NoiseModel.default_sample_size = 5
noise = NoiseModel(param_mean=param_m, param_covariance=param_cov)
parameters = noise.run()

# ------- Simulate dynamics -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
results = sim.run(np.linspace(0, 5000, 100))

# In the measurement model
fk = FractionalKilling(simulation_result=results,
                       dataset=data,
                       measured_values={'viability':['cPARP_obs', 'time']},
                       observables=['cPARP_obs'],
                       experimental_conditions=cell_viability[['TRAIL_conc']])

fk.process.set_params()
fk.process.remove_step('log_scale')
fk.process.remove_step('interpolate')
fk.process.remove_step('sample_average')

fk.process.add_step(('ddx', Scale(columns='cPARP_obs', scale_fn=derivative)),0)
fk.process.add_step(('at_max_t', ScaleGroups(groupby='simulation', scale_fn=where_max, **{'var':'cPARP_obs'})),1)
fk.process.add_step(('log10',Scale(columns='cPARP_obs', scale_fn='log10')),2)
fk.process.add_step(('polynomial',Scale(columns='cPARP_obs', scale_fn=polynomial_features, **{'degree':2})), 4)
fk.process.add_step(('sample_average', SampleAverage(columns='viability', drop_columns='simulation',
                                                    groupby={'TRAIL_conc'}, apply_noise=False)))
print(fk.run())
