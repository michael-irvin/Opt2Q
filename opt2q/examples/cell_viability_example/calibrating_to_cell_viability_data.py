# MW Irvin -- Lopez Lab -- 2018-12-19

"""
=================================================
Apoptosis Model Calibrated to Cell Viability Data
=================================================

Nominal observations provide information about the quantifiable attributes (i.e. markers) on which they depend.

For example, programmed cell death (apoptosis) depends on caspase activity; as such, apoptotic cells will more likely
have similar caspase activity that differs from that of surviving cells.

`Albeck and Sorger 2015 <http://msb.embopress.org/content/11/5/803.long>`_ find that the maximum rate of change in
caspase indicator, and the time when that maximum occurs, predicts cellular commitment to apoptosis with 83% accuracy.

The following uses the :class:`~opt2q.measurement.FractionalKilling` to calibrate a model of apoptosis to cell viability
measurements by Albeck and Sorger.
"""

import numpy as np
from opt2q.examples.cell_viability_example.cell_viability_likelihood_fn import likelihood_fn
from scipy.optimize import differential_evolution

# Todo: make a better way of updating num_sims
num_sims = 1
params_for_update = likelihood_fn.noise_model.param_mean[['TRAIL_conc']].drop_duplicates().reset_index(drop=True)
params_for_update['num_sims'] = num_sims
likelihood_fn.noise_model.update_values(param_mean=params_for_update)

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
        likelihood_fn,
        bounds=[(-3, 3),
                (-3, 3),
                (1, 3),
                (0, 1),
                (0, 1),
                (0, 1),
                (-100, 100),
                (-100, 100),
                (-100, 100),
                (-100, 100),
                (-100, 100),
                (-10, 10)])

print(x)
np.save('calibrated_params_cell_viability_scipy_diff_evolution.npy', x)


