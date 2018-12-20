# MW Irvin -- Lopez Lab -- 2018-10-01
import numpy as np
from opt2q.examples.western_blot_example.apoptosis_model_likelihood_fn import likelihood_fn
from scipy.optimize import differential_evolution

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
        likelihood_fn,
        bounds=[(-3, 3),        # x
                (0, 1),         # x
                (0, 1),         # x
                (0, 1),         # x
                (0, 1),         # x
                (-3, 3),        # x
                (-100, 100),    # x
                (-1, 1),        # x
                (-3, 3),        # x
                (-100, 100),    # x
                (-1, 1),        # x1
                (-1, 1),        # x1
                (-1, 1)])       # x12

print(x)
np.save('calibrated_params_scipy_diff_evolution.npy', x)
