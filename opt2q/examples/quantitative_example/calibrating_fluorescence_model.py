# MW Irvin -- Lopez Lab -- 2018-10-01
import numpy as np
from opt2q.examples.quantitative_example.fluorescence_likelihood_fn import likelihood_fn
from scipy.optimize import differential_evolution

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
        likelihood_fn,
        bounds=[(-3, 3),        # kc3
                (-3, 1),        # kc4
                (1, 3)])        # L_0

print(x)
np.save('calibrated_params_scipy_diff_evolution.npy', x)
