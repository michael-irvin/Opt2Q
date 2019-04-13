# MW Irvin -- Lopez Lab -- 2018-10-01
import numpy as np
from opt2q.examples.quantitative_example.fluorescence_likelihood_fn import likelihood_fn
from scipy.optimize import differential_evolution

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
        likelihood_fn,
        bounds=[(-8,  -2),   # float  kc0
                (-5,   1),   # float  kc2
                (-11, -5),   # float  kf3
                (-5,   1),   # float  kc3
                (-10, -2),   # float  kf4
                (-8,   4)],  # float  kr7
        )

print(x)
np.save('calibrated_params_scipy_diff_evolution.npy', x)
