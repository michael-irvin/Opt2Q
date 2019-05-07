# MW Irvin -- Lopez Lab -- 2019-02-07
import numpy as np
from opt2q.examples.western_blot_example.western_blot_likelihood_fn import likelihood_fn
from scipy.optimize import differential_evolution

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
        likelihood_fn,
        bounds=[(-7,  -3),    # float  kc0
                (-5,   1),    # float  kc2
                (-11, -6),    # float  kf3
                (-5,   1),    # float  kc3
                (-10, -4),    # float  kf4
                (-8,   4),    # float  kr7
                (0,    0.5),  # float  kc2_cv
                (0,    0.5),  # float  kc3_cv
                (-1,   1)])   # float  kc2_kc3_cor

print(x)
np.save('calibrated_params_scipy_diff_evolution.npy', x)
