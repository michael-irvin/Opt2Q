# MW Irvin -- Lopez Lab -- 2019-02-07
import numpy as np
from opt2q.examples.western_blot_example.western_blot_likelihood_fn import likelihood_fn
from scipy.optimize import differential_evolution

# Differential Evolution Optimization of likelihood fn
x = differential_evolution(
        likelihood_fn,
        bounds=[( 2, 6),   # float  C_0
                (-2, 4),   # float  kc3
                (-2, 4),   # float  kc4
                (-8,-4),   # float  kf3
                (-8,-4),   # float  kf4
                (0, 1),    # float  kf3_cv
                (0, 1),    # float  kf4_cv
                (0, 1)])   # float  kf3_kf4_vor

print(x)
np.save('calibrated_params_scipy_diff_evolution.npy', x)
