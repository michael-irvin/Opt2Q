# MW Irvin -- Lopez Lab -- 2019-05-01

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
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin

from scipy.stats import norm, laplace, uniform
from opt2q.examples.cell_viability_example.cell_viability_likelihood_fn import likelihood_fn


# Model Inference via PyDREAM
sampled_params_0 = [SampledParam(norm, loc=-5, scale=1.0),           # float  kc0 -- 95% bounded in (-7,  -3)
                    SampledParam(norm, loc=-2, scale=1.5),           # float  kc2 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-8.5, scale=1.25),        # float  kf3 -- 95% bounded in (-11, -6)
                    SampledParam(norm, loc=-2, scale=1.5),           # float  kc3 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-7, scale=1.5),           # float  kf4 -- 95% bounded in (-10, -4)
                    SampledParam(norm, loc=-2, scale=3.0),           # float  kr7 -- 95% bounded in (-8,   4)
                    SampledParam(uniform, loc=[0], scale=[0.5]),     # float  kc2_cv -- bounded in [0,  0.5]
                    SampledParam(uniform, loc=[0], scale=[0.5]),     # float  kc3_cv -- bounded in [0,  0.5]
                    SampledParam(uniform, loc=[-1.0], scale=[3.0]),  # float  kc2_kc3_cor -- bounded in [-1.0,  1.0]
                    SampledParam(laplace, loc=0.0, scale=10),        # float  LR coef
                    SampledParam(laplace, loc=0.0, scale=10),        # float  LR coef
                    SampledParam(laplace, loc=0.0, scale=10),        # float  LR coef
                    SampledParam(laplace, loc=0.0, scale=10),        # float  LR coef
                    SampledParam(laplace, loc=0.0, scale=10),        # float  LR coef
                    SampledParam(laplace, loc=0.0, scale=1),         # float  LR intercept
                    ]


def likelihood(x):
    # PyDREAM maximizes likelihood_fn while Opt2Q returns neg log-likelihood
    return -likelihood_fn(x)


n_chains = 3
n_iterations = 10000
model_name = 'PyDream_CellViability_20190613'

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       gamma_levels=4,
                                       adapt_gamma=True,
                                       history_thin=1,
                                       model_name=model_name,
                                       verbose=True)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(model_name + str(chain) + '_' + str(total_iterations),
                sampled_params[chain])
        np.save(model_name + str(chain) + '_' + str(total_iterations), log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(model_name + str(total_iterations) + '.txt', GR)


