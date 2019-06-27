# MW Irvin -- Lopez Lab -- 2018-05-06

import numpy as np
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin

from scipy.stats import norm
from opt2q.examples.quantitative_example.fluorescence_likelihood_fn import likelihood_fn

# Model Inference via PyDREAM
sampled_params_0 = [SampledParam(norm, loc=-5, scale=1.0),     # float  kc0 -- 95% bounded in (-7,  -3)
                    SampledParam(norm, loc=-2, scale=1.5),     # float  kc2 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-8.5, scale=1.25),  # float  kf3 -- 95% bounded in (-11, -6)
                    SampledParam(norm, loc=-2, scale=1.5),     # float  kc3 -- 95% bounded in (-5,   1)
                    SampledParam(norm, loc=-7, scale=1.5),     # float  kf4 -- 95% bounded in (-10, -4)
                    SampledParam(norm, loc=-2, scale=3.0),     # float  kr7 -- 95% bounded in (-8,   4)
                    ]


def likelihood(x):
    # PyDREAM maximizes likelihood_fn while Opt2Q returns neg log-likelihood
    return -likelihood_fn(x)


n_chains = 5
n_iterations = 10000
model_name = 'PyDream_Fluorescence_20190506'

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
        np.save(model_name + str(chain) + '_' + str(total_iterations) + '_' + 'parameters', sampled_params[chain])
        np.save(model_name + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(model_name + str(total_iterations) + '.txt', GR)
