# MW Irvin -- Lopez Lab -- 2019-07-31

"""
Case 13: TRAIL, DISC, Bid, Caspase-8 and -3 model. Calibrates apoptosis model to data where TRAIL, DISC stability, Caspase-3 activity
and Bid expression are independent variables:
    (IV) TRAIL concentration, DISC stability via si-CHIP, Bid expression modulation via knockdown and Caspase-3 activity
         via inhibitor
    (DV) Concentration of DISC, Caspase-3 and -8, Bid, PARP (and post translational modifications thereof).
    Cell viability was modeled as function of cell death marker, cPARP.
"""

import datetime as dt
import numpy as np
from apoptosis_model_calibrations.case_1_mega_model_calibration import likelihood
from apoptosis_model_calibrations.generate_likelihood_fn import param_priors

from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
# This is equivalent to the "mega model"; case_1_mega_model_calibration.py


sampled_params_0 = param_priors

n_chains = 4
n_iterations = 1000
max_iterations = 10000
now = dt.datetime.now()
model_name = f'PyDream_case_13_{now.year}{now.month}{now.day}'

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
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters', sampled_params[chain])
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt(model_name + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.isnan(GR).any() or np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

        # append sample with a re-run of the pyDream algorithm
        while not converged or (total_iterations < max_iterations):
            total_iterations += n_iterations
            print("Saved Results")
            print(total_iterations)
            print(not converged and (total_iterations < max_iterations))

            sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                               likelihood=likelihood,
                                               niterations=n_iterations,
                                               nchains=n_chains,
                                               multitry=False,
                                               gamma_levels=4,
                                               adapt_gamma=True,
                                               history_thin=1,
                                               model_name=model_name,
                                               verbose=True,
                                               restart=True,  # restart at the last sampled position
                                               start=starts)

            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters',
                        sampled_params[chain])
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(n_chains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt(model_name + str(total_iterations) + '.txt', GR)

            if np.all(GR < 1.2):
                converged = True
