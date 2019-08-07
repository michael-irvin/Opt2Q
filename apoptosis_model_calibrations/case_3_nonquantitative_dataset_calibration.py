# MW Irvin -- Lopez Lab -- 2019-07-30

"""
Case 3: "non-quantitative only model" Calibrating apoptosis model to all non-quantitative measurement in the dataset
    (IV) TRAIL concentration, Caspase-3 and -8 Inhibition, Bid knockdown (BidKD)
    (DV) DISC formation, Caspase-8 and -3 and Bid concentration, PARP activity
    Cell Viability is modeled by cell death marker, cPARP.
"""

import numpy as np
import datetime as dt
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin

from apoptosis_model_calibrations.generate_likelihood_fn import param_priors, generate_likelihood_fn
from apoptosis_model_calibrations.compile_apoptosis_data import CompileData, data_file


data = data_file[data_file['Measurement Model'].str.match(r'^FractionalKilling$') |
                 data_file['Measurement Model'].str.match(r'^WesternBlot')].dropna(axis=1, how='all')
compiled_dataset = CompileData()
compiled_dataset.run(data)

sampled_params_0 = param_priors
neg_log_likelihood = generate_likelihood_fn(compiled_dataset, n_sims=200, n_timepoints=100)


def likelihood(x):
    # PyDREAM maximizes likelihood_fn while Opt2Q returns neg log-likelihood
    return -neg_log_likelihood(x)


n_chains = 4
n_iterations = 10000
now = dt.datetime.now()
model_name = f'PyDream_case_3_{now.year}{now.month}{now.day}'

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


