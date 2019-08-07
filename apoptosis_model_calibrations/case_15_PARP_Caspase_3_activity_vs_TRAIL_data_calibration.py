# MW Irvin -- Lopez Lab -- 2019-07-31

"""
Case 15: Calibrating apoptosis model to experiments investigating the affect of
    (IV) TRAIL concentration on
    (DV) PARP Caspase-3 activity
    All data types (including Cell Viability which is model by cell death marker, cPARP) are used in this calibration.
"""

import numpy as np
import datetime as dt
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin

from apoptosis_model_calibrations.generate_likelihood_fn import param_priors, generate_likelihood_fn
from apoptosis_model_calibrations.compile_apoptosis_data import CompileData, data_file

# We only want measured values columns that correspond to cPARP and PARP 'C3_active_obs' and 'C3_inactive_obs' observables
mv_columns = set(k for k, v in CompileData.model_observables.items() if set(v) -
                 {'cPARP_obs', 'PARP_obs', 'C3_active_obs', 'C3_inactive_obs'} == set())
data_columns = set()
for col in mv_columns:  # Add st. dev columns
    data_columns |= set(data_file.columns[data_file.columns.str.match(f'{col}')])

annotating_columns = CompileData.annotating_columns
data = data_file[data_file[list(data_columns)].notnull().any(axis=1)][list(data_columns | annotating_columns)]

# We only want the rows in which the 'Condition' column is a control (i.e. TRAIL is the only independent variable)
data = data[data.Condition.str.match(r'^Control$') | data.Condition.str.match(r'^siCtrl$')]

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
model_name = f'PyDream_case_10_{now.year}{now.month}{now.day}'

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


