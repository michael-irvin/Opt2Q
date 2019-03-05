# MW Irvin -- Lopez Lab -- 2019-02-18

"""
To use this:
First, *manually* set `noise_model_sample_size` to 10000, in the western_blot_likelihood_fn python module.
Next, run this file.
"""

import numpy as np
import pandas as pd
from opt2q.examples.western_blot_example.western_blot_likelihood_fn import likelihood_fn

sim_results = likelihood_fn.simulator.run(np.linspace(0, 32400, 10))


def get_likelihood(num_sims, sim_res):
    sims = sim_res.opt2q_dataframe.groupby('simulation')
    sim_results_n = pd.concat([sims.get_group(group) for group in
                               np.random.choice(list(sims.groups.keys()), num_sims, replace=False)])

    l_val = 0
    for key, measurement in likelihood_fn.measurement_models.items():
        measurement._update_sim_res_df(sim_results_n)
        l_val += measurement.likelihood()
    return l_val


def likelihood_variability(num_sims, num_likelihood_calculations):
    return np.std([get_likelihood(num_sims, sim_results) for x in range(num_likelihood_calculations)])


list_num_eval = [4, 10, 50, 200, 500, 1000, 5000, 10000]
variability = []
for x in list_num_eval:
    v = likelihood_variability(x, 20)
    print(v)
    variability.append(v)

# variability = [likelihood_variability(x, 20)for x in list_num_eval]

print(variability)
