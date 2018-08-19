"""
Scatter plot of noise model results
-----------------------------------
Simple example of the Opt2Q noise simulation.

"""

import pandas as pd
from opt2q.noise import NoiseModel
from matplotlib import pyplot as plt

mean = pd.DataFrame([['kcat', 500, 'high_activity'],
                     ['kcat', 100, 'low_activity' ],
                     ['vol',   10, 'high_activity'],
                     ['vol',   10, 'low_activity' ]],
                    columns=['param', 'value', 'experimental_treatment'])
cov = pd.DataFrame([['vol', 'kcat', 1.0], ['vol', 'vol', 3.0]], columns=['param_i', 'param_j', 'value'])
experimental_treatments = NoiseModel(param_mean=mean, param_covariance=cov)
