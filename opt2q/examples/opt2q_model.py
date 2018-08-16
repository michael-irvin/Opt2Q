import pandas as pd
import numpy as np

from opt2q.noise import NoiseModel

param_mean = pd.DataFrame([['vol',   10, 'wild_type',    False],
                           ['kr',   100, 'low_affinity', np.NaN],
                           ['kcat', 100, 'low_affinity', np.NaN],
                           ['vol',   10, 'pt_mutation',  True],
                           ['kr',  1000, 'pt_mutation',  False],
                           ['kcat',  10, 'pt_mutation',  True]],
                          columns=['param', 'value', 'exp_condition', 'apply_noise'])

param_cov = pd.DataFrame([['kr', 'kcat', 0.1, 'low_affinity']],
                         columns=['param_i','param_j','value', 'exp_condition'])

noise_model = NoiseModel(param_mean=param_mean, param_covariance=param_cov)

noise_model_2 = NoiseModel(param_mean = pd.DataFrame([['vol',   20, 'pt_mutation',  True],
                                                      ['kr',  1000, 'pt_mutation',  False],
                                                      ['kcat',  50, 'pt_mutation',  True]],
                                                     columns=['param', 'value', 'exp_condition', 'apply_noise']))
