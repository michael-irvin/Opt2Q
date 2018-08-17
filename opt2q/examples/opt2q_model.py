import pandas as pd
import numpy as np

from opt2q.noise import NoiseModel
from opt2q.calibrator import objective_function
param_mean = pd.DataFrame([['a', 2.0, 3, True],
                           ['b', 0.0, 1, True]],
                          columns=['param', 'value', 'num_sims', 'apply_noise'])
param_cov = pd.DataFrame([['a', 'c', 0.1]], columns=['param_i', 'param_j', 'value'])
NoiseModel.default_param_values = {'c':3.0}
NoiseModel.default_sample_size=10000000000000000000000000000000000000000

nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
test = nm._add_noisy_values(nm.param_mean, nm.param_covariance, nm.experimental_conditions_dataframe)
print(test.mean())
print(test.cov())
quit()


param_mean = pd.DataFrame([['vol',   10, 'wild_type',    False],
                           ['kr',   100, 'high_affinity', np.NaN],
                           ['kcat', 100, 'high_affinity', np.NaN],
                           ['vol',   10, 'pt_mutation',  True],
                           ['kr',  1000, 'pt_mutation',  False],
                           ['kcat',  10, 'pt_mutation',  True]],
                          columns=['param', 'value', 'exp_condition', 'apply_noise'])

param_cov = pd.DataFrame([['kr', 'kcat', 0.1, 'high_affinity']],
                         columns=['param_i','param_j','value', 'exp_condition'])

noise_model_1 = NoiseModel(param_mean=param_mean, param_covariance=param_cov)

noise_model_2 = NoiseModel(param_mean=pd.DataFrame([['vol',   20, 'pt_mutation',  True],
                                                    ['kr',  1000, 'pt_mutation',  False],
                                                    ['kcat',  50, 'pt_mutation',  True]],
                                                   columns=['param', 'value', 'exp_condition', 'apply_noise']))


@objective_function(noise1=noise_model_1, noise2=noise_model_2)
def my_func(x):
    # The 'vol' param for the 'pt_mutations' exp_condition has a distribution that is a mixture of two log-normal pdfs
    vol1, vol2 = x[0], x[1]
    n1, n2 = x[2], 100 - x[2]

    my_func.noise1.update_values(param_mean=pd.DataFrame([['vol', vol1]], columns=['param', 'value']))
    my_func.noise2.update_values(param_mean=pd.DataFrame([['vol', vol2]], columns=['param', 'value']))
    my_func.noise1.update_values(param_mean=pd.DataFrame([['pt_mutation', n1]], columns=['exp_condition', 'num_sims']))
    my_func.noise2.update_values(param_mean=pd.DataFrame([['pt_mutation', n2]], columns=['exp_condition', 'num_sims']))

    # The 'high_affinity' exp_condition applies noise to 'kr' and 'kcat'
    my_func.noise1.update_values(
        param_mean=pd.DataFrame([['kr', 'high_affinity', x[3]]], columns=['param', 'exp_condition', 'value']),
        param_covariance=pd.DataFrame([['kr', x[4], 'high_affinity']], columns=['param_i', 'value', 'exp_condition'])
    )

    print(my_func.noise1.param_mean)
    print(my_func.noise2.param_mean)
    print(my_func.noise1.param_covariance)
    print(my_func.noise1.experimental_conditions_dataframe)
    print(my_func.noise2.experimental_conditions_dataframe)
    return


my_func([17, 18, 19, 20, 21])
