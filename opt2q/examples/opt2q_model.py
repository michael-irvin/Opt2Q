import pandas as pd
from opt2q.noise import NoiseModel
from opt2q.calibrator import objective_function
from matplotlib import pyplot as plt
import numpy as np

from opt2q.simulator import Simulator
from pysb.examples.michment import model
from opt2q.examples.apoptosis_model import model


# simulate noise
percent = 0.80
n = 1000
nS = int(n*percent)
nM = n - nS

S_phase = pd.DataFrame([['kr', 1000., True, nS]], columns=['param', 'value', 'apply_noise', 'num_sims'])
M_phase = pd.DataFrame([['kr', 300., True, nM]], columns=['param', 'value', 'apply_noise', 'num_sims'])

noise_model_S = NoiseModel(param_mean=S_phase)
noise_model_M = NoiseModel(param_mean=M_phase)

parameters = pd.concat([noise_model_S.run(), noise_model_M.run()],
                       ignore_index=True, sort=False).drop(columns=['simulation'])
parameters['simulation'] = range(len(parameters))

fig, ax = plt.subplots()
parameters[['kr']].plot(kind='hist', density=True, ax=ax, bins=20, alpha=0.5)
parameters[['kr']].plot.kde(ax=ax, legend=False)
plt.show()

# simulate dynamics
sim = Simulator(model=model, param_values=parameters)
results = sim.run(np.linspace(0, 50, 50)).opt2q_dataframe

cm = plt.get_cmap('tab10')
fig1, ax = plt.subplots(figsize=(8, 6))
for name, group in results.groupby(by='simulation'):
    group.plot.line(y='Product', ax=ax, color=cm.colors[0], alpha=0.01, legend=False)
plt.show()

# simulate measurement

quit()
new_params = pd.DataFrame([[np.nan, 'normal', 1],
                           [10.0,   'slow',   1],
                           [1.0e3,  'fast',   1]],
                          columns=['kcat', 'condition', 'experiment'])
sim = Simulator(model=model, param_values=new_params)
results = sim.run(np.linspace(0, 50, 50))

results_df = results.opt2q_dataframe

cm = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(8,6))
for i, (label, df) in enumerate(results_df.groupby(['experiment', 'condition'])):
    df.plot.line(y='Product', ax=ax, label=label, color=cm.colors[i])
plt.legend()
plt.show()

mean = pd.DataFrame([['kcat', 500, 'high_activity'],
                     ['kcat', 100, 'low_activity' ],
                     ['vol',   10, 'high_activity'],
                     ['vol',   10, 'low_activity' ]],
                    columns=['param', 'value', 'experimental_treatment'])
cov = pd.DataFrame([['vol', 'kcat', 25.0], ['vol', 'vol', 3.0]], columns=['param_i', 'param_j', 'value'])
NoiseModel.default_sample_size = 1000
experimental_treatments = NoiseModel(param_mean=mean, param_covariance=cov)
parameters = experimental_treatments.run()

# plot
cm = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(8,6))
for i, (label, df) in enumerate(parameters.groupby('experimental_treatment')):
    df.plot.scatter(x='kcat', y='vol', ax=ax, label=label, color=cm.colors[i])
plt.legend()
plt.show()


param_mean = pd.DataFrame([['vol',        10,       'wild_type',       False],
                           ['kr',        100,       'high_affinity',   np.NaN],
                           ['kcat',      100,       'high_affinity',   np.NaN],
                           ['vol',        10,       'pt_mutation',     True],
                           ['kr',       1000,       'pt_mutation',     False],
                           ['kcat',       10,       'pt_mutation',     True]],
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

    print(my_func.noise1.run().head(10))
    # print(my_func.noise1.experimental_conditions_dataframe)
    # print(my_func.noise2.experimental_conditions_dataframe)
    return


my_func([17, 18, 19, 20, 21])
