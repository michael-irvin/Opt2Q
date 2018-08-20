"""
Scatter plot of noise model results
-----------------------------------
Simple example of the Opt2Q noise simulation.

"""

import pandas as pd
from opt2q.noise import NoiseModel
from matplotlib import pyplot as plt


mean = pd.DataFrame([['kcat', 200, 'high_activity', 200],
                     ['kcat', 100, 'low_activity' , 200],
                     ['vol',   10, 'high_activity', 100],
                     ['vol',   10, 'low_activity' , 100]],
                    columns=['param', 'value', 'experimental_treatment', 'num_sims'])
cov = pd.DataFrame([['vol', 'kcat', 30.0], ['vol', 'vol', 3.0]], columns=['param_i', 'param_j', 'value'])
NoiseModel.default_sample_size = 200
experimental_treatments = NoiseModel(param_mean=mean, param_covariance=cov)
parameters = experimental_treatments.run()

cm = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(8,6))
for i, (label, df) in enumerate(parameters.groupby('experimental_treatment')):
    df.plot.scatter(x='kcat', y='vol', ax=ax, label=label, color=cm.colors[i])
plt.legend()
plt.show()
