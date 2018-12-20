# MW Irvin -- Lopez Lab -- 2018-09-19
"""
======================
Ordinal Logistic Model
======================
Ordinal measurements provide ordinal information about the quantities they measure.

For example, western blot measurements map protein abundance into ordinal categories. It can describe the relative
abundance of a protein as greater or less than a reference, but obscures the interval between them.

`Albeck and Sorger 2008 <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0060299>`_ illustrate
this ordinal transformation in figures comparing western blot measurements of a cell death protein (cPARP) to normalized
measurements of  EC-RP fluorescence (a proxy for cPARP abundance).

The following uses an ordinal :class:`~opt2q.measurement.base.LogisticClassifier` to model the ordinal transformation of
western blot measurements in the Albeck and Sorger paper mentioned above. Data were extracted from these figures using
:class:`WebPlotDigitizer <https://apps.automeris.io/wpd/>`

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from opt2q.measurement.base import Interpolate, LogisticClassifier, Scale

# data
western_blot = pd.read_csv('Albeck_Sorger_WB.csv')
fluorescence = pd.read_csv('Albeck_Sorger_2008_Fluorescence.csv')

# simulation
x = Interpolate('time', ['fluorescence', '1-fluorescence'], [3, 4, 5, 7, 9], groupby=['n']).transform(fluorescence)
x_scaled = Scale(['fluorescence', '1-fluorescence'], scale_fn='log10').transform(x)
lc = LogisticClassifier(western_blot, column_groups={'cPARP': ['fluorescence'], 'PARP': ['1-fluorescence']},
                        classifier_type='ordinal_eoc')
result = lc.transform(x_scaled)

# --- plot results ---
# plot fluorescence data
cm = plt.get_cmap('tab10')
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey='all', gridspec_kw={'width_ratios':[2, 1]})
for n, df in fluorescence.groupby('n'):
    df.plot(x='time', y='fluorescence', label='trail '+str(n), ax=ax, color=cm.colors[0], style='.-')

# plot measurement model
domain = pd.DataFrame({'fluorescence': np.logspace(-2, 0, 10), '1-fluorescence': np.logspace(-2, 0, 10)})
x = Scale(['fluorescence', '1-fluorescence'], scale_fn='log10').transform(domain)
lc.do_fit_transform = False
lc_results = lc.transform(x)

cPARP_results = lc_results.filter(regex='cPARP')
cPARP_results['fluorescence'] = domain['fluorescence']
for col in sorted(list(set(cPARP_results.columns)-{'fluorescence'})):
    cPARP_results.plot(y='fluorescence', x=col, ax=ax1, label=col)
ax1.set_title('Probability')
plt.savefig('fluorescence.png')
plt.show()

# simulated western blot
result['loc1'] = 0.3
result['loc2'] = 0.7

result_obs = {'cPARP': 'loc1', 'PARP':'loc2'}
result_size = {'cPARP': [0, 200, 400, 600, 800], 'PARP': [200, 500, 800]}
fig, ax = plt.subplots(figsize=(8, 3))
for label, df in result.groupby('time'):
    for col in [i for i in df.columns if '__' in i]:
        obs, level = tuple([k_rvs[::-1] for k_rvs in col[::-1].split('__')][::-1])
        df.plot.scatter(x='time', y=result_obs[obs], ax=ax, s=result_size[obs][int(level)],
                        alpha=0.5*np.mean(df[col])**2)
plt.ylim((0, 1))
plt.savefig('simulated_wb.png')
plt.show()

# ---- repeat with new coefficient values ----
# simulated western blot with new values
lc.coefficients = {'cPARP': {'coef_': np.array([10])}}  # one way to do it
lc.set_params(**{'coefficients__cPARP': {'coef_': np.array([9])}})  # other way to do it
result = lc.transform(x_scaled)

# plot fluorescence data
cm = plt.get_cmap('tab10')
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey='all', gridspec_kw={'width_ratios':[2, 1]})
for n, df in fluorescence.groupby('n'):
    df.plot(x='time', y='fluorescence', label='trail '+str(n), ax=ax, color=cm.colors[0], style='.-')

# plot measurement model
domain = pd.DataFrame({'fluorescence': np.logspace(-2, 0, 10), '1-fluorescence': np.logspace(-2, 0, 10)})
x = Scale(['fluorescence', '1-fluorescence'], scale_fn='log10').transform(domain)
lc.do_fit_transform = False
lc_results = lc.transform(x)

cPARP_results = lc_results.filter(regex='cPARP')
cPARP_results['fluorescence'] = domain['fluorescence']
for col in sorted(list(set(cPARP_results.columns)-{'fluorescence'})):
    cPARP_results.plot(y='fluorescence', x=col, ax=ax1, label=col)
ax1.set_title('Probability')
plt.savefig('fluorescence.png')
plt.show()

# simulated western blot
result['loc1'] = 0.3
result['loc2'] = 0.7

result_obs = {'cPARP': 'loc1', 'PARP':'loc2'}
result_size = {'cPARP': [0, 200, 400, 600, 800], 'PARP': [200, 500, 800]}
fig, ax = plt.subplots(figsize=(8, 3))
for label, df in result.groupby('time'):
    for col in [i for i in df.columns if '__' in i]:
        obs, level = tuple([k_rvs[::-1] for k_rvs in col[::-1].split('__')][::-1])
        df.plot.scatter(x='time', y=result_obs[obs], ax=ax, s=result_size[obs][int(level)],
                        alpha=0.5*np.mean(df[col])**2)
plt.ylim((0, 1))
plt.savefig('simulated_wb.png')
plt.show()

