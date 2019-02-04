# MW Irvin -- Lopez Lab -- 2019-02-02
"""
==============================
Western Blot Measurement Model
==============================

Presenting a western blot measurement model that is based on an empirical relationship, that Albeck and Sorger
described, between normalized fluorescence and western blot measurements of cPARP. This model also considers key
steps in the western blot process that engender its ordinal scale.

The western blot features sample prep (e.g. cell lysis) that essentially averages the intracellular contents of
individual cells. The variability between cells is lost, but can contribute to the variance in concentration of prepped
samples.

Technical noise sources obscure the interval between ordinal measurements but retains ordering information such
as "greater than" and "less than".

`Albeck and Sorger 2008 <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0060299>`_ illustrate
this ordinal transformation in figures comparing western blot measurements of a cell death protein (cPARP) to normalized
measurements of  EC-RP fluorescence (a proxy for cPARP abundance).

The following uses an ordinal :class:`~opt2q.measurement.base.LogisticClassifier` to model the ordinal transformation of
western blot measurements in the Albeck and Sorger paper mentioned above. Data were extracted from these figures using
:class:`WebPlotDigitizer <https://apps.automeris.io/wpd/>`.

The following describes individual steps (or :class:`~opt2q.measurement.base.Transform`) that compose the measurement
model pipeline.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from opt2q.measurement.base import Interpolate, LogisticClassifier, Scale, SampleAverage

# ------- Data -------
script_dir = os.path.dirname(__file__)
fluorescence_file_path = os.path.join(script_dir, 'Albeck_Sorger_2008_Fluorescence.csv')
western_blot_file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')

fluorescence = pd.read_csv(fluorescence_file_path)
western_blot = pd.read_csv(western_blot_file_path)

fig, ax = plt.subplots()
fluorescence.groupby('n').plot(x='time', y='fluorescence', ax=ax, color='green', style='.-', legend=False)
ax.set_xlabel('time [hrs]')
ax.set_ylabel('fluorescence')
ax.set_title('EC-RP Fluorescence in Cells Treated with 10ng/mL TRAIL')
fig.show()

# ======= Measurement Pipeline ========
# ------- Interpolate ---------
# Interpolate fluorescence data at time-points in Western blot
interpolate = Interpolate('time',                                 # independent variable
                          ['fluorescence', '1-fluorescence'],     # dependent variable(s)
                          [3, 4, 5, 7, 9],                        # new values
                          groupby='n'                             # do a separate interpolation for each unique "n"
                          )
interpolated_fl = interpolate.transform(fluorescence)

# ------- Sample Average -------
# Model's cell lysis step
sample_avr = SampleAverage(columns=['fluorescence', '1-fluorescence'],  # columns to average
                           groupby=['time'],                            # do a separate average for each time point
                           drop_columns=['n'],                          # drop 'n'
                           apply_noise=True,                            # simulate noise associated with the sample prep
                           variances=0.01                               # variance of the applied noise
                           )
sample_avr_fl = sample_avr.transform(interpolated_fl)

# plot sample average
fig, (axes, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), sharey='all', gridspec_kw={'width_ratios':[2, 1, 1]})
vp = axes.violinplot(dataset=[sample_avr_fl[sample_avr_fl.time == 3]['fluorescence'],
                              sample_avr_fl[sample_avr_fl.time == 4]['fluorescence'],
                              sample_avr_fl[sample_avr_fl.time == 5]['fluorescence'],
                              sample_avr_fl[sample_avr_fl.time == 7]['fluorescence'],
                              sample_avr_fl[sample_avr_fl.time == 9]['fluorescence']])
for parts in vp['bodies']:
    parts.set_facecolor('green')
    parts.set_edgecolor('green')
axes.set_xticklabels([1, 3, 4, 5, 7, 9])
axes.set_xlabel('time [hrs]')
axes.set_ylabel('simulated sample average')
axes.set_title("Simulated Concentration. of Prepped Samples")

# ------- Log Transform -------
# Log transform reflects the multiplicative error observed in Western blots (and biological processes in general).
log_scale = Scale(columns=['fluorescence', '1-fluorescence'],
                  scale_fn='log10')
scaled_fl = log_scale.transform(sample_avr_fl)

# ------- Ordinal Logistic Classifier -------
classifier = LogisticClassifier(western_blot,                                # Supervised learning requires targets
                                column_groups={'cPARP': ['fluorescence'],    # Assign each target a list of features
                                               'PARP': ['1-fluorescence']},  # do
                                classifier_type='ordinal_eoc'                # Ordinal classifier w/ empirical
                                )                                            # ordering constraint
result = classifier.transform(scaled_fl)

# plot classifier
classifier_domain = np.logspace(-3, 0.02, 100)
features = pd.DataFrame({'fluorescence': classifier_domain,'1-fluorescence': classifier_domain})
scaled_features = log_scale.transform(features)

classifier.do_fit_transform = False  # Important. Set to False, so trained model transforms new 'out-of-sample' values.
lc_results = classifier.transform(scaled_features)

cPARP_results = lc_results.filter(regex='cPARP')
cPARP_results['fluorescence'] = features['fluorescence']
for col in sorted(list(set(cPARP_results.columns)-{'fluorescence'})):
    cPARP_results.plot(y='fluorescence', x=col, ax=ax2, label=col)

PARP_results = lc_results[['PARP__0', 'PARP__1', 'PARP__2']]
PARP_results['1-fluorescence'] = features['1-fluorescence']
for col in sorted(list(set(PARP_results.columns)-{'1-fluorescence'})):
    PARP_results.plot(y='1-fluorescence', x=col, ax=ax3, label=col)

ax2.tick_params(axis='x',
                which='both',
                bottom=False,
                labelbottom=False)
ax2.set_xlabel("Probability")
ax2.set_title("""Probability of blot  
    category vs Fluorescence""")

ax3.tick_params(axis='x',
                which='both',
                bottom=False,
                labelbottom=False)
ax3.set_xlabel("Probability")
ax3.set_title("""Probability of blot  
    category vs Fluorescence""")

plt.show()

# plot simulated western blot
result['loc1'] = 0.1
result['loc2'] = 0.2
result_obs = {'cPARP': 'loc1', 'PARP':'loc2'}
result_size = {'cPARP': [1, 3, 5, 8, 11], 'PARP': [3, 7, 10]}

cm = plt.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(6, 2))
for label, df in result.groupby('time'):
    for col in [i for i in df.columns if '__' in i]:
        obs, level = tuple([k_rvs[::-1] for k_rvs in col[::-1].split('__')][::-1])
        df.plot.scatter(x='time', y=result_obs[obs], ax=ax, marker='_',
                        lw=result_size[obs][int(level)], s=500,
                        alpha=0.01*np.mean(df[col])**2, color=cm.colors[1])

ax.set_title("Simulated Western Blot")
ax.set_yticks([0.1, 0.2])
ax.set_yticklabels(['cPARP', 'PARP'], fontdict={'fontsize': 12})
ax.set_ylabel("")
ax.set_xlabel("time [hrs]")
plt.ylim((0.0, 0.3))
plt.show()
