# MW Irvin -- Lopez Lab -- 2018-02-02

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt2q.noise import NoiseModel

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')
western_blot = pd.read_csv(file_path)

# represent higher ordinal categories larger rectangles
parp_blot_sizes = [3, 7, 10]
western_blot['PARP_sizes'] = western_blot['PARP'].apply(lambda x: parp_blot_sizes[x])
western_blot['PARP_loc'] = 0.2

cparp_blot_sizes = [1, 3, 5, 8, 11]
western_blot['cPARP_sizes'] = western_blot['cPARP'].apply(lambda x: cparp_blot_sizes[x])
western_blot['cPARP_loc'] = 0.1

cm = plt.get_cmap('tab20c')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4), sharex='all')
axes = {10: (ax1, cm.colors[6]), 50: (ax2, cm.colors[5]), 250: (ax3, cm.colors[4])}
for label, df in western_blot.groupby('TRAIL_conc'):
    axes[label][0].scatter(x=df['time'], y=df['PARP_loc'], lw=df['PARP_sizes'],
                s=500, marker='_', color=axes[label][1])

    axes[label][0].scatter(x=df['time'], y=df['cPARP_loc'], lw=df['cPARP_sizes'],
                           s=500, marker='_', color=axes[label][1])

    axes[label][0].set_title(f'{label} ng/mL TRAIL')
    axes[label][0].set_ylim(0.0, 0.3)
    axes[label][0].set_yticks([0.1, 0.2])
    axes[label][0].set_yticklabels(['cPARP', 'PARP'], fontdict={'fontsize': 12})
    axes[label][0].set_xticks([0, 1, 2, 3, 4, 5, 7, 9])


plt.xlabel('time [hrs]')
plt.show()

# ======= Dynamical Model ========
# ------- Noise Model --------
# Params
ligand = pd.DataFrame([['L_0',   600,  10, False],      # 'TRAIL_conc' column annotates experimental treatments
                       ['L_0',  3000,  50, False],      # 600 copies per cell corresponds to 10 ng/mL TRAIL
                       ['L_0', 15000, 250, False]],
                      columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

C_0, kc3, kc4, kf3, kf4 = (1e4, 1e-2, 1e-2, 1e-6, 1e-6)
k_values = pd.DataFrame([['kc3', kc3, False],
                         ['kc4', kc4, False],
                         ['kf3', kf3, True],
                         ['kf4', kf4, True],
                         ['C_0', C_0, True]],
                        columns=['param', 'value', 'apply_noise'])\
    .iloc[np.repeat(range(5), 3)]                       # Repeat for each of the 3 experimental treatments
k_values['TRAIL_conc'] = np.tile([10, 50, 250], 5)      # Repeat for each of the 5 parameter
param_means = pd.concat([ligand, k_values], sort=False)

kf3_var, kf4_var, kf3_kf4_covariance = (4e-14, 4e-14, 1e-14)
param_variances = pd.DataFrame([['kf3', 'kf3', kf3_var],
                                ['kf4', 'kf4', kf4_var],
                                ['kf3', 'kf4', kf3_kf4_covariance]],  # Covariance between 'kf3' and kf4'
                               columns=['param_i', 'param_j', 'value'])

NoiseModel.default_coefficient_of_variation = 0.25      # 'C_0' takes default variability of 25%
NoiseModel.default_sample_size = 500

# Noise Model
noise = NoiseModel(param_mean=param_means, param_covariance=param_variances)
parameters = noise.run()

# plot parameters
cm = plt.get_cmap('tab20b')
fig, ax = plt.subplots(figsize=(4, 4))
ax.hist(parameters['C_0'], bins=20, alpha=0.4, density=True, color=cm.colors[12])
ax.tick_params(axis='y',
               which='both',
               left=False,
               labelleft=False)
plt.xlabel('Caspase copies per cell')
plt.title('Variability in Caspase Conc. Parameter')
plt.show()

parameters[['kf3', 'kf4']].plot.scatter(x='kf3', y='kf4', color=cm.colors[12] , alpha=0.1)
plt.xlim(0, 0.001)
plt.ylim(0, 0.001)
plt.show()