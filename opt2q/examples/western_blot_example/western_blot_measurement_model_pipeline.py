# MW Irvin -- Lopez Lab -- 2018-02-02

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')
western_blot = pd.read_csv(file_path)

# represent higher ordinal categories larger circles
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
