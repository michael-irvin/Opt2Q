# MW Irvin -- Lopez Lab -- 2019-02-14
"""
================================
Cell Viability Measurement Model
================================

Presenting a cell viability measurement model that is based on an empirical relationship, between initiator caspase
activity and cell death, model by `Roux and Sorger (2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4461398/>` .
They use a fluorescent indicator to monitor initiator caspase activity.

Informative features in their classifier model (i.e. classifying cell death and survival) are 1.) maximum initiator
caspase activity and 2.) the time at which initiator caspase activity maximizes.

The following describes the individual steps (or :class:`transforms <opt2q.measurement.base.Transform>`) that encode
the features and classifier model needed to model cell death as a function of capsase activity.
"""
import os
import pandas as pd
from matplotlib import pyplot as plt

# ------- Data -------
script_dir = os.path.dirname(__file__)
cell_viability_file_path = os.path.join(script_dir, 'Roux_Sorger_2015_cell_viability.tsv')
cell_death_10_file_path = os.path.join(script_dir, 'Roux_Sorger_2015_cell_parameters_TRAIL_10ng-mL.txt')
cell_death_25_file_path = os.path.join(script_dir, 'Roux_Sorger_2015_cell_parameters_TRAIL_25ng-mL.txt')
cell_death_50_file_path = os.path.join(script_dir, 'Roux_Sorger_2015_cell_parameters_TRAIL_50ng-mL.txt')

cell_viability = pd.read_csv(cell_viability_file_path, sep='\t')
cell_death_10 = pd.read_csv(cell_death_10_file_path, sep='\t', header=1)
cell_death_25 = pd.read_csv(cell_death_25_file_path, sep='\t', header=1)
cell_death_50 = pd.read_csv(cell_death_50_file_path, sep='\t', header=1)

# plot manually calculated cell-viability
cv_data = [cell_death_10['Surviving'].sum()/float(cell_death_10['Cell #'].max()),
           cell_death_25['Surviving'].sum()/float(cell_death_25['Cell #'].max()),
           cell_death_50['Surviving'].sum()/float(cell_death_50['Cell #'].max())]

plt.figure(figsize=(4, 4.5))
plt.bar([0, 1, 2], cv_data, color=(35/256., 59/256., 246/256., 0.6))
plt.xticks([0, 1, 2], [10, 25, 50])
plt.xlabel('TRAIL Concentration [ng/mL]')
plt.ylim((0, 1))
plt.title('Cell Viability')
plt.show()
