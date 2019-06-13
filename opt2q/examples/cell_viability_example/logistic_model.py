# MW Irvin -- Lopez Lab -- 2018-12-10
"""
=========================
Logistic Classifier Model
=========================

Nominal observations provide information about the quantifiable attributes (i.e. markers) on which they depend.

For example, programmed cell death (apoptosis) depends on caspase activity; as such, apoptotic cells will more likely
have similar caspase activity that differs from that of surviving cells.

`Albeck and Sorger 2015 <http://msb.embopress.org/content/11/5/803.long>`_ find that the maximum rate of change in
caspase indicator, and the time when that maximum occurs, predicts cellular commitment to apoptosis with 83% accuracy.

The following uses the :class:`~opt2q.measurement.base.LogisticClassifier` to model the relationship between cell
death observations by Albeck and Sorger, and capsase activity. Cell death observations and corresponding caspase
activity measurements were downloaded from their lab `website <http://lincs.hms.harvard.edu/roux-molsystbiol-2015/>_.

"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt2q.measurement.base import LogisticClassifier, Scale
from opt2q.measurement.base.functions import polynomial_features

# data
cell_death_data = pd.read_csv('cell_death_data.csv', skiprows=[0])
cell_death_data = cell_death_data.assign(log10_k=np.log10(cell_death_data['k'].clip(lower=1.5e-7)))

# ------- plot data -------
cm = plt.get_cmap('tab10')
groups = cell_death_data.groupby('Surviving')
names = ['Dead', 'Surviving']

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['log10_k'], group['Time of Max C8 activity (tau)'],
            marker='o', linestyle='', ms=10, alpha=0.7, label=names[name], color=cm.colors[name])
ax.set_xlabel('log10(k), k=rate of change in Caspase indicator')
ax.set_ylabel('Time of Max Caspase activity (tau)')
ax.set_title('Apoptosis in cells treated with 10ng TRAIL')
ax.legend()

# ------- classifier ------
lc = LogisticClassifier(dataset=cell_death_data,
                        column_groups={'Surviving': ['log10_k', 'Time of Max C8 activity (tau)']},
                        classifier_type='nominal')
result = lc.transform(cell_death_data[['log10_k', 'Time of Max C8 activity (tau)', 'Cell #']])  # 'Cell #' is the index

# ------- plot classifier results --------
x1 = np.linspace(-7, -4, 10)
x2 = np.linspace(50, 600, 10)
nx1, nx2 = np.meshgrid(x1, x2)

grid = np.c_[nx1.ravel(), nx2.ravel()]
lc.set_params(**{'do_fit_transform':False})  # Don't re-fit the model coefficients.
probs = lc.transform(pd.DataFrame(grid, columns=['log10_k', 'Time of Max C8 activity (tau)'])).values[:, 1].reshape(nx1.shape)
cs = ax.contour(nx1, nx2, probs, colors=['black'], alpha=0.75)
ax.clabel(cs, inline=1, fontsize=10)
plt.show()

# ------- trying polynomial basis function -------
scale = Scale(columns=['log10_k', 'Time of Max C8 activity (tau)'], scale_fn=polynomial_features, degree=2)
scaled_x = scale.transform(cell_death_data[['log10_k', 'Time of Max C8 activity (tau)', 'Cell #']])

# Note: This would normally work, but LogisticClassifier may not track new columns that contain whitespace.
# The PySB simulation result and Opt2Q transform do not have whitespace.
lc_poly = LogisticClassifier(dataset=cell_death_data,
                             column_groups={'Surviving': ['log10_k', 'Time of Max C8 activity (tau)']},
                             classifier_type='nominal')

# Manually track added column names:
# lc_poly = LogisticClassifier(dataset_fluorescence=cell_death_data,
#                              column_groups={'Surviving': set(scaled_x.columns)-{'Cell #'}},
#                              classifier_type='nominal')

lc_poly.transform(scaled_x)  # 'Cell #' is the index


# ------- plot classifier results --------
cm = plt.get_cmap('tab10')
groups = cell_death_data.groupby('Surviving')
names = ['Dead', 'Surviving']

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['log10_k'], group['Time of Max C8 activity (tau)'],
            marker='o', linestyle='', ms=10, alpha=0.7, label=names[name], color=cm.colors[name])
ax.set_xlabel('log10(k), k=rate of change in Caspase indicator')
ax.set_ylabel('Time of Max Caspase activity (tau)')
ax.set_title('Apoptosis in cells treated with 10ng TRAIL')
ax.legend()

x1 = np.linspace(-7, -4, 10)
x2 = np.linspace(50, 600, 10)
nx1, nx2 = np.meshgrid(x1, x2)

grid = np.c_[nx1.ravel(), nx2.ravel()]
lc_poly.set_params(**{'do_fit_transform':False})  # Don't re-fit the model coefficients.
scaled_grid = scale.transform(pd.DataFrame(grid, columns=['log10_k', 'Time of Max C8 activity (tau)']))
probabilities = lc_poly.transform(scaled_grid).values[:, 1].reshape(nx1.shape)
cs = ax.contour(nx1, nx2, probabilities, colors=['black'], alpha=0.75)  # , levels=np.linspace(0.1, 0.9, 15))
ax.clabel(cs, inline=1, fontsize=10)
plt.show()