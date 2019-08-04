# MW Irvin -- Lopez Lab -- 2019-06-24

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from opt2q.simulator import Simulator
from opt2q.examples.apoptosis_model_ import model
from opt2q.measurement.base import LogisticClassifier, Scale, Standardize
from opt2q.measurement.base.functions import polynomial_features
from opt2q.examples.cell_viability_example.calibrating_to_cell_viability_data import sampled_params_0, n_chains

chain_history = np.load('PyDREAM_CellViability_20190507_DREAM_chain_history.npy')

chain_log_post_0 = np.load('PyDREAM_CellViability_201905070_10000.npy')
chain_log_post_1 = np.load('PyDREAM_CellViability_201905071_10000.npy')
chain_log_post_2 = np.load('PyDREAM_CellViability_201905072_10000.npy')


n_params = len(sampled_params_0)
param_names = ['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc2_cv', 'kc3_cv', 'kc2_kc3_cor', 'LR coef 1', 'LR coef 2',
               'LR coef 3', 'LR coef 4', 'LR coef 5', 'LR intercept']
parameters = chain_history.reshape(int(len(chain_history)/n_params), n_params)

idx_highest_posterior_params = np.argmax(chain_log_post_2)
highest_posterior_params = parameters[2::3, :][np.argmax(chain_log_post_2), :]
print(highest_posterior_params)
quit()
cm = plt.get_cmap('tab10')

# Posterior Values Per Chain Convergence
# plt.plot(chain_log_post_0, alpha=0.8, label='chain 0')
# plt.plot(chain_log_post_1, alpha=0.8, label='chain 1')a
# plt.plot(chain_log_post_2, alpha=0.8, label='chain 2')
#
# plt.title("Log Posterior")
# plt.xlabel('Iteration')
# plt.legend()
# plt.show()
#
# plt.plot(range(2000, 10000), chain_log_post_0[2000:], alpha=0.8, label='chain 0')
# plt.plot(range(2000, 10000), chain_log_post_1[2000:], alpha=0.8, label='chain 1')
# plt.plot(range(2000, 10000), chain_log_post_2[2000:], alpha=0.8, label='chain 2')
#
# plt.title("Log Posterior")
# plt.xlabel('Iteration')
# plt.legend()
# plt.show()
#
# # Convergence of Parameter Values Per Chain
# for param_num in range(n_params):
#     for chain_num in range(n_chains):
#         plt.plot(parameters[chain_num::n_chains, param_num], label='chain %s' % chain_num, alpha=0.3)
#     plt.legend()
#     plt.title('Parameter %s; %s' % (param_num, param_names[param_num]))
#     plt.ylabel('parameter value')
#     plt.xlabel('iteration')
#     plt.show()
#
# # Parameter Values
# for param_num in range(n_params):
#     init_dist = sampled_params_0[param_num].dist
#     x = np.linspace(init_dist.ppf(0.01), init_dist.ppf(0.99), 100)
#     plt.plot(x, init_dist.pdf(x), '--', color=cm.colors[0], label='initial')
#
#     param_samples = parameters[2000*n_chains:, param_num]
#     xs = np.linspace(np.min(param_samples), np.max(param_samples), 100)
#     density = gaussian_kde(param_samples)
#
#     plt.hist(parameters[2000*n_chains:, param_num], normed=True, bins=30, alpha=0.5, color=cm.colors[4])
#     plt.plot(xs, density(xs), label='final', color=cm.colors[0])
#     plt.legend()
#     plt.xlabel('parameter value')
#     plt.title('Marginal distribution of parameter %s (Cell Viability)' % param_names[param_num])
#     plt.show()
#
# # Joint Parameter Value Distributions
# params_df = pd.DataFrame(parameters[2000*n_chains:, :], columns=param_names)
# g = sns.pairplot(params_df, diag_kind="kde", markers="+",
#                  plot_kws=dict(s=50, color=cm.colors[0], edgecolor=cm.colors[0], linewidth=1, alpha=0.01),
#                  diag_kws=dict(shade=True, color=cm.colors[0]))
# for i, j in zip(*np.triu_indices_from(g.axes, 1)):
#     g.axes[i, j].set_visible(False)
# plt.show()


# ======== Cell Death Classifier =========
cell_death_data = pd.read_csv('cell_death_data.csv', skiprows=[0])
cell_death_data = cell_death_data.assign(log10_k=np.log10(cell_death_data['k'].clip(lower=1.5e-7)))
# The classifier coefficients correspond columns (by name in alphabetical order).
# Therefore columns names should be the same in the calibrated model.
cell_death_data = cell_death_data.rename(columns={"log10_k": "cPARP_obs", "Time of Max C8 activity (tau)": "time"})

# ------- trying polynomial basis function -------
scale = Scale(columns=['cPARP_obs', 'time'], scale_fn=polynomial_features, **{'degree': 2})
scaled_x = scale.transform(cell_death_data[['cPARP_obs', 'time', 'Cell #']])

# Note: This would normally work, but LogisticClassifier may not track new columns that contain whitespace.
# The PySB simulation result and Opt2Q transform do not have whitespace.
lc_poly = LogisticClassifier(dataset=cell_death_data,
                             column_groups={'Surviving': ['cPARP_obs', 'time']},
                             classifier_type='nominal')

lc_poly.transform(scaled_x)  # 'Cell #' is the index

# plot dead and surviving cells
groups = cell_death_data.groupby('Surviving')
names = ['Dead', 'Surviving']

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['cPARP_obs'], group['time'],
            marker='o', linestyle='', ms=10, alpha=0.7, label=names[name], color=cm.colors[name])
ax.set_xlabel('log10(k), k=rate of change in Caspase indicator')
ax.set_ylabel('Time of Max Caspase activity (tau)')
ax.set_title("""Apoptosis in cells treated with 10ng TRAIL with
             Opt2Q Calibrated Model of 50% Survival Probability Threshold""")
ax.legend()

x1 = np.linspace(-7, -4, 10)
x2 = np.linspace(50, 600, 10)
nx1, nx2 = np.meshgrid(x1, x2)

grid = np.c_[nx1.ravel(), nx2.ravel()]
for i in np.random.randint(2000, 10000, 50):
    x = parameters[i, -6:]
    viability_coef = np.array([[x[0],         # :  (-100, 100),   float
                                x[1],         # :  (-100, 100),   float
                                x[2],         # :  (-100, 100),   float
                                x[3],         # :  (-100, 100),   float
                                x[4]]])       # :  (-100, 100),   float
    viability_intercept = np.array([x[5]])    # :  (-10, 10)]     float

    measurement_model_params = {'coefficients__Surviving__coef_': viability_coef,
                                'coefficients__Surviving__intercept_': viability_intercept}
    lc_poly.set_params(**measurement_model_params)
    lc_poly.set_params(**{'do_fit_transform': False})  # Don't re-fit the model coefficients.

    scaled_grid = scale.transform(pd.DataFrame(grid, columns=['cPARP_obs', 'time']))

    std_scaled_grid = Standardize(columns=['time', 'cPARP_obs', 'time^2', 'time$cPARP_obs', 'cPARP_obs^2'],
                                  groupby=None).transform(scaled_grid)
    probabilities = lc_poly.transform(std_scaled_grid).values[:, 1].reshape(nx1.shape)
    cs = ax.contour(nx1, nx2, probabilities, colors=['black'], alpha=0.25, levels=[0.5])
    # ax.clabel(cs, inline=1, fontsize=10)
plt.show()

sim = Simulator(model=model, solver='scipyode')
sim_results_0 = sim.run(np.linspace(0, 5000, 100))
results_df_0 = sim_results_0.opt2q_dataframe
results_df_0 = results_df_0.reset_index()
results_df_time_hrs_0 = results_df_0['time'].apply(lambda x: x/3600)  # convert to hrs for plot

n_0 = np.random.randint(2000, 10000, 100)
dynamical_params = 10**parameters[n_0, :6]
sim.param_values = pd.DataFrame(dynamical_params,
                                columns=['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7'])
sim_results = sim.run(np.linspace(0, 5000, 100))

results_df = sim_results.opt2q_dataframe
results_df = results_df.reset_index()
results_df_time_hrs = results_df['time'].apply(lambda x: x/3600)  # convert to hrs for plot

fig = plt.figure()
for s, df in results_df.groupby('simulation'):
    plt.plot(df['time'], df['BID_obs'], color=cm.colors[0], alpha=0.5, label='calibrated params')
plt.plot(results_df_time_hrs_0, results_df_0['BID_obs'], '--', color=cm.colors[0], alpha=0.5, label='starting params')
plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results (tBID)')
custom_lines = [Line2D([0], [0], color=cm.colors[0], linestyle='--'),
                Line2D([0], [0], color=cm.colors[0])]
plt.legend(custom_lines, ['starting params', 'calibrated params'])
plt.savefig('fig1.png')
fig.show()

fig = plt.figure()
for s, df in results_df.groupby('simulation'):
    plt.plot(df['time'], df['cPARP_obs'], color=cm.colors[1], alpha=0.5, label='calibrated params')
plt.plot(results_df_time_hrs_0, results_df_0['cPARP_obs'], '--', color=cm.colors[1], alpha=0.5, label='starting params')
plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results (cPARP)')
custom_lines = [Line2D([0], [0], color=cm.colors[1], linestyle='--'),
                Line2D([0], [0], color=cm.colors[1])]
plt.legend(custom_lines, ['starting params', 'calibrated params'])
plt.savefig('fig2.png')
fig.show()