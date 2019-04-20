# MW Irvin -- Lopez Lab -- 2018-02-02

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from opt2q.data import DataSet
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import WesternBlot
from opt2q.examples.apoptosis_model_ import model


# ======= Measurement Model Pipeline ========

# ------- Data -------
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Albeck_Sorger_WB.csv')
western_blot = pd.read_csv(file_path)
western_blot['time'] = western_blot['time'].apply(lambda x: x*3600)  # Convert to [s] (as is in the PySB model).

# ------- Noise Model --------
noise_model_sample_size = 300

# Params
ligand = pd.DataFrame([['L_0',   600,  10, False],      # 'TRAIL_conc' column annotates experimental treatments
                       ['L_0',  3000,  50, False],      # 600 copies per cell corresponds to 10 ng/mL TRAIL
                       ['L_0', 15000, 250, False]],
                      columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

kc0, kc2, kf3, kc3, kf4, kr7 = (1.0e-05, 1.0e-02, 3.0e-08, 1.0e-02, 1.0e-06, 1.0e-02)
k_values = pd.DataFrame([['kc0', kc0, True],
                         ['kc2', kc2, True],   # co-vary with kc3
                         ['kf3', kf3, False],
                         ['kc3', kc3, True],
                         ['kf4', kf4, False],
                         ['kr7', kr7, False]],
                        columns=['param', 'value', 'apply_noise'])\
    .iloc[np.repeat(range(5), 3)]                       # Repeat for each of the 3 experimental treatments
k_values['TRAIL_conc'] = np.tile([10, 50, 250], 5)      # Repeat for each of the 5 parameter
param_means = pd.concat([ligand, k_values], sort=False)

kc2_cv, kc3_cv, kc2_kc3_cor = (0.2, 0.2, 0.25)
kc2_var, kc3_var, kc2_kc3_cov = ((kc2 * kc2_cv) ** 2, (kc3 * kc3_cv) ** 2, kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor)
param_variances = pd.DataFrame([['kc2', 'kc2', kc2_var],
                                ['kc3', 'kc3', kc3_var],
                                ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance between 'kf3' and kf4'
                               columns=['param_i', 'param_j', 'value'])

NoiseModel.default_coefficient_of_variation = 0.25      # 'kc_0' takes default variability of 25%
NoiseModel.default_sample_size = noise_model_sample_size

# Noise Model
noise = NoiseModel(param_mean=param_means, param_covariance=param_variances)
parameters = noise.run()

# ------- Dynamical Model -------
sim = Simulator(model=model, param_values=parameters, solver='cupsoda',
                integrator_options={'n_blocks': 64, 'memory_usage': 'global', 'vol': 4e-15})

results = sim.run(np.linspace(0, 32400, 100))

# ------- Measurement Model -------
# A separate western blot was done each experimental condition
experimental_conditions = [10, 50, 250]
dataset = {ec: DataSet(data=western_blot[western_blot['TRAIL_conc'] == ec], measured_variables=['cPARP', 'PARP'])
           for ec in experimental_conditions}

western_blot_models = {ec: WesternBlot(simulation_result=results,
                                       dataset=dataset[ec],
                                       measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
                                       observables=['PARP_obs', 'cPARP_obs'],
                                       experimental_conditions=pd.DataFrame([ec], columns=['TRAIL_conc']),
                                       time_points=[0, 3600, 7200, 10800, 14400, 18000, 25200, 32400])
                       for ec in experimental_conditions}

for ec in experimental_conditions:
    western_blot_models[ec].process.set_params(sample_average__sample_size=200, sample_average__noise_term=1000)

sim_wb_results = {ec: western_blot_models[ec].run() for ec in experimental_conditions}


# ========================================================
# ============= Plotting Measurement Model ===============

# The rest of this file just encodes the plots

# ------- Plot Data -------
# represent higher ordinal categories larger rectangles
parp_loc = 0.2
cparp_loc = 0.1
blot_sizes = [1, 3, 5, 8, 11]

western_blot['PARP_sizes'] = western_blot['PARP'].apply(lambda x: blot_sizes[x])
western_blot['PARP_loc'] = parp_loc

western_blot['cPARP_sizes'] = western_blot['cPARP'].apply(lambda x: blot_sizes[x])
western_blot['cPARP_loc'] = cparp_loc

western_blot['time'] = western_blot['time'].apply(lambda x: x/3600)  # Convert back to [hrs].


cm = plt.get_cmap('Accent')
fig = plt.figure(figsize=(6, 4))
gs = GridSpec(3, 3)

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :], sharey=ax1)
ax3 = plt.subplot(gs[2, :], sharey=ax1)

axes = {10: (ax1, cm.colors[0]), 50: (ax2, cm.colors[1]), 250: (ax3, cm.colors[2])}

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

fig.suptitle("Western Blot Data", size=16)
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

plt.show()

# ------- plot parameters --------
cm = plt.get_cmap('tab20b')
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(parameters['kc0'], bins=20, alpha=0.4, density=True, color=cm.colors[12])
ax.tick_params(axis='y',
               which='both',
               left=False,
               labelleft=False)
ax.set_xticklabels([0.5, 1.0, 1.5, 2.0])
plt.xlim(4e-6, 2.25e-5)
plt.xticks([5.0e-6, 1.0e-5, 1.5e-5, 2.0e-5])
plt.xlabel('kc0 •1e-6 [1/s]')
plt.title('Simulated variability in DISC formation parameter, kc0')
plt.show()

fig2, ax = plt.subplots(figsize=(5, 5))
parameters[['kc2', 'kc3']].plot.scatter(x='kc2', y='kc3', color=cm.colors[12], alpha=0.1, ax=ax)
ax.set_xticklabels([0.5, 1.0, 1.5, 2.0])
ax.set_yticklabels([0.5, 1.0, 1.5, 2.0])
plt.title("Simulated co-varying kc2 and kc3 parameters")
plt.xlim(4e-3, 2.25e-2)
plt.ylim(4e-3, 2.25e-2)
plt.xlabel('kc2 •1e-2 [1/s]')
plt.ylabel('kc3 •1e-2 [1/s]')
plt.xticks([5.0e-3, 1.0e-2, 1.5e-2, 2.0e-2])
plt.yticks([5.0e-3, 1.0e-2, 1.5e-2, 2.0e-2])
plt.show()

# ------- plot dynamics -------
results_df = results.opt2q_dataframe.reset_index()
results_df['time_axis'] = results_df['time'].apply(lambda x: x/3600)
cm = plt.get_cmap('Accent')

fig3, ax = plt.subplots(figsize=(6, 5))
legend_handles = []
for i, (label, df) in enumerate(results_df.groupby('TRAIL_conc')):
    ec_label = f'{label} ng/mL TRAIL'
    legend_handles.append(mpatches.Patch(color=cm.colors[i], label=ec_label))
    for name, group in df.groupby(by='simulation'):
        group.plot.line(x='time_axis', y='cPARP_obs', ax=ax, color=cm.colors[i], alpha=0.05, legend=False)
ax.set_xlabel('time [hrs]')
ax.set_ylabel('cPARP [copies per cell]')
ax.set_title('Simulated cPARP Dynamics')
ax.legend(handles=legend_handles)
plt.show()

fig4, ax = plt.subplots(figsize=(6, 5))
cm = plt.get_cmap('Accent')
legend_handles = []
for i, (label, df) in enumerate(results_df.groupby('TRAIL_conc')):
    ec_label = f'{label} ng/mL TRAIL'
    legend_handles.append(mpatches.Patch(color=cm.colors[i], label=ec_label))
    for name, group in df.groupby(by='simulation'):
        group.plot.line(x='time_axis', y='PARP_obs', ax=ax, color=cm.colors[i], alpha=0.05, legend=False)
ax.set_xlabel('time [hrs]')
ax.set_ylabel('PARP [copies per cell]')
ax.set_title('Simulated PARP Dynamics')
ax.legend(handles=legend_handles)
plt.show()


# ------- measurement model -------
def alpha(x):
    # represent probabilities via transparency (i.e. alpha)
    return 0.5 + 5 * (x - 0.5) / np.sqrt(100 * (x - 0.5)**2 + 1)


sample_avr_results = {}
for i, ec in enumerate(experimental_conditions):
    fig = plt.figure(figsize=(15, 15))

    gs = GridSpec(7, 8)

    ax01 = plt.subplot(gs[:3, :3])
    ax02 = plt.subplot(gs[:3, 3:6],  sharey=ax01)
    ax03 = plt.subplot(gs[:3, 6:],   sharey=ax01)
    ax11 = plt.subplot(gs[3:6, :3])
    ax12 = plt.subplot(gs[3:6, 3:6], sharey=ax11)
    ax13 = plt.subplot(gs[3:6, 6:],  sharey=ax11)
    ax22 = plt.subplot(gs[6:, 3:6])

    # plot simulation results
    for name, group in results_df[results_df.TRAIL_conc == ec].groupby(by='simulation'):
        group.plot.line(x='time_axis', y='PARP_obs', ax=ax01, color=cm.colors[i], legend=False, alpha=0.1)
        group.plot.line(x='time_axis', y='cPARP_obs', ax=ax11, color=cm.colors[i], legend=False, alpha=0.1)

    ax01.set_title('Simulated PARP dynamics')
    ax01.set_ylabel('PARP [copies per cell]')
    ax01.set_xlabel('time [hrs]')
    ax01.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


    ax11.set_title('Simulated cPARP dynamics')
    ax11.set_ylabel('cPARP [copies per cell]')
    ax11.set_xlabel('time [hrs]')
    ax11.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # plot sample average (cell lysis step)
    interpolated_results = western_blot_models[ec].interpolation.transform(
        results_df[['cPARP_obs', 'PARP_obs','time', 'simulation','TRAIL_conc']])

    sample_avr_results[ec] = western_blot_models[ec].process.get_step('sample_average').transform(interpolated_results)
    sample_avr_results[ec]['time'] = sample_avr_results[ec]['time'].apply(lambda x: x/3600)

    vp0 = ax02.violinplot(dataset=[sample_avr_results[ec][(sample_avr_results[ec].time == 0)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 1)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 2)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 3)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 4)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 5)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 7)]['PARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 9)]['PARP_obs']])

    vp1 = ax12.violinplot(dataset=[sample_avr_results[ec][(sample_avr_results[ec].time == 0)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 1)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 2)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 3)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 4)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 5)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 7)]['cPARP_obs'],
                                   sample_avr_results[ec][(sample_avr_results[ec].time == 9)]['cPARP_obs']])
    for parts in vp0['bodies']:
        parts.set_color(cm.colors[i])
    for part_names in ('cbars', 'cmins', 'cmaxes'):
        parts = vp0[part_names]
        parts.set_edgecolor(cm.colors[i])

    for parts in vp1['bodies']:
        parts.set_color(cm.colors[i])
    for part_names in ('cbars', 'cmins', 'cmaxes'):
        parts = vp1[part_names]
        parts.set_edgecolor(cm.colors[i])

    ax02.set_xticklabels([0, 0, 1, 2, 3, 4, 5, 7, 9])
    ax02.set_xlabel('time [hrs]')
    ax02.set_ylabel('simulated sample average')
    ax02.set_title("Simulated PARP Concentration. of Prepped Samples")
    ax02.legend(handles=[mpatches.Patch(color=cm.colors[i], label=f'{ec} ng/mL TRAIL')], loc='lower left')

    ax12.set_xticklabels([0, 0, 1, 2, 3, 4, 5, 7, 9])
    ax12.set_ylabel('simulated sample average')
    ax12.set_title("Simulated cPARP Concentration. of Prepped Samples")
    ax12.set_xlabel('time [hrs]')

    # plot probability of categories
    western_blot_models[ec].process.set_params(
        classifier__do_fit_transform=False)  # Important for out-of-sample calculations
    western_blot_models[ec].process.remove_step('sample_average')
    measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(0, 1000000, 100),
                                        'cPARP_obs': np.linspace(0, 1000000, 100)})
    measurement_model_x['TRAIL_conc'] = 10  # needs a column in common with the data
    measurement_model_y = western_blot_models[ec].process.transform(measurement_model_x)

    PARP_results = measurement_model_y.filter(regex='^PARP')
    PARP_results['PARP'] = measurement_model_x['PARP_obs']

    for col in sorted(list(set(PARP_results.columns) - {'PARP'})):
        PARP_results.plot(y='PARP', x=col, ax=ax03, label=col)

        ax03.set_title('Probability')

    western_blot_models[ec].process.set_params(
        classifier__do_fit_transform=False)  # Important for out-of-sample calculations
    western_blot_models[ec].process.remove_step('sample_average')
    measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(0, 1000000, 100),
                                        'cPARP_obs': np.linspace(0, 1000000, 100)})
    measurement_model_x['TRAIL_conc'] = ec  # needs a column in common with the data
    measurement_model_y = western_blot_models[ec].process.transform(measurement_model_x)

    cPARP_results = measurement_model_y.filter(regex='cPARP')
    cPARP_results['cPARP'] = measurement_model_x['cPARP_obs']

    for col in sorted(list(set(cPARP_results.columns) - {'cPARP'})):
        cPARP_results.plot(y='cPARP', x=col, ax=ax13, label=col)

        ax03.set_title('Probability')

    # plot simulated Western blot
    avr_cat_prob = sim_wb_results[ec].groupby('time').mean()
    time_axis = avr_cat_prob.index / 3600

    parp_locs = np.full_like(time_axis, parp_loc)
    cparp_locs = np.full_like(time_axis, cparp_loc)

    avr_cat_prob_parp_cols = avr_cat_prob.filter(regex='^PARP').columns
    avr_cat_prob_cparp_cols = avr_cat_prob.filter(regex='cPARP').columns

    for j, level in enumerate(avr_cat_prob_parp_cols):
        color_ = np.asarray([list(cm.colors[i]) + [alpha(a)] for a in avr_cat_prob[level]])
        ax22.scatter(x=time_axis, y=parp_locs, s=500, marker='_', lw=blot_sizes[j], color=color_)
    for j, level in enumerate(avr_cat_prob_cparp_cols):
        color_ = np.asarray([list(cm.colors[i]) + [alpha(a)] for a in avr_cat_prob[level]])
        ax22.scatter(x=time_axis, y=cparp_locs, s=500, marker='_', lw=blot_sizes[j], color=color_)

    ax22.set_title(f'Simulated Western Blot {ec} ng/mL TRAIL')
    ax22.set_ylim(0.0, 0.3)
    ax22.set_yticks([0.1, 0.2])
    ax22.set_yticklabels(['cPARP', 'PARP'], fontdict={'fontsize': 12})
    ax22.set_xticks([0, 1, 2, 3, 4, 5, 7, 9])
    ax22.set_xlabel('time [hrs]')

    fig.suptitle(f"Western Blot Measurement Model {ec} ng/mL TRAIL", x=0.5, y=1.0, va='top', size=20)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"{i}plt.png")
    plt.show()

# Plot simulated Western Blot
fig = plt.figure(figsize=(6, 4))
gs = GridSpec(3, 3)

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :], sharey=ax1)
ax3 = plt.subplot(gs[2, :], sharey=ax1)

axes = {10: (ax1, cm.colors[0]), 50: (ax2, cm.colors[1]), 250: (ax3, cm.colors[2])}

for i, ec in enumerate(experimental_conditions):
    avr_cat_prob = sim_wb_results[ec].groupby('time').mean()
    time_axis = avr_cat_prob.index / 3600

    parp_locs = np.full_like(time_axis, parp_loc)
    cparp_locs = np.full_like(time_axis, cparp_loc)

    avr_cat_prob_parp_cols = avr_cat_prob.filter(regex='^PARP').columns
    avr_cat_prob_cparp_cols = avr_cat_prob.filter(regex='cPARP').columns

    for j, level in enumerate(avr_cat_prob_parp_cols):
        color_ = np.asarray([list(cm.colors[i]) + [alpha(a)] for a in avr_cat_prob[level]])
        axes[ec][0].scatter(x=time_axis, y=parp_locs, s=500, marker='_', lw=blot_sizes[j], color=color_)
    for j, level in enumerate(avr_cat_prob_cparp_cols):
        color_ = np.asarray([list(cm.colors[i]) + [alpha(a)] for a in avr_cat_prob[level]])
        axes[ec][0].scatter(x=time_axis, y=cparp_locs, s=500, marker='_', lw=blot_sizes[j], color=color_)

    axes[ec][0].set_title(f'{ec} ng/mL TRAIL')
    axes[ec][0].set_ylim(0.0, 0.3)
    axes[ec][0].set_yticks([0.1, 0.2])
    axes[ec][0].set_yticklabels(['cPARP', 'PARP'], fontdict={'fontsize': 12})
    axes[ec][0].set_xticks([0, 1, 2, 3, 4, 5, 7, 9])

plt.xlabel('time [hrs]')

fig.suptitle("Simulated Western Blot", size=16)
gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

plt.show()
