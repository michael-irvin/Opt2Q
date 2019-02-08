# MW Irvin -- Lopez Lab -- 2018-02-08
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from opt2q.examples.western_blot_example.western_blot_likelihood_fn import likelihood_fn
from decimal import Decimal

calibrated_parameters = [2.35343618,  1.87440897, -0.61746058, -4.00650177, -4.22771324, 0.2, 0.2, 0.25]

# ======= Calibrated Parameters ========
# Model results with the Starting Parameters are plotted in
# western_blot_measurement_model_pipeline.py

likelihood_fn(calibrated_parameters)
parameters = likelihood_fn.noise_model.run()

# ------- plot parameters --------
cm = plt.get_cmap('tab20b')
fig, ax = plt.subplots(figsize=(4, 4))
ax.hist(parameters['C_0'], bins=20, alpha=0.4, density=True, color=cm.colors[12])
ax.tick_params(axis='y',
               which='both',
               left=False,
               labelleft=False)
plt.xlabel('Caspase copies per cell')
plt.title('Simulated variability in Caspase parameter')
plt.show()

kf3_min, kf3_max = parameters['kf3'].min(), parameters['kf3'].max()
kf4_min, kf4_max = parameters['kf4'].min(), parameters['kf4'].max()
kf3_range = kf3_max -kf3_min
kf4_range = kf4_max -kf4_min

kf3_ticks = [kf3_min + 0.2 * kf3_range,
             kf3_min + 0.4 * kf3_range,
             kf3_min + 0.6 * kf3_range,
             kf3_min + 0.8 * kf3_range]
kf4_ticks = [kf4_min + 0.2 * kf4_range,
             kf4_min + 0.4 * kf4_range,
             kf4_min + 0.6 * kf4_range,
             kf4_min + 0.8 * kf4_range]

kf3_exp = len(Decimal(kf3_max).as_tuple()[1]) + Decimal(kf3_max).as_tuple()[2] - 1
kf4_exp = len(Decimal(kf4_max).as_tuple()[1]) + Decimal(kf4_max).as_tuple()[2] - 1

fig2, ax = plt.subplots(figsize=(5, 5))
parameters[['kf3', 'kf4']].plot.scatter(x='kf3', y='kf4', color=cm.colors[12], alpha=0.1, ax=ax)
ax.set_xticklabels(['%.2f'%Decimal(x/(10**kf3_exp)) for x in kf3_ticks])
ax.set_yticklabels(['%.2f'%Decimal(x/(10**kf4_exp)) for x in kf4_ticks])
plt.title("Simulated co-varying kf3 and kf4 parameters")
plt.xlim(kf3_min, kf3_max)
plt.ylim(kf4_min, kf4_max)
plt.xlabel(f'kf3 [•1e{kf3_exp}]')
plt.ylabel(f'kf4 [•1e{kf4_exp}]')

plt.xticks(kf3_ticks)
plt.yticks(kf4_ticks)
plt.show()

# ------- plot dynamics -------
results = likelihood_fn.simulator.run(np.linspace(0, 32400, 100))
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
experimental_conditions = likelihood_fn.measurement_models.keys()
western_blot_models = likelihood_fn.measurement_models
sim_wb_results = {ec: likelihood_fn.measurement_models[ec].results for ec in experimental_conditions}

# represent higher ordinal categories larger rectangles
parp_loc = 0.2
cparp_loc = 0.1
blot_sizes = [1, 3, 5, 8, 11]

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
    measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(0, 100000, 100),
                                        'cPARP_obs': np.linspace(0, 100000, 100)})
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
    measurement_model_x = pd.DataFrame({'PARP_obs': np.linspace(0, 100000, 100),
                                        'cPARP_obs': np.linspace(0, 100000, 100)})
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
