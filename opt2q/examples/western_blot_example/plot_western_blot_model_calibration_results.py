# MW Irvin -- Lopez Lab -- 2018-02-08
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from opt2q.examples.apoptosis_model_ import model
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.examples.western_blot_example.western_blot_likelihood_fn import likelihood_fn, param_means, param_variances
from decimal import Decimal


calibrated_parameters = [-3.60467941, -0.38469258, -5.88974386, -2.20091485, -2.49022636,
                         -7.09583405, 0.25, 0.25, 0.0]  # 0.29699746, 0.85080678, -0.33342348]

# ======== Starting Parameters ========
param_means['apply_noise'] = False
param_means = param_means.reset_index(drop=True)

noise_model = NoiseModel(param_mean=param_means, param_covariance=param_variances)
starting_params = noise_model.run()

# ========= Calibrated Parameters w/o Noise added ==========
x = calibrated_parameters
kc0 = 10 ** x[0]    # :  [(-7,  -3),   # float  kc0
kc2 = 10 ** x[1]    # :   (-5,   1),   # float  kc2
kf3 = 10 ** x[2]    # :   (-11, -6),   # float  kf3
kc3 = 10 ** x[3]    # :   (-5,   1),   # float  kc3
kf4 = 10 ** x[4]    # :   (-10, -4),   # float  kf4 (The simulator is sensitive to high values).
kr7 = 10 ** x[5]    # :   (-8,   4),   # float  kr7

kc2_cv = x[6]       # :   (0, 1),      # float  kc2_cv
kc3_cv = x[7]       # :   (0, 1),      # float  kc3_cv
kc2_kc3_cor = x[8]  # :   (-1, 1)])    # float  kc2_kc3_cor

kc2_var = (kc2 * kc2_cv) ** 2
kc3_var = (kc3 * kc3_cv) ** 2
kc2_kc3_covariance = kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor

# noise model
# start_time = time.time()
ligand = pd.DataFrame([['L_0', 600, 10, False],  # 'TRAIL_conc' column annotates experimental treatments
                       ['L_0', 3000, 50, False],  # 600 copies per cell corresponds to 10 ng/mL TRAIL
                       ['L_0', 15000, 250, False]],
                      columns=['param', 'value', 'TRAIL_conc', 'apply_noise'])

k_values = pd.DataFrame([['kc0', kc0, True],
                         ['kc2', kc2, True],  # co-vary with kc3
                         ['kf3', kf3, False],
                         ['kc3', kc3, True],
                         ['kf4', kf4, False],
                         ['kr7', kr7, False]],
                        columns=['param', 'value', 'apply_noise']) \
    .iloc[np.repeat(range(5), 3)]  # Repeat for each of the 3 experimental treatments
k_values['TRAIL_conc'] = np.tile([10, 50, 250], 5)  # Repeat for each of the 5 parameter

new_params = pd.concat([ligand, k_values], sort=False)
new_params_cov = pd.DataFrame([['kc2', 'kc2', kc2_var],
                               ['kc3', 'kc3', kc3_var],
                               ['kc2', 'kc3', kc2_kc3_covariance]],  # Covariance between 'kc2' and kc3'
                              columns=['param_i', 'param_j', 'value'])

noise_model.update_values(param_mean=new_params, param_covariance=new_params_cov)
calibrated_params = noise_model.run()

# -------- Simulator --------
sim_results_0 = Simulator(model, tspan=np.linspace(0, 32400, 100), param_values=starting_params).run().opt2q_dataframe
sim_results_n = Simulator(model, tspan=np.linspace(0, 32400, 100), param_values=calibrated_params).run().opt2q_dataframe

# -------- Plot Simulation Results wo Normalization -------
cm = plt.get_cmap('Accent')
sim_res_groups_n = sim_results_n.groupby('TRAIL_conc')

fig = plt.figure()
for i, group in enumerate(sim_results_0.groupby('TRAIL_conc')):
    label, df = group
    _df = df.reset_index()
    _df_n = sim_res_groups_n.get_group(label)
    time_hrs = _df['time'].apply(lambda x: x/3600.)

    plt.plot(time_hrs, _df_n['BID_obs'], color=cm.colors[i], label=f'{label} ng/mL calibrated params')
    plt.plot(time_hrs, _df['BID_obs'], '--', color=cm.colors[i], label=f'{label} ng/mL starting params')

plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results (Bid Truncation)')
plt.legend()
fig.show()

fig = plt.figure()
for i, group in enumerate(sim_results_0.groupby('TRAIL_conc')):
    label, df = group
    _df = df.reset_index()
    _df_n = sim_res_groups_n.get_group(label)
    time_hrs = _df['time'].apply(lambda x: x / 3600.)

    plt.plot(time_hrs, _df_n['cPARP_obs'], color=cm.colors[i], label=f'{label} ng/mL calibrated params')
    plt.plot(time_hrs, _df['cPARP_obs'], '--', color=cm.colors[i], label=f'{label} ng/mL starting params')

plt.xlabel('time [hrs]')
plt.ylabel('protein [copies per cell]')
plt.title('Simulation Results (cPARP)')
plt.legend()
fig.show()

# ======= Calibrated Parameters ========
# Model results with the Starting Parameters are plotted in
# western_blot_measurement_model_pipeline.py

likelihood_fn(calibrated_parameters)
parameters = likelihood_fn.noise_model.run()

# ------- plot parameters --------
cm = plt.get_cmap('tab20b')
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.hist(parameters['C_0'], bins=20, alpha=0.4, density=True, color=cm.colors[12])
# ax.tick_params(axis='y',
#                which='both',
#                left=False,
#                labelleft=False)
# plt.xlabel('Caspase copies per cell')
# plt.title('Simulated variability in Caspase parameter')
# plt.show()

kc2_min, kc2_max = parameters['kc2'].min(), parameters['kc2'].max()
kc3_min, kc3_max = parameters['kc3'].min(), parameters['kc3'].max()
kc2_range = kc2_max - kc2_min
kc3_range = kc3_max - kc3_min

kc2_ticks = [kc2_min + 0.2 * kc2_range,
             kc2_min + 0.4 * kc2_range,
             kc2_min + 0.6 * kc2_range,
             kc2_min + 0.8 * kc2_range]
kc3_ticks = [kc3_min + 0.2 * kc3_range,
             kc3_min + 0.4 * kc3_range,
             kc3_min + 0.6 * kc3_range,
             kc3_min + 0.8 * kc3_range]

kc2_exp = len(Decimal(kc2_max).as_tuple()[1]) + Decimal(kc2_max).as_tuple()[2] - 1
kc3_exp = len(Decimal(kc3_max).as_tuple()[1]) + Decimal(kc3_max).as_tuple()[2] - 1

fig2, ax = plt.subplots(figsize=(5, 5))
parameters[['kc2', 'kc3']].plot.scatter(x='kc2', y='kc3', color=cm.colors[12], alpha=0.1, ax=ax)
ax.set_xticklabels(['%.2f'%Decimal(x/(10**kc2_exp)) for x in kc2_ticks])
ax.set_yticklabels(['%.2f'%Decimal(x/(10**kc3_exp)) for x in kc3_ticks])
plt.title("Simulated co-varying kc2 and kc3 parameters")
plt.xlim(kc2_min, kc2_max)
plt.ylim(kc3_min, kc3_max)
plt.xlabel(f'kc2 [•1e{kc2_exp}]')
plt.ylabel(f'kc3 [•1e{kc2_exp}]')

plt.xticks(kc2_ticks)
plt.yticks(kc3_ticks)
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
