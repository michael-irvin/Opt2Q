# MW Irvin -- Lopez Lab -- 2018-10-10

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from measurement_model_demo.immunoblot_measurement_model_calibration import burn_in_len, n_chains, n_iterations, \
    synthetic_immunoblot_data
from measurement_model_demo.apoptosis_model import model
from measurement_model_demo.generate_synthetic_immunoblot_dataset import parameters, IC_RP__n_cats, EC_RP__n_cats
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import LogisticClassifier, ScaleToMinMax, Interpolate


# ======== UPDATE THIS PART ===========
# Update this part with the new log-p, parameter files, etc
script_dir = os.path.dirname(__file__)
calibration_folder = 'immunoblot_calibration_results'
plot_title = 'Immunoblot Calibration'

burn_in_len = burn_in_len
number_of_traces = n_chains

chain_history_file = [os.path.join(script_dir, calibration_folder, f) for f in
                      os.listdir(os.path.join(script_dir, calibration_folder))
                      # if '20191031' in f and 'chain_history' in f][0]
                      # if 'cauchy_priors_2021210' in f and 'chain_history' in f][0]
                      if 'calibration_202129' in f and 'chain_history' in f][0]

log_p_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                            os.listdir(os.path.join(script_dir, calibration_folder))
                            # if '20191031' in f and 'log_p' in f])
                            # if 'cauchy_priors_2021210' in f and 'log_p' in f])
                            if 'calibration_202129' in f and 'log_p' in f])

parameter_file_paths_ = sorted([os.path.join(script_dir, calibration_folder, f) for f in
                                os.listdir(os.path.join(script_dir, calibration_folder))
                                # if '20191031' in f and 'parameters' in f])
                                # if 'cauchy_priors_2021210' in f and 'parameters' in f])
                                if 'calibration_202129' in f and 'parameters' in f])

# reorder traces to be in numerical order (i.e. 1000 before 10000).
file_order = [str(n_iterations*n) for n in range(1, int(len(log_p_file_paths_)/number_of_traces)+1)]
log_p_file_paths = []
parameter_file_paths = []
for file_num in file_order:
    log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
    parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]

param_names = ['coefficients__tBID_blot__coef_',
               'coefficients__tBID_blot__theta_1',
               'coefficients__tBID_blot__theta_2',
               'coefficients__tBID_blot__theta_3',
               'coefficients__tBID_blot__theta_4',
               'coefficients__cPARP_blot__coef_',
               'coefficients__cPARP_blot__theta_1',
               'coefficients__cPARP_blot__theta_2',
               'coefficients__cPARP_blot__theta_3']

# =====================================
# chain history
chain_history = np.load(chain_history_file)

# get parameters
log_p_traces = []
parameter_samples = []
for trace_num in range(number_of_traces):
    log_p_trace = np.concatenate(np.roll([np.load(os.path.join(script_dir, calibration_folder, lp))
                                          for lp in log_p_file_paths if f'_{trace_num}_' in lp], 0))
    log_p_traces.append(log_p_trace)

    parameter_sample = np.concatenate(np.roll([np.load(os.path.join(script_dir, calibration_folder, pp))
                                               for pp in parameter_file_paths if f'_{trace_num}_' in pp], 0))
    parameter_samples.append(parameter_sample)

num_params = len(param_names)

# plot calibration metrics
cm = plt.get_cmap('tab10')

for trace in log_p_traces:
    plt.plot(trace)

plt.title('Immunoblot Calibration Posterior Trace')
plt.ylabel('log(Posterior)')
plt.xlabel('iteration')
plt.show()

params_of_interest = range(len(param_names))
for j in params_of_interest:
    s = pd.DataFrame({f'trace {n}': parameter_samples[n][burn_in_len:, j] for n in range(len(parameter_samples))})
    ax = s.plot.hist(alpha=0.5, bins=20)
    s.plot(kind='kde', ax=ax, secondary_y=True)

    # plt.hist(param_array[burn_in_len:, j], normed=True)
    plt.title(param_names[j])
    plt.show()

# plot data and model
sim = Simulator(model=model, param_values=parameters, solver='cupsoda')
sim_results = sim.run(np.linspace(0, synthetic_immunoblot_data.data['time'].max(), 100))
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1]})

ax1.plot(results['time'].values, results['tBID_obs'].values/max(results['tBID_obs'].values),
         label=f'tBID simulations', color=cm.colors[1])
ax1.scatter(x=synthetic_immunoblot_data.data['time'],
            y=synthetic_immunoblot_data.data['tBID_blot'].values/(IC_RP__n_cats-1),
            s=10, color=cm.colors[1], label=f'tBID blot data', alpha=0.5)
ax1.legend()


# plot classifier model predictions
def make_classifier_params(x):
    x = abs(x)
    c0 = x[0]
    t1 = x[1]
    t2 = t1 + x[2]
    t3 = t2 + x[3]
    t4 = t3 + x[4]

    c5 = x[5]
    t6 = x[6]
    t7 = t6 + x[7]
    t8 = t7 + x[8]
    return {'coefficients__tBID_blot__coef_': np.array([c0]),
            'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
            'coefficients__cPARP_blot__coef_': np.array([c5]),
            'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5}


log_ps = np.concatenate([lpt for lpt in log_p_traces])
params = np.concatenate([p for p in parameter_samples])
idx = np.argmax(log_ps[burn_in_len:])

# # set up classifier
x_scaled = ScaleToMinMax(columns=['tBID_obs', 'cPARP_obs'])\
            .transform(results[['time', 'tBID_obs', 'cPARP_obs']])
x_int = Interpolate('time', ['tBID_obs', 'cPARP_obs'], synthetic_immunoblot_data.data['time'])\
            .transform(x_scaled)
lc = LogisticClassifier(synthetic_immunoblot_data,
                        column_groups={'tBID_blot': ['tBID_obs'], 'cPARP_blot': ['cPARP_obs']},
                        do_fit_transform=True,
                        classifier_type='ordinal_eoc')

lc.set_up(x_int)
lc.do_fit_transform = False

# # classifier prediction
for idx in np.random.choice(range(burn_in_len, len(params)), 100):
    lc.set_params(**make_classifier_params(params[idx]))
    plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})
    lc_results = lc.transform(plot_domain)
    cPARP_results = lc_results.filter(regex='cPARP_blot')
    tBID_results = lc_results.filter(regex='tBID_blot')

    for n, col in enumerate(sorted(list(tBID_results.columns))):
        ax2.plot(tBID_results[col].values, np.linspace(0, 1, 100), label=col, color=cm.colors[n], alpha=0.1)

a = 50
true_mm_params = {'coefficients__cPARP_blot__coef_': np.array([a]),
                  'coefficients__cPARP_blot__theta_': np.array([0.03, 0.20, 0.97]) * a,
                  'coefficients__tBID_blot__coef_': np.array([a]),
                  'coefficients__tBID_blot__theta_': np.array([0.03, 0.4, 0.82, 0.97]) * a}
lc.set_params(**true_mm_params)
plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})
lc_results = lc.transform(plot_domain)
cPARP_results = lc_results.filter(regex='cPARP_blot')
tBID_results = lc_results.filter(regex='tBID_blot')

for n, col in enumerate(sorted(list(tBID_results.columns))):
    ax2.plot(tBID_results[col].values, np.linspace(0, 1, 100), label=col, color=cm.colors[n], linewidth=2)

plt.show()
