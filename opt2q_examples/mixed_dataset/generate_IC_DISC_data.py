import pandas as pd
import numpy as np
from opt2q.data import DataSet
from opt2q.measurement.base import ScaleToMinMax, Interpolate, LogisticClassifier
from opt2q_examples.mixed_dataset.calibration_setup import set_up_simulator as sim_setup_immuno
from opt2q_examples.apoptosis_model import model
from matplotlib import pyplot as plt
from opt2q_examples.plot_tools import plot


# ------- Save Dataset -----------
save_dataset = False

# ------- Create Dummy Dataset to Train a Classifier -------
n = 300  # measurements every n seconds.
m = 4
dummy_df = pd.DataFrame({'time': range(0, 18001, n),
                         'IC_DISC_localization': (list(range(m))*500)[:len(range(0, 18001, n))]})
dummy_dataset = DataSet(dummy_df, measured_variables={'IC_DISC_localization': 'ordinal'})

# ------- Simulate Model --------
sim = sim_setup_immuno(model)
results = sim.run().opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

# set up classifier
x_scaled = ScaleToMinMax(columns=['C8_DISC_recruitment_obs']).transform(results[['time', 'C8_DISC_recruitment_obs']])
x_int = Interpolate('time', ['C8_DISC_recruitment_obs'], dummy_df['time']).transform(x_scaled)
lc = LogisticClassifier(dummy_dataset,
                        column_groups={'IC_DISC_localization': ['C8_DISC_recruitment_obs']},
                        do_fit_transform=True,
                        classifier_type='ordinal_eoc')
lc.set_up(x_int)

a = 25
lc.set_params(** {'coefficients__IC_DISC_localization__coef_': np.array([a]),
                  'coefficients__IC_DISC_localization__theta_': np.array([0.05, 0.40, 0.85])*a})

lc.do_fit_transform = False
lc_results = lc.transform(x_int)

# ------- Simulate Model --------

# plot classifier
lc.do_fit_transform = False
plot_domain = pd.DataFrame({'C8_DISC_recruitment_obs': np.linspace(0, 1, 100)})
lc_curve = lc.transform(plot_domain)


# ------- Generate Synthetic Data ------
IC_DISC_localization_cols = lc_results.filter(regex='IC_DISC_localization__').columns

lc_results['IC_DISC_localization'] = lc_results.apply(lambda x: np.random.choice(
    [int(c.split('__')[1]) for c in IC_DISC_localization_cols],
    p=[x[c] for c in IC_DISC_localization_cols]), axis=1)

immunoblot_data = lc_results[['time', 'IC_DISC_localization']]
synthetic_immunoblot_data = DataSet(immunoblot_data,
                                    measured_variables={'IC_DISC_localization': 'ordinal'})

# ------ How well does the classifier recover the preset parameters --------
# set up classifier
x_scaled2 = ScaleToMinMax(columns=['C8_DISC_recruitment_obs']).transform(results[['time', 'C8_DISC_recruitment_obs']])
x_int2 = Interpolate('time', ['C8_DISC_recruitment_obs'], synthetic_immunoblot_data.data['time']).transform(x_scaled2)
lc2 = LogisticClassifier(synthetic_immunoblot_data,
                         column_groups={'IC_DISC_localization': ['C8_DISC_recruitment_obs']},
                         do_fit_transform=True,
                         classifier_type='ordinal_eoc')
lc2.set_up(x_int2)
lc2.transform(x_int2)

# plot fitted classifier
lc2.do_fit_transform = False
lc2_curve = lc2.transform(plot_domain)

if __name__ == '__main__':
    cm = plt.get_cmap('tab10')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), sharey='all', gridspec_kw={'width_ratios': [2, 1, 1]})
    ax1.scatter(x=synthetic_immunoblot_data.data['time'],
                y=synthetic_immunoblot_data.data['IC_DISC_localization'].values / m,
                s=100, color=cm.colors[5], label=f'IC-DISC localization data', alpha=0.5)
    ax1.plot(x_scaled['time'], x_scaled['C8_DISC_recruitment_obs'],
             color=cm.colors[5], label=f'simulated IC-DISC localization')
    ax1.legend()

    for col in sorted(list(lc_curve.columns)):
        ax2.plot(lc_curve[col].values, np.linspace(0, 1, 100), label=col)
        ax3.plot(lc2_curve[col].values, np.linspace(0, 1, 100), label=col)
        ax3.axvline(x=0.5, linestyle='--', color='k', alpha=0.5)

    ax1.set_title('Classification of Simulated Caspase localization to the DISC')
    ax1.set_xlabel('time [seconds]')
    ax1.set_ylabel('fluorescence [AU]')
    ax2.set_xlabel('category probability')
    ax2.legend()
    ax3.set_xlabel('fitted category probability')
    plt.show()

print(lc2.get_params()['coefficients__IC_DISC_localization__coef_'])


if save_dataset:
    import pickle
    import datetime as dt

    now = dt.datetime.now()

    with open(f'synthetic_IC_DISC_localization_blot_dataset_{now.year}_{now.month}_{now.day}.pkl', 'wb') as output:
        pickle.dump(synthetic_immunoblot_data, output, pickle.HIGHEST_PROTOCOL)

    with open(f'synthetic_IC_DISC_localization_blot_dataset_{now.year}_{now.month}_{now.day}.pkl', 'rb') as data_input:
        loaded_dataset = pickle.load(data_input)
