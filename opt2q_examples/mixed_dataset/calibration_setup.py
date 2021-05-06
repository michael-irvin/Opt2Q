import pandas as pd
import numpy as np
from opt2q.simulator import Simulator
from opt2q.measurement.base import Pipeline, ScaleToMinMax, Interpolate, LogisticClassifier


def set_up_simulator(model):
    param_names = [p.name for p in model.parameters_rules()][:-6]
    parameters = pd.DataFrame([10**np.load('true_params.npy')[:len(param_names)]],
                              columns=param_names)

    sim = Simulator(model=model, param_values=parameters, solver='scipyode',
                    solver_options={'integrator': 'lsoda'}, tspan=np.linspace(0, 20160, 100))
    return sim


def set_up_immunoblot(sim, dataset):
    results = sim.run().opt2q_dataframe.reset_index().rename(columns={'index': 'time'})

    measurement_model = Pipeline(steps=[('x_scaled', ScaleToMinMax(columns=['C8_DISC_recruitment_obs'])),
                                        ('x_int', Interpolate('time', ['C8_DISC_recruitment_obs'], dataset.data['time'])),
                                        ('classifier', LogisticClassifier(
                                            dataset,
                                            column_groups={'IC_DISC_localization': ['C8_DISC_recruitment_obs']},
                                            do_fit_transform=False,
                                            classifier_type='ordinal_eoc'))])
    measurement_model.transform(results[['time', 'C8_DISC_recruitment_obs']])
    return measurement_model


if __name__ == '__main__':
    import pickle
    from opt2q_examples.apoptosis_model import model

    with open(f'synthetic_IC_DISC_localization_blot_dataset_2020_10_18.pkl', 'rb') as data_input:
        loaded_dataset = pickle.load(data_input)

    test_sim = set_up_simulator(model)
    wb = set_up_immunoblot(test_sim, loaded_dataset)
