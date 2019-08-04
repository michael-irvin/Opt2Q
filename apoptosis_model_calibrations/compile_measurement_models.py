# MW Irvin -- Lopez Lab -- 2019-07-26

"""
Set-up measurement models to be apply consistent *in silico* measurements to all experiments in the study.
"""
import math
import numpy as np
from opt2q.measurement.base.transforms import CumulativeComputation, ScaleGroups, Interpolate, Pipeline, \
    LogisticClassifier, SampleAverage
from opt2q.measurement.base.functions import derivative, where_max


def compile_cparp_dependent_fractional_killing_model(measurement_model):
    """
    Use the following feature to predict cell death:
    1. cPARP concentration
    2. max rate of change in cPARP concentration (this will be the "switch rate" if switch dynamics arises)
    3. time @ unset of switch dynamics (even if it occurs after the time points in the experiment).
    4. Area under the cPARP vs. time curve.
    """
    mock_data = measurement_model.experimental_conditions_df.copy()
    if mock_data.shape[0] == 1:  # skl classifier requires at least two categories (i.e. rows) in dataset
        mock_data = mock_data.iloc[np.tile(mock_data.index, 2)].reset_index(drop=True)
    mock_data['Fraction Killed'] = np.tile([0, 1], math.ceil(len(mock_data) / 2))[:len(mock_data)]

    interpolate = Interpolate(independent_variable_name='time',
                              dependent_variable_name=['time__delay', 'cPARP_obs'],
                              new_values=measurement_model.experimental_conditions_df.drop_duplicates().reset_index(
                                  drop=True),
                              groupby='simulation')
    measurement_model.interpolation_ds = interpolate
    measurement_model.interpolation = interpolate

    measurement_model.process = Pipeline(
        steps=[('max_ddx', ScaleGroups(groupby='simulation',  # Derivative of cPARP curves
                                       columns=['cPARP_obs'],
                                       scale_fn=derivative,
                                       keep_old_columns=True)),
               # Feature: Time-delay until cPARP reaches a point of inflection (or max rate of change).
               ('delay', ScaleGroups(groupby='simulation',  # Time at max ddx (i.e. lag time)
                                     columns=['time', 'cPARP_obs__max_ddx'],
                                     scale_fn=where_max, **{'var': 'cPARP_obs__max_ddx', 'drop_var': True},
                                     keep_old_columns=True)),
               # Feature: Max rate of change in cPARP. At a point of inflection, this is the "switch rate"
               ('cummax', CumulativeComputation(groupby='simulation',
                                                columns=['cPARP_obs__max_ddx'],
                                                operation='max',
                                                keep_old_columns=False)),  # overwrite max_ddx
               # Feature: Area under the curve of cPARP signal
               ('auc', CumulativeComputation(groupby='simulation',
                                             columns=['cPARP_obs'],
                                             operation='sum',
                                             keep_old_columns=True,
                                             **{'parse_columns': False})),  # only transform cPARP_obs
               # Simulate Fractional Killing Measurement
               ('interpolate', interpolate),
               ('classifier', LogisticClassifier(mock_data,
                                                 do_fit_transform=False,
                                                 classifier_type='nominal',
                                                 column_groups={'Fraction Killed': ['cPARP_obs',
                                                                                    'cPARP_obs__auc',
                                                                                    'cPARP_obs__max_ddx',
                                                                                    'time__delay']})),
               ('sample_average', SampleAverage(columns=['Fraction Killed'],
                                                drop_columns='simulation',
                                                groupby=list(
                                                    set(measurement_model.experimental_conditions_df.columns) -
                                                    {'simulation'}),
                                                apply_noise=False))
               ])
    return measurement_model


