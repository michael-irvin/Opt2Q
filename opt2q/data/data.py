# MW Irvin -- Lopez Lab -- 2018-08-23
import pandas as pd
from numbers import Number


class DataSet(object):
    """
    Formats Data for use in Opt2Q models

    Parameters
    ----------
    data: :class:`~pandas.DataFrame`
        Dataframe with values of the measured variables.
        optionally additional columns indexing experimental conditions and/or 'time'.

    measured_variables: list or dict
        As list, it lists the column in ``data`` that are measured values.

        As dict, it names the columns in ``data`` that are measured values and gives their measurement type:
        'quantitative', 'semi-quantitative', 'ordinal' or 'nominal.

        Columns must exist in ``data``.

    manipulated_variables: dict
        param_mean and param_cov arguments of the :class:`~opt2q.noise.NoiseModel`

    observables:
    """

    def __init__(self, data, measured_variables=None, manipulated_variable=None, *args, **kwargs):
        self.data = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
        self._measured_variables = self._check_measured_variables(data, measured_variables)
        self.measured_variables = {'a': 'default', 'b': 'default', 'c': 'default'}
        self.observables = {1, 2, 3}  # Todo: Measurement model requires vector-like attr, 'observables'.
        self.experimental_conditions = pd.DataFrame()  # This will also contain time-points (if necessary).

    @staticmethod
    def _check_measured_variables(data, measured_vars) -> dict:
        """
        Measured Variables must be in data.columns.

        If dict, it must meet column constraints measurement_types must meet columns constraints.

        Returns
        -------
        dict
        """
        return []
