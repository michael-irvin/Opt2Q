# MW Irvin -- Lopez Lab -- 2018-08-23
import pandas as pd
import numpy as np
from numbers import Number
from opt2q.utils import _is_vector_like, _convert_vector_like_to_list, _list_the_errors


class DataSet(object):
    """
    Formats Data for use in Opt2Q models

    Parameters
    ----------
    data: :class:`~pandas.DataFrame`
        Dataframe with values of the measured variables. Additional columns index experimental conditions, etc.

    measured_variables: list or dict
        As list, it lists the column in ``data`` that are measured values.

        As dict, it names the columns in ``data`` that are measured values and gives their measurement type:
        'quantitative', 'semi-quantitative', 'ordinal' or 'nominal.

        Columns must exist in ``data``.

    manipulated_variables: dict
        param_mean and param_cov arguments of the :class:`~opt2q.noise.NoiseModel`

    measurement_error: float, list or dict (optional)
        Variance of the data. Or the probability of error of the reported categorical data.
        As a dict, variance terms (floats) are indexed by the name(s) of the measured_variables.
        Todo: Redo the measurement_error! Allow it to vary with time, experimental condition, etc.
        Todo: Define weather it is relative or absolute error
        Todo: Allow updating the error term by the calibrator.

    kwargs: dict
        Additional options:

        ``observable``: list
            Names observables in a PySB :class:`models <pysb.core.Model>` that this dataset_fluorescence should require.
        ``use_common_ordinal_classifier: bool
            Assumes all ordinal measured values are classified via a single common classifier.
            Set to true when using WesternBlotPTM.

    """

    measurement_types = ['quantitative', 'semi-quantitative', 'ordinal', 'nominal']

    def __init__(self, data, measured_variables, manipulated_variable=None, measurement_error=None, *args, **kwargs):
        measured_vars = self._convert_measured_variables_to_dict(measured_variables)
        self.data = self._check_data(data, measured_vars)
        self.experimental_conditions = self._get_experimental_conditions_df(self.data, measured_vars)
        self.measured_variables = measured_vars

        observables = kwargs.get('observables', [])
        self._observables = self._check_observables(observables)

        self._measurement_error = self._check_measurement_error(measurement_error, self.measured_variables)
        self.measurement_error_df = self._get_errors_df()

        # ordinal error term
        use_common_classifier = kwargs.get('use_common_ordinal_classifier', False)
        self._ordinal_errors_matrices = dict()
        _ordinal_variables_names = [k for k, v in self.measured_variables.items() if v is 'ordinal']
        _ordinal_error_matrices = self._get_ordinal_errors_matrices(
            self._ordinal_errors_matrices, _ordinal_variables_names, single_classifier=use_common_classifier)
        _ordinal_vars_df_one_hot_representation = self._one_hot_transform_of_data(_ordinal_variables_names)

        self.ordinal_errors_df = self._apply_ordinal_errors_matrices_to_one_hot_data(
            _ordinal_vars_df_one_hot_representation, _ordinal_variables_names, single_classifier=use_common_classifier)
        self._ordinal_variables_names = _ordinal_variables_names

    def _convert_measured_variables_to_dict(self, measured_vars) -> dict:
        """
        Measured variables must be in data.columns.

        If dict, it must meet column constraints measurement_types must meet columns constraints.

        Returns
        -------
        dict
        """
        if isinstance(measured_vars, dict):
            valid_types = set(self.measurement_types) | {'default'}
            for k, v in measured_vars.items():
                if v not in valid_types:
                    raise ValueError(
                        "'measured_variables' can only be 'quantitative', 'semi-quantitative', 'ordinal' or 'nominal'. "
                        "Not '{}'.".format(v))
            return measured_vars

        if _is_vector_like(measured_vars):
            return {i: 'default' for i in measured_vars}
        else:
            raise ValueError("measured_variables must be either dict or list. Not {}."
                             .format(type(measured_vars).__name__))

    @staticmethod
    def _check_data(data_, measured_vars):
        """
        Must be a dataframe. With columns mentioned in ``measured_variables``.

        Additional columns (including 'time' columns) are considered experimental conditions settings.
        """
        if not isinstance(data_, pd.DataFrame):
            raise ValueError("'data' can only be a pandas DataFrame. Not a {}".format(type(data_).__name__))

        data = data_.reset_index()
        if 'index' in data:
            data = data.drop(columns=['index'])

        data_cols = set(data.columns)
        required_cols = set(measured_vars.keys())

        if required_cols - data_cols != set():
            missing_cols = list(required_cols - data_cols)
            raise ValueError("'measured_variables' mentioned these variables not present in the data: "
                             + _list_the_errors(missing_cols)+".")
        return data

    @staticmethod
    def _get_experimental_conditions_df(data, measured_vars):
        data_cols = set(data.columns)
        measured_cols = set(measured_vars.keys())
        extra_cols = list(data_cols-measured_cols)
        return data[extra_cols]

    @property
    def observables(self):
        """These constitute required observables in the PySB :class:`models <pysb.core.Model>`"""
        return self._observables

    @observables.setter
    def observables(self, v):
        self._observables = self._check_observables(v)

    @staticmethod
    def _check_observables(obs):
        if _is_vector_like(obs):
            return obs
        else:
            raise ValueError("'observables' must be list")

    @staticmethod
    def _make_default_ordinal_errors_matrix(size, measurement_error=None):
        """
        Return square numpy array like this:
        array([[ 0.95 ,  0.05 ,  0.   ,  0.   ,  0.   ],
               [ 0.025,  0.95 ,  0.025,  0.   ,  0.   ],
               [ 0.   ,  0.025,  0.95 ,  0.025,  0.   ],
               [ 0.   ,  0.   ,  0.025,  0.95 ,  0.025],
               [ 0.   ,  0.   ,  0.   ,  0.05 ,  0.95 ]])
        If size = 0, return a numpy array of size 0
        """
        if size == 0:
            return np.array([])

        if measurement_error is None:
            err = 0.05

        # make user measurement error is a float between 0 and 1.
        elif isinstance(measurement_error, float) and 0. < measurement_error < 1:
            err = measurement_error

        else:
            raise ValueError('Ordinal Data Protocols permit only one unique value of '
                             'measurement_error per observable Ordinal data')
        if size == 1:
            return np.array([err])

        c = np.eye(size, k=0) * (1.0 - err) + np.eye(size, k=-1) * (err / 2.0) + np.eye(size, k=1) * (err / 2.0)
        c[(0, -1), (1, -2)] *= 2

        return c

    @staticmethod
    def _check_measurement_error(measurement_error, measured_variables):
        """
        Returns a dictionary or error terms for each measured value.

        Parameter
        ---------
        measurement_error:
        """
        if isinstance(measurement_error, float):
            return {k: measurement_error for k in measured_variables.keys()}

        if measurement_error is None:
            measurement_error = dict()

        if isinstance(measurement_error, dict):
            for k, v in measured_variables.items():
                measurement_error.update({k:measurement_error.get(k, 0.05 if v is 'ordinal' else 0.20)})
        else:
            raise ValueError('measurement_error must be a float or dict with float item')
        return measurement_error

    def _get_errors_df(self):
        errors_df = pd.DataFrame()
        for k, v in self.measured_variables.items():
            if v is not 'ordinal':
                errors_df[k+'__error'] = self._measurement_error[k] * self.data[k]

            # set zero and negative values equal to 10% of the smallest positive value in the error_df
            errors_df_min = errors_df[errors_df > 0].min()
            errors_df = errors_df.clip(0.1 * errors_df_min, axis=1)

        errors_df[self.experimental_conditions.columns] = self.experimental_conditions
        return errors_df

    def _get_ordinal_errors_matrices(self, ordinal_matrix_dict, ordinal_variables_names, single_classifier=False):
        """
        Return a dict of ordinal error matrices.

        Parameters
        ----------
        ordinal_variables_names = list of measured_variables to create an ordinal errors matrix for.

        single_classifier: bool, all matrices are the same size. Otherwise they are as large as the number of unique
        categories named in the data.
        """
        if single_classifier:
            unique_cats = set()
            for k in ordinal_variables_names:
                unique_cats |= set(self.data[k].unique())
            err_mat_size = len(unique_cats)
            for k in ordinal_variables_names:
                ordinal_matrix_dict.update(
                    {k: self._make_default_ordinal_errors_matrix(
                        err_mat_size, measurement_error=self._measurement_error[k])})
        else:
            for k in ordinal_variables_names:
                err_mat_size = len(self.data[k].unique())
                ordinal_matrix_dict.update(
                    {k: self._make_default_ordinal_errors_matrix(
                        err_mat_size, measurement_error=self._measurement_error[k])}
                )
        return ordinal_matrix_dict

    def _one_hot_transform_of_data(self, ordinal_variables_names):
        """
        Does a one-hot transformation on the ordinal columns in the data, and names the columns to match the transform
        results.

        :return:
        """
        data_for_likelihood = self.data.copy()
        for val in ordinal_variables_names:
            data_i = pd.get_dummies(self.data[val])
            data_i.rename(columns={j: '{}__{}'.format(str(val), str(j)) for j in data_i.columns}, inplace=True)
            data_i_cols = data_i.columns
            data_for_likelihood[data_i_cols] = data_i[data_i_cols]
        return data_for_likelihood

    def _apply_ordinal_errors_matrices_to_one_hot_data(self, one_hot_transformed_data_df, ordinal_variables_names,
                                                       single_classifier=False):
        if len(one_hot_transformed_data_df) == 0:
            return one_hot_transformed_data_df[ordinal_variables_names]

        errors_df = pd.DataFrame()
        for ord_var in ordinal_variables_names:
            data_df = one_hot_transformed_data_df.filter(regex='^{}__'.format(ord_var))
            if single_classifier:
                all_cats = [ord_var+'__'+cs.split('__')[1] for cs in one_hot_transformed_data_df.filter(regex='__').columns]
                missing_cats = list(set(all_cats) - set(data_df.columns))
                data_df = pd.concat([pd.DataFrame(np.zeros((len(data_df), len(missing_cats))), columns=missing_cats),
                                     data_df], axis=1)
            cols = sorted(data_df.columns)
            errors_df[cols] = pd.DataFrame(data_df[cols].values.dot(self._ordinal_errors_matrices[ord_var]),
                                           columns=cols)
        errors_df[self.experimental_conditions.columns] = self.experimental_conditions
        return errors_df
