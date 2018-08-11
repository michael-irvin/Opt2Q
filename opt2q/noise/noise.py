"""
Tools for Simulating Extrinsic Noise and Experimental Conditions
"""

# MW Irvin -- Lopez Lab -- 2018-08-08
from opt2q.utils import _list_the_errors, MissingParametersErrors
from pysb.bng import generate_equations
import pandas as pd
import numpy as np


class NoiseModel(object):
    """
    Models extrinsic noise effects and experimental conditions as variations in the values of parameters in a PySB
    :class:`~pysb.core.Model`.

    Generates a :class:`pandas.DataFrame` of noisy and/or static values. The generated values dictate values of `PySB`
    :class:`~pysb.core.Model` :class:`~pysb.core.Parameter` and/or :class:`~pysb.core.ComplexPattern` (model species).

    The :run:`~opt2q.noise.NoiseModel.run` method returns a :class:`pandas.DataFrame` formatted for use in the Opt2Q
    :class:`opt2q.simulator.Simulator`.

    Parameters
    ----------
    param_mean: :class:`pandas.DataFrame` (optional)
        Object names and their mean (values in a DataFrame with the following columns:

        `param` (column): `str`
            object name
        `value` (column): `float`
            value (or average value, when noise is applied) for the parameter
        `apply_noise` (column): `bool` (optional)
            When True, this parameter is drawn from a underlying probability distribution. False by default, unless the parameter (and experimental condition) is also mentioned in the `covariance`.
        `num_sims`(column): `int` (optional)
            Sample size of the parameters.
            A :attr:`default_sample_size <opt2q.noise.NoiseModel.default_sample_size>` of 50 is used if this
            column is absent and `apply_noise` is True for any parameter in the experimental condition. If `apply_noise`
            is False, this defaults to 1.
        Additional columns designate experimental conditions. (optional)
            These columns cannot have the following names: 'param_name', 'value', 'apply_noise', 'param_i', 'param_j',
            'covariance', 'num_sims'.

            .. note::
                Each unique row, in these additional columns, designates a different experimental condition. If no
                additional columns are present, a single unnamed experimental condition is provided by default.

    param_covariance: :class:`pandas.DataFrame` (optional)
        Object names and their covariance values in a DataFrame with the following columns:

        `param_i` (column): `str`
            Model object name
        `param_j` (column): `str`
            Model object name
        `value` (column):`float`
            Covariance between model objects `param_i` and `param_j`
        Additional columns designate experimental conditions. (optional)
            These columns cannot have the following names: 'param_name', 'value', 'apply_noise', 'param_i', 'param_j',
            'covariance', 'num_sims'.

            .. note::
                Each unique row, in these additional columns, designates a different experimental condition. If no
                additional columns are present, a single unnamed experimental condition is provided by default.

        **Pending code** (Todo): Currently num_sims column is not read as part of the covariance dataframe. Add that option.

    model: `PySB` :class:`~pysb.core.Model` (optional)

    kwargs: dict, (optional)
        Dictionary of keyword arguments:

        ``noise_simulator``: Function that applies noise for the parameters. Defaults to :func:`~opt2q.noise.multivariate_log_normal_fn`
    """
    required_columns = {'param_mean':{'param', 'value'},
                        'param_covariance':{'param_i', 'param_j', 'value'}}
    other_useful_columns = {'simulation', 'num_sims', 'apply_noise'}
    default_param_values = None  # Use dict to designated defaults for 'param' and 'value'.

    def __init__(self, param_mean=None, param_covariance=None, model=None):
        _param_mean = self._check_required_columns(param_mean, var_name='param_mean')
        _param_covariance = self._check_required_columns(param_covariance, var_name='param_covariance')
        _param_mean, _param_covariance, \
            _exp_con_cols, _exp_con_df = self._check_experimental_condition_cols(_param_mean, _param_covariance)

        _param_mean = self._add_params_from_param_covariance(_param_mean, _param_covariance)
        _param_mean = self._add_apply_noise_col(_param_mean)

        if _param_mean.shape[0] != 0 and _param_mean['value'].isnull().values.any():
            _param_mean = self._add_missing_param_values(_param_mean, model=model)

        self._param_mean = _param_mean
        self._param_covariance = _param_covariance

    # setup
    def _check_required_columns(self, param_df, var_name ='param_mean'):
        """
        First check of param_mean and param_covariance. Checks that the DataFrame as the required column names.

        Parameters
        ----------
        param_df: :class:`~pandas.DataFrame` or None
            param_means or param_covariance argument passed upon instantiation.

        var_name: str (optional)
            Name of the variable (:class:`~pandas.DataFrame`) who's columns need checking.
            Currently 'param_mean' and 'param_covariance' only.

        Returns
        -------
        param_df: :class:`~pandas.DataFrame`
            Returns empty :class:`~pandas.DataFrame` if ``param_df`` is None.
        """

        if param_df is None:
            return pd.DataFrame()

        try:
            if param_df.shape[0] == 0:
                return pd.DataFrame(columns=list(set(self.required_columns[var_name])|set(param_df.columns)))

            if self.required_columns[var_name] - set(param_df.columns) == set([]):  # df has required cols.
                return param_df
            else:
                note = "'{}' must be a pd.DataFrame with the following column names: ".format(var_name) + \
                       _list_the_errors(self.required_columns[var_name] - set(param_df.columns)) + "."
                raise ValueError(note)
        except KeyError:
            raise KeyError("'{}' is not supported".format(var_name))

    def _check_experimental_condition_cols(self, param_m, param_c):
        not_exp_cols = self._these_columns_cannot_annotate_exp_cons()
        mean_exp_cols = set(param_m.columns) - not_exp_cols
        cov_exp_cols = set(param_c.columns) - not_exp_cols

        if mean_exp_cols == cov_exp_cols == set([]):
            return param_m, param_c, mean_exp_cols, pd.DataFrame()
        else:
            return self._copy_experimental_conditions_to_second_df(param_m, mean_exp_cols, param_c, cov_exp_cols)

    def _these_columns_cannot_annotate_exp_cons(self):
        """
        Return column names (set) prohibited from annotating experimental conditions
        """
        _cols = set([])  #
        for param_name, req_cols in self.required_columns.items():
            _cols |= req_cols

        return _cols | self.other_useful_columns

    def _copy_experimental_conditions_to_second_df(self, df1, df1_cols, df2, df2_cols):
        """
        Copies experimental conditions columns to a dataframe that lacks experimental conditions columns.
        """
        _cols_ = np.array([df1_cols, df2_cols])
        has_cols = _cols_ != set([])
        exp_cols = _cols_[has_cols]
        if len(exp_cols) == 1:  # only one DataFrame has additional columns
            _dfs_ = [df1, df2]
            exp_cols = list(exp_cols[0])
            df_with_cols, df_without_cols = _dfs_[list(has_cols).index(True)], _dfs_[list(has_cols).index(False)]
            exp_cols_only_df = df_with_cols[exp_cols].drop_duplicates()
            num_unique_exp_rows = len(exp_cols_only_df)
            len_df_without_cols = len(df_without_cols)

            try:
                expanded_df_without_cols = pd.concat([df_without_cols] * num_unique_exp_rows, ignore_index=True)
                expanded_df_without_cols[exp_cols] = pd.DataFrame(np.repeat(
                    exp_cols_only_df.values, len_df_without_cols, axis=0),
                    columns=exp_cols)
                return [(expanded_df_without_cols, df_with_cols)[i] for i in _cols_ != set([])]\
                       + [exp_cols, exp_cols_only_df]
            except ValueError:
                return tuple((pd.DataFrame(columns=exp_cols), df_with_cols)[i] for i in _cols_ != set([]))
        else:
            return self._combine_experimental_conditions(df1, df1_cols, df2, df2_cols)

    @staticmethod
    def _combine_experimental_conditions(df1, df1_cols, df2, df2_cols):
        """
        Combines the experimental conditions DataFrames of df1 and df2
        """
        if df1_cols == df2_cols:
            exp_cols = list(df1_cols)
            df1_exp_idx = df1[exp_cols].drop_duplicates()
            df2_exp_idx = df2[exp_cols].drop_duplicates()
            combined_exp_idx = pd.concat([df1_exp_idx, df2_exp_idx], ignore_index=True).drop_duplicates()
            return df1, df2, exp_cols, combined_exp_idx
        else:
            raise AttributeError("Means and Covariances must have the same experiment indices")

    @staticmethod
    def _combine_param_i_j(param_c):
        """
        Combines the param_i and param_j columns. This is useful for adding params mentioned in param_covariance to
        param_mean
        """
        param_c_i = param_c.rename(columns={'param_i': 'param'}, copy=True).drop(columns=['param_j', 'value'])
        param_c_j = param_c.rename(columns={'param_j': 'param'}, copy=True).drop(columns=['param_i', 'value'])
        return pd.concat([param_c_i, param_c_j], ignore_index=True).drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _add_apply_noise_col(_df, default_value=False):
        if _df.shape[0] > 0:
            try:
                _df['apply_noise'].fillna(default_value, inplace=True)
            except KeyError:
                _df['apply_noise'] = default_value
        return _df

    def _add_params_from_param_covariance(self, param_m, param_c):
        """
        Any parameters mentioned in ``param_covariance`` must also appear in ``param_mean``.  This adds the parameter
        names overwrites the ``apply_noise`` column for with to True.
        """
        if param_c.shape[0] == 0:
            return param_m

        if param_m.shape[0] == 0:
            _param_m = pd.DataFrame(columns=list(set(param_m.columns)|{'param', 'value'}))  # make it possible to merge
        else:
            _param_m = param_m

        params_from_c = self._combine_param_i_j(param_c)
        params_from_c = self._add_apply_noise_col(params_from_c, default_value=True)
        added_params = pd.merge(params_from_c.drop(columns=['apply_noise']), _param_m, how='outer')
        return params_from_c.combine_first(added_params)

    def _add_missing_param_values(self, mean, model=None):
        """
        If parameters appear in 'param_covariance' but are absent in 'param_mean', try to fill them in with parameter
        values from the model.

        Creates/Updates: self.default_param_values
        """
        if self.default_param_values is not None:
            mean['value'] = mean['value'].fillna(mean['param'].map(self.default_param_values))
        elif model is not None:
            self.default_param_values = self._get_parameters_from_model(model)
            mean['value'] = mean['value'].fillna(mean['param'].map(self.default_param_values))
        if mean.isnull().values.any():
            raise MissingParametersErrors("'param_covariance' contains parameters that are absent from 'param_mean'."
                                          " Please add these parameters to 'param_mean' or include a PySB model")
        return mean

    @staticmethod
    def _get_parameters_from_model(_model):
        generate_equations(_model)
        return {p.name: p.value for p in _model.parameters}

    @property
    def param_mean(self):
        return self._param_mean

    @param_mean.setter
    def param_mean(self, val):
        self._param_mean = val

    def update_param_mean(self):
        """
        Updates the param_mean DataFrame with values from a similarly shaped column (i.e. same columns). This method is
        intended for the :class:`~opt2q.calibrator.ObjectiveFunction`, primarily.
        """
        pass

    def run(self):
        """
        Returns a :class:`pandas.DataFrame` of noisy and/or static values of PySB` :class:`~pysb.core.Model`,
        :class:`~pysb.core.Parameter` and/or :class:`~pysb.core.ComplexPattern` (model species).
        """
        pass