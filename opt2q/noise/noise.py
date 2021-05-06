"""
Tools for Simulating Extrinsic Noise and Experimental Conditions
"""

# MW Irvin -- Lopez Lab -- 2018-08-08
from opt2q.utils import _list_the_errors, MissingParametersErrors, UnsupportedSimulator, DuplicateParameterError
from pysb.bng import generate_equations
import pandas as pd
import numpy as np


def multivariate_log_normal_fn(mean, covariance, n, names_column='param', *args, **kwargs):
    """
    Simulates extrinsic noise by applying random log-normal variation to a set of values.

    Parameters
    ----------
    mean: :class:`pandas.DataFrame`
        Object names and their mean values in a DataFrame with the following columns:

        'param' (required column)
            The name of this column can be changed by passing an argument to this function's ``name_column`` parameter.
            The names or the objects.
        'value' (required column)
            Mean (average) value (float) of the objects

    covariance: :class:`pandas.DataFrame`
        Object names and their covariance values.

            Column and index names must only be the names of parameters.
            These names must also be present in the `param_name` column of this function's ``mean`` parameter.

    n: `int`
        number of samples from the distribution

    names_column: `str` (optional)
        name of the column in this function's ``mean`` parameter that has the names of the objects
        that will have noise applied to them. Defaults to "param_name".

    atol: 'float` (optional)
        Minimum value for means and values along the diagonal. This avoids nans. Defaults to 1.0e-8.


    Returns
    -------
    A :class:`pandas.DataFrame` whose columns are the object names and each row contains a sample of the objects' values.

    """

    # clip zeros to prevent breaking log-norm function
    atol = 0.01*covariance[covariance > 0].min().min()
    atol_m = 0.01*mean[mean.value > 0]['value'].min()

    _mean = mean.set_index(names_column).clip(lower=atol_m).astype(float, errors='ignore')
    _cov = covariance[_mean.index].reindex(_mean.index)

    _cov_diagonal = pd.DataFrame([np.clip(np.diag(_cov), atol, np.inf)], columns=_mean.index)
    for cov_i in _mean.index:
        _cov.at[cov_i, cov_i] = _cov_diagonal[cov_i]

    mean_t_mean = np.product(np.meshgrid(_mean.values, _mean.values), axis=0)

    mu = np.log(_mean.values[:, 0]) - 0.5 * np.log(1 + np.diag(_cov.values) / np.diag(mean_t_mean))
    # mu = np.log(_mean.values[:, 0])/(np.sqrt(np.diag(_cov.values) / (np.diag(mean_t_mean))+1))

    cov = np.log((_cov / mean_t_mean) + 1)

    return pd.DataFrame(np.exp(np.random.multivariate_normal(mu, cov, n)), columns=_mean.index.values)
    # return pd.DataFrame(np.random.multivariate_normal(_mean.values[:,0], _cov, n), columns=_mean.index.values)


class NoiseModel(object):
    """
    Models extrinsic noise effects and experimental conditions as variations in the values of parameters in a PySB
    :class:`~pysb.core.Model`.

    Generates a :class:`pandas.DataFrame` of noisy and/or static values. The generated values dictate values of `PySB`
    :class:`~pysb.core.Model` :class:`~pysb.core.Parameter` and/or :class:`~pysb.core.ComplexPattern` (model species).

    The :meth:`~opt2q.noise.NoiseModel.run` method returns a :class:`pandas.DataFrame` formatted for use in the Opt2Q
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
            When True, this parameter is drawn from a underlying probability distribution. False by default, unless the
            parameter (and experimental condition) is also mentioned in the `covariance`.
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
        Object names and their variance or covariance values in a DataFrame with the following columns:

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

        **Pending code** (Todo): Currently num_sims column is not read as part of the covariance dataframe. Add that option?

    model: `PySB` :class:`~pysb.core.Model` (optional)

    options: dict, (optional)
        Dictionary of keyword arguments:

        ``noise_simulator``: Function that applies noise for the parameters. Defaults to :func:`~opt2q.noise.multivariate_log_normal_fn`
        ``noise_simulator_kwargs``: Dictionary of kwargs passed to the noise simulator

    Attributes
    ----------
    default_param_values:(dict)
        The dictionary of parameter names and values. Any parameters named in param_covariance must appear in param_mean. If not, they can be retrieved here.

    default_sample_size: (int)
        Class attribute dictates num of simulations of a noisy value when not specified via the ``num_sim`` column of the ``param_means`` parameter.

    default_coefficient_of_variation: (float)
        Between 0 and 1. Default coefficient of variation, used when a value has noise applied to it but a variance value is not set.

    supported_noise_simulators: (dict)
        Dictionary of supported noise simulator names and functions
    """
    supported_noise_simulators = {'multivariate_log_norm': multivariate_log_normal_fn}
    required_columns = {'param_mean': {'param', 'value'}, 'param_covariance': {'param_i', 'param_j', 'value'}}
    other_useful_columns = {'simulation', 'num_sims', 'apply_noise'}

    default_param_values = None  # Use dict to designated defaults for 'param' and 'value'.
    default_sample_size = 50  # int only
    default_coefficient_of_variation = 0.2

    def __init__(self, param_mean=None, param_covariance=None, model=None, **options):
        # input settings
        _param_mean = self._check_required_columns(param_mean, var_name='param_mean')
        _param_covariance = self._check_required_columns(param_covariance, var_name='param_covariance')
        _param_mean, _param_covariance, \
            _exp_con_cols, _exp_con_df = self._check_experimental_condition_cols(_param_mean, _param_covariance)

        self._check_for_duplicate_rows(_param_mean, _exp_con_cols, var_name='param_mean')
        self._check_for_duplicate_rows(_param_covariance, _exp_con_cols, var_name='param_covariance')

        _param_mean = self._add_params_from_param_covariance(_param_mean, _param_covariance)

        if _param_mean.shape[0] != 0 and _param_mean['value'].isnull().values.any():
            _param_mean = self._add_missing_param_values(_param_mean, model=model)

        _param_mean = self._add_apply_noise_col(_param_mean)

        self._exp_cols_df = self._add_num_sims_col_to_experimental_conditions_df(_param_mean, _exp_con_df, _exp_con_cols)
        self._exp_con_cols = _exp_con_cols
        self._param_mean = _param_mean
        self._param_covariance = _param_covariance

        # simulator setting
        noise_simulator = options.get('noise_simulator', 'multivariate_log_norm')
        self._noise_simulator = self._check_noise_simulator(noise_simulator)
        self._noise_simulator_kwargs = options.get('noise_simulator_kwargs', {})

        self._run = [self._simulate, self._simulate_groups][self._exp_cols_df.shape[0] > 1]

    # Setup
    def _check_required_columns(self, param_df, var_name='param_mean'):
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

    def _check_for_duplicate_rows(self, param_df, exp_col, var_name='param_mean'):
        """
        Raises ValueError the same parameter appears in the same experiment twice
        """
        if param_df.shape[0] is 0:
            return
        pertinent_cols = list(exp_col | self.required_columns[var_name]-{'value'})
        if param_df[pertinent_cols].drop_duplicates().shape[0] != param_df.shape[0]:
            raise DuplicateParameterError("'{}' contains a duplicate parameter.".format(var_name))

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
                return tuple([(expanded_df_without_cols, df_with_cols)[i] for i in _cols_ != set([])]
                             + [set(exp_cols), exp_cols_only_df])

            except ValueError:   # breaks when df_with_out_columns is of len 0.
                return tuple([(pd.DataFrame(columns=list(set(exp_cols)|set(df_without_cols.columns))), df_with_cols)[i]
                              for i in _cols_ != set([])] + [set(exp_cols), exp_cols_only_df])
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
            return df1, df2, set(exp_cols), combined_exp_idx
        else:
            raise AttributeError("Means and Covariances use the same columns to index experiments")

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

        if mean[['value']].isnull().values.any():
            raise MissingParametersErrors("'param_covariance' contains parameters that are absent from 'param_mean'."
                                          " Please add these parameters to 'param_mean' or include a PySB model")
        return mean

    @staticmethod
    def _get_parameters_from_model(_model):
        generate_equations(_model)
        return {p.name: p.value for p in _model.parameters}

    def _add_num_sims_col_to_experimental_conditions_df(self, param_mean, exp_con_df, exp_con_cols):
        """
        Adds num_sims column (i.e. sample size) to the dataframe. If not already there.

        Group by experimental conditions then apply num_sims, as either max of the ``num_sims`` or default for if
        ``apply_noise`` is True or Not.

        Requires an ``apply_noise`` column which is added (if not already present by self._add_apply_noise_col).

        Parameters
        ----------
        param_mean: :class:`~pandas.DataFrame`

            - Must have column named ``apply_noise``.
            - The columns annotating the experimental conditions must be the same as exp_con_df.columns.
            - There should be the same number of unique e experimental conditions rows as there are rows in exp_con_df

        exp_con_df: :class:`~pandas.DataFrame`
            The experimental conditions in the measurement.
            Must be the same length as the number of unique experimental conditions rows in exp_con_df param_mean
            Or pd.DataFrame() (if exp_con_cols is an empty set)
            This is created by self._check_experimental_condition_cols

        exp_con_cols: set
            The columns designating experimental conditions. This is created by self._check_experimental_condition_cols
        """

        if param_mean.shape[0] is 0:
            return pd.DataFrame()

        if 'num_sims' in param_mean.columns:
            if len(exp_con_cols) > 0:
                param_num_sims = param_mean.groupby(list(exp_con_cols)).apply(self._set_num_sims_as_max)
                return self._merge_num_sims_w_ec(param_num_sims, exp_con_cols, exp_con_df)
            else:
                param_num_sims = self._set_num_sims_as_max(param_mean)
        else:
            if len(exp_con_cols) > 0:
                param_num_sims = param_mean.groupby(list(exp_con_cols)).apply(self._set_num_sims_as_default)
                return self._merge_num_sims_w_ec(param_num_sims, exp_con_cols, exp_con_df)
            else:
                param_num_sims = self._set_num_sims_as_default(param_mean)

        return param_num_sims[['num_sims']].drop_duplicates()

    @staticmethod
    def _set_num_sims_as_max(group):
        group['num_sims'] = group['num_sims'].max()
        return group

    def _set_num_sims_as_default(self, group):
        """
        Define the ``num_sims`` of the ``param_mean`` :class:`~pandas.DataFrame` or group as the default based on if
        ``apply_noise`` is True or not.
        """
        group['num_sims'] = self.default_sample_size if group.apply_noise.any() else 1
        return group

    @staticmethod
    def _merge_num_sims_w_ec(params_num_sims , exp_con_cols, exp_con_df, how='outer'):
        exp_cols = list({'num_sims'} | exp_con_cols)
        num_sims = params_num_sims[exp_cols].drop_duplicates()
        return exp_con_df.merge(num_sims, how=how)

    # Methods used by the calibrator.objective_function
    def update_values(self, param_mean=None, param_covariance=None):
        """
        Replaces rows of the DataFrame

        .. note:: Updates cannot introduce new values in the 'params' and experimental conditions columns
                    nor can it introduce new columns.

        Examples
        --------
        >>> from opt2q.noise import NoiseModel
        >>> mean_values = pd.DataFrame([['A', 1.0, 'KO'], ['B', 1.0, 'WT'], ['A', 1.0, 'WT']],
        ...                            columns=['param', 'value', 'ec'])
        >>> noise_model = NoiseModel(param_mean=mean_values)
        >>> noise_model.update_values(param_mean=pd.DataFrame([['A', 2]], columns=['param', 'value']))
        >>> print(noise_model.param_mean)
            param   value   'ec'
        0   'A'     2.0     'KO'
        1   'A'     2.0     'WT'
        2   'B'     1.0     'WT'

        Notice how the DataFrame in update_values does not mention 'ec'. In this case *all* 'A' take the updated value.

        >>> from opt2q.noise import NoiseModel
        >>> mean_values = pd.DataFrame([['A', 1.0, 'KO'], ['B', 1.0, 'WT'], ['A', 1.0, 'WT']],
        ...                            columns=['param', 'value', 'ec'])
        >>> noise_model = NoiseModel(param_mean=mean_values)
        >>> noise_model.update_values(param_mean=pd.DataFrame([['KO', 2]], columns=['ec', 'num_sims']))
        >>> print(noise_model.param_mean)
            param   value   ec      num_sims
        0   'A'     2.0     'KO'    2
        1   'A'     2.0     'WT'    1
        2   'B'     1.0     'WT'    1

        >>> print(noise_model._exp_con_cols)
            ec      num_sims
        0   'KO'    2
        1   'WT'    1
        """
        if param_mean is not None:
            updated_param_mean = self._update_param_mean(param_mean)
            if 'num_sims' in updated_param_mean.columns:
                _exp_cols = self._update_exp_con_df(updated_param_mean, self._exp_con_cols, self._exp_cols_df)
                if 'num_sims' not in self._param_mean.columns:
                    updated_param_mean = updated_param_mean.drop(columns=['num_sims'])
                self._check_updated_df(updated_param_mean, "param_mean")
                self._param_mean = updated_param_mean
                self._exp_cols_df = _exp_cols
            else:
                self._check_updated_df(updated_param_mean, "param_mean")
                self._param_mean = updated_param_mean

        if param_covariance is not None:
            updated_param_covariance = self._update_param_covariance(param_covariance)
            try:
                self._check_updated_df(updated_param_covariance, "param_covariance")
            except ValueError:  # Try switching the param axes
                param_covariance = param_covariance.rename(columns={'param_i':'param_j', 'param_j':'param_i'})
                updated_param_covariance = self._update_param_covariance(param_covariance)
                self._check_updated_df(updated_param_covariance, "param_covariance")
            self._param_covariance = updated_param_covariance

    def _update_param_mean(self, param_mean):
        """
        Updates the param_mean DataFrame with values from a similarly shaped column (i.e. same columns).
        """
        index_for_update = list((self._exp_con_cols | {'param'}).intersection(set(param_mean.columns)))
        old_means = self._param_mean.set_index(index_for_update)
        updates = param_mean.set_index(index_for_update)
        new_means = updates.combine_first(old_means).reset_index()
        new_means['apply_noise'] = new_means['apply_noise'].astype(bool)
        return new_means

    def _update_param_covariance(self, param_covariance):
        """
        Updates the param_mean DataFrame with values from a similarly shaped column (i.e. same columns). This method is
        intended for the :class:`~opt2q.calibrator.ObjectiveFunction`, primarily.
        """
        index_for_update = list((self._exp_con_cols | {'param_i', 'param_j'}).intersection(set(param_covariance.columns)))
        old_cov = self._param_covariance.set_index(index_for_update)
        updates = param_covariance.set_index(index_for_update)
        return updates.combine_first(old_cov).reset_index()

    def _update_exp_con_df(self, param_mean, exp_con_cols, exp_con_df):
        if len(exp_con_cols) > 0:
            param_num_sims = param_mean.groupby(list(exp_con_cols)).apply(self._set_num_sims_as_max)
            old_exp_cons = exp_con_df.set_index(list(exp_con_cols))
            new_exp_cons = param_num_sims[list(exp_con_cols|{'num_sims'})].set_index(list(exp_con_cols))
            new_exp_cons = new_exp_cons.combine_first(old_exp_cons).reset_index().drop_duplicates()
            new_exp_cons['num_sims'] = new_exp_cons['num_sims'].astype(int)
            return new_exp_cons.reset_index(drop=True)
        else:
            param_num_sims = self._set_num_sims_as_max(param_mean)[['num_sims']].astype(int)
            return param_num_sims.drop_duplicates().reset_index(drop=True)

    def _check_updated_df(self, df, var_name):
        if df.shape != self.__getattribute__(var_name).shape:
            raise ValueError("Your update includes experimental conditions and 'param' not present in 'param_mean'."
                             "Updates to 'params' and experimental conditions columns are forbidden.")

    # Accessing Attributes from the Outside
    @property
    def param_mean(self):
        return self._param_mean

    @property
    def param_covariance(self):
        return self._param_covariance

    @property
    def experimental_conditions_dataframe(self):
        return self._exp_cols_df

    # Run Method and Assoc.
    def _check_noise_simulator(self, _simulator_name):
        try:
            return self.supported_noise_simulators[_simulator_name]
        except KeyError:
            raise UnsupportedSimulator("{} is not a supported noise simulator".format(_simulator_name))

    def run(self):
        """
        Returns a :class:`pandas.DataFrame` of noisy and/or static values of PySB` :class:`~pysb.core.Model`,
        :class:`~pysb.core.Parameter` and/or :class:`~pysb.core.ComplexPattern` (model species).

        This serves as an input to the :class:`~opt2q.simulator.Simulator`.
        """
        simulated = self._run(self.param_mean, self.param_covariance, self.experimental_conditions_dataframe)
        simulated = simulated.reset_index().rename(columns={'index': 'simulation'})
        return simulated

    def _simulate_groups(self, mean, cov, exp):
        simulated = pd.DataFrame()
        for idx, row in exp.iterrows():
            exp_id = pd.DataFrame(row).T
            mean_i = mean
            cov_i = cov

            for col in self._exp_con_cols:
                mean_conditional = mean[col] == row[col]
                cov_conditional = cov[col] == row[col]
                mean_i = mean_i[mean_conditional]
                cov_i = cov_i[cov_conditional]

            sim_i = self._simulate(mean_i, cov_i, exp_id)
            simulated = pd.concat((simulated, sim_i), ignore_index=True, sort=False)
        return simulated

    def _simulate(self, mean, cov, exp):
        """
        Generates a DataFrame of parameter values formatted for use in the Opt2Q simulator.

        The column names are taken from `param` and the experimental conditions columns.

        Parameters
        ----------
        mean: :class:`~pandas.DataFrame`
            Mean param values. This class's :attr:`~opt2q.noise.NoiseModel.param_means` attribute or a group thereof.
        cov: :class:`~pandas.DataFrame`
            Param covariance values. This class's :attr:`~opt2q.noise.NoiseModel.param_covariance` attribute or a group
            thereof.
        exp: :class:`~pandas.DataFrame`
            Experimental Conditions index This class's :attr:`~opt2q.noise.NoiseModel.param_covariance` attribute or a
            group thereof.
        """
        if mean.shape[0] == 0:
            return pd.DataFrame(columns=self._exp_con_cols)
        else:
            fixed_params = self._add_fixed_values(mean, exp)
            noisy_params = self._add_noisy_values(mean, cov, exp)
            exp_indices = self._apply_exp_idx(exp)

            simulated = pd.DataFrame()
            simulated[fixed_params.columns] = fixed_params
            simulated[noisy_params.columns] = noisy_params
            simulated[list(self._exp_con_cols)] = exp_indices
            return simulated

    @staticmethod
    def _add_fixed_values(_mean, exp, names_col='param'):
        """
        Repeats the values in ``_mean`` for which 'apply_noise' is False. The column is the param name.

        .. note:: returns an index named 'param'. This is resolved by the method that calls this method.

        Parameters
        ----------
        _mean: :class:`~pandas.DataFrame`
            Mean param values. This class's :attr:`~opt2q.noise.NoiseModel.param_means` attribute or a group thereof.
        exp: :class:`~pandas.DataFrame`
            Experimental Conditions index This class's :attr:`~opt2q.noise.NoiseModel.param_covariance` attribute or a
            group thereof.
        names_col: (str), optional
            Defaults to 'param'. This will become useful when Opt2Q supports PySB model components (e.g. initials).

        Return
        ------
        :class:`~pandas.DataFrame`
        """
        n = exp['num_sims'].values[0]

        fixed_terms = _mean[_mean['apply_noise'] == False]
        fixed_terms = fixed_terms[[names_col, 'value']].set_index(names_col).T
        fixed_terms = pd.DataFrame(np.repeat(fixed_terms.values, n, axis=0), columns=fixed_terms.columns)
        return fixed_terms

    def _add_noisy_values(self, _mean, _cov, exp, names_col='param'):
        """
        Applies noise to parameters.

        Parameters
        ----------
        _mean: :class:`~pandas.DataFrame`
            Mean param values. This class's :attr:`~opt2q.noise.NoiseModel.param_means` attribute or a group thereof.

        .. note::

            It is essential that _mean includes parameters mentioned in _cov. The
            :class:`~opt2q.noise.NoiseModel._add_params_from_param_covariance` method makes sure
        """
        n = int(exp['num_sims'].values[0])
        varied_terms = _mean[_mean['apply_noise'] == True]

        if varied_terms.shape[0] is 0:
            return pd.DataFrame()

        cov_mat = self._create_covariance_matrix(varied_terms, _cov)
        return self._noise_simulator(varied_terms[['param', 'value']], cov_mat, n, names_column=names_col, **self._noise_simulator_kwargs)

    def _create_covariance_matrix(self, _means, _covariances):
        """
        Returns :class:`~pandas.DataFrame` covariance matrix the columns and indices are the parameter names
        """
        params_array = _means[['param', 'value']].values
        len_cov_mat = params_array.shape[0]
        cov_mat_diagonal = (np.asarray(params_array[:, 1], dtype=float) * self.default_coefficient_of_variation)**2

        # square matrix of NaNs length of dict_means
        raw_covariance_matrix = np.full((len_cov_mat, len_cov_mat), np.NaN)
        np.fill_diagonal(raw_covariance_matrix, cov_mat_diagonal)
        incomplete_cov_mat = pd.DataFrame(raw_covariance_matrix, index=params_array[:, 0], columns=params_array[:, 0])

        complete_covariance_mat = self._update_covariance_matrix(params_array, _covariances, incomplete_cov_mat)
        return complete_covariance_mat.fillna(0.)

    @staticmethod
    def _update_covariance_matrix(_mean_params_array, covariances_df, incomplete_covariance_matrix):
        # Covariance updates
        try:
            covariance_updates = covariances_df.pivot('param_i', 'param_j', 'value')
            covariance_updates = covariance_updates.reindex(index=_mean_params_array[:, 0],
                                                            columns=_mean_params_array[:, 0])
            covariance_updates.update(covariance_updates.transpose())  # Todo: Make sure p_i, p_j == p_j, p_i
        except KeyError:
            covariance_updates = pd.DataFrame()

        # Complete the covariance matrix here.
        incomplete_covariance_matrix.update(covariance_updates)
        complete_covariance_matrix = incomplete_covariance_matrix
        return complete_covariance_matrix

    def _apply_exp_idx(self, exp):
        cols = list(self._exp_con_cols)
        n = exp['num_sims'].values[0]
        return pd.DataFrame(np.repeat(exp[cols].values, n, axis=0), columns=cols)
