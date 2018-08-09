"""
Tools for Simulating Extrinsic Noise and Experimental Conditions
"""

# MW Irvin -- Lopez Lab -- 2018-08-08
from opt2q.utils import _list_the_errors
import pandas as pd


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

    def __init__(self, param_mean=None, param_covariance=None):

        _param_mean = self._check_required_columns(param_mean, var_name='param_mean')
        _param_mean = self._add_apply_noise_col(_param_mean)
        self._param_mean = _param_mean

        _param_covariance = self._check_required_columns(param_covariance, var_name='param_covariance')

        pass

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
            if self.required_columns[var_name] - set(param_df.columns) == set([]): # df has required cols.
                return param_df
            else:
                note = "'{}' must be a pd.DataFrame with the following column names: ".format(var_name) + \
                       _list_the_errors(self.required_columns[var_name] - set(param_df.columns)) + "."
                raise ValueError(note)
        except KeyError:
            raise KeyError("'{}' is not supported".format(var_name))

    @staticmethod
    def _add_apply_noise_col(mean_df):
        if mean_df.shape[0] > 0:
            try:
                mean_df['apply_noise'].fillna(False, inplace=True)
            except KeyError:
                mean_df['apply_noise'] = False
        return mean_df

    def run(self):
        """
        Returns a :class:`pandas.DataFrame` of noisy and/or static values of PySB` :class:`~pysb.core.Model`,
        :class:`~pysb.core.Parameter` and/or :class:`~pysb.core.ComplexPattern` (model species).
        """
        pass

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