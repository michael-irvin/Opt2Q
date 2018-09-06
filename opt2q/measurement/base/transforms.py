# MW Irvin -- Lopez Lab -- 2018-08-24
import pandas as pd
import numpy as np
import inspect
from scipy.interpolate import interp1d
from opt2q.utils import *


class Transform(object):
    """
    Base class for measurement processes, i.e. transformations to the simulation results that constitute a simulation of
    the measurement process.

    This class can change the number rows and/or columns; as the underlying calculations can apply to individual columns
    and rows, or to groups of columns and/or rows. Some transforms do not support certain grouped calculations while
    others require it.

    This class can also depend on free-parameters.
    """
    set_params_fn = dict()

    def __init__(self):
        pass

    # clearer __repr__
    def __repr__(self):
        sig = inspect.signature(self.__init__)
        sig_args, sig_kw = getattr(self, '_signature_params', ((), dict()))
        sig_str = sig.bind_partial(*sig_args, **sig_kw).__repr__().split('BoundArguments ')[1][:-1]
        name_str = self.__class__.__name__
        return '{}{}'.format(name_str, sig_str)

    # get params
    def get_params(self, transform_name=None):
        """
        Returns a dictionary of parameter names and values for the :class:`~opt2q.measurement.base.Transform`.

        Return parameters in the model.

        Parameters
        ----------
        transform_name: str (optional)
            Name of the process. This is prepended to the parameter-name; e.g. 'process__parameter'.
        """
        return self._get_params(transform_name=transform_name)

    def _get_params(self, transform_name=None):
        if transform_name is not None:
            return self._get_params_named_process(transform_name)

        _params = self._get_params_custom_params(dict())

        for (k, v) in self.__dict__.items():
            if k[0] == '_' or k == 'set_params_fn':  # ignore private attrs
                pass
            else:
                self._get_params_update_params_dict(_params, k, v)
        return _params

    def _get_params_custom_params(self, _params):
        for (k, v) in getattr(self, '_get_params_dict', dict()).items():
            self._get_params_update_params_dict(_params, k, v)
        return _params

    def _get_params_update_params_dict(self, _params, k, v):
        if isinstance(v, dict):
            self._get_params_next_layer(k, v, _params)
        else:
            _params.update({k: v})

    def _get_params_named_process(self, name):
        _params = dict()

        specific_params_dict = getattr(self, '_get_params_dict', dict())
        for (k, v) in specific_params_dict.items():
            self._get_params_update_params_dict(_params, '{}__{}'.format(name, k), v)

        for (k, v) in self.__dict__.items():
            if k[0] == '_' or k == 'set_params_fn':
                pass
            else:
                self._get_params_update_params_dict(_params, '{}__{}'.format(name, k), v)
        return _params

    def _get_params_next_layer(self, layer_name, layer_val, params):
        for (k, v) in layer_val.items():
            if isinstance(v, dict):
                self._get_params_next_layer('{}__{}'.format(layer_name, k), v, params)
            else:
                params.update({'{}__{}'.format(layer_name, k): v})

    # set params
    def set_params(self, **params):
        """
        Sets parameters of a :class:`~opt2q.measurement.base.transform.Transform` class.

        Only parameters mentioned in the class's :attr:`~opt2q.measurement.base.transform.Transform.set_params_fn` can
        be updated by this method. All others are ignored.

        Parameters
        ----------
        params: dict
            parameter names and values
        """
        params_dict = self._build_params_dict(params)
        self._set_params(**params_dict)

    def _set_params(self, **params):
        params_present = set(params.keys()).intersection(self.set_params_fn.keys())
        for k in params_present:
            if isinstance(params[k], dict):
                self.set_params_fn[k](**params[k])
            else:
                self.set_params_fn[k](params[k])

        # for k, v in params.items():
        #     try:
        #         if isinstance(v, dict):
        #             self.set_params_fn[k](**v)
        #         else:
        #             self.set_params_fn[k](v)
        #     except KeyError:
        #         continue

    def _build_params_dict(self, params):
        """
        Builds a nested dict of values for the parameters being changed. Names are presented in reverse order.

        Example
        -------

        """

        # if not hasattr(self, '_process_params'):
        #     self._process_params = self.get_params()
        param_dict = {}
        for k, v in params.items():
            names_list = self._parse_param_names(k)
            current_level = param_dict
            for name in names_list[:-1]:
                if name not in current_level:
                    current_level[name] = {}
                current_level = current_level[name]
            current_level[names_list[-1]] = v
        return param_dict

    @staticmethod
    def _parse_param_names(name):
        """
        Converts the param names in the dict passed to set_params (e.g. ``'A__b___adaba_do'``) to a list of the names of
        the referenced parameters (e.g. ``['A', 'b', 'adaba_do']``). Names are separated by double underscores.

        .. note:: Parameters cannot have underscore prefixes prefixes are not
        """
        return [k_rvs[::-1] for k_rvs in name[::-1].split('__')][::-1]


class Interpolate(Transform):
    """
    Interpolates values in numeric (dependent variable) columns of a :class:`~pandas.DataFrame` at new values of an
    independent-variable.

    Represents dependent variable columns of a :class:`~pandas.DataFrame` as a function of an independent-variable
    column, and uses it to find new values of the dependent variables. The DataFrame is passed to this class's
    :meth:`~opt2q.measurement.base.transform.Interpolate.transform` method. But, names of the independent and dependent
    variables, and the new values are supplied at instantiation.

    Parameter
    ---------
    independent_variable_name: str
        The column name of the independent variable.

    dependent_variable_name: str or list of strings
        The column name(s) of the dependent variable(s)

    new_values: vector-like or :class:`~pandas.DataFrame`
        The values of the independent variable in the interpolation. Additional columns annotate experimental
        conditions, or etc.

    groupby: str, or list optional
        The name of the column(s) by which to group the operation by. Each unique value in this column denotes a
        separate group. Defaults to None or to the unique experimental conditions rows in ``new_values``.

        Your group-by column(s) should identify unique rows of the experimental conditions. If multiple experimental
        conditions' rows appear in the same group, the interpolation is repeated for each unique row. If the rows
        differ between ``x`` (what is passed to the :meth:`~opt2q.measurement.base.transform.Interpolate.transform`
        method) and ``new_values``, the values in ``x`` are retained.

    options: dict
         Dictionary of keyword arguments:

        ``interpolation_method_name`` (str): interpolation method that gets passed to the `SciPy`
        :class:`~scipy.interpolate.interp1d` class. Defaults to 'cubic'.

    .. note::
        The columns mentioned in ``independent_variable_name``, ``dependent_variable_name`` and ``new_values`` and
        ``groupby``

    Attributes
    ----------
    set_params_fn: dict
        Dictionary of methods that handle updates of certain parameters. Only parameter mentioned here can be
        updated dynamically using the :meth:`~opt2q.measurement.base.transforms.Transform` method.
    """

    def __init__(self, independent_variable_name, dependent_variable_name, new_values, groupby=None, **options):
        super(Interpolate).__init__()

        self._independent_variable_name, \
            self._dependent_variable_name, \
            self._iv_and_dv_names_set = self._check_independent_and_dependent_variables(independent_variable_name,
                                                                                        dependent_variable_name)

        self._new_values, self._new_val_extra_cols = self._check_new_values(new_values, independent_variable_name)
        self._new_val_has_extra_cols = len(self._new_val_extra_cols) > 0

        # How to carryout the interpolation
        self._group_by = self._check_group_by(groupby, self._iv_and_dv_names_set)
        self._interpolate = [self._interpolation_in_groups,
                             self._interpolation_not_in_groups][self._group_by is None]
        self._get_new_values_per_group = [self._use_same_new_x_for_each_group,
                                          self._get_new_x_for_each_group][self._new_val_has_extra_cols]
        self._transform = [self._transform_new_values_simple,
                           self._transform_new_values_extra_cols][self._new_val_has_extra_cols]

        self._interpolation_method_name = options.get('interpolation_method_name', 'cubic')

        self.set_params_fn = {'new_values': self._set_new_values,
                              'interpolation_method_name': self._set_interpolation_method_name}

    # set up
    @staticmethod
    def _check_independent_and_dependent_variables(iv, dv):
        iv_set = {iv}
        if _is_vector_like(dv):
            dv_set = _convert_vector_like_to_set(dv)
        else:
            dv_set = {dv}
            dv = [dv]
        return iv, dv, iv_set | dv_set

    def _check_new_values(self, new_val, iv_col_name):
        """
        Checks that the new values are a pd.DataFrame or are vector-like.

        If ``new_val`` is a dataframe, it must have a a column of numeric values named after the independent variable
        """
        if isinstance(new_val, pd.DataFrame) and new_val.shape[1] > 1:
            if iv_col_name not in new_val._get_numeric_data().columns:
                raise ValueError("'new_values' must have an column named '{}'.".format(iv_col_name))
            else:
                return new_val, list(set(new_val.columns) - {iv_col_name})
        elif _is_vector_like(new_val):
            return pd.DataFrame(_convert_vector_like_to_list(new_val), columns=[self._independent_variable_name]), []
        else:
            raise ValueError("'new_values' must be vector-like list of numbers or a pd.DataFrame.")

    @staticmethod
    def _check_group_by(group_by, iv_dv_set):
        """Raise ValueError if group_by includes iv or dv columns"""
        if isinstance(group_by, str):
            group_by = [group_by]
        if _is_vector_like(group_by) and len(iv_dv_set.intersection(_convert_vector_like_to_set(group_by))) == 0:
            return _convert_vector_like_to_list(group_by)
        elif group_by is not None:
            raise ValueError("group_by must be a column name or list of column names and cannot have the "
                             "independent or dependent variable columns' names.")

    # properties
    @property
    def new_values(self):
        return self._new_values

    @new_values.setter
    def new_values(self, v):
        self._set_new_values(v)

    def _set_new_values(self, val):
        new_val, _extra_cols = self._check_new_values(val, self._independent_variable_name)
        has_extra_cols = len(_extra_cols) > 0
        self._transform = [self._transform_new_values_simple,
                           self._transform_new_values_extra_cols][has_extra_cols]
        self._get_new_values_per_group = [self._use_same_new_x_for_each_group,
                                          self._get_new_x_for_each_group][has_extra_cols]
        self._new_values = new_val
        self._new_val_extra_cols = _extra_cols
        self._new_val_has_extra_cols = has_extra_cols

    @property
    def interpolation_method_name(self):
        return self._interpolation_method_name

    @interpolation_method_name.setter
    def interpolation_method_name(self, v):
        self._set_interpolation_method_name(self, v)

    def _set_interpolation_method_name(self, v):
        self._interpolation_method_name = v

    @property
    def _signature_params(self):
        return (self._independent_variable_name,
                self._dependent_variable_name,
                'DataFrame(shape={})'.format(self.new_values.shape)), \
               {'interpolation_method_name': 'cubic'}

    @property
    def _get_params_dict(self):
        return {
            'new_values':  self.new_values,
            'interpolation_method_name': self.interpolation_method_name
        }

    # transform
    def transform(self, x):
        """
        Interpolates, in a DataFrame, x, the values in numeric column(s) relative to new values of an independent
        variable.

        Parameter
        ---------
        x: :class:`~pandas.DataFrame`
            The independent and dependent variables utilized to generate the interpolant.

            The independent and dependent variable column names (specified at instantiation of this class) must be in x.
            The independent variable cannot have repeat values within the same group.

        :class:`~opt2q.measurement.base.transforms.Interpolate` class.
        """
        return self._transform(x)

    def _transform_new_values_extra_cols(self, x):
        """
        Do interpolation when the new_values has additional columns annotating experimental conditions.
        """
        x_trimmed_rows = self._intersect_x_and_new_values_experimental_condition(x, self.new_values)
        x_extra_cols = list(set(x.columns) - self._iv_and_dv_names_set)
        return self._interpolate(x_trimmed_rows, x_extra_cols)

    def _transform_new_values_simple(self, x):
        # new_values is one group
        x_extra_cols = list(set(x.columns) - self._iv_and_dv_names_set)
        return self._interpolate(x, x_extra_cols)

    def _intersect_x_and_new_values_experimental_condition(self, x, new_val):
        """
        Get the rows of x that have experimental-conditions in common with those present in ``self.new_values``
        """
        try:
            return pd.merge(x, new_val, on=self._new_val_extra_cols, suffixes=('', '_y'))[x.columns].drop_duplicates().reset_index(drop=True)
        except KeyError as error:
            raise KeyError("'new_values' contains columns not present in x: " + _list_the_errors(error.args))

    def _interpolation_not_in_groups(self, x, x_extra_cols):
        """run the interpolation on the whole dataframe x"""
        if len(x_extra_cols) == 0:
            return self._interpolate_values_no_repeats(x, self.new_values)
        else:
            return self._interpolate_values_w_repeats(x, self.new_values, x_extra_cols)

    def _interpolation_in_groups(self, x, x_extra_cols):
        """group x by the group column(s) and for every group run the interpolation"""
        interpolation_result = pd.DataFrame()
        for name, group in x.groupby(self._group_by):
            # rows of new_x present in group's extra columns
            new_x_for_this_group = self._get_new_values_per_group(group, x_extra_cols)
            group_interpolation_result = self._interpolate_values_w_repeats(group, new_x_for_this_group, x_extra_cols)
            interpolation_result = pd.concat([interpolation_result, group_interpolation_result], ignore_index=True, sort=False)
        return interpolation_result

    def _get_new_x_for_each_group(self, group, x_extra_cols):
        # rows of new_x present in group
        return self._intersect_x_and_new_values_experimental_condition(self.new_values, group[x_extra_cols])

    def _use_same_new_x_for_each_group(self, *args):
        return self.new_values

    def _interpolate_values_w_repeats(self, x, new_x, x_extra_cols):
        """
        If x's additional columns have multiple unique rows, repeat the interpolation for each unique row.

        This prevents any loss of extraneous information. Do this for *all* DataFrames that have additional columns.
        """
        len_unique_ec_rows_x = x[x_extra_cols].drop_duplicates().shape[0]
        range_unique_ec_rows_x = range(len_unique_ec_rows_x)
        len_new_x = new_x.shape[0]

        new_x_for_interpolation = x[x_extra_cols].iloc[
            np.tile(range_unique_ec_rows_x, len_new_x)].reset_index(drop=True)

        new_x_for_interpolation[self._independent_variable_name] = np.repeat(
            new_x[self._independent_variable_name].values, len_unique_ec_rows_x)

        return self._interpolate_values_no_repeats(x, new_x_for_interpolation)

    def _interpolate_values_no_repeats(self, x, new_x):
        """If the only x only has 'iv' and 'dv', do not try to do repeats of the interpolation."""
        for dv in self._dependent_variable_name:
            try:
                cubic_spline_fn = interp1d(x[self._independent_variable_name], x[dv],
                                           kind=self._interpolation_method_name)

            except ValueError:  # If too few time-points in the simulation result, the model
                self._interpolation_method_name = 'linear'
                cubic_spline_fn = interp1d(x[self._independent_variable_name], x[dv],
                                           kind=self._interpolation_method_name)

            new_x[dv] = new_x[self._independent_variable_name].apply(cubic_spline_fn)
        return new_x


class Pipeline(object):
    pass