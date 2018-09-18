# MW Irvin -- Lopez Lab -- 2018-08-24
import inspect
import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy import optimize
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mord import LogisticSE, obj_margin, grad_margin
from opt2q.utils import *
from opt2q.measurement.base.functions import TransformFunction, log_scale
from opt2q.data import DataSet


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
        if hasattr(self, '_signature_params'):
            sig_args, sig_kw = self._signature_params
            sig_str = sig.bind_partial(*sig_args, **sig_kw).__repr__().split('BoundArguments ')[1][:-1]
        else:
            sig_str = sig
        name_str = self.__class__.__name__
        return '{}{}'.format(name_str, sig_str)

    # get params
    def get_params(self, transform_name=None) -> dict:
        """
        Returns a dictionary of parameter names and values for the :class:`~opt2q.measurement.base.Transform`.

        Parameters
        ----------
        transform_name: str (optional)
            Name of the transform. This is prepended to the parameter-name with a double-underscore separating.
            E.g. 'transform__parameter'.
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

    def _build_params_dict(self, params):
        """
        Builds a nested dict of values for the parameters being changed. Names are presented in reverse order.

        Example
        -------

        """
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

    def transform(self, x, **kwargs):
        return x


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
        The name of the column(s) by which to group the operation. Each unique value in this column denotes a
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
        self._set_interpolation_method_name(v)

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
    def transform(self, x, **kwargs):
        """
        Interpolates, in a DataFrame, x, the values in numeric column(s) relative to new values of an independent
        variable.

        Parameter
        ---------
        x: :class:`~pandas.DataFrame`
            The independent and dependent variables utilized to generate the interpolant.

            The independent and dependent variable column names (specified at instantiation of this class) must be in x.
            The independent variable cannot have repeat values within the same group.
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
            return pd.merge(x, new_val, on=self._new_val_extra_cols, suffixes=('', '_y'))[x.columns]\
                .drop_duplicates().reset_index(drop=True)
        except KeyError as error:
            raise KeyError("'new_values' contains columns not present in x: " + _list_the_errors(error.args))

    def _interpolation_not_in_groups(self, x, x_extra_cols):
        """run the interpolation on the whole dataframe x"""
        if len(x_extra_cols) == 0:
            return self._interpolate_values_no_repeats(x, self.new_values)
        else:
            return self._interpolate_values_w_repeats(x, self.new_values, x_extra_cols)

    def _interpolation_in_groups(self, x, x_extra_cols):
        """group x by the group column(s) and run the interpolation for every group"""
        interpolation_result = pd.DataFrame()
        for name, group in x.groupby(self._group_by):
            # rows of new_x present in group's extra columns
            new_x_for_this_group = self._get_new_values_per_group(group, x_extra_cols)
            group_interpolation_result = self._interpolate_values_w_repeats(group, new_x_for_this_group, x_extra_cols)
            interpolation_result = pd.concat([interpolation_result, group_interpolation_result],
                                             ignore_index=True, sort=False)
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


def _threshold_fit(X, y, alpha, n_class, mode='SE',
                   max_iter=1000, verbose=False, tol=1e-12,
                   sample_weight=None):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss, and
    and forces empirical order constraints.

    This is a modification of Fabian Pedregosa-Izquierdo's `code`_.

    .. _code: https://github.com/fabianp/mord

    Parameters
    ----------
    mode : string (optional)
        one of {'AE', '0-1', 'SE'}. Defaults to 'SE' (Squared error)

    max_iter: int (optional)
        Maximum number of iterations to perform. Defaults to 1000

    tol: float (optional)
        Acceptance tolerance. Defaults to 1e-12

    verbose: bool
        Dictates whether to display output

    sample_weight: vector-like (optional)
        Sample weights
    """

    X, y = check_X_y(X, y, accept_sparse='csr')
    unique_y = np.sort(np.unique(y))
    if not np.all(unique_y == np.arange(unique_y.size)):
        raise ValueError(
            'Values in y must be %s, instead got %s'
            % (np.arange(unique_y.size), unique_y))

    n_samples, n_features = X.shape

    # convert from c to theta
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class-1)] = 1.

    if mode == 'AE':
        # loss forward difference
        loss_fd = np.ones((n_class, n_class - 1))
    elif mode == '0-1':
        loss_fd = np.diag(np.ones(n_class - 1)) + \
            np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
        loss_fd[-1, -1] = 1  # border case
    elif mode == 'SE':
        a = np.arange(n_class-1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None])**2 - (a - b[:, None]+1)**2)
    else:
        raise NotImplementedError

    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1]:] = np.arange(n_class - 1)
    options = {'maxiter': max_iter, 'disp': verbose}
    bounds = [(0, None)] * n_features + \
             [(None, None)] * (n_class - 1)

    sol = optimize.minimize(obj_margin, x0, method='L-BFGS-B',
                            jac=grad_margin, bounds=bounds, options=options,
                            args=(X, y, alpha, n_class, loss_fd, L, sample_weight),
                            tol=tol)
    if verbose and not sol.success:
        print(sol.message)

    w, c = sol.x[:X.shape[1]], sol.x[X.shape[1]:]
    theta = L.dot(c)
    return w, theta


class LogisticEOC(LogisticSE):
    """
    Ordinal logistic regression that constrains the order of the features to match that of the empirical measurement,
    i.e. higher ordinal classes correspond to higher values in the feature-space.  This constraint is described by W.F.
    Young Young, Forrest W [1].

    This class modifies the ordinal logistic regression modeled by Fabian Pedregosa-Izquierdo's (explained `here`_). He
    defines n-class - 1 *parallel* hyperplanes that serve as thresholds between adjacent classes. The fact that they are
    parallel simplifies the model and decreases the number of required free-parameters.

    .. _here: https://pythonhosted.org/mord/

    .. [1] "Quantitative analysis of qualitative data." Psychometrika 46, no. 4 (1981): 357-388.
    """
    def fit(self, X, y, sample_weight=None):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 1e-3:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = _threshold_fit(
            X, y_tmp, self.alpha, self.n_class_,
            mode='SE', verbose=self.verbose, max_iter=self.max_iter,
            sample_weight=sample_weight)
        return self


class LogisticClassifier(Transform):
    """
    Uses a Logistic model to classify values in a DataFrame into ordinal or nominal categories.

    Logistic regression is a supervised classification model, and requires a dataset with class labels for each
    observable (or group of observables).  Todo: What will that look like for Dead vs. Surviving -> Apoptosis vs. Necr.?

    Parameters
    ----------
    dataset: :class:`~opt2q.data.DataSet`, or :class:`~pandas.DataFrame`
        Values of the measured variables and additional columns annotating experimental conditions. The measured
        variables are name the same as those mentioned in columns or column_groups.

        The column or column-group names must appear in the dataset.

    columns: str or list of strings, optional
        The name(s) of the column(s) of the :class:`~pandas.DataFrame`, ``x``, (passed to the
        :meth:`~opt2q.measurement.base.transforms.LogisticClassifier.transform` method) that will be classified into
        categories.

        All column names must also appear in the ``dataset``.

        All non-numeric columns in ``x`` are ignored (even if they appear in this argument).

        Defaults to the numeric columns present in ``x`` that also exist in the ``dataset``.

    column_groups: dict, optional
        Groups of columns of the :class:`~pandas.DataFrame`, ``x``, (passed to the
        :meth:`~opt2q.measurement.base.transforms.LogisticClassifier.transform` method) that will be classified into
        categories. Each group is constitutes the features that the classification is based on. Each group name (str)
        (or dict key). This name is a column name in the ``dataset``. Defaults to ``columns`` argument.

        All non-numeric columns are ignored (even if they appear in this argument).

        All column-group names must also appear in the ``dataset``.

    group_features: bool, optional
        If True, all columns serve as a single feature space for by the classification is depends. If False, each column
        serves as the feature for a separate classification.  This parameter is irrelevant when ``column_groups`` is
        used. If ``group_features`` is True, you must specify a name for the group via ``group_name``. Defaults to
        False.

    group_name: str, optional
        If ``group_features`` is True, you must specify a name for the group. This group must appear in the ``dataset``.

    do_fit_transform: bool, optional
        When True, Simply fit the classifier to values of ``x`` and data ``y``. When False, use a previous fit to
        simulate the classification; useful for cross-validating using a test set, or manually adjusting the classifier
        coefficients. Defaults to True.

    classifier_type: str, optional
        One of the supported classifiers: 'nominal' (logistic regression), 'ordinal' (ordinal logistic regression),
        'ordinal_eoc' (ordinal logistic regression with empirical ordering constraint).

    """

    classifiers = {'ordinal': LogisticSE,  # Ordinal Logistic Regression
                   'ordinal_eoc': LogisticEOC,  # Ordinal Logistic Regression w/ Empirical Order Constraint
                   'nominal': LogisticRegression}

    _classifier_attribute = {'ordinal': ['coef_', 'theta_'],
                             'ordinal_eoc': ['coef_', 'theta_'],
                             'nominal': ['coef_', 'intercept_']}

    def __init__(self, dataset, columns=None, column_groups=None, group_features=False,
                 group_name=None, do_fit_transform=True, classifier_type='nominal'):
        super(LogisticClassifier).__init__()

        # set columns
        columns_set, columns_dict = self._check_columns(columns, column_groups)
        columns_dict = self._put_columns_in_single_group(columns_set, group_name) \
            if group_features and columns is not None \
            else columns_dict
        self._columns_set = columns_set
        self._columns_dict = columns_dict

        # set data df
        data_df = self._check_dataset(dataset, columns_dict)
        self._data_df = data_df

        # set params dictating transform steps
        self._transform_get_columns = \
            self._transform_get_columns_from_x if columns_set == set() else self._transform_get_columns_from_column_dict

        self._do_fit_transform = do_fit_transform is True or not isinstance(do_fit_transform, bool)  # non-bool -> True
        self._get_transform = self._get_transform_w_fit if self.do_fit_transform else self._get_transform_wo_fit
        self._logistic_models_dict = dict()  # this will be updated by self._get_transform_fn()
        self._classifier, self._classifier_params = self._check_classifier_type(classifier_type)

        # set_params method params
        self.set_params_fn = {'do_fit_transform': self._set_do_fit_transform}

    def _check_columns(self, cols, col_groups):
        """
        Check attributes of the columns.

        cols, col_groups have to *actually* be str
        because self.transform() will modify them with a suffix ('__n')
        """
        if cols is not None and col_groups is not None:
            raise ValueError("Use only 'columns' or 'column_groups'. Not both.")

        if cols is not None:
            column_set, column_dict = self._convert_columns_to_dict(cols)
        elif col_groups is not None:
            column_set, column_dict = self._get_columns_from_column_dict(col_groups)
        else:
            return set(), dict()
        return column_set, column_dict

    @staticmethod
    def _convert_columns_to_dict(cols):
        """Return a dict of names (str or int): [name] for the column names in ``cols`` (as consistent with column_groups).

        Call this method only when cols is not None."""
        if _is_vector_like(cols):
            column_list = _convert_vector_like_to_list(cols)
            return column_list, {i: [i] for i in column_list}
        elif isinstance(cols, str) or isinstance(cols, int):  # pandas defaults to int columns.
            return [cols], {cols: [cols]}
        else:
            raise ValueError("columns can only a str or list of str.")  # call this method only when cols is not None.

    @staticmethod
    def _get_columns_from_column_dict(column_dict):
        """
        Make a list of the columns involved in column_dict.

        This will be used to check if these columns are in ``x`` and not in ``groupby``.
        """
        col_set = set([])
        for k, v in column_dict.items():
            if _is_vector_like(v):
                column_dict.update({k: _convert_vector_like_to_list(v)})
                col_set |= _convert_vector_like_to_set(v)
            else:
                column_dict.update({k: [v]})
                col_set |= {v}
        return col_set, column_dict

    @staticmethod
    def _put_columns_in_single_group(columns_set, group_name):
        """
        Call this only if ``group_features`` is True.

        Returns column_dict {'group_name': [columns in column_set]}
        """
        if not isinstance(group_name, str) and not isinstance(group_name, int):
            raise ValueError("'group_name' must be a string.")
        return {group_name: list(columns_set)}

    def _check_dataset(self, dataset, columns_dict):
        """
        Check dataset for required attributes. Raise Error if not DataSet or pd.DataFrame
        """
        if isinstance(dataset, pd.DataFrame):
            # any mentioned columns can be measured variables.
            self._check_that_dataset_has_required_columns(dataset.columns, columns_dict)
            return dataset

        if isinstance(dataset, DataSet):
            # only consider categorical data
            dataset_cols = [k for k, v in dataset.measured_variables.items() if v in ('default', 'nominal', 'ordinal')]
            self._check_that_dataset_has_required_columns(dataset_cols, columns_dict)
            return dataset.data  # use the intersection of columns in dataset and x

    @staticmethod
    def _check_that_dataset_has_required_columns(dataset_columns, columns_dict):
        """
        The Dataset.measured_variables_columns must have the column (or group) names mentioned in ``columns_dict``
        (i.e. the columns_dict.keys())
        """
        if set(columns_dict.keys()) - set(dataset_columns) != set():
            raise ValueError("The 'dataset' must have the following nominal or ordinal measured-variables columns: " +
                             _list_the_errors(list(set(columns_dict.keys()) - set(dataset_columns))))

    def _check_classifier_type(self, classifier_type):
        if classifier_type not in self.classifiers.keys():
            raise ValueError("'{}' is not a supported classifier type.".format(classifier_type))
        else:
            return self.classifiers[classifier_type], self._classifier_attribute[classifier_type]

    # params dict
    @property
    def _get_params_dict(self):
        return {'do_fit_transform': self.do_fit_transform}

    # fit_transform settings
    @property
    def do_fit_transform(self):
        return self._do_fit_transform

    @do_fit_transform.setter
    def do_fit_transform(self, v):
        self._set_do_fit_transform(v)

    def _set_do_fit_transform(self, v):
        self._do_fit_transform = v is True or not isinstance(v, bool)  # non-bool -> True
        self._get_transform = self._get_transform_w_fit if self._do_fit_transform else self._get_transform_wo_fit

    # coefficients
    @property
    def coefficients(self):
        """Only retrieve coefficients from the models when asked for them, via this property."""
        logistic_models_dict = dict()  # dictionary of models indexed by column or column_group or (as given by key).
        coefficients = dict()
        for k, v in logistic_models_dict:
            coefficients.update({k, logistic_models_dict[k].coef_})  # {k: {'coef_': np.array, 'intercept_': np.array}}
        return coefficients

    @coefficients.setter
    def coefficients(self, v):
        """Coefficients must be updated manually but must be first establish via """
        self._set_coefficients(v)

    def _set_coefficients(self, v):
        self._check_that_coefficients_are_the_correct_shape()

    def _check_for_coefficients(self):
        """
        Raises AttributeError if this class does not have coefficients. (Coefficients are produced by the
        logistic regression fit method.

        If ``do_fit_transform`` is False, this function is called.
        """
        pass

    def _check_that_coefficients_are_the_correct_shape(self):
        """
        This class requires that the transform be previously fit to data. This will establish the
        coefficients' shape.

        For every row in the new coefficients' dataframe. check that the shape is same as before.
        """
        pass

    def _get_coefficients_from_model(self):
        """Gets coefficients a fit logistic regression model and assembles them into a DataFrame with indexing
        columns equal to the groupby columns + the column group names. The remaining columns are integers counting
        up from zero (these correspond to the indices of the flattened array that is passed to the logistic
        regression model's ``coef_`` attribute."""
        pass

    # transform
    def transform(self, x, **kwargs):
        """
        Return a :class:`~pandas.DataFrame` with describing the probability of membership in ordinal categories
        described for given observables.
        """

        # we need a function that either gets the intersection of x and self._columns_set or returns the non-numeric
        # columns of x.
        return self._transform(x, self._data_df)

    def _transform(self, x, y):
        """
        Runs the logistic regression
        """

        columns_set, columns_dict = self._transform_get_columns(x)
        x_extra_columns = set(x.columns) - columns_set
        y_extra_columns = set(y.columns) - set(self._columns_dict.keys())
        combined_x_y = self._prep_data(x, y, list(self._columns_dict.keys()), x_extra_columns, y_extra_columns)

        result_df = pd.DataFrame()
        for y_col, x_col in columns_dict.items():
            # prep y
            y_ = combined_x_y[y_col].astype(int) if combined_x_y.dtypes[y_col] == float else combined_x_y[y_col]
            # get model
            model = self._transform_get_logistic_model(combined_x_y[x_col], y_, y_col)
            # predict results
            results = model.predict_proba(combined_x_y[x_col])
            results_columns = ['{}__{}'.format(str(y_col), cat) for cat in np.unique(y_)]
            result_df[results_columns] = pd.DataFrame(results, columns=results_columns)

        # add exp conditions
        result_df[list(x_extra_columns)] = combined_x_y[list(x_extra_columns)]
        return result_df

    @staticmethod
    def _prep_data(x, y, y_cols, x_extra_cols, y_extra_cols):
        shared_columns = list(x_extra_cols.intersection(y_extra_cols))
        data_blocks = pd.merge(x[shared_columns], y[shared_columns]).drop_duplicates().reset_index(drop=True)

        # repeat the blocks so that they are consistent
        x_blocks = x.groupby(shared_columns)
        y_blocks = y.groupby(shared_columns)

        prepped_data = pd.DataFrame()
        for name, block in data_blocks.groupby(shared_columns):
            x_idx = x_blocks.groups[name]
            y_idx = y_blocks.groups[name]

            # repeat y for every unique row in x.
            x_idx_repeats = np.repeat(x_idx, len(y_idx))
            y_idx_repeats = np.tile(y_idx, len(x_idx))

            prepped_block = x.iloc[x_idx_repeats].reset_index(drop=True)
            prepped_block[y_cols] = y.iloc[y_idx_repeats][y_cols].reset_index(drop=True)

            prepped_data = pd.concat([prepped_data, prepped_block], ignore_index=True, sort=False)

        return prepped_data

    @staticmethod
    def _get_scalable_columns(x) -> set:
        try:
            scalable_cols = set(x._get_numeric_data().columns)
        except AttributeError:
            raise TypeError("x must be a pandas.DataFrame")
        return scalable_cols

    def _transform_get_columns_from_x(self, x) -> (set, dict):
        """
        When columns and column-groups are unset by user. Get the numeric columns of x that are also in the data.
        Return columns-dict.
        """
        scalable_cols = self._get_scalable_columns(x)
        columns_in_x_and_in_dataset = scalable_cols.intersection(set(self._data_df.columns))
        return self._convert_columns_to_dict(columns_in_x_and_in_dataset)

    def _transform_get_columns_from_column_dict(self, x) -> (set, dict):
        scalable_cols = self._get_scalable_columns(x)
        if self._columns_set - scalable_cols != set():
            missing_cols = self._columns_set - scalable_cols
            raise ValueError("'x' is missing the following numeric columns: " + _list_the_errors(missing_cols))
        return self._columns_set, self._columns_dict

    def _transform_get_logistic_model(self, x, y, col):
        """
        Return the logistic model.
        """
        try:
            logistic_model = self._logistic_models_dict[col]
            return self._get_transform(logistic_model, x=x, y=y)

        except KeyError:  # will catch on first iteration.
            logistic_model = self._get_transform_w_fit(self._classifier(), x=x, y=y)
            self._logistic_models_dict.update({col: logistic_model})
            return logistic_model

    @staticmethod
    def _get_transform_wo_fit(logistic_model, **kwargs):
        """
        Return model without fitting it
        """
        return logistic_model

    @staticmethod
    def _get_transform_w_fit(logistic_model, **kwargs):
        """
        Return model after running model.fit(x, y)
        """
        x = kwargs.get('x')
        y = kwargs.get('y')

        logistic_model.fit(x, y)  # make sure the logistic_model.fit() method update updates the DataFrame
        return logistic_model


class Scale(Transform):
    """
    Conducts *simple* scaling of values in a DataFrame

    By simple, I mean the scaling depends *only* on the values, (no references or other function(s) is needed).

    Parameters
    ----------
    columns: str or list of strings, optional
        The column name(s) of the variable(s) being scaled. Defaults to all numeric columns of the
        :class:`~pandas.DataFrame`, ``x``, that is passed to the
        :meth:`~opt2q.measurement.base.transforms.Scale.transform` method.

        All non-numeric columns are ignored (even if they where named in this argument).

    scale_fn: str or func, optional
        As a string is must name one of the functions in
        :attr:`~opt2q.measurement.base.transforms.Scale.scale_functions`
        Defaults to 'log2'

    Attributes
    ----------
    scale_functions: dict
        Dictionary of scaling functions and their names.
    """
    scale_functions = {'log2': (log_scale, {'base': 2, 'clip_zeros': True}),
                       'log10': (log_scale, {'base': 10, 'clip_zeros': True}),
                       'loge': (log_scale, {'base': np.e, 'clip_zeros': True})}

    def __init__(self, columns=None, scale_fn='log2', **scale_fn_kwargs):
        super(Scale).__init__()
        self._columns, self._columns_set = self._check_columns(columns)
        self._scale_fn, self._scale_fn_kwargs = self._check_scale_fn(scale_fn)
        self._scale_fn_kwargs.update(scale_fn_kwargs)

        # update scale_fn repr to reflect user defined kwargs
        self._scale_fn.signature(**self.scale_fn_kwargs)

        self.set_params_fn = {'scale_fn': self._set_scale_fn,
                              'scale_fn_kwargs': self._set_scale_fn_kwargs}

    @staticmethod
    def _check_columns(cols):
        if cols is None:
            return None, None
        if _is_vector_like(cols):
            cols_set = _convert_vector_like_to_set(cols)
            cols = _convert_vector_like_to_list(cols)
        else:
            cols_set = {cols}
            cols = [cols]
        return cols, cols_set

    def _check_scale_fn(self, scale_fn) -> tuple:
        if isinstance(scale_fn, str) and scale_fn in self.scale_functions.keys():
            return self.scale_functions[scale_fn]
        elif isinstance(scale_fn, str):
            raise ValueError("'scale_fn' must be in 'scale_functions'. '{}' is not.".format(scale_fn))
        elif isinstance(scale_fn, TransformFunction):
            return scale_fn, dict()
        elif callable(scale_fn):
            return TransformFunction(scale_fn), dict()
        else:
            raise ValueError(
                "'scale_fn' must be callable or a str in 'scale_functions'. '{}' is neither.".format(scale_fn)
            )

    @property
    def _get_params_dict(self):
        return {'scale_fn': self.scale_fn, 'scale_fn_kwargs': self.scale_fn_kwargs}

    @property
    def scale_fn(self):
        return self._scale_fn

    @scale_fn.setter
    def scale_fn(self, val):
        self._set_scale_fn(val)

    def _set_scale_fn(self, v):
        self._scale_fn, self._scale_fn_kwargs = self._check_scale_fn(v)  # new scale_fn, new kwargs

    @property
    def scale_fn_kwargs(self):
        return self._scale_fn_kwargs

    @scale_fn_kwargs.setter
    def scale_fn_kwargs(self, v):
        self._set_scale_fn_kwargs(**v)

    def _set_scale_fn_kwargs(self, **kw):
        self._scale_fn_kwargs.update(kw)
        self._scale_fn.signature(**self.scale_fn_kwargs)

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
        if 'scale_fn' in params:
            self._set_params(scale_fn=params.pop('scale_fn'))  # update scale_fn before updating scale_fn_kwargs

        params_dict = self._build_params_dict(params)
        self._set_params(**params_dict)

    # transform
    def transform(self, x, **kwargs):
        """
        Scale values in a :class:`~pandas.DataFrame`, x.

        Parameter
        ---------
        x: :class:`~pandas.DataFrame`
            The values to be scaled.
        """
        try:
            scalable_cols = set(x._get_numeric_data().columns)
        except AttributeError:
            raise TypeError("x must be a DataFrame")
        if self._columns is not None:
            cols_to_scale = scalable_cols.intersection(self._columns_set)
        else:
            cols_to_scale = scalable_cols

        scaled_df = self.scale_fn(x[list(cols_to_scale)], **self.scale_fn_kwargs)
        remaining_cols = list(set(x.columns)-cols_to_scale)

        scaled_df[remaining_cols] = x[remaining_cols]
        return scaled_df


class Standardize(Transform):
    """
    Standardizes (i.e. scales values to have zero mean and unit variance) of values in a DataFrame

    Parameters
    ----------
    columns: str or list of strings, optional
        The column name(s) of the variable(s) being scaled. Defaults to all numeric columns of the
        :class:`~pandas.DataFrame`, ``x``, that is passed to the
        :meth:`~opt2q.measurement.base.transforms.Scale.transform` method.

        All non-numeric columns are ignored (even if they where named in this argument).

    groupby: str, or list, optional
        The name of the column(s) by which to group the operation. Each unique value in the column(s) denotes a
        separate group. Defaults to None, and conducts a single operation along the length on the whole DataFrame.

    do_fit_transform: bool, optional
        When True, Simply scale the values of ``x`` to a (column-wise) mean of zero and unit variance.
        When False, use scaling parameters from most the previous scaling to transform  ``x``.
    """
    def __init__(self, columns=None, groupby=None, do_fit_transform=True):
        super(Standardize).__init__()
        self._columns, self._columns_set = self._check_columns(columns)
        self._group_by = self._check_group_by(groupby, self._columns_set)  # returns list or None

        # non-bool defaults to True
        self._do_fit_transform = do_fit_transform is True or not isinstance(do_fit_transform, bool)

        # transform related methods
        self._transform_scale_fn_dict = dict()
        self._transform_get_scale_fn = [self._scale_fn_wo_fit, self._scale_fn_w_fit][self.do_fit_transform]
        self._transform = [self._transform_not_in_groups, self._transform_in_groups][self._group_by is not None]

        # set params
        self.set_params_fn = {'do_fit_transform': self._set_do_fit_transform}

    @staticmethod
    def _check_columns(cols):
        if cols is None:
            return None, set([])
        if _is_vector_like(cols):
            cols_set = _convert_vector_like_to_set(cols)
            cols = _convert_vector_like_to_list(cols)
        else:
            cols_set = {cols}
            cols = [cols]
        return cols, cols_set

    @staticmethod
    def _check_group_by(group_by, columns):
        if group_by is None:
            return group_by
        if _is_vector_like(group_by):
            groups = _convert_vector_like_to_set(group_by)
        elif isinstance(group_by, str) or isinstance(group_by, int):
            groups = {group_by}
        else:
            raise ValueError("groupby must be a string or list of strings")

        # cannot group-by columns that are getting scaled.
        if len(groups.intersection(columns)) > 0:
            raise ValueError("columns and groupby cannot be have any of the same column names.")
        else:
            return list(groups)

    @property
    def _get_params_dict(self):
        return {'do_fit_transform': self.do_fit_transform}

    @property
    def do_fit_transform(self):
        return self._do_fit_transform

    @do_fit_transform.setter
    def do_fit_transform(self, v):
        self._set_do_fit_transform(v)

    def _set_do_fit_transform(self, v):
        # non-bool defaults to True
        self._do_fit_transform = v is True or not isinstance(v, bool)
        self._transform_get_scale_fn = [self._scale_fn_wo_fit, self._scale_fn_w_fit][v]

    def _scale_fn_wo_fit(self, x_scale_cols_only, name='__default'):
        if name in self._transform_scale_fn_dict:
            return self._transform_scale_fn_dict[name]
        return self._scale_fn_w_fit(x_scale_cols_only, name)

    def _scale_fn_w_fit(self, x_scale_cols_only, name='__default'):
        self._transform_scale_fn_dict[name] = StandardScaler().fit(x_scale_cols_only)
        return self._transform_scale_fn_dict[name]

    def transform(self, x, **kwargs):
        """
        Standardize values in ``x``.
        """
        try:
            scalable_cols = set(x._get_numeric_data().columns)
        except AttributeError:
            raise TypeError("x must be a DataFrame")

        if self._columns is not None:
            cols_to_scale = scalable_cols.intersection(self._columns_set)
        else:
            cols_to_scale = scalable_cols

        return self._transform(x, cols_to_scale)

    def _transform_in_groups(self, x, _scale_these_cols):
        _scale_these_cols -= set(self._group_by)
        cols = list(_scale_these_cols)
        remaining_cols = list(set(x.columns)-_scale_these_cols)
        scaled_df = pd.DataFrame()
        for name, group in x.groupby(self._group_by):
            scaled_ = self._transform_get_scale_fn(group[cols], name=name).transform(group[cols])
            scaled_group = pd.DataFrame(scaled_, columns=cols)
            scaled_group[remaining_cols] = group[remaining_cols].reset_index(drop=True)
            scaled_df = pd.concat([scaled_df, scaled_group], ignore_index=True, sort=False)
        return scaled_df

    def _transform_not_in_groups(self, x, _scale_these_cols):
        cols = list(_scale_these_cols)
        remaining_cols = list(set(x.columns) - _scale_these_cols)
        scaled_ = self._transform_get_scale_fn(x[cols], name='__default').transform(x[cols])
        scaled_df = pd.DataFrame(scaled_, columns=cols)
        scaled_df[remaining_cols] = x[remaining_cols]
        return scaled_df


class Pipeline(Transform):
    """
    Runs a series of transformation steps on an Opt2Q :class:`~pysb.simulator.SimulationResult`'s ``opt2q_dataframe``.

    The transformation steps are :class:`~opt2q.measurement.base.transforms.Transform` class instances.

    This class serves as the process attribute of various measurement models.

    Parameters
    ----------
    steps: list (optional)
        Lists the Pipeline's steps. It is a list of tuples: ('name', 'transform'); where 'name' is the name (str) of the
        step, and 'transform' is a :class:`~opt2q.measurement.base.transforms.Transform`.

    .. note::
        Do not name your steps (int)! The :meth:`~opt2q.measurement.base.transforms.Pipeline.remove_step` method will
        treat ints as indices and not names of the steps.

    """
    def __init__(self, steps=None):
        super(Pipeline).__init__()

        self.steps = self._check_steps(steps)

    # Todo: Add a rename step.

    @staticmethod
    def _check_steps(_steps):
        if _steps is None:
            return []
        names = [x[0] for x in _steps]
        if len(names) != len(set(names)):
            raise ValueError('Each steps must have a unique name. Duplicate steps are not allowed.')

        for name, transformation in _steps:
            if isinstance(transformation, Transform):
                pass
            else:
                raise ValueError('Each step must be a Transform class instance. {} is not.'.format(transformation))
        return _steps

    # for repr and get params
    @property
    def _signature_params(self):
        return (), {'steps': self.steps}

    @property
    def _get_params_dict(self):
        return dict()

    # manage steps
    def remove_step(self, index):
        """
        Removes a step from the pipeline of process steps

        Parameters
        ----------
        index: int or str
            As an int, it is the index of the step being removed. As str, it names the removed step.

        Example
        -------
        >>> from opt2q.measurement.base import Pipeline, Interpolate,
        >>> process = Pipeline(steps=[('interpolate', Interpolate('iv', 'dv', np.array([1, 2, 3])))])
        >>> process.remove_step('interpolate')
        >>> print(process.steps)
        []
        """
        if isinstance(index, int):
            del self.steps[index]
        if isinstance(index, str):
            index = [x for x, y in enumerate(self.steps) if y[0] == index][0]
            del self.steps[index]

    def add_step(self, step, index=None):
        """
        Adds a step to the process :class:`~opt2q.measurement.base.transforms.Pipeline`.

        Parameters
        ----------
        step: tuple
            The name (str) and :class:`~opt2q.measurement.base.transforms.Transform` class being added.

        index: int or str (optional)
            As an int, it is the index of the new step. As str, it names the the step the new step will be inserted
            *after*. Defaults to the last index in the list of process steps.

        Example
        -------
        >>> from opt2q.measurement.base import Pipeline, Interpolate
        >>> process = Pipeline()
        >>> process.add_step(('interpolate', Interpolate('iv', 'dv', np.array([0.0]))))
        >>> print(process.steps)
        [('interpolate', Interpolate(independent_variable_name='iv', dependent_variable_name=['dv'], new_values='DataFrame(shape=(1, 1))', options={'interpolation_method_name': 'cubic'}))]
        """
        # if a step with that name is already in there replace it.
        names = [x[0] for x in self.steps]
        if step[0] in names:
            self.remove_step(step[0])

        step_ = self._check_steps([step])[0]
        if index is None:
            index = len(self.steps)
        if isinstance(index, int):
            self.steps.insert(index, step_)
            return
        if isinstance(index, str):
            # insert after named process step
            idx = [x for x, y in enumerate(self.steps) if y[0] == index][0]+1
            self.steps.insert(idx, step_)
            return

    def get_params(self, transform_name=None):
        """
        Returns a dictionary of parameter names and values for the :class:`~opt2q.measurement.base.Pipeline`.

        The name of each step is prepended to the parameter name with a double-underscore separating.
        E.g 'step_name__parameter

        Parameters
        ----------
        transform_name: str (optional)
            Name of the transform. This is prepended to the parameter-name with a double-underscore separating.
            E.g. 'transform__parameter', or 'transform__step_name__parameter'

        Examples
        --------
        >>> #example pending

        """
        if transform_name is not None:
            return self._get_pipeline_params_transform_name(transform_name)

        params = dict()
        for name, transformation in self.steps:
            params.update(transformation.get_params(name))
        return params

    def _get_pipeline_params_transform_name(self, transform_name):
        params = dict()
        for name, transformation in self.steps:
            params.update(transformation.get_params('{}__{}'.format(transform_name, name)))
        return params

    def set_params(self, **params):
        """
        Set the params of the :class:`~opt2q.measurement.base.Process`.

        Parameters
        ----------
        params: dict
            Parameter names and values. The parameter names are formatted with the name of the step, the parameter name
            separated by double underscores. If the parameter is a dictionary (e.g. 'scale_fn_kwargs' parameter) keys in
            the dictionary are represented by appending a double underscore and the key.

            For example: A process step, 'scale', has a dict of kwargs, 'scale_fn_kwargs'. The kwarg, 'base' should
            appear in ``params`` as 'scale__scale_fn_kwargs__base'.


        .. note::
            Parameter names cannot have underscore prefixes.

        """
        steps = OrderedDict(self.steps)
        params_dict = self._build_params_dict(params)  # we have to set all the parameters for a Process at once.
        for k, v in params_dict.items():
            try:
                steps[k].set_params(**v)
            except KeyError as error:
                warnings.warn("The process does not have the following step(s): " + _list_the_errors(error.args))
                continue

    def transform(self, x, **kwargs):
        """
        Runs a series of transformations on the ``opt2q_dataframe`` or ``dataframe`` found in the
        :class:`~pysb.simulator.SimulationResult` passed to the :class:`~opt2q.measurement.base.MeasurementModel` model.

        .. note:: Each transformation must have a 'transform' method.

        Parameters
        ----------
        x: :class:`~pandas.DataFrame`
            Ideally an ``opt2q_dataframe`` from a :class:`~pysb.simulator.SimulationResult` , but any DataFrame will '
            suffice. The specific requirements of ``x`` depend on the individual
            :class:`~opt2q.measurement.base.transforms.Pipeline` steps.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Results of the series of transformations in the pipeline.
        """
        xt = x
        for name, transformation in self.steps:
            xt = transformation.transform(xt)
            # Note: All transformations must have a run method. If they are sub-class of process, they will.
        return xt


class RenameColumns(Transform):
    """Renames columns of the DataFrame."""
    pass
