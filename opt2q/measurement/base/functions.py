# MW Irvin -- Lopez Lab -- 2018-09-07
"""
Suite of Functions used in Measurement Models
"""
import numpy as np
import pandas as pd
from numba import double, jit
import inspect

from sklearn.preprocessing import PolynomialFeatures


class TransformFunction(object):
    """
    Behaves like a function but has additional features that aid implementation in Opt2Q.

    This class take functions that can take (as its first argument) and return a :class:`~pandas.DataFrame`.

    ..note::
        If the function generates new columns, use the
        :meth:`~opt2q.measurement.base.functions.TransformFunction.replace_white_space_in_new_columns` method
        to replace the whitespace in column names with "$".

    Parameters
    ----------
    f: function or callable:
        Must accept for its first argument a :class:`~pandas.DataFrame`.
        If provided a :class:`~pandas.DataFrame` is must return a :class:`~pandas.DataFrame`.
    """
    def __init__(self, f):
        self._f = f
        self._sig = inspect.signature(self._f)
        self._sig_string = self._signature()

    # args and kwargs to be passed to the function can be presented in the __repr__
    def _signature(self, **kwargs) -> str:

        sig = self._sig
        ba = sig.bind_partial(**kwargs)
        ba.apply_defaults()
        sig_str = '('
        for name,  param in sig.parameters.items():
            if name in ba.arguments:
                sig_str += str(param.replace(default=ba.arguments[name]))
            else:
                sig_str += param.name
            sig_str += ', '
        sig_str = sig_str[:-2] + ')'
        return sig_str

    def signature(self, **kwargs):
        """
        Update the values of arguments in ``self._f`` signature (for the __repr__). Even

        This helps keep up with what the function is receiving from a particular class.

        Updates self._sig
        """
        self._sig_string = self._signature(**kwargs)

    @property
    def _sig_str(self):
        return getattr(self, '_sig_string', self._sig)

    # clearer __repr__
    def __repr__(self):
        sig = inspect.signature(self.__init__)
        if hasattr(self, '_signature_params'):
            sig_args, sig_kw = self._signature_params
            sig_str = sig.bind_partial(*sig_args, **sig_kw).__repr__().split('BoundArguments ')[1][:-1]
        else:
            sig_str = self._sig_str
        name_str = self._f.__name__
        return '{}{}'.format(name_str, sig_str)

    def __call__(self, x, *args, **kwargs):
        self._x = x
        return self._f(x, *args, **kwargs)

    # pre-processing methods
    @staticmethod
    def clip_zeros(x) -> pd.DataFrame:
        """
        clip zero values to 10% of the smallest number greater than zero in the dataframe.

        Parameters
        ----------
        x: :class:`pandas.DataFrame`
        """
        x = pd.DataFrame(x)
        x_min = x[x > 0].min()
        x = x.clip(0.1*x_min, axis=1)
        return x

    def replace_white_space_in_new_columns(self, new_x, new_char='$'):
        """
        Replaces whitespace with ``new_char`` in newly created columns of the dataframe passed to ``self._f``.

        Since the columns annotating experimental conditions can naturally contain whitespace, whitespace
        is not tracked by the :class:`~opt2q.measurement.base.transforms.Transform` classes. This means
        new columns can potentially get ignored in subsequent operations.
        """
        new_cols = set(new_x.columns) - set(self._x.columns)

        return new_x.rename(columns={s: s.replace(" ", new_char) for s in new_cols})


def transform_function(fn):
    """
    Decorator that endows a function with attributes of the
    :class:`~opt2q.measurement.base.functions.TransformFunction`

    Use this decorator on functions that can take (as its first argument) and return a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    fn: function or callable:
        Must accept for its first argument a :class:`~pandas.DataFrame`.
        If provided a :class:`~pandas.DataFrame` is must return a :class:`~pandas.DataFrame`.

    Returns
    -------
    :class:`~opt2q.measurement.base.functions.TransformFunction` instance

    """
    return TransformFunction(fn)


@transform_function
def log_scale(x, base=10, clip_zeros=True):
    """
    Log-Scales the values in an array

    Parameters
    ----------
    x: :class:`pandas.DataFrame`

    base: float, optional
        Log base. Defaults to base 10.
    clip_zeros: bool, optional
        If True, clip the values to 10% of the lowest value greater than 0. For example: [0, 1, 2] -> [0.1, 1, 2]
        clip_zeros returns a :class:`pandas.DataFrame`.

        .. note:: A column of all zeros is replaced with NaNs.
    """
    if clip_zeros:
        x = log_scale.clip_zeros(x)
    return np.log(x).divide(np.log(base))


@transform_function
def polynomial_features(x, degree=2, interaction_only=False, include_bias=False):
    """
    Polynomial expansion on the columns of ``x``. Uses Scikit-Learn's
    :class:`~sklearn.preprocessing.data.PolynomialFeatures` class, and takes the same args.

    Parameters
    ----------
    x: :class:`pandas.DataFrame`

    degree: int, optional
        Degree of the resulting polynomial. Defaults to 2

    interaction_only: bool, optional
        When True, it returns only the products of tow or more columns but not the squares etc of single columns.
        Defaults to False.

    include_bias: bool, optional
        When true, a column of ones is included in the result. Defaults to False.
    """

    p = PolynomialFeatures(degree, interaction_only=interaction_only, include_bias=include_bias)
    px = pd.DataFrame(p.fit_transform(x), columns=p.get_feature_names(x.columns))
    return polynomial_features.replace_white_space_in_new_columns(px, new_char='$')


@transform_function
def derivative(x):
    """
    Returns using second order accurate central differences in the interior points and second order forward
    and backward differences for the first and last points respectively.
    """
    return pd.DataFrame(np.gradient(x, axis=0), columns=x.columns)


@transform_function
def cummax(x):
    return x.cummax(axis=0)

@transform_function
def cummin(x):
    return x.cummin(axis=0)

@transform_function
def cumsum(x):
    return x.cumsum(axis=0)


@transform_function
def column_max(x):
    return pd.DataFrame(x.max()).T


@transform_function
def column_min(x):
    return pd.DataFrame(x.min()).T


@transform_function
def where_max(x, var=None, drop_var=False):
    """
    Return row of x where ``var`` is max.
    If ``drop_var`` is true, it drops the ``var`` column after the transformation.
    """
    idx_max = x[var].idxmax()
    res = pd.DataFrame([x.loc[idx_max].values], columns=x.columns)
    if drop_var:
        res = res.drop(columns=[var])
    return res


@transform_function
def where_min(x, var=None, drop_var=False):
    """
    Return row of x where ``var`` is min.
        If ``drop_var`` is true, it drops the ``var`` column after the transformation.

    """
    idx_min = x[var].idxmin()
    res = pd.DataFrame([x.loc[idx_min].values], columns=x.columns)
    if drop_var:
        res = res.drop(columns=[var])
    return res


def fast_linear_interpolate_fillna(values, indices):
    # result = np.zeros_like(values, dtype=np.float32)
    result = values

    for idx in range(indices.shape[0]):
        x = indices[idx, 0]
        y = indices[idx, 1]

        value = values[x, y]
        if x == 0:
            new_val = value
        elif x == len(values[:, 0]) - 1:
            new_val = value
        elif np.isnan(value):  # interpolate
            lid = 0
            while True:
                lid += 1
                left = values[x - lid, y]
                if not np.isnan(left):
                    break
            rid = 0
            while True:
                rid += 1
                right = values[x + rid, y]
                if not np.isnan(right):
                    break

            new_val = left + (values[x, 0] - values[x - lid, 0]) * (right - left) / (values[x + rid, 0] - values[x - lid, 0])

        else:
            new_val = value

        result[x, y] = new_val
    return result


fast_linear_interpolate_fillna = jit(double[:, :](double[:, :], double[:, :]))(fast_linear_interpolate_fillna)
