# MW Irvin -- Lopez Lab -- 2018-09-07
"""
Suite of Functions used in Measurement Models
"""
import numpy as np
import pandas as pd
import inspect


class TransformFunction(object):
    """
    Behaves like a function but has additional features that aid implementation in Opt2Q.

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
