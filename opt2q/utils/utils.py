"""
Background Stuff
"""
import warnings
import numpy as np
import pandas as pd
import time



# MW Irvin -- Lopez Lab -- 2018-08-08
def _list_the_errors(iterable_arg):
    """
    For error messages that list a bunch of things that have gone wrong.

    e.g. raise ValueError(_list_the_errors([a, b, c]) + " went wrong") would print "a, b, and, c went wrong".
    """
    iterable_arg = list(iterable_arg)
    if len(iterable_arg) > 1:
        note = " ".join("'{}',".format(k) for k in iterable_arg[:-1]) + " and '{}'".format(iterable_arg[-1])
    else:
        note = "'{}'".format(iterable_arg[0])
    return note


class MissingParametersErrors(Exception):
    pass


class UnsupportedSimulator(Exception):
    """
    Raised when unsupported simulator is supplied.
    """
    pass


class DuplicateParameterError(Exception):
    """
    Raised when NoiseModel detects duplicated parameter
    """
    pass


class UnsupportedSimulatorError(Exception):
    """
    Raised when unsupported :mod:`~pysb.simulator` class is supplied.
    """
    pass


class IncompatibleFormatWarning(Warning):
    pass


class CupSodaNotInstalledWarning(Warning):
    pass


def incompatible_format_warning(_var):
    """
    Warns that the supplied parameter has an Opt2Q-incompatible format.

    In this case, the simulator can still return simulation results, but they may not have a format compatible for use
    with other Opt2Q functions.
    """
    warnings.warn(
        'The supplied {} may not be formatted for use in other Opt2Q modules. Proceeding anyway.'.format(_var),
        category=IncompatibleFormatWarning)


def _is_vector_like(vector_like_obj):
    # if pd.Series, pd.DataFrame w/ one column, 1d np.array, list, tuple, set
    if isinstance(vector_like_obj, (pd.Series, list, tuple, set)):
        return True
    elif isinstance(vector_like_obj, (pd.DataFrame, np.ndarray)):  # and ``has no experiment columns``
        return len(vector_like_obj.shape) == 1 or \
               (len(vector_like_obj.shape) == 2 and
                (1 in vector_like_obj.shape or 0 in vector_like_obj.shape))
    else:  # not vector-like format
        return False


def _convert_vector_like_to_list(vector_like_obj):
    if not isinstance(vector_like_obj, (pd.DataFrame, np.ndarray)):
        return list(vector_like_obj)
    elif vector_like_obj.size is not 0:
        v_obj = np.asarray(vector_like_obj)
        observables = list(v_obj.reshape(max(v_obj.shape)))
        return list(observables)
    else:
        return list([])


def _convert_vector_like_to_set(vector_like_obj):
    if not isinstance(vector_like_obj, (pd.DataFrame, np.ndarray)):
        return set(vector_like_obj)
    elif vector_like_obj.size is not 0:
        v_obj = np.asarray(vector_like_obj)
        observables = list(v_obj.reshape(max(v_obj.shape)))
        return set(observables)
    else:
        return set([])


def profile(func):
    """
    A timer decorator
    """

    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value

    return function_timer


