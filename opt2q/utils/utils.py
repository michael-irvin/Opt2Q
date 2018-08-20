"""
Background Stuff
"""


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
