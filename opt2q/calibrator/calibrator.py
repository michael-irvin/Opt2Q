"""
Tools for calibrating an Opt2Q Model
"""
import inspect


# MW Irvin -- Lopez Lab -- 2018-08-07


class ObjectiveFunction(object):
    """
    Behaves as a function but possess attributes that aid its implementation as an objective_function in conventional
    optimizers
    """
    def __init__(self, f):
        self._f = f

        arg_spec = inspect.getfullargspec(self._f)
        self.arg_names = list(arg_spec[0])
        self.arg_defaults = arg_spec[3]

        if self.arg_defaults is None:
            self.arg_defaults = []
        else:
            self.arg_defaults = list(self.arg_defaults)

        self.len_defaults = len(self.arg_defaults)
        self.len_names = len(self.arg_names)

    def __repr__(self):
        """
        Represents function the way it would be typed. Allows for easy copy-paste implementation.

        :return: str
        """
        f_name = '{}('.format(self._f.__name__)
        if self.len_names == 1:
            f_name += '{})'.format(self.arg_names[0])
        else:
            f_name += ', '.join('{}'.format(k) for k in
                                self.arg_names[:min(self.len_names-self.len_defaults, self.len_names-1)]) +\
                      ''.join(', {}={}'.format(self.arg_names[k + self.len_names - self.len_defaults],
                                               self.arg_defaults[k]) for k in range(self.len_defaults)) + \
                      ')'
        return f_name

    def __call__(self, x_, *args, **kwargs):
        return self._f(x_, *args, **kwargs)


class objective_function(object):
    """
    Decorator that creates an objective function using an Opt2Q noise and simulation model, and Opt2Q measurement
    model(s)

    """
    def __init__(self, **kwargs):
        # Process Decorator arguments
        self.__dict__.update(kwargs)

    def __call__(self, _f):
        """
        Parameters
        ----------
        _f: func
            User Defined Objective Function

        Returns
        -------
        :class:`~opt2q.calibrator.ObjectiveFunction` instance

        """
        obj_f = ObjectiveFunction(_f)
        obj_f.__dict__.update(self.__dict__)
        return obj_f
