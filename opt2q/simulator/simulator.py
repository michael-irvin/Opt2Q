# MW Irvin -- Lopez Lab -- 2018-08-19
import warnings
import logging
from pysb.simulator import ScipyOdeSimulator
from opt2q.utils import UnsupportedSimulatorError


class Simulator(object):
    """
    Conducts numerical simulation of a dynamical model, and formats the simulation results to be compatible with other
    `Opt2Q` functions.

    This class uses the :mod:`~pysb.simulator` in `PySB <http://pysb.org/>`_ and will accept the same parameters. It
    can return `Opt2Q`-compatible formats if supplied `Opt2Q`-compatible formats. (See following)

    Parameters
    ----------
    solver : str, optional
        The name of a supported PySB solver. Defaults to :class:`~pysb.simulator.ScipyOdeSimulator`

    solver_options: dict, optional
        Dictionary of the keyword arguments to be passed to PySB solver. Examples include:

        * ``integrator``: Choice of integrator, including ``vode`` (default),
          ``zvode``, ``lsoda``, ``dopri5`` and ``dop853``. See
          :py:class:`scipy.integrate.ode <scipy.integrate.ode>`
          for further information.
        * ``cleanup``: Boolean, `cleanup` argument used for
          :func:`pysb.bng.generate_equations <pysb.bng.generate_equations>` call
        * ``use_theano``: Boolean, option of using theano in the `scipyode` solver

    integrator_options: dict, optional
         A dictionary of keyword arguments to supply to the integrator. See
         :py:class:`scipy.integrate.ode <scipy.integrate.ode>`
    """

    supported_solvers = {'scipyode': ScipyOdeSimulator}

    def __init__(self, model, solver='scipyode', solver_options=None, integrator_options=None):
        # Solver
        self.solver = self._check_solver(solver)
        self.solver_kwargs = self._get_solver_kwargs(solver_options)
        self._add_integrator_options_dict(integrator_options)
        self.sim = self.solver(model, **self.solver_kwargs)  # solver instantiates model and generates_equations
        self.model = model

        # Warnings Log
        self._capture_warnings_setting = self._warning_settings
        # The logging.captureWarnings (when set to True) redirects all warnings to a logging package.
        # Undo this to display Opt2Q generated warnings
        logging.captureWarnings(False)

    def _check_solver(self, _solver):
        try:
            return self.supported_solvers[_solver]
        except KeyError:
            raise UnsupportedSimulatorError("This simulator does not support {}".format(_solver))

    @staticmethod
    def _get_solver_kwargs(solver_opts):
        if solver_opts is not None:
            return dict(solver_opts)
        else:
            return {}

    def _add_integrator_options_dict(self, integrator_options):
        """
        Adds integrator_options to self.solver_kwargs. Updates self.solver_kwargs

        Parameters
        ----------
        integrator_options: (dict)
            A dictionary of keyword arguments to supply to the scipy integrator.
        """

        if integrator_options is not None:
            self.solver_kwargs.update({'integrator_options': dict(integrator_options)})
        else:
            self.solver_kwargs.update({'integrator_options': {}})

    @staticmethod
    def _warning_settings():
        """
        Returns the current warning Setting
        """
        return warnings.showwarning.__name__ == "_showwarning"
