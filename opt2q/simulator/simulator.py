# MW Irvin -- Lopez Lab -- 2018-08-19
import warnings
import logging
import numpy as np
import pandas as pd
from pysb.simulator import ScipyOdeSimulator, CupSodaSimulator
from opt2q.utils import UnsupportedSimulatorError, incompatible_format_warning, CupSodaNotInstalledWarning, \
    DaeSimulatorNotInstalledWarning
import subprocess
import sys
import os
import socket
import platform

try:
    from pysb.pathfinder import get_path
    # Path to cupSODA executable
    get_path('cupsoda')
    do_not_use_cupsoda = False
except Exception:
    do_not_use_cupsoda = True


class Simulator(object):
    """
    Conducts numerical simulation of a dynamical model, and formats the simulation results to be compatible with other
    `Opt2Q` functions.

    This class uses the :mod:`~pysb.simulator` in `PySB <http://pysb.org/>`_ and will accept the same parameters. It
    can return `Opt2Q`-compatible formats if supplied `Opt2Q`-compatible formats. (See following)

    Parameters
    ----------
    model : `PySB` :class:`~pysb.core.Model`
        A PySB rule-based model of system's dynamics

    tspan: vector-like (optional)
            time points over which to conduct the integration.

            This must be specified either with the Opt2Q :class:`Simulator <opt2q.simulator.Simulator>` class or it's
            ``run`` method.

    param_values : :class:`pandas.DataFrame`, vector-like or dict, (optional)
        The Opt2Q compatible format is a :class:`~pandas.DataFrame`

            - It's column names (`str`) must be the names of the `PySB model`'s :class:`~pysb.core.Parameter` objects.
            - An optional column, named `simulation` of consecutive integers (int) designates the simulation number.
            - Additional columns designate experimental conditions (note: these additional columns should match those in
                the `initials`)

        As vector-like or dict, it be consistent with ``param_values`` parameter expected in the PySB
        :mod:`~pysb.simulator`.

    initials : :class:`pandas.DataFrame`, vector-like or dict, (optional)
        The Opt2Q compatible format is a :class:`~pandas.DataFrame`

            - It's column names must be :class:`~pysb.core.ComplexPattern` objects present in the `PySB model`. They can
                be accessed via ``model.species``.
            - Additional columns designate experimental conditions (note: these additional columns should match those in
                the `param_values`)

        As vector-like or dict, it must be consistent with ``initials`` parameter expected in the
        PySB :mod:`~pysb.simulator`.

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

    kwargs: dict
        Dictionary of keyword arguments.

        *``check_updates``: (bool) If True, check updates to ``param_values`` and ``initials`` for Opt2Q-compatibility

    Attributes
    ----------
    check_updates: bool
        If True check new parameter and initials objects for compatibility with the solver, etc. Defaults to True
    """

    supported_solvers = {'scipyode': ScipyOdeSimulator, 'cupsoda': CupSodaSimulator}

    def __init__(self, model, tspan=None, param_values=None, initials=None, solver='scipyode', solver_options=None,
                 integrator_options=None, **kwargs):
        # Solver
        self.solver_kwargs = self._get_solver_kwargs(solver_options)
        self._add_integrator_options_dict(integrator_options)

        self.solver = self._check_solver(solver)
        self.sim = self.solver(model, **self.solver_kwargs)  # solver instantiates model and generates_equations

        self.model = model

        # Warnings Log
        self._capture_warnings_setting = self._warning_settings()  # When True, redirects warnings to logging package.
        logging.captureWarnings(False)  # Undo this to display Opt2Q generated warnings

        # experimental conditions columns
        self._exp_conditions_columns = None

        # Components (parameters and initials)
        self._model_params_names = [p.name for p in self.model.parameters]
        self._model_param_values = [p.value for p in self.model.parameters]
        self._model_params_dict = dict(zip(self._model_params_names, self._model_param_values))

        self._check_updates = kwargs.get('check_updates', True)
        self._update_components = [self._update_components_wo_check,
                                   self._update_components_w_check][self._check_updates]
        self._component_update_methods = {'param_values': self._update_params_wo_check,
                                          'initials': self._update_initials_wo_check}

        self._params_are_compatible, \
            self._param_values, \
            self._param_values_run, \
            self._initials_are_compatible, \
            self._initials, \
            self._initials_run = self._check_components(self.model, param_values, initials)

        # time axis
        self.tspan = tspan

    def _check_solver(self, _solver):
        if do_not_use_cupsoda and _solver is 'cupsoda':
            warnings.warn("You cannot use the 'cupsoda' solver. "
                          "The program cupSODA was not found in the default search path(s) for your operating system."
                          "The 'scipyode' solver will be used instead and may take much longer for simulation.",
                          category=CupSodaNotInstalledWarning)

            # Default (scipyode w/'lsoda') takes mxstep while cupsoda takes max_step.
            self.solver_kwargs.update({'integrator': 'lsoda'})
            if 'max_steps' in self.solver_kwargs['integrator_options'].keys():
                self.solver_kwargs['integrator_options'].\
                    update({'mxstep': self.solver_kwargs['integrator_options'].pop('max_steps')})

            for cupsoda_setting in {'n_blocks', 'memory_usage', 'vol', 'gpu'}:  # drop all cupsoda only terms
                if cupsoda_setting in self.solver_kwargs['integrator_options'].keys():
                    self.solver_kwargs['integrator_options'].pop(cupsoda_setting)
                if cupsoda_setting in self.solver_kwargs.keys():
                    self.solver_kwargs.pop(cupsoda_setting)

            return self.supported_solvers['scipyode']

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

    def _check_components(self, _model, _params, _initials):
        """
        Check parameters and initials for Opt2Q-compatibility, experimental treatment column names, etc.
        """
        model_names = set(self._model_params_names)
        model_initials = set(_model.species)

        # check params
        params_are_compatible, params_not_in_model, checked_params, params_for_run \
            = self._check_params(_params, model_names)

        # check initials
        initials_are_compatible, initials_not_in_model, checked_initials, initials_for_run = \
            self._check_initials(_initials, model_initials)

        # check that additional columns are equivalent for initials and param_values
        if params_are_compatible and initials_are_compatible:
            try:
                cols = list(params_not_in_model | {'simulation'})
                pd.testing.assert_frame_equal(checked_params[cols], checked_initials[cols], check_dtype=False)
            except (AssertionError, KeyError):
                raise ValueError("The experimental conditions columns of 'initials' and 'param_values' DataFrames"
                                 "must be equal")
        if params_are_compatible:
            self._exp_conditions_columns = list(params_not_in_model|{'simulation'})
        elif initials_are_compatible:
            self._exp_conditions_columns = list(initials_not_in_model|{'simulation'})

        return params_are_compatible, checked_params, params_for_run, \
            initials_are_compatible, checked_initials, initials_for_run

    def _check_params(self, _params, model_names):
        if isinstance(_params, pd.DataFrame):
            params_are_compatible, params_not_in_model, params_in_model \
                = self._is_compatible(set(_params.columns), model_names, var_name='param_values')
            checked_params = self._check_components_df(_params, params_are_compatible)
            params_for_run = self._params_for_run(checked_params, params_in_model, self._model_params_names,
                                                  self._model_param_values)
        else:
            params_are_compatible, checked_params = self._check_components_not_df(_params, 'param_values')
            params_not_in_model = None
            params_for_run = checked_params
        return params_are_compatible, params_not_in_model, checked_params, params_for_run

    def _check_initials(self, _initials, model_initials):
        if isinstance(_initials, pd.DataFrame):
            initials_are_compatible, initials_not_in_model, initials_in_model \
                = self._is_compatible(set(_initials.columns), model_initials, var_name='initials')
            checked_initials = self._check_components_df(_initials, initials_are_compatible)
            initials_for_run = self._initials_for_run(checked_initials, list(initials_in_model))
        else:
            initials_are_compatible, checked_initials = self._check_components_not_df(_initials, 'initials')
            initials_not_in_model = None
            initials_for_run = checked_initials
        return initials_are_compatible, initials_not_in_model, checked_initials, initials_for_run

    @staticmethod
    def _is_compatible(df_cols, model_components, var_name='param_values'):
        """
        Check that params or initials dataframe has additional columns indexing experimental treatments etc.

        Returns
        -------
        bool
            True if Opt2Q compatible
        set
            Components shared by df and the model_components
        """
        components_not_in_model = df_cols - model_components
        components_in_model = df_cols.intersection(model_components)
        if len(components_not_in_model) > 0 and len(components_in_model) > 0:
            return True, components_not_in_model, components_in_model
        else:
            incompatible_format_warning(var_name)
            return False, components_not_in_model, components_in_model

    @staticmethod
    def _check_components_not_df(_component, var_name):
        if _component is not None:
            incompatible_format_warning(var_name)
        return False, _component

    def _check_components_df(self, _values, _compatible):
        if _compatible and _values.shape[0] != 0:
            return self._add_simulations_column(_values)
        elif len(_values) == 0:
            return None  # an empty dataframe breaks the pysb simulator
        else:  # opt2q incompatible dataframe
            return _values

    @staticmethod
    def _add_simulations_column(df):
        if 'simulation' in df.columns:
            df.sort_values(['simulation'], inplace=True)
            if not np.array_equal(df['simulation'].values, np.arange(df.shape[0])):
                raise ValueError("The 'simulation' column of param_values and initials"
                                 " can only be consecutive integers starting from 0")
        else:
            df['simulation'] = np.arange(df.shape[0])
        return df

    def _params_for_run(self, pre_params, params_col, model_params_names, model_param_values):
        """
        Complete DataFrame using parameters and values from the PySB model.
        """
        if pre_params is None:
            return None
        else:
            pre_param_cols = list(params_col)
            _shape = (pre_params.shape[0], len(model_params_names))
            completed_df = pd.DataFrame(np.ones(_shape)*model_param_values, columns=model_params_names)
            completed_df[pre_param_cols] = pre_params[pre_param_cols]
            completed_df.fillna(self._model_params_dict, inplace=True)
            return completed_df.values

    @staticmethod
    def _initials_for_run(pre_initials, _initials_cols):
        """Simulator takes initials as a dict. Convert df to dict"""
        if pre_initials is None:
            return None
        else:
            return pre_initials[_initials_cols].to_dict('list')

    @property
    def check_updates(self):
        return self._check_updates

    @check_updates.setter
    def check_updates(self, val):
        if val is True:
            self._update_components = self._update_components_w_check
        elif val is False:
            self._update_components = self._update_components_wo_check
        else:
            raise ValueError("'check_updates can only be bool (True or False)")
        self._check_updates = val

    def _update_components_w_check(self, **kw):
        param_values = kw.get('param_values', self.param_values)
        initials = kw.get('initials', self.initials)
        self._params_are_compatible, \
            self._param_values, \
            self._param_values_run, \
            self._initials_are_compatible, \
            self._initials, \
            self._initials_run = self._check_components(self.model, param_values, initials)

    def _update_components_wo_check(self, **kw):
        for k, v in kw.items():
            self._component_update_methods[k](v)

    def _update_params_wo_check(self, params):
        if isinstance(params, pd.DataFrame):
            params_in_model = set(params.columns).intersection(set(self._model_params_names))
            self._param_values_run = self._params_for_run(params, params_in_model, self._model_params_names,
                                                          self._model_param_values)
            self._param_values = params
        else:
            self._param_values = params
            self._param_values_run = params

    def _update_initials_wo_check(self, initials):
        if isinstance(initials, pd.DataFrame):
            initials_in_model = set(initials.columns).intersection(set(self.model.species))
            self._initials_run = self._initials_for_run(initials, list(initials_in_model))
            self._initials = initials
        else:
            self._initials = initials
            self._initials_run = initials

    @property
    def param_values(self):
        return self._param_values

    @param_values.setter
    def param_values(self, val):
        self._update_components(param_values=val)

    @property
    def initials(self):
        return self._initials

    @initials.setter
    def initials(self, val):
        self._update_components(initials=val)

    @staticmethod
    def affinitize_to(target_CPU_ID):
        # Affinitize to the desired target CPU ID (Only on Linux)
        response = subprocess.run(['taskset', '-p', '-c', str(target_CPU_ID), str(os.getpid())],
                                  stdout=subprocess.PIPE)
        if response.returncode != 0:
            print("taskset:")
            print(response.stderr)
            print(" terminating execution.")
            sys.exit(-1)

    @staticmethod
    def report_affinitization():
        # (Only on Linux)
        response = subprocess.run(['taskset', '-acp', str(os.getpid())],
                                  stdout=subprocess.PIPE)
        if response.returncode != 0:
            print("taskset:")
            print(response.stderr)
            print(" terminating execution.")
            sys.exit(-1)
        CPU_IDs = response.stdout.decode().split()[-1]
        hostname = socket.gethostname()
        print("Process " + str(os.getpid()) + " affinitized to:" +
              " logical CPU(s) " + CPU_IDs +
              " on host " + hostname + ".")

    def run(self, tspan=None, param_values=None, initials=None, check_updates=True, **run_kwargs):
        """
        Runs a simulation and returns a :class:`~pysb.simulator.SimulationResult`; which may also contain an
        Opt2Q compatible results dataframe ``results.opt2q_dataframe``.

        Parameters
        ----------
        tspan: vector-like (optional)
            time points over which to conduct the integration.

            This must be specified either with the Opt2Q :class:`Simulator <opt2q.simulator.Simulator>` class or it's
            ``run`` method.

        param_values:  :class:`pandas.DataFrame`, vector-like or dict, (optional)
            Same as parameter ``param_values`` in :class:`Simulator <opt2q.simulator.Simulator>` (see above).

        initials:  :class:`pandas.DataFrame`, vector-like or dict, (optional)
            Same as parameter ``initials`` in :class:`Simulator <opt2q.simulator.Simulator>` (see above).

        check_updates: Boolean, optional
            When `True`, `initials` and `param_values` are checked for compatibility with `Opt2Q` before running the
            simulation.

        _run_kwargs: dict, optional
            Additional kwargs supplied to the solver.

        Returns
        -------
            A set of trajectories for the PySB model `species` and `observables` over the time points specified by
            ``tspan``. This is presented as a PySB :class:`~pysb.simulator.SimulationResult`, with an additional
            ``results.opt2q_dataframe`` that may include additional indices annotating the experimental conditions
            modeled by each trajectory.

        Notes
        -----
            Currently, ``param_values`` and ``initials`` must have identical experimental indices.

            The :class:`~pysb.simulator.SimulationResult` object that gets returned does not save/load ``results.opt2q_dataframe``.
        """
        if 'affinitize_to' in run_kwargs:
            target_cpu_id = run_kwargs.pop('affinitize_to')
            if 'Linux' in platform.platform():
                self.affinitize_to(target_cpu_id)

        if run_kwargs.get('report_affinitization', False):
            self.report_affinitization()

        self.check_updates = check_updates

        if tspan is not None:
            self.tspan = tspan
        if initials is not None:
            self.initials = initials
        if param_values is not None:
            self.param_values = param_values

        # start_time = time.time()
        results = self.sim.run(tspan=self.tspan,
                               initials=self._initials_run,
                               param_values=self._param_values_run,
                               **run_kwargs)
        results.opt2q_dataframe = self.opt2q_dataframe(results.dataframe)

        return results

    def opt2q_dataframe(self, df):
        if self._params_are_compatible:
            exp_index = self.param_values[self._exp_conditions_columns]
        elif self._initials_are_compatible:
            exp_index = self.initials[self._exp_conditions_columns]
        else:
            return df

        new_df = df.copy()
        try:
            new_df.reset_index(level=['simulation', 'time'], inplace=True)
        except KeyError:
            new_df['simulation'] = 0
            new_df.reset_index(level=['time'], inplace=True)

        new_df = new_df.merge(exp_index, how='inner', on=['simulation'])
        new_df.set_index('time', inplace=True)
        return new_df

