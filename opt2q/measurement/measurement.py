# MW Irvin -- Lopez Lab -- 2018-08-24
"""
Suite of Measurement Models
"""
import pandas as pd
import numpy as np
from scipy import stats
from opt2q.utils import _is_vector_like, _convert_vector_like_to_list, parse_column_names, _list_the_errors
from opt2q.measurement.base.base import MeasurementModel
from opt2q.measurement.base.transforms import Interpolate, LogisticClassifier, Pipeline, Scale, Standardize, \
    SampleAverage, ScaleToMinMax

from timeit import default_timer as timer
from datetime import timedelta




class Fluorescence(MeasurementModel):
    """
    Simulates Fluorescence Measurements.

    Conducts a series of transformations on a :class:`~pysb.simulator.SimulationResult` to represent attributes of the
    Fluorescence (semi-quantitative) measurement. The fluorescent intensity of a molecular indicator scales with protein
    abundance such that it could be modeled with a monotonic (commonly linear) function.

    Parameters
    ----------
    simulation_result: :class:`~pysb.simulator.SimulationResult`
        Results of a `PySB` Model simulation. Produced by the run methods in the `PySB`
        (:meth:`~pysb.simulator.ScipyOdeSimulator.run`) and `Opt2Q` (:meth:`~opt2q.simulator.Simulator.run`)
        simulators. Since the Western Blot measurement makes use of a classifier, this simulation result should have a
        large number of simulations.

    dataset: :class:`~opt2q.data.DataSet`, optional
        Measured values and associated attributes (e.g. experimental conditions names) which dictate the rows of the
        :class:`~pysb.simulator.SimulationResult` ``opt2q_dataframe`` pertain to the data.

    measured_values: dict, optional
        A dictionary of (keys) measured variables (as named in the DataSet) and a list of corresponding PySB model
        observables. If a ``dataset_fluorescence`` is provided, ``measured_value`` is required.

    observables: vector-like, optional
        Lists the names (str) of the PySB PySB :class:`pysb.core.Model` observables and/or species involved in the
        measurement.

        These observables apply to all the experimental conditions involved in the measurement. Observables not
        mentioned in the ``simulation_result`` and/or ``dataset_fluorescence`` (if supplied) are ignored.

    time_points: vector-like, optional
        Lists the time-points involved in the measurement. Defaults to the time points in the
        :class:`~pysb.simulator.SimulationResult`. You can also assign time-points using a 'time' in the
        ``experimental_conditions`` argument (as follows).

    experimental_conditions: :class:`~pandas.DataFrame`, optional
        The experimental conditions involved in the measurement model. Defaults to experimental conditions in
        the ``dataset_fluorescence`` (if present) or the ``simulation_result``.

        You can add a 'time' column to specify time-points that are specific to the individual experimental conditions.
        NaNs in this column will be replace by the ``time_points`` values or the time-points mentioned in the
        :class:`~pysb.simulator.SimulationResult`

    Attributes
    ----------
    interpolation_ds: :class:`~opt2q.measurement.base.transforms.Interpolate`
        Interpolates values based on time_points mentioned in the experimental conditions in the ``data_set``.
    interpolation: :class:`~opt2q.measurement.base.transform.Interpolate`
        Interpolates values based on time_points in the ``simulation_result`` (or those manually specified via the
        ``time_points`` and/or ``experimental_conditions`` arguments).
    results: :class:`~pandas.DataFrame`
        Results of the simulated Fluorescence measurement
    process: :class:`~opt2q.measurement.base.transforms.Pipeline`
        Series of steps needed to model the measurement.
        .. note:: Deleting or changing the names of preset steps can cause problems.
    """

    def __init__(self, simulation_result, dataset=None, measured_values=None, observables=None, time_points=None,
                 experimental_conditions=None, interpolate_first=True):

        super().__init__(simulation_result,
                         dataset=dataset,
                         observables=observables,
                         time_points=time_points,
                         experimental_conditions=experimental_conditions)

        _measured_values, _process_observables = self._get_process_observables(self._dataset, measured_values)
        _groupby_columns = self._get_groupby_columns()
        self._measured_values = measured_values

        if self._dataset is None:
            self._run = self._run_process
            self.interpolation_ds = None
            self._likelihood = self._likelihood_raise_error
        else:
            self._run = self._run_with_dataset
            self.interpolation_ds = Interpolate('time',
                                                list(_process_observables),
                                                self._dataset_experimental_conditions_df,
                                                groupby='simulation')
            self._likelihood = self._get_likelihood

        self.interpolation = Interpolate('time',
                                         list(_process_observables),
                                         self.experimental_conditions_df,
                                         groupby='simulation')

        self.process = Pipeline(steps=[('normalize', ScaleToMinMax(feature_range=(0, 1),
                                                                   columns=list(_process_observables),
                                                                   groupby=_groupby_columns,
                                                                   do_fit_transform=True))])

        self._interpolate_first = False if not interpolate_first else True  # Non-bool Defaults to True
        self._add_interpolate_step()

        self._results_cols = list(set(_process_observables) | (set(self.experimental_conditions_df.columns)) |
                                  {'time', 'simulation'})  # columns involved in generating the result

    def _get_groupby_columns(self):
        """
        Return a default groupby for scaling transforms are carried out in groups. Defaults to the
        experimental conditions columns in the result. If no columns are present, it groups by 'simulation'.
        """
        exp_con_cols = set(self._experimental_conditions_df.columns)-{'time', 'simulation'}
        if len(exp_con_cols) == 0:
            return ['simulation']
        else:
            return list(exp_con_cols)

    def _get_process_observables(self, dataset, measured_values):
        """
        Observables passed to the process

        If a dataset_fluorescence is provided,
        """
        obs = self.observables
        if dataset is None:
            return None, obs
        else:
            return self._check_measured_values_dict(measured_values, dataset)

    def _check_measured_values_dict(self, measured_values_dict, dataset) -> (dict, set):
        """
        Check that measured_values keys are in the dataset_fluorescence.measured_variables.

        Look for observables mentioned in measured_values_dict that are not in self.observables, and add them.
        """
        obs = self.observables  # set
        data_cols = [k for k, v, in dataset.measured_variables.items() if v in
                     ('default', 'quantitative', 'semi-quantitative')]

        if measured_values_dict is None:
            raise ValueError("You must provide 'measured_values'.")

        for k, v in measured_values_dict.items():
            if k not in data_cols: raise ValueError(
                "'measured_values' contains a variable, '{}', not mentioned"
                " as a 'quantitative' or 'semi-quantitative' variable in the 'dataset_fluorescence'."
                .format(k))

            if _is_vector_like(v):  # look for the v in the default_observables
                measured_obs = _convert_vector_like_to_list(v)
                for i in measured_obs:  # if v is an observable mentioned in the
                    if isinstance(i, str):
                        mentioned_obs = i.split('__')[0]
                        obs |= {mentioned_obs} if mentioned_obs in self._default_observables else set()
            else:
                raise ValueError("'measured_values' must be a dict of list")

        return measured_values_dict, obs

    @property
    def interpolate_first(self):
        return self._interpolate_first

    @interpolate_first.setter
    def interpolate_first(self, val):
        self._interpolate_first = False if not val else True  # Non-bool Defaults to True
        self._add_interpolate_step()

    def _add_interpolate_step(self):
        if self.interpolate_first:
            self.process.add_step(('interpolate', self.interpolation), index=0)
        elif 'classifier' in [s[0] for s in self.process.steps]:
            self.process.add_step(('interpolate', self.interpolation), 'classifier')
        else:
            self.process.add_step(('interpolate', self.interpolation))

    def _replace_interpolate_step(self, step):
        idx = [x for x, y in enumerate(self.process.steps) if y[0] == 'interpolate'][0]
        self.process.remove_step('interpolate')
        self.process.steps.insert(idx, ('interpolate', step))

    def run(self, **kwargs):
        """
        Returns measurement results

        Parameters
        ----------
        kwargs: dict, optional
            ``use_dataset``, bool
                True, this method transforms only experimental conditions mentioned in the data. When False,
                the predictions will include experimental conditions in the simulation result that are not
                present in the dataset_fluorescence. Defaults to True
            ``do_fit_transform``, bool optional
                When True all process steps with a 'do_fit_transform' have 'do_fit_transform' temporarily set
                to True. When False all process steps have their 'do_fit_transform' temporarily set to False
                (This is necessary for out-of-sample calculations).
                Defaults to the whatever the preset settings are for the individual process steps.
        """
        return self._run(**kwargs)

    def _run_process(self, **kwargs):
        """
        Run the measurement process without considering the dataset_fluorescence. This is used in the absence of a dataset_fluorescence
        """

        if 'do_fit_transform' in kwargs:
            do_fit_transform = kwargs.pop('do_fit_transform')
            return self._run_process_set_do_fit_transform(do_fit_transform, **kwargs)
        else:
            sim_results = self._trim_sim_results_to_have_only_rows_in_experimental_conditions_df(
                use_dataset=kwargs.get('use_dataset', False))
            self.results = self.process.transform(sim_results[self._results_cols])
            # self.results = self.process.transform(self.simulation_result_df[self._results_cols])
            return self.results

    def _run_process_set_do_fit_transform(self, do_fit_transform, **kwargs):
        original_do_fit_transform_settings = {k: v for k, v in self.process.get_params().items()
                                              if 'do_fit_transform' in k}
        new_do_fit_transform_settings = {k: do_fit_transform for k in self.process.get_params().keys()
                                         if 'do_fit_transform' in k}

        self.process.set_params(**new_do_fit_transform_settings)
        sim_results = self._trim_sim_results_to_have_only_rows_in_experimental_conditions_df(
            use_dataset=kwargs.get('use_dataset', False))
        self.results = self.process.transform(sim_results[self._results_cols])
        # self.results = self.process.transform(self.simulation_result_df[self._results_cols])
        self.process.set_params(**original_do_fit_transform_settings)

        return self.results

    def _run_with_dataset(self, **kwargs):
        use_dataset = kwargs.get('use_dataset', True)

        if use_dataset and 'interpolate' in [x[0] for x in self.process.steps]:
            self._replace_interpolate_step(self.interpolation_ds)
            kwargs.update({'use_dataset': True})
            result_ds = self._run_process(**kwargs)
            self._replace_interpolate_step(self.interpolation)
            self.results = result_ds
            return self.results
        else:
            return self._run_process(**kwargs)

    def likelihood(self, use_all_dataset_obs=True, use_all_dataset_exp_cond=True):
        """
        Calculates the negative log-likelihood assuming the error of measured values of fluorescence has a
        normal distribution.
        """
        return self._likelihood()

    def _get_likelihood(self):
        """
        Return scalar value of the negative log-likelihood.

        Parameters
        ----------
        use_all_dataset_obs: bool (optional)
            If observables are supplied via the ``observables`` argument and a :class:`~opt2q.data.DataSet`, the
            likelihood, by default, uses *all* the observables mentioned in the dataset_fluorescence (even if they are absent from
            the ``observables`` argument).

            If False, the likelihood uses only  the subset of observables mentioned in both the DataSet and the
            ``observables`` arguments.

        use_all_dataset_exp_cond: bool (optional)
            If experimental conditions are supplied via the ``experimental_conditions`` argument and a
            :class:`~opt2q.data.DataSet`, the likelihood uses, by default, *all* the experimental conditions mentioned
            in the dataset_fluorescence (even if they are absent from the ``experimental_conditions`` argument).

            If False, the likelihood uses only the subset of observables mentioned in both the DataSet and the
            ``experimental_conditions`` arguments.
        """

        self.results = self.run(use_dataset=True)
        result_columns_and_corresponding_data_col = self._get_results_columns_that_go_with_data()
        error_cols_list = [k + '__error' for k in self._measured_values.keys()]

        y_ = self._dataset.data
        y_[error_cols_list] = self._dataset.measurement_error_df[error_cols_list]
        exp_conditions_cols = list(self._dataset.experimental_conditions.columns)
        y = y_.merge(self.results, how='outer', on=exp_conditions_cols).drop_duplicates().reset_index(drop=True)

        log_likelihood = 0
        for obs in self._measured_values.keys():
            sim_cols = result_columns_and_corresponding_data_col[obs]
            if len(sim_cols) == 1:
                y_sim_ = y[list(sim_cols)].values[:, 0]
                y_data_ = y[obs].values
                # non-zero variance values
                y_error_ = np.sqrt(y[obs + '__error'].values)

                log_likelihood += np.sum(-stats.norm.logpdf(y_sim_, loc=y_data_, scale=y_error_))
            else:
                raise ValueError("The measurement model result for {} can only have one column. It had the following: ".
                                 format(obs) + _list_the_errors(sim_cols) + ".")
        return log_likelihood

    def _get_results_columns_that_go_with_data(self):
        """
        So the measured values in the data do not necessarily have the same names as those in the simulation results.

        For each k in the measured_values keys, get the columns in the simulation_result that correspond.
        """
        x_col_set = set(self.results.columns)
        return {k: parse_column_names(x_col_set, set(v)).intersection(x_col_set) for k, v in self._measured_values.items()}

    def _likelihood_raise_error(self):
        raise ValueError("likelihood requires a dataset")
