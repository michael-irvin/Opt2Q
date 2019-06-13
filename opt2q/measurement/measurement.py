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


class WesternBlot(MeasurementModel):
    """
    Simulates a Western Blot Measurements.

    Conducts a series of transformation on a :class:`~pysb.simulator.SimulationResult` to represents attributes of the
    Western blot (ordinal) measurement.

    Parameters
    ----------
    simulation_result: :class:`~pysb.simulator.SimulationResult`
        Results of a `PySB` Model simulation. Produced by the run methods in the `PySB`
        (:meth:`~pysb.simulator.ScipyOdeSimulator.run`) and `Opt2Q` (:meth:`~opt2q.simulator.Simulator.run`)
        simulators. Since the Western Blot measurement makes use of a classifier, this simulation result should have a
        large number of simulations.

    dataset: :class:`~opt2q.data.DataSet`
        Measured values and associated attributes (e.g. experimental conditions names) which dictate the rows of the
        :class:`~pysb.simulator.SimulationResult` ``opt2q_dataframe`` pertain to the data.

    measured_values: dict
        A dictionary of (keys) measured variables (as named in the DataSet) and a list of corresponding PySB model observables.

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
        Results of the simulated Western blot measurement
    process: :class:`~opt2q.measurement.base.transforms.Pipeline`
        Series of steps needed to model the measurement.
        .. note:: Deleting or changing the names of preset steps can cause problems.
    """

    def __init__(self, simulation_result, dataset, measured_values, observables=None, time_points=None,
                 experimental_conditions=None):

        super().__init__(simulation_result,
                         dataset=dataset,
                         observables=observables,
                         time_points=time_points,
                         experimental_conditions=experimental_conditions)

        _measured_values, _process_observables = self._check_measured_values_dict(measured_values, dataset)
        self._measured_values_dict = _measured_values

        self.interpolation_ds = Interpolate('time',
                                            list(_process_observables),
                                            self._dataset_experimental_conditions_df,
                                            groupby='simulation')
        self.interpolation = Interpolate('time',
                                         list(_process_observables),
                                         self.experimental_conditions_df,
                                         groupby='simulation')

        self.process = Pipeline(
            steps=[('sample_average', SampleAverage(columns=list(_process_observables), drop_columns='simulation',
                                                    groupby=list(set(
                                                        self.experimental_conditions_df.columns) - {'simulation'}),
                                                    apply_noise=True,
                                                    variances=0.0,
                                                    sample_size=10000)),
                   ('log_scale', Scale(columns=list(_process_observables), scale_fn='log10')),
                   ('standardize', Standardize(columns=list(_process_observables), groupby=None)),
                   ('classifier', LogisticClassifier(self._dataset, column_groups=_measured_values,
                                                     do_fit_transform=False, classifier_type='ordinal_eoc'))])
        self.results = None
        self._results_cols = list(set(_process_observables) | (set(self.experimental_conditions_df.columns)) |
                                  {'time', 'simulation'})

    def _check_that_measured_values_is_dict(self, measured_values):
        if isinstance(measured_values, dict):
            return self._check_that_dict_items_are_vector_like(measured_values)
        else:
            raise ValueError("'measured_values' must be a dict.")

    @staticmethod
    def _check_that_dict_items_are_vector_like(dict_in):
        dict_out = dict()
        for k, v in dict_in.items():
            dict_out.update({k: _convert_vector_like_to_list(v)} if _is_vector_like(v) else {k:[v]})
        return dict_out

    def _check_measured_values_dict(self, measured_values_dict, dataset) -> (dict, set):
        """
        Check that measured_values keys are in the dataset_fluorescence.measured_variables.

        Look for observables mentioned in measured_values_dict that are not in self.observables, and add them.
        """
        obs = self.observables  # set
        # Todo: change to: obs = set() if self._observables is None else self._observables  # set
        data_cols = [k for k, v, in dataset.measured_variables.items() if v in ('default', 'ordinal')]

        for k, v in measured_values_dict.items():
            if k not in data_cols: raise ValueError(
                "'measured_values' contains a variable, '{}', not mentioned as an ordinal variable in the 'dataset_fluorescence'."
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

    def run(self, use_dataset=True):
        """
        Run the measurement transform process

        Parameters
        ----------
        use_dataset: bool
            When true, it will run the simulation and return values for all the experiments mentioned in the dataset_fluorescence.
            Otherwise, it will return values for all the experiments specified in the ``experimental_conditions``
            argument or in the simulation result whether it is mentioned in the data or not (This is useful for out
            of sample calculations).
        """

        if use_dataset:
            x_ds = self.interpolation_ds.transform(self.simulation_result_df[self._results_cols])
            result_ds = self.process.transform(x_ds)
            self.results = result_ds
            return self.results

        else:
            original_do_fit_transform_settings = {k: v for k, v in self.process.get_params().items()
                                                  if 'do_fit_transform' in k}
            do_not_do_fit_transform = {k: False for k in self.process.get_params().keys()
                                       if 'do_fit_transform' in k}

            self.process.set_params(**do_not_do_fit_transform)
            x = self.interpolation.transform(self.simulation_result_df[self._results_cols])
            self.results = self.process.transform(x)
            self.process.set_params(**original_do_fit_transform_settings)

            return self.results

    def likelihood(self, use_all_dataset_obs=True, use_all_dataset_exp_cond=True):
        """
        Return scalar value of the likelihood.

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
        ordinal_errors = self._dataset._ordinal_errors_df

        exp_conditions_cols = list(self._dataset.experimental_conditions.columns)
        columns_for_merge = exp_conditions_cols + ['simulation']

        # The sample average step drops the 'simulation' column and changes the number of rows.
        # Add the 'simulation' column back to the results dataframe so that it can merge with the _ordinal_errors_df
        if 'simulation' not in self.results.columns:  # sample_average drops 'simulation' column
            sample_avr_step = [y[1] for y in self.process.steps if y[0] == 'sample_average'][0]
            sims = range(sample_avr_step.sample_size) if sample_avr_step._apply_noise else [0]
            sims_repeats = int(self.results.shape[0]/len(sims))

            self.results = self.results.assign(simulation=np.tile(sims, sims_repeats))

        # Duplicate rows of _ordinal_errors_df to match self.results (which can have many simulations per data-point).
        ordinal_category_dist = ordinal_errors.merge(self.results[columns_for_merge], how='outer',
                                                     on=exp_conditions_cols).drop_duplicates().reset_index(drop=True)

        # Probability of the predictions and the data referencing the same ordinal category
        category_names = ordinal_category_dist.filter(regex='__').columns
        likelihood_per_category_and_simulation = pd.DataFrame(
            self.results[category_names].values*ordinal_category_dist[category_names].values,
            columns=category_names)
        likelihood_per_category_and_simulation[columns_for_merge] = ordinal_category_dist[columns_for_merge]

        # Marginal probability estimates. Average probability for the simulations assigned to a particular data-point.
        likelihood_per_category = likelihood_per_category_and_simulation.\
            groupby(list(self._dataset.experimental_conditions.columns)).\
            mean().reset_index().drop(['simulation'], axis=1)

        # Sum probabilities of the categories mentioned for a particular measured variable
        transposed_likelihoods = likelihood_per_category.transpose().reset_index()
        transposed_likelihoods['index'] = [this.split("__")[0] for this in transposed_likelihoods['index']]
        likelihoods = transposed_likelihoods.groupby('index').sum(numeric_only=True).transpose()

        # Sum neg-log likelihood
        return np.sum(-np.log(np.array(
                likelihoods[
                    likelihoods.columns.difference(list(self._dataset.experimental_conditions.columns))]
                .values)
                .astype(float)
               ))


class FractionalKilling(MeasurementModel):
    """
    Simulates Measurements of Fractional Killing.

    Conducts a series of transformations on a :class:`~pysb.simulator.SimulationResult` to represent attributes of a
    measurement fractional killing. While fractional killing is quantitative, it reports the proportion of cells
    undergoing a certain fate (nominal).

    Parameters
    ----------
    simulation_result: :class:`~pysb.simulator.SimulationResult`
        Results of a `PySB` Model simulation. Produced by the run methods in the `PySB`
        (:meth:`~pysb.simulator.ScipyOdeSimulator.run`) and `Opt2Q` (:meth:`~opt2q.simulator.Simulator.run`)
        simulators. Since the fractional killing measurement makes use of a classifier, this simulation result should
        have a large number of simulations.

    dataset: :class:`~opt2q.data.DataSet`
        Measured values and associated attributes (e.g. experimental conditions names) which dictate the rows of the
        :class:`~pysb.simulator.SimulationResult` ``opt2q_dataframe`` pertain to the data. The pertinent measured values
        in the dataset_fluorescence can only have values between 0, 1.

    measured_values: dict
        Relate measured variable, key, (as named in the DataSet) to a list of corresponding columns in the ``dataframe``
        or ``opt2q_dataframe`` of the :class:`~pysb.simulator.SimulationResult`. The dict can only have one key (one
        measured variable).

    observables: vector-like, optional
        Lists the names (str) of the PySB :class:`pysb.core.Model` observables and/or species involved in the
        measurement.

        These observables apply to all the experimental conditions involved in the measurement. Observables not
        mentioned in the ``simulation_result`` and/or ``dataset_fluorescence`` (if supplied here) are ignored.

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

    interpolate_first: bool
        When True, the 'interpolation' step is the first step of the measurement
        :attr:``~opt2q.measurement.measurement.FractionalKilling.process`. Otherwise, it appears after all other prepocessing steps
        and before the 'classifier' step. Defaults to True.

    Attributes
    ----------
    process: :class:`~opt2q.measurement.base.transforms.Pipeline`
        Series of steps needed to model the measurement.
        .. note:: Do not remove the 'classifier' step. If you replace it, don't change it's name.

    """

    def __init__(self, simulation_result, dataset, measured_values, observables=None, time_points=None,
                 experimental_conditions=None, interpolate_first=True, time_dependent=True):

        super().__init__(simulation_result,
                         dataset=dataset,
                         observables=observables,
                         time_points=time_points,
                         experimental_conditions=experimental_conditions,
                         time_dependent=time_dependent)

        _measured_values, _process_observables = self._check_measured_values_dict(measured_values, dataset)
        self._measured_values_dict = _measured_values
        self._measured_values_names = list(_measured_values.keys())

        _data_columns = list(measured_values.keys())
        _mock_data_col = _data_columns[0]
        self._mock_dataset = self._create_mock_dataset(self._dataset.data, _data_columns)

        self._data_error_col_name = list(set(self._dataset.measurement_error_df.columns) -
                                         set(self._dataset_experimental_conditions_df.columns))
        self.process = Pipeline(
            steps=[('log_scale', Scale(columns=list(_process_observables), scale_fn='log10')),
                   ('standardize', Standardize(columns=list(_process_observables), groupby=None)),
                   ('classifier', LogisticClassifier(self._mock_dataset, column_groups=_measured_values,
                                                     do_fit_transform=False, classifier_type='nominal')),
                   ('sample_average', SampleAverage(columns=_mock_data_col, drop_columns='simulation',
                                                    groupby=list(set(self.experimental_conditions_df.columns) -
                                                                 {'simulation'}), apply_noise=False))])

        if self._time_dependent:
            self.interpolation_ds = Interpolate('time',
                                                list(_process_observables),
                                                self._dataset_experimental_conditions_df,
                                                groupby='simulation')

            self.interpolation = Interpolate('time',
                                             list(_process_observables),
                                             self.experimental_conditions_df,
                                             groupby='simulation')
            self._interpolate_first = False if not interpolate_first else True  # Non-bool Defaults to True

            self._add_interpolate_step()

        self.results = None
        self._results_cols = list(set(_process_observables) | (set(self.experimental_conditions_df.columns)) |
                                  {'time', 'simulation'})  # columns involved in generating the result

    @staticmethod
    def _create_mock_dataset(data, data_columns):
        # todo make mock data something that can also be user-defined.
        other_columns = list(set(data.columns)-set(data_columns))
        num_categories = len(data_columns)+1
        len_data = data.shape[0]
        data_ = data[other_columns].copy()
        data_[data_columns[0]] = np.tile(np.arange(num_categories), int(len_data/num_categories)+1)[:len_data].astype(int)
        return data_

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
        else:
            idx = [x for x, y in enumerate(self.process.steps) if y[0] == 'classifier'][0]
            self.process.steps.insert(idx, ('interpolate', self.interpolation))

    def _replace_interpolate_step(self, step):
        idx = [x for x, y in enumerate(self.process.steps) if y[0] == 'interpolate'][0]
        self.process.remove_step('interpolate')
        self.process.steps.insert(idx, ('interpolate', step))

    def _check_measured_values_dict(self, measured_values_dict, dataset) -> (dict, set):
        """
        Check that measured_values has only one key, and it is in the dataset_fluorescence.measured_variables.

        Check that the measured_values have values between 0 and 1.

        Look for observables mentioned in measured_values_dict that are not in self.observables, and add them.
        """

        # Measured_values has only one key
        if len(measured_values_dict) > 1:
            raise ValueError("'measured_values' cannot have multiple keys.")

        obs = self.observables  # set
        data_cols = [k for k, v, in dataset.measured_variables.items() if v is not 'ordinal']

        # Measured_values name exists in DataSet and has values between 0 and 1.
        for k, v in measured_values_dict.items():
            if k not in data_cols: raise ValueError(
                "'measured_values' contains a variable, '{}', not mentioned as in the 'dataset_fluorescence'."
                .format(k))
            if max(dataset.data[k]) > 1 or min(dataset.data[k]) < 0:
                raise ValueError("The variable, '{}', in the 'dataset_fluorescence', can only have values between 0.0 and 1.0".format(k))

            # Corresponding columns in the simulation result
            if _is_vector_like(v):  # look for the v in the default_observables
                measured_obs = _convert_vector_like_to_list(v)
                for i in measured_obs:  # if v is an observable mentioned in the simulation result.
                    # But what if you want to reference only a feature that gets created via a proceeding scaling step?
                    if isinstance(i, str):
                        mentioned_obs = i.split('__')[0]
                        obs |= {mentioned_obs} if mentioned_obs in self._default_observables|{'time'} else set()
            else:
                raise ValueError("'measured_values' must be a dict of lists")

        return measured_values_dict, obs

    def run(self, use_dataset=True):
        """
        Run the measurement model and return predicted values of the fraction of cells killed.

        Parameters
        ----------
        use_dataset, bool
            True, this method transforms only experimental conditions mentioned in the data. When False,
            it will do "out of sample" predictions; i.e. doing the transform on experimental conditions in
            simulation result but not in the dataset_fluorescence.
        """
        if use_dataset and 'interpolate' in [x[0] for x in self.process.steps]:
            self._replace_interpolate_step(self.interpolation_ds)
            result_ds = self.process.transform(self.simulation_result_df[self._results_cols])
            self._replace_interpolate_step(self.interpolation)
            self.results = result_ds

        else:  # set 'do_fit_transform' to false to enable out-of-sample predictions.
            original_do_fit_transform_settings = {k: v for k, v in self.process.get_params().items()
                                                  if 'do_fit_transform' in k}
            do_not_do_fit_transform = {k: False for k in self.process.get_params().keys()
                                       if 'do_fit_transform' in k}

            self.process.set_params(**do_not_do_fit_transform)
            self.results = self.process.transform(self.simulation_result_df[self._results_cols])
            self.process.set_params(**original_do_fit_transform_settings)

        return self.results

    def setup(self, use_dataset=True):
        """
        Set-up the measurement model.

        Parameters
        ----------
        use_dataset, bool
            True, this method transforms only experimental conditions mentioned in the data. When False,
            it will do "out of sample" predictions; i.e. doing the transform on experimental conditions in
            simulation result but not in the dataset_fluorescence.
        """
        if use_dataset and 'interpolate' in [x[0] for x in self.process.steps]:
            self._replace_interpolate_step(self.interpolation_ds)
            result_ds = self.process.set_up(self.simulation_result_df[self._results_cols])
            self._replace_interpolate_step(self.interpolation)
            self.results = result_ds

        else:  # set 'do_fit_transform' to false to enable out-of-sample predictions.
            original_do_fit_transform_settings = {k: v for k, v in self.process.get_params().items()
                                                  if 'do_fit_transform' in k}
            do_not_do_fit_transform = {k: False for k in self.process.get_params().keys()
                                       if 'do_fit_transform' in k}

            self.process.set_params(**do_not_do_fit_transform)
            self.results = self.process.set_up(self.simulation_result_df[self._results_cols])
            self.process.set_params(**original_do_fit_transform_settings)

        return self.results

    def run_classification(self, use_dataset=True):
        """
        Run the measurement model and return predictions of cell death vs. survival.

        Parameters
        ----------
        use_dataset, bool
            True, this method transforms only experimental conditions mentioned in the data. When False,
            the predictions will include experimental conditions in the simulation result that are not
            present in the dataset_fluorescence.
        """
        pass

    def likelihood(self, **kwargs):
        """
        Calculates the negative log-likelihood assuming the measured values of fractional killing have a
        beta distribution.
        """
        self.results = self.run(use_dataset=True)

        y_name = self._measured_values_names[0]
        y_sim_name = y_name + '__1'
        # y_err_name = y_name + '__error'
        y_ = self.results.merge(self._dataset.data)

        y = y_.merge(self._dataset.measurement_error_df,
                     on=list(self._dataset_experimental_conditions_df.columns))

        y_sim_ = y[y_sim_name].values
        y_data_ = y[y_name].values
        y_error = y[self._data_error_col_name].values

        y_sim = np.clip(y_sim_, .0001, .999)
        y_data = np.clip(y_data_, .0001, .999)
        measurement_error = np.array(
            np.clip(y_error, 0.0001, 0.5))  # a standard deviation of 0.50 is random chance for fractional data
        phi = np.clip((y_data * (1 - y_data) - measurement_error ** 2) / measurement_error ** 2, 0.0001, np.inf)
        y_data_idx = range(len(y_data))

        return np.sum([-stats.beta.logpdf(
            y_sim[i],
            y_data[i] * phi[i],
            (1 - y_data[i]) * phi[i]) for i in y_data_idx])


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

        # Likelihood: if dataset_fluorescence is none Likelihood is raise error else Likelihood is normal pdf of data and simulation.

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
        else:
            idx = [x for x, y in enumerate(self.process.steps) if y[0] == 'classifier'][0]
            self.process.steps.insert(idx, ('interpolate', self.interpolation))

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
            self.results = self.process.transform(self.simulation_result_df[self._results_cols])
            return self.results

    def _run_process_set_do_fit_transform(self, do_fit_transform, **kwargs):
        original_do_fit_transform_settings = {k: v for k, v in self.process.get_params().items()
                                              if 'do_fit_transform' in k}
        new_do_fit_transform_settings = {k: do_fit_transform for k in self.process.get_params().keys()
                                         if 'do_fit_transform' in k}

        self.process.set_params(**new_do_fit_transform_settings)
        self.results = self.process.transform(self.simulation_result_df[self._results_cols])
        self.process.set_params(**original_do_fit_transform_settings)

        return self.results

    def _run_with_dataset(self, **kwargs):
        use_dataset=kwargs.get('use_dataset', True)

        if use_dataset and 'interpolate' in [x[0] for x in self.process.steps]:
            self._replace_interpolate_step(self.interpolation_ds)
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
        y = y_.merge(self.results,how='outer',on=exp_conditions_cols).drop_duplicates().reset_index(drop=True)

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
        raise ValueError("likelihood requires a dataset_fluorescence")
