# MW Irvin -- Lopez Lab -- 2018-08-24
"""
Suite of Measurement Models
"""
from opt2q.utils import _is_vector_like, _convert_vector_like_to_list
from opt2q.measurement.base.base import MeasurementModel
from opt2q.measurement.base.transforms import Interpolate, LogisticClassifier, Pipeline, Scale, Standardize, SampleAverage
import pandas as pd
import numpy as np


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
        mentioned in the ``simulation_result`` and/or ``dataset`` (if supplied) are ignored.

    time_points: vector-like, optional
        Lists the time-points involved in the measurement. Defaults to the time points in the
        :class:`~pysb.simulator.SimulationResult`. You can also assign time-points using a 'time' in the
        ``experimental_conditions`` argument (as follows).

    experimental_conditions: :class:`~pandas.DataFrame`, optional
        The experimental conditions involved in the measurement model. Defaults to experimental conditions in
        the ``dataset`` (if present) or the ``simulation_result``.

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
                 experimental_conditions=None,):

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
                                                    groupby=list(set(self.experimental_conditions_df.columns)-
                                                                 {'simulation'}), apply_noise=True, variances=0.0)),
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
        Check that measured_values keys are in the dataset.measured_variables.

        Look for observables mentioned in measured_values_dict that are not in self.observables, and add them.
        """
        obs = self.observables  # set
        # Todo: change to: obs = set() if self._observables is None else self._observables  # set
        data_cols = [k for k, v, in dataset.measured_variables.items() if v in ('default', 'ordinal')]

        for k, v in measured_values_dict.items():
            if k not in data_cols: raise ValueError(
                "'measured_values' contains a variable, '{}', not mentioned as an ordinal variable in the 'dataset'."
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
        if use_dataset:
            x_ds = self.interpolation_ds.transform(self.simulation_result_df[self._results_cols])
            result_ds = self.process.transform(x_ds)
            self.results = result_ds
            return self.results

        else:
            current_classifier_do_fit_transform = self.process.get_params()['classifier__do_fit_transform']

            self.process.set_params(**{'classifier__do_fit_transform': False})
            x = self.interpolation.transform(self.simulation_result_df[self._results_cols])
            self.results = self.process.transform(x)

            self.process.set_params(**{'classifier__do_fit_transform': current_classifier_do_fit_transform})
        return self.results

    def likelihood(self, use_all_dataset_obs=True, use_all_dataset_exp_cond=True):
        """
        Return scalar value of the likelihood.

        Parameters
        ----------
        use_all_dataset_obs: bool (optional)
            If observables are supplied via the ``observables`` argument and a :class:`~opt2q.data.DataSet`, the
            likelihood, by default, uses *all* the observables mentioned in the dataset (even if they are absent from
            the ``observables`` argument).

            If False, the likelihood uses only  the subset of observables mentioned in both the DataSet and the
            ``observables`` arguments.

        use_all_dataset_exp_cond: bool (optional)
            If experimental conditions are supplied via the ``experimental_conditions`` argument and a
            :class:`~opt2q.data.DataSet`, the likelihood uses, by default, *all* the experimental conditions mentioned
            in the dataset (even if they are absent from the ``experimental_conditions`` argument).

            If False, the likelihood uses only the subset of observables mentioned in both the DataSet and the
            ``experimental_conditions`` arguments.
        """

        self.results = self.run(use_dataset=True)
        ordinal_errors = self._dataset._ordinal_errors_df
        exp_conditions_cols = list(self._dataset.experimental_conditions.columns)
        columns_for_merge = exp_conditions_cols + ['simulation']

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
        in the dataset can only have values between 0, 1.

    measured_values: dict
        Relate measured variable, key, (as named in the DataSet) to a list of corresponding columns in the ``dataframe``
        or ``opt2q_dataframe`` of the :class:`~pysb.simulator.SimulationResult`. The dict can only have one key (one
        measured variable).

    observables: vector-like, optional
        Lists the names (str) of the PySB :class:`pysb.core.Model` observables and/or species involved in the
        measurement.

        These observables apply to all the experimental conditions involved in the measurement. Observables not
        mentioned in the ``simulation_result`` and/or ``dataset`` (if supplied here) are ignored.

    time_points: vector-like, optional
        Lists the time-points involved in the measurement. Defaults to the time points in the
        :class:`~pysb.simulator.SimulationResult`. You can also assign time-points using a 'time' in the
        ``experimental_conditions`` argument (as follows).

    experimental_conditions: :class:`~pandas.DataFrame`, optional
        The experimental conditions involved in the measurement model. Defaults to experimental conditions in
        the ``dataset`` (if present) or the ``simulation_result``.

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
                 experimental_conditions=None, interpolate_first=True):

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
        self._interpolate_first = False if not interpolate_first else True  # Non-bool Defaults to True

        # Note: Pipeline.transform needs to be updated to look for variations in the name in addition to the name itself
        # Polynomial expansion creates RIP1^2 and RIP1 RIP3 and RIP3^2 from RIP1 and RIP3.
        # But If RIP3p is present, the transform has to ignore it... how?
        # ADD __ to the operation. And say take RIP1 + RIP1__^2, RIP1__ RIP3__,
        # State that in the Transform Function that changes to column names will not be tracked automatically unless the
        # changes include __

        # An alternatively way to track is to look for new columns in x... x in x1 that are not in x0.. those
        # columns can only come from...
        # But this approach would mean that ALL new columns funnel into the next transform (which might not be what
        #   you're wanting to do.

        # Look for word breaks in the observables.. This should work because observables inharently don't have
        # word breaks. Let the delimiter by the space and the __


        self.process = Pipeline(
            steps=[('log_scale', Scale(columns=list(_process_observables), scale_fn='log10')),
                   ('standardize', Standardize(columns=list(_process_observables), groupby=None)),
                   # todo: Make Mock Data for Classifier (Supervised Learning requires labeled dataset)
                   ('classifier', LogisticClassifier(self._dataset.data, column_groups=_measured_values,
                                                     do_fit_transform=False, classifier_type='nominal')),
                   # The `_process_observables` should no longer be in the Dataframe....
                   ('sample_average', SampleAverage(columns=list(_process_observables), drop_columns='simulation',
                                                    groupby=list(set(self.experimental_conditions_df.columns) -
                                                                 {'simulation'}), apply_noise=False))
                   ])
        self._add_interpolate_step()

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
        Check that measured_values has only one key, and is in the dataset.measured_variables.

        Check that the measured_values have values between 0 and 1.

        Look for observables mentioned in measured_values_dict that are not in self.observables, and add them.
        """
        if len(measured_values_dict) > 1:
            raise ValueError("'measured_values' cannot have multiple keys.")

        obs = self.observables  # set
        # Todo: change to: obs = set() if self._observables is None else self._observables  # set
        data_cols = [k for k, v, in dataset.measured_variables.items() if v is not 'ordinal']

        for k, v in measured_values_dict.items():
            if k not in data_cols: raise ValueError(
                "'measured_values' contains a variable, '{}', not mentioned as in the 'dataset'."
                .format(k))
            if max(dataset.data[k]) > 1 or min(dataset.data[k]) < 0:
                raise ValueError("The variable, '{}', in the 'dataset', can only have values between 0.0 and 1.0".format(k))

            if _is_vector_like(v):  # look for the v in the default_observables
                measured_obs = _convert_vector_like_to_list(v)
                for i in measured_obs:  # if v is an observable mentioned in the
                    if isinstance(i, str):
                        mentioned_obs = i.split('__')[0]
                        obs |= {mentioned_obs} if mentioned_obs in self._default_observables else set()
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
            the predictions will include experimental conditions in the simulation result that are not
            present in the dataset.
        """
        if use_dataset:
            self._replace_interpolate_step(self.interpolation_ds)
            result_ds = self.process.transform(self.simulation_result_df[self._results_cols])
            self._replace_interpolate_step(self.interpolation)

            self.results = result_ds
            return self.results

        else:
            current_classifier_do_fit_transform = self.process.get_params()['classifier__do_fit_transform']

            self.process.set_params(**{'classifier__do_fit_transform': False})
            self.results = self.process.transform(self.simulation_result_df[self._results_cols])

            self.process.set_params(**{'classifier__do_fit_transform': current_classifier_do_fit_transform})
        return self.results

    def run_classification(self, use_dataset=True):
        """
        Run the measurement model and return predictions of cell death vs. survival.

        Parameters
        ----------
        use_dataset, bool
            True, this method transforms only experimental conditions mentioned in the data. When False,
            the predictions will include experimental conditions in the simulation result that are not
            present in the dataset.
        """
        pass


class Fluorescence(MeasurementModel):
    """
    Simulates Fluorescence Measurements.

    Conducts a series of transformations on a :class:`~pysb.simulator.SimulationResult` to represent attributes of the
    Fluorescence (semi-quantitative) measurement.

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
        A dictionary of (keys) measured variables (as named in the DataSet) and a list of corresponding PySB model observables.

    observables: vector-like, optional
        Lists the names (str) of the PySB PySB :class:`pysb.core.Model` observables and/or species involved in the
        measurement.

        These observables apply to all the experimental conditions involved in the measurement. Observables not
        mentioned in the ``simulation_result`` and/or ``dataset`` (if supplied) are ignored.

    time_points: vector-like, optional
        Lists the time-points involved in the measurement. Defaults to the time points in the
        :class:`~pysb.simulator.SimulationResult`. You can also assign time-points using a 'time' in the
        ``experimental_conditions`` argument (as follows).

    experimental_conditions: :class:`~pandas.DataFrame`, optional
        The experimental conditions involved in the measurement model. Defaults to experimental conditions in
        the ``dataset`` (if present) or the ``simulation_result``.

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
                 experimental_conditions=None):

        super().__init__(simulation_result,
                         dataset=dataset,
                         observables=observables,
                         time_points=time_points,
                         experimental_conditions=experimental_conditions)

        # Note self.observables can be user defined or all mentioned in the PySB model.
        # - DS - MV
        #   measured_values = self.observables
        #   process observables = self.observables (defaults to ALL obs in model)
        #   likelihood = Raise Error

        # - DS + MV
        #   measured_values = arg: measured_values.keys()
        #   process observables = arg: measured_values.values() + whatever is in self._observables (or {}, dft)
        #   likelihood = Raise Error

        # + DS - MV
        #   measured_values = dataset.measured_variables.keys() if not 'ordinal' and in self._default_observables
        #   process observables = the same as measured_values
        #   likelihood = _likelihood

        # + DS + MV
        #   measured_values = measured_values.keys() (Use the same function as above)
        #   process observables = self.observables
        #   likelihood = _likelihood

        # ---- which experimental conditions? ----
        # Differs if the DataSet is present or not.

        # ---- Likelihood ----
        # Raise and error if DataSet is None.


