# MW Irvin -- Lopez Lab -- 2018-08-23
from opt2q.utils import _is_vector_like, _convert_vector_like_to_set, _convert_vector_like_to_list, _list_the_errors
from opt2q.measurement.base.transforms import Transform
from opt2q.data import DataSet as Opt2qDataSet
import pandas as pd
import numpy as np
import warnings


class MeasurementModel(object):
    """
    Base class for the measurement model.

    This base class interprets the PySB :class:`~pysb.simulator.SimulationResult`, :class:`~opt2q.data.DataSet` and/or
    user-inputs to define important attributes of the measurement model.

    Parameters
    ----------
    simulation_result: :class:`~pysb.simulator.SimulationResult`
        Results of a `PySB` Model simulation. Produced by the run methods in the `PySB`
        (:meth:`~pysb.simulator.ScipyOdeSimulator.run`) and `Opt2Q` (:meth:`~opt2q.simulator.Simulator.run`)
        simulators.

    dataset: :class:`~opt2q.data.DataSet`, optional
        Measured values and associated attributes, e.g. 'observables', 'experimental_conditions', that dictate what
        rows of the simulation_result should apply to the measurement model.

        The :meth:`opt2q.measurement.base.likelihood` method *requires" the :class:`~opt2q.data.DataSet`, and will refer
        to the DataSet's attributes, even when such attributes were defined manually via ``time_points``,
        ``observables``, ``experimental_conditions`` arguments.

        .. note::
            The DataSet's experimental_conditions columns must be the same as those in the
            :class:`~pysb.simulator.SimulationResult`

    observables: vector-like, optional
        Lists the names (str) of the PySB model observables and/or species involved in the measurement.

        These observables apply to all the experimental conditions involved in the measurement. Observables not
        mentioned in the ``simulation_result`` and/or ``dataset_fluorescence`` (if supplied) are ignored.

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

    """
    def __init__(self, simulation_result,  dataset=None, observables=None, time_points=None,
                 experimental_conditions=None, time_dependent=True, **kwargs):

        self._time_dependent = True if time_dependent is not False else False  # Defaults to True

        opt2q_df, pysb_df = self._check_simulation_result(simulation_result)
        exp_con_df, exp_con_cols = self._get_ec_from_sim_result(opt2q_df, pysb_df)  # df and cols include 'time'

        self._dataset = self._check_dataset(dataset)  # returns None or an Opt2Q DataSet

        self._default_observables, \
            self._dataset_observables,\
            self._observables = self._get_observables(pysb_df, self._dataset, observables)

        self._required_observables = self._get_required_observables(self._observables, self._dataset_observables)

        if _is_vector_like(time_points):
            self._time_points = _convert_vector_like_to_list(time_points)  # create only if user-specifies

        self._default_time_points = self._get_default_time_points(time_points, self._dataset, pysb_df)

        # keep dfs w/ nans for easy updating with updates to default time-points.
        self._dataset_ec_df_pre = self._get_experimental_conditions_from_dataset(
            exp_con_df, exp_con_cols, dataset)
        self._ec_df_pre = self._get_experimental_conditions_from_user(
            exp_con_df, exp_con_cols, self._dataset_ec_df_pre, experimental_conditions)
        if self._time_dependent:
            self._dataset_experimental_conditions_df = self._add_default_time_points_to_experimental_conditions(
                self._dataset_ec_df_pre, self._default_time_points)
            self._experimental_conditions_df = self._add_default_time_points_to_experimental_conditions(
                self._ec_df_pre, self._default_time_points
            )
        else:
            self._dataset_experimental_conditions_df = self._dataset_ec_df_pre
            self._experimental_conditions_df = self._ec_df_pre

        self._opt2q_df = opt2q_df
        # Todo: Add @property and @...setter for simulation_result, dataset_fluorescence!!
        # Todo: Add @property and @...setter for observables, time_points, experimental_conditions!!

    @staticmethod
    def _check_simulation_result(simulation_result_obj):
        """
        Gets pysb and opt2q results dataframes from simulation result.
        """
        try:
            pysb_df = simulation_result_obj.dataframe.copy()
            opt2q_df = getattr(simulation_result_obj, 'opt2q_dataframe', pysb_df).copy()
        except AttributeError:
            raise ValueError('simulation_result must be a PySB or Opt2Q SimulationResult')

        pysb_df.reset_index(inplace=True)
        opt2q_df.reset_index(inplace=True)
        if 'simulation' not in pysb_df.columns:
            pysb_df['simulation'] = 0
            opt2q_df['simulation'] = 0
        return opt2q_df, pysb_df

    def _get_ec_from_sim_result(self, opt2q_df, pysb_df):
        """
        Get experimental conditions dataframe from simulation result and make the 'time' column is NaNs.

        The NaNs will be replaced by default time-points once they are established.

        Return
        ------
        :class:`~pandas.DataFrame`
            Simulation Result's Experimental Conditions Dataframe
        set
            DataFrame columns
        """
        ec_cols = set(opt2q_df.columns) - set(pysb_df.columns)
        list_ec_cols = list(ec_cols)
        default_ec_df = opt2q_df[list_ec_cols]
        if default_ec_df.shape[1] == 0:
            if self._time_dependent:
                return pd.DataFrame([np.NaN], columns=['time']), ec_cols | {'time'}
            else:
                return pd.DataFrame(), ec_cols
        else:
            default_ec_df = default_ec_df.drop_duplicates().reset_index(drop=True)
            if not self._time_dependent:
                return default_ec_df, ec_cols
            else:
                default_ec_df['time'] = np.NaN
                return default_ec_df, ec_cols | {'time'}

    @staticmethod
    def _check_dataset(ds):
        if ds is None or isinstance(ds, Opt2qDataSet):
            return ds
        else:
            raise ValueError("'dataset' must be an Opt2Q DataSet.")

    def _get_observables(self, pysb_df, dataset=None, observables=None):
        """
        Returns list of observables' names

        Takes observables from ``observables`` or ``dataset_fluorescence`` arguments, if not None.
        If both are supplied, observables are the intersection of ``observables`` and ``dataset_fluorescence``.

        Defaults to all observables mentioned in the ``simulation_result``.
        """
        default_observables = set(pysb_df.columns)-{'time', 'simulation'}

        if dataset is not None:
            dataset_obs = self._get_obs_from_dataset(dataset, default_observables)
        else:
            dataset_obs = None

        if observables is not None:
            user_obs = self._check_observables(observables, default_observables)
            if dataset_obs is not None and len(user_obs-dataset_obs) > 0 and len(dataset_obs) > 0:
                # The likelihood uses dataset_obs or the intersection of dataset_obs and user_obs.
                warnings.warn("The 'observables' contain values not mentioned in the dataset_fluorescence. "
                              "They will be ignored in the likelihood calculation.")
        else:
            user_obs = None

        return default_observables, dataset_obs, user_obs

    @staticmethod
    def _get_required_observables(user_obs, dataset_obs):
        """Any simulation-result update must have these observables"""
        req_obs = set([])
        for obs_set in (user_obs, dataset_obs):
            if obs_set is not None:
                req_obs |= obs_set
        return req_obs

    @staticmethod
    def _get_obs_from_dataset(dataset, default_observables):
        """Returns observables in dataset_fluorescence that are also in the simulation result"""
        dataset_obs = set(dataset.observables)
        if len(dataset_obs - default_observables) > 0:
            warnings.warn('The supplied dataset_fluorescence has observables not present in the simulation result. '
                          'They will be ignored.')
        dataset_obs = dataset_obs.intersection(default_observables)
        return dataset_obs

    @staticmethod
    def _check_observables(observables, default_observables):
        """
        Check that 'observables' is vector-like and remove observables not present in the simulation_result.

        Parameters
        ----------
        observables: vector-like
        default_observables: set
        """
        if _is_vector_like(observables):
            user_obs = _convert_vector_like_to_set(observables)
            if len(user_obs - default_observables) > 0:
                warnings.warn("Observables not present in the simulation result will be ignored.")
            return user_obs.intersection(default_observables)
        else:
            raise ValueError('observables must be vector-like')

    @staticmethod
    def _get_default_time_points(times, dataset, sim_res_df):
        if _is_vector_like(times):
            return _convert_vector_like_to_list(times)
        elif dataset is not None and set(dataset.experimental_conditions.columns) == {'time'}:
            return _convert_vector_like_to_list(dataset.experimental_conditions)
        else:
            return _convert_vector_like_to_list(sim_res_df['time'].unique())

    def _get_experimental_conditions_from_dataset(self, sim_result_ec_df, sim_result_ec_cols_set, dataset):
        """Returns the following DataFrames of experimental conditions and time-points:

         - dataset_ec_df: None or a Dataframe with times and/or experimental conditions from the DataSet.
         - user_ec_df: DataFrame with times and experimental conditions taken from user inputs or the simulation result.
         - ec: DataFrame of experimental conditions taken first from user, then dataframe or finally simulation result.

         We need to keep the NaN here because if the user seeks to update the time_points, the experimental conditions
         that are using default time-points would have to be updated (these experimental conditions have to be
         remembered).
        """
        if dataset is None:
            return None  # Likelihood requires dataset_fluorescence.
        else:
            dataset_ec_df = dataset.experimental_conditions
            dataset_ec_cols = dataset_ec_df.columns
            dataset_ec_cols_set = set(dataset_ec_cols) | {'time'} if self._time_dependent else set(dataset_ec_cols)
            self._check_that_cols_match(dataset_ec_cols_set, sim_result_ec_cols_set)

        if set(dataset_ec_cols) == {'time'}:
            return dataset_ec_df  # vector-like DataFrame

        elif 'time' not in dataset_ec_cols and self._time_dependent:
            dataset_ec_df = self._add_time_column_w_nans(dataset_ec_df)

        merge_on_cols = list(set(dataset_ec_cols) - {'time'})
        merge_dataset_ec_df = self._intersect_w_sim_result_ec(dataset_ec_df, sim_result_ec_df, dataset_ec_cols,
                                                              on=merge_on_cols,
                                                              var_name='DataSet.experimental_conditions')
        return merge_dataset_ec_df[list(sim_result_ec_cols_set)]

    def _get_experimental_conditions_from_user(self, sim_result_ec_df, sim_result_ec_cols_set, dataset_ec_df, usr_ec):
        if usr_ec is None and dataset_ec_df is None:
            user_ec_df = sim_result_ec_df[list(sim_result_ec_cols_set-{'time'})]
            if self._time_dependent:
                user_ec_df = self._add_time_column_w_nans(user_ec_df)
            return user_ec_df
        elif usr_ec is None:
            return dataset_ec_df

        user_ec_cols = set(usr_ec.columns)
        if len(user_ec_cols - sim_result_ec_cols_set) > 0:
            raise ValueError("The following experimental conditions columns are not in the simulation "
                             "result, and cannot be used: " + _list_the_errors(user_ec_cols - sim_result_ec_cols_set))
        if 'time' not in usr_ec.columns and self._time_dependent:
            usr_ec = self._add_time_column_w_nans(usr_ec)
        merge_on_cols = list(user_ec_cols - {'time'})
        merge_ec_df = self._intersect_w_sim_result_ec(usr_ec, sim_result_ec_df, list(sim_result_ec_cols_set),
                                                      on=merge_on_cols)
        return merge_ec_df[list(sim_result_ec_cols_set)]

    @staticmethod
    def _add_time_column_w_nans(df):
        df = df.assign(time=pd.Series(np.full(max(len(df), 1), np.NaN)))

        # if df.shape[0] is 0:
        #     df = df.assign(time=pd.Series(np.full(max(len(df), 1), np.NaN)))
        #     # df['time'] = [np.NaN]
        # else:
        #     df = df.assign(time=pd.Series(np.full(len(df), np.NaN)))
        #     # df['time'] = np.NaN
        return df

    @staticmethod
    def _intersect_w_sim_result_ec(ec_df, sim_res_ec_df, sim_res_ec_cols,
                                   var_name='experimental_conditions', how='inner', suffixes=('', '_y'), **kw):
        """
        Trim DataFrame to contain only rows present in the simulation result. Show warning if dropped rows.

        Parameters
        ----------
        ec_df: Test DataFrame
        sim_res_ec_df: Simulation Result DataFrame (Experimental Conditions' columns only). No duplicate rows.
        sim_res_ec_cols: (list) Experimental Conditions' columns
        var_name: (str) 'experimental_conditions' or 'DataSet.experimental_conditions'
        """
        on = kw.get('on', sim_res_ec_cols)
        if len(on) == 0:
            return ec_df
        trimmed_df = pd.merge(ec_df, sim_res_ec_df, how=how, on=on, suffixes=suffixes)
        trimmed_df = trimmed_df.drop_duplicates().reset_index(drop=True)

        if trimmed_df[on].shape[0] < ec_df[on].drop_duplicates().shape[0] and kw.get('show_warning', True):
            warnings.warn(
                "The '{}' dataframe contains rows that are not present in the simulation result".format(var_name)
            )
        return trimmed_df

    @staticmethod
    def _check_that_cols_match(test_exp_cols, ref_exp_cols, test_name='DataSet.experimental_conditions',
                               ref_name='simulation_result'):
        if ref_exp_cols != test_exp_cols:
            raise ValueError("'{}' column names ({}) must match those in the '{}', ({}).".
                             format(test_name, test_exp_cols, ref_name, ref_exp_cols))

    @staticmethod
    def _add_default_time_points_to_experimental_conditions(df, default_times, column='time'):
        """
        Replaces NaN in the 'time' column of the dataframe with the measurement model's default time-points.
        """
        if df is None:
            return df

        len_default = len(default_times)
        idx_of_nans = df[df[column].isna()].index
        df_w_nans = df.loc[idx_of_nans.repeat(len_default)].drop(columns=[column])
        df_w_nans[column] = np.tile(default_times, len(idx_of_nans))

        df_wo_nans = df[~df[column].isna()]
        return pd.concat([df_wo_nans, df_w_nans], ignore_index=True, sort=False)

    @property
    def time_points(self):
        """Return list of *default* time points."""
        return getattr(self, '_time_points', self._default_time_points)

    @time_points.setter
    def time_points(self, tps):
        if _is_vector_like(tps):
            self._default_time_points = _convert_vector_like_to_list(tps)
            self._time_points = _convert_vector_like_to_list(tps)
        else:
            raise ValueError("'time_points' must be vector-like")
        self._dataset_experimental_conditions_df = self._add_default_time_points_to_experimental_conditions(
            self._dataset_ec_df_pre, self.time_points)
        self._experimental_conditions_df = self._add_default_time_points_to_experimental_conditions(
            self._ec_df_pre, self.time_points
        )

    @property
    def observables(self):
        if self._observables is not None:
            return self._observables
        else:
            return self._default_observables

    @property
    def experimental_conditions_df(self):
        return self._experimental_conditions_df

    @property
    def simulation_result_df(self):
        """Returns the opt2q dataframe of the simulation result."""
        return self._opt2q_df

    @simulation_result_df.setter
    def simulation_result_df(self, v):
        self._opt2q_df = self._update_sim_res_df(v)

    def update_simulation_result(self, sim_res):
        """
        Update the measurement model's :class:`~pysb.simulator.SimulationResult`. This is useful to the calibrator which
        updates generates a new :class:`SimulationResults <pysb.simulator.SimulationResult>` as it optimizes the PySB
        parameters.

        .. note::
            These updates cannot change the experimental conditions, and must retain all observables mentioned in
            ``observables`` and those in the ``dataset_fluorescence``.

        Parameter
        ---------
        sim_res: :class:`~pysb.simulator.SimulationResult`
            Results of a `PySB` Model simulation. Produced by the run methods in the `PySB`
            (:meth:`~pysb.simulator.ScipyOdeSimulator.run`) and `Opt2Q` (:meth:`~opt2q.simulator.Simulator.run`)
            simulators.
        """
        self._update_simulation_result(sim_res)

    def _update_simulation_result(self, sim_res):
        opt2q_df, pysb_df = self._check_simulation_result(sim_res)
        self._opt2q_df = self._update_sim_res_df(opt2q_df)

    def _update_sim_res_df(self, sim_res_df):
        if 'time' not in sim_res_df.columns:
            sim_res_df.reset_index(inplace=True)

        if len(self._required_observables - set(sim_res_df.columns)) != 0:
            raise ValueError("This simulation result is missing the following required observables: " +
                             _list_the_errors(list(self._required_observables - set(sim_res_df.columns))))

        # todo: check that the experimental conditions in dataset_fluorescence and ec are the ones in the updated sim_res.

        if not hasattr(self, '_time_points') and self._time_dependent:
            # update default_time_points because this will overwrite NaN dataset_ec_df and ec_df
            self._default_time_points = self._get_default_time_points(None, self._dataset, sim_res_df)

            self._dataset_experimental_conditions_df = self._add_default_time_points_to_experimental_conditions(
                self._dataset_ec_df_pre, self._default_time_points)
            self._experimental_conditions_df = self._add_default_time_points_to_experimental_conditions(
                self._ec_df_pre, self._default_time_points
            )
        return sim_res_df

    def _trim_sim_results_to_have_only_rows_in_experimental_conditions_df(self, use_dataset=True):
        ec_df = self._dataset_experimental_conditions_df if use_dataset else self.experimental_conditions_df
        ec_cols = list(set(self.experimental_conditions_df.columns) - {'time', 'simulation'})
        if len(ec_cols) > 0:
            sim_results = self._intersect_w_sim_result_ec(ec_df[ec_cols], self.simulation_result_df, ec_cols)
        else:
            sim_results = self.simulation_result_df
        return sim_results

    def likelihood(self, use_all_dataset_obs=True, use_all_dataset_exp_cond=True):
        """
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

        Returns
        -------
        float

        """
        pass

    def run(self):
        """
        Simulates the measurement.

        Runs the process
        """
        pass

    def process(self):
        """
        Series of transformations to the simulation result that model the experimental measurement process.
        """
        pass