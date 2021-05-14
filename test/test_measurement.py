# MW Irvin -- Lopez Lab -- 2018-08-23
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule
from pysb.bng import generate_equations
from pysb.testing import *
from opt2q.simulator import Simulator
from opt2q.measurement.base import MeasurementModel, Scale
from opt2q.data import DataSet
import pandas as pd
import unittest
import warnings
np.random.seed(10)


class TestSolverModel(object):
    @with_model
    def setUp(self):
        Monomer('A', ['a'])
        Monomer('B', ['b'])

        Parameter('ksynthA', 100)
        Parameter('ksynthB', 100)
        Parameter('kbindAB', 100)

        Parameter('A_init', 0)
        Parameter('B_init', 0)

        Initial(A(a=None), A_init)
        Initial(B(b=None), B_init)

        Observable("A_free", A(a=None))
        Observable("B_free", B(b=None))
        Observable("AB_complex", A(a=1) % B(b=1))

        Rule('A_synth', None >> A(a=None), ksynthA)
        Rule('B_synth', None >> B(b=None), ksynthB)
        Rule('AB_bind', A(a=None) + B(b=None) >> A(a=1) % B(b=1), kbindAB)

        self.model = model

        # Convenience shortcut for accessing model monomer objects
        self.mon = lambda m: self.model.monomers[m]
        generate_equations(self.model)

        # Hack to prevent weird fails after assertDictEqual is called
        self.test_non_opt2q_params = None
        self.test_non_opt2q_params_df = None

        self.sim = Simulator(self.model)
        self.sim.param_values = pd.DataFrame([[100, 'WT', 1],
                                              [100, 'KO', 1],
                                              [30, 'DKO', 2]],
                                             columns=['kbindAB', 'condition', 'experiment'])

        self.sim_result = self.sim.run(tspan=np.linspace(0, 10, 3),
                                       initials=pd.DataFrame({self.model.species[1]: [100, 0, 0],
                                                              'condition': ['WT', 'KO', 'DKO'],
                                                              'experiment': [1, 1, 2]}))
        self.sim_result_2 = self.sim.run(tspan=np.linspace(0, 8, 3),
                                         initials=pd.DataFrame({self.model.species[1]: [100, 0, 0],
                                                                'condition': ['WT', 'KO', 'DKO'],
                                                                'experiment': [1, 1, 2]}))

    def tearDown(self):
        self.model=None
        self.mon=None
        self.test_non_opt2q_params = None
        self.test_non_opt2q_params_df = None


class TestMeasurementModel(TestSolverModel, unittest.TestCase):
    def test_check_simulation_result(self):
        with self.assertRaises(ValueError) as error:
            mm = MeasurementModel(self.sim_result)
            mm._check_simulation_result("simulation_result")
        self.assertTrue(error.exception.args[0] == 'simulation_result must be a PySB or Opt2Q SimulationResult')

    def test_check_simulation_result_2(self):
        mm = MeasurementModel(self.sim_result)
        test1, test2 = mm._check_simulation_result(self.sim_result)
        target1 = self.sim_result.opt2q_dataframe.copy()
        target1.reset_index(inplace=True)
        target2 = self.sim_result.dataframe.copy()
        target2.reset_index(inplace=True)
        pd.testing.assert_frame_equal(test1, target1)
        pd.testing.assert_frame_equal(test2, target2)

    def test_check_dataset(self):
        ds = DataSet(pd.DataFrame(), [])
        mm = MeasurementModel(self.sim_result)
        mm._check_dataset(ds)

    def test_check_dataset_bad_input(self):
        with self.assertRaises(ValueError) as error:
            mm = MeasurementModel(self.sim_result)
            mm._check_dataset('ds')
        self.assertTrue(error.exception.args[0] == "'dataset' must be an Opt2Q DataSet.")

    def test_check_observables(self):
        mm = MeasurementModel(self.sim_result)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test = mm._check_observables({'AB_complex', 'Nonexistent_Obs'}, mm._default_observables)
            assert str(w[0].message) == "Observables not present in the simulation result will be ignored."
        target = {'AB_complex'}
        self.assertSetEqual(test, target)

    def test_check_observables_bad_input(self):
        with self.assertRaises(ValueError) as error:
            mm = MeasurementModel(self.sim_result)
            mm._check_observables(DataSet, mm._default_observables)
        self.assertTrue(error.exception.args[0] == 'observables must be vector-like')

    def test_get_observables(self):
        ds = DataSet(pd.DataFrame(), [])
        ds.observables = ['AB_complex', 'AB_complex']
        ds.experimental_conditions=pd.DataFrame([['WT', 1, 0.0],
                                                 ['KO', 1, 0.0],
                                                 ['DKO', 2, 0.0]],
                                                columns=['condition', 'experiment', 'time'])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mm = MeasurementModel(self.sim_result, dataset=ds, observables=['A_free', 'AB_complex'])
        assert str(w[-1].message) == "The 'observables' contain values not mentioned in the dataset_fluorescence. " \
                                     "They will be ignored in the likelihood calculation."
        target = ({'AB_complex'}, {'A_free', 'AB_complex'})
        test = (mm._dataset_observables, mm._observables)
        for i, xi in enumerate(target):
            self.assertSetEqual(xi, test[i])
        self.assertSetEqual(mm._required_observables, {'A_free', 'AB_complex'})

    def test_get_default_time_points_user_spec(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([2, 4, 6], columns=['time'])
        o2_df, ps_df = mm._check_simulation_result(self.sim_result)
        test = mm._get_default_time_points([1, 2, 3], ds, ps_df)
        target = [1, 2, 3]
        self.assertListEqual(test, target)

    def test_get_default_time_points_dataset(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([2, 4, 6], columns=['time'])
        o2_df, ps_df = mm._check_simulation_result(self.sim_result)
        test = mm._get_default_time_points(None, ds, ps_df)
        target = [2, 4, 6]
        self.assertListEqual(test, target)

    def test_get_default_time_points_dataset_None(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([2, 4, 6], columns=['time'])
        o2_df, ps_df = mm._check_simulation_result(self.sim_result)
        test = mm._get_default_time_points(None, None, ps_df)
        target = list(np.linspace(0, 10, 3))
        self.assertListEqual(test, target)

    def test_get_default_time_points_dataset_df_w_ec(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([2, 4, 6], columns=['ec'])
        o2_df, ps_df = mm._check_simulation_result(self.sim_result)
        test = mm._get_default_time_points(None, ds, ps_df)
        target = list(np.linspace(0, 10, 3))
        self.assertListEqual(test, target)

    def test_check_that_cols_match(self):
        with self.assertRaises(ValueError) as error:
            mm = MeasurementModel(self.sim_result)
            mm._check_that_cols_match({'condition'}, {'experiment', 'condition'})
        self.assertTrue(error.exception.args[0] ==
                        "'DataSet.experimental_conditions' column names ({'condition'}) must match those in "
                        "the 'simulation_result', ({'condition', 'experiment'})." or
                        error.exception.args[0] ==
                        "'DataSet.experimental_conditions' column names ({'condition'}) must match those in "
                        "the 'simulation_result', ({'experiment', 'condition'})."
                        )

    def test_get_experimental_conditions_from_dataset_ds_None(self):
        # If dataset_fluorescence is None return None and ec_cols is None.
        mm = MeasurementModel(self.sim_result)
        test = mm._get_experimental_conditions_from_dataset(pd.DataFrame(), set([]), None)
        assert test is None

    def test_get_experimental_conditions_from_dataset_ds_empty_df(self):
        # Add a time axis with just NaNs.
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame()
        test = mm._get_experimental_conditions_from_dataset(pd.DataFrame([1, 2, 3], columns=['time']), {'time'}, ds)
        pd.testing.assert_frame_equal(pd.DataFrame([np.NaN], columns=['time']), test)

    def test_get_experimental_conditions_from_dataset_ds_no_time_axis(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([1, 2, 3], columns=['a'])
        test = mm._get_experimental_conditions_from_dataset(
            pd.DataFrame([[1, 2],
                          [2, 2],
                          [3, 3]],
                         columns=['a', 'time']),
            {'time', 'a'},
            ds)
        pd.testing.assert_frame_equal(pd.DataFrame([[1, np.NaN],
                                                    [2, np.NaN],
                                                    [3, np.NaN]],
                                                   columns=['a', 'time']),
                                      test[['a', 'time']])

    def test_get_experimental_conditions_from_dataset_ds_drop_rows(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([[1, 2],
                                                   [2, 2]],
                                                  columns=['a', 'time'])
        with warnings.catch_warnings(record=True) as w:
            test = mm._get_experimental_conditions_from_dataset(
                pd.DataFrame([[2, 2],
                              [3, 3]],
                             columns=['a', 'time']),
                {'time', 'a'}, ds)
        assert str(w[-1].message) == "The 'DataSet.experimental_conditions' dataframe contains rows that " \
                                     "are not present in the simulation result"
        pd.testing.assert_frame_equal(pd.DataFrame([[2, 2]],
                                                   columns=['a', 'time']),
                                      test[['a', 'time']])

    def test_get_experimental_conditions_from_dataset_ds_has_time_axis_only(self):
        mm = MeasurementModel(self.sim_result)
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame([1, 2, 3], columns=['time'])
        test = mm._get_experimental_conditions_from_dataset(pd.DataFrame(columns=['time']), {'time'}, ds)
        target = pd.DataFrame([1, 2, 3], columns=['time'])
        pd.testing.assert_frame_equal(test, target)

    def test_get_experimental_conditions_from_user_ec_None(self):
        # ec is None
        sim_res_df = pd.DataFrame([[1, 2, 0.1], [2, 2, 0.2], [3, 2, 0.1], [4, 2, 0.2]], columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = None
        user_df = None
        target = pd.DataFrame([[1, 2, np.NaN], [2, 2,  np.NaN], [3, 2,  np.NaN], [4, 2,  np.NaN]],
                              columns=['ec1', 'ec2', 'time'])
        mm = MeasurementModel(self.sim_result)
        test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        pd.testing.assert_frame_equal(test[['ec1', 'ec2', 'time']], target[['ec1', 'ec2', 'time']], check_dtype=False)

    def test_get_experimental_conditions_from_user_ec_None_empty_sim_results_df(self):
        # ec is None and sim_results has no additional experimental conditions cols.
        sim_res_df = pd.DataFrame()
        sim_res_cols = {'time'}
        dataset_df = None
        user_df = None
        target = pd.DataFrame([np.NaN], columns=['time'])
        mm = MeasurementModel(self.sim_result)
        test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_get_experimental_conditions_from_user_ec_None_use_DataSet(self):
        # ec is None but dataset_fluorescence is not. Use dataset_fluorescence
        sim_res_df = pd.DataFrame(columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = pd.DataFrame([[1, 2, np.NaN], [2, 2, np.NaN], [4, 2, np.NaN]], columns=['ec1', 'ec2', 'time'])
        user_df = None
        target = pd.DataFrame([[1, 2, np.NaN], [2, 2, np.NaN], [4, 2, np.NaN]], columns=['ec1', 'ec2', 'time'])

        mm = MeasurementModel(self.sim_result)
        test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        pd.testing.assert_frame_equal(test[['ec1', 'ec2', 'time']], target[['ec1', 'ec2', 'time']], check_dtype=False)

    def test_get_experimental_conditions_from_user_ec_has_forbidden_cols(self):
        # ec has extra columns; not in the sim_result
        sim_res_df = pd.DataFrame(columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = pd.DataFrame([[1, 2, np.NaN], [2, 2, np.NaN], [4, 2, np.NaN]], columns=['ec1', 'ec2', 'time'])
        user_df = pd.DataFrame([[1, 2, np.NaN]], columns=['ec3', 'ec4', 'time'])

        mm = MeasurementModel(self.sim_result)
        with self.assertRaises(ValueError) as error:
            mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        self.assertTrue(error.exception.args[0] ==
                        "The following experimental conditions columns are not in the simulation result,"
                        " and cannot be used: 'ec3' and 'ec4'" or
                        "The following experimental conditions columns are not in the simulation result, "
                        "and cannot be used: 'ec4' and 'ec3'")

    def test_get_experimental_conditions_from_user_ec_has_no_time_axis(self):
        # ec has ec-columns but no time axis. Add a bunch of Nans for it.
        sim_res_df = pd.DataFrame([[1, 2, 0.0], [2, 2, 0.0], [4, 2, 0.0]], columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = pd.DataFrame()
        user_df = pd.DataFrame([[1, 2], [2, 2], [4, 2]], columns=['ec1', 'ec2'])
        target = pd.DataFrame([[1, 2, np.NaN], [2, 2, np.NaN], [4, 2, np.NaN]], columns=['ec1', 'ec2', 'time'])

        mm = MeasurementModel(self.sim_result)
        test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        pd.testing.assert_frame_equal(test[['ec1', 'ec2', 'time']], target[['ec1', 'ec2', 'time']], check_dtype=False)

    def test_get_experimental_conditions_from_user_ec_has_not_all_ec_cols(self):
        # ec has less the total cols. They should be added. Don't drop any rows. Keep time points in EC.
        sim_res_df = pd.DataFrame([[1, 2, 0.0], [2, 2, 0.0], [4, 2, 0.0]], columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = pd.DataFrame()
        user_df = pd.DataFrame([[1, 2], [2, 2], [4, np.NaN]], columns=['ec1', 'time'])
        target = pd.DataFrame([[1, 2, 2], [2, 2, 2], [4, 2, np.NaN]], columns=['ec1', 'ec2', 'time'])

        mm = MeasurementModel(self.sim_result)
        test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        pd.testing.assert_frame_equal(test[['ec1', 'ec2', 'time']], target[['ec1', 'ec2', 'time']], check_dtype=False)

    def test_get_experimental_conditions_from_user_ec_has_not_all_ec_cols_drop_rows_and_warn(self):
        # ec has less than the total cols. They should be added.
        # Drop the [5, NaN] row and warn that its not in the sim-results
        sim_res_df = pd.DataFrame([[1, 2, 0.0], [2, 2, 0.0], [4, 2, 0.0]], columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = pd.DataFrame()
        user_df = pd.DataFrame([[1, 2], [2, 2], [5, np.NaN]], columns=['ec1', 'time'])
        target = pd.DataFrame([[1, 2, 2], [2, 2, 2]], columns=['ec1', 'ec2', 'time'])

        mm = MeasurementModel(self.sim_result)

        with warnings.catch_warnings(record=True) as w:
            test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        assert str(w[-1].message) == "The 'experimental_conditions' dataframe contains rows that " \
                                     "are not present in the simulation result"

        pd.testing.assert_frame_equal(test[['ec1', 'ec2', 'time']], target[['ec1', 'ec2', 'time']], check_dtype=False)
        self.assertSetEqual(set(test.columns), set(target.columns))

    def test_get_experimental_conditions_from_user_ec_has_not_all_ec_cols_less_than_total_number_of_rows(self):
        # ec has less than the total cols. They should be added.
        # ec has less than the total rows. These absent rows are left out, without warning.
        sim_res_df = pd.DataFrame([[1, 2, 0.0], [2, 2, 0.0], [4, 2, 0.0]], columns=['ec1', 'ec2', 'time'])
        sim_res_cols = {'ec1', 'ec2', 'time'}
        dataset_df = pd.DataFrame()
        user_df = pd.DataFrame([[1, 2], [2, 2]], columns=['ec1', 'time'])
        target = pd.DataFrame([[1, 2, 2], [2, 2, 2]], columns=['ec1', 'ec2', 'time'])

        mm = MeasurementModel(self.sim_result)

        with warnings.catch_warnings(record=True) as w:
            test = mm._get_experimental_conditions_from_user(sim_res_df, sim_res_cols, dataset_df, user_df)
        assert len(w) == 0  # i.e. No warning raised

        pd.testing.assert_frame_equal(test[['ec1', 'ec2', 'time']], target[['ec1', 'ec2', 'time']], check_dtype=False)
        self.assertSetEqual(set(test.columns), set(target.columns))

    def test_add_default_time_points_to_experimental_conditions_some_nans(self):
        mm = MeasurementModel(self.sim_result)
        df = pd.DataFrame([[1,2], [2, np.NaN], [3, 3], [4, np.NaN]], columns=['a', 'b'])
        default_times = [0.1, 0.5, 1.0]
        target = pd.DataFrame([[1, 2],
                               [3, 3],
                               [2, 0.1],
                               [2, 0.5],
                               [2, 1.0],
                               [4, 0.1],
                               [4, 0.5],
                               [4, 1.0]], columns=['a', 'b'])
        test = mm._add_default_time_points_to_experimental_conditions(df, default_times, column='b')
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_add_default_time_points_to_experimental_condition_all_nans(self):
        mm = MeasurementModel(self.sim_result)
        df = pd.DataFrame([[2, np.NaN], [4, np.NaN]], columns=['a', 'b'])
        default_times = [0.1, 0.5, 1.0]
        target = pd.DataFrame([[2, 0.1],
                               [2, 0.5],
                               [2, 1.0],
                               [4, 0.1],
                               [4, 0.5],
                               [4, 1.0]], columns=['a', 'b'])
        test = mm._add_default_time_points_to_experimental_conditions(df, default_times, column='b')
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_add_default_time_points_to_experimental_conditions_no_nans(self):
        mm = MeasurementModel(self.sim_result)
        df = pd.DataFrame([[1, 2],[3, 3]], columns=['a', 'b'])
        default_times = [0.1, 0.5, 1.0]
        target = pd.DataFrame([[1, 2],
                               [3, 3]], columns=['a', 'b'])
        test = mm._add_default_time_points_to_experimental_conditions(df, default_times, column='b')
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_add_default_time_points_to_experimental_conditions(self):
        mm = MeasurementModel(self.sim_result, experimental_conditions=pd.DataFrame([['WT', np.NaN],
                                                                                     ['KO',   3.50]],
                                                                                    columns=['condition', 'time']))
        mm.time_points = [0.10, 0.20]
        test = mm._experimental_conditions_df
        target = pd.DataFrame([['KO', 3.50, 1],
                               ['WT', 0.10, 1],
                               ['WT', 0.20, 1]],
                              columns=['condition', 'time', 'experiment'])
        pd.testing.assert_frame_equal(test[['condition', 'time', 'experiment']], target[['condition', 'time', 'experiment']])

    def test_time_points_setter_bad_input(self):
        mm = MeasurementModel(self.sim_result)
        self.assertListEqual(mm.time_points, list(np.linspace(0, 10, 3)))
        with self.assertRaises(ValueError) as error:
            mm.time_points = "bad input"
        self.assertTrue(error.exception.args[0] == "'time_points' must be vector-like")

    def test_get_ec_from_sim_result(self):
        mm = MeasurementModel(self.sim_result)
        o2df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['time', 'value'])
        test1, test2 = mm._get_ec_from_sim_result(o2df, o2df)
        target1 = pd.DataFrame([np.NaN], columns=['time'])
        target2 = {'time'}
        pd.testing.assert_frame_equal(test1, target1)
        self.assertSetEqual(test2, target2)

    def test_update_simulation_result(self):
        mm = MeasurementModel(self.sim_result,
                              experimental_conditions=pd.DataFrame([['WT', np.NaN],
                                                                    ['KO', 3.50]],
                                                                   columns=['condition', 'time']))
        mm.update_simulation_result(self.sim_result_2)
        test = mm._experimental_conditions_df
        target = pd.DataFrame([['KO', 3.50, 1],
                               ['WT', 0.00, 1],
                               ['WT', 4.00, 1],
                               ['WT', 8.00, 1]],
                              columns=['condition', 'time', 'experiment'])
        pd.testing.assert_frame_equal(test[['condition', 'time', 'experiment']],
                                      target[['condition', 'time', 'experiment']])

    def test_update_sim_res_df(self):
        ds = DataSet(pd.DataFrame(), [])
        ds.observables = ['AB_complex', 'A_free']
        ds.experimental_conditions = pd.DataFrame([['WT', 1, 0.0],
                                                   ['KO', 1, 0.0],
                                                   ['DKO', 2, 0.0]],
                                                  columns=['condition', 'experiment', 'time'])

        mm = MeasurementModel(self.sim_result, dataset=ds, observables=['A_free', 'AB_complex'])
        with self.assertRaises(ValueError) as error:
            mm._update_sim_res_df(pd.DataFrame(columns=['A', 'B']))
        print(error.exception.args[0])
        self.assertTrue(error.exception.args[0] ==
                        "This simulation result is missing the following required observables: "
                        "'A_free', and 'AB_complex'" or
                        error.exception.args[0] ==
                        "This simulation result is missing the following required observables: "
                        "'AB_complex', and 'A_free'"
                        )

    def test_simulation_result_df_setter(self):
        mm = MeasurementModel(self.sim_result,
                              experimental_conditions=pd.DataFrame([['WT', np.NaN],
                                                                    ['KO', 3.50]],
                                                                   columns=['condition', 'time']))
        mm.simulation_result_df = self.sim_result_2.opt2q_dataframe
        test = mm._experimental_conditions_df
        target = pd.DataFrame([['KO', 3.50, 1],
                               ['WT', 0.00, 1],
                               ['WT', 4.00, 1],
                               ['WT', 8.00, 1]],
                              columns=['condition', 'time', 'experiment'])
        pd.testing.assert_frame_equal(test[['condition', 'time', 'experiment']],
                                      target[['condition', 'time', 'experiment']])


