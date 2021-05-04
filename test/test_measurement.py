# MW Irvin -- Lopez Lab -- 2018-08-23
import numpy as np
from pysb import Monomer, Parameter, Initial, Observable, Rule
from pysb.bng import generate_equations
from pysb.testing import *
from opt2q.simulator import Simulator
from opt2q.measurement.base import MeasurementModel, SampleAverage, Scale
from opt2q.measurement import WesternBlot, FractionalKilling, Fluorescence, WesternBlotPTM
from opt2q.data import DataSet
# from opt2q.examples.cell_viability_example.cell_viability_likelihood_fn import cell_viability_model
from opt2q.utils import parse_column_names
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

    def test_get_obs_from_dataset(self):
        ds = DataSet(pd.DataFrame(), [])
        ds.experimental_conditions = pd.DataFrame(columns=['condition', 'experiment'])
        ds.observables = ['AB_complex', 'Nonexistent_Obs']
        mm = MeasurementModel(self.sim_result)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test = mm._get_obs_from_dataset(ds, mm._default_observables)
            print(w[-1].message)
            assert str(w[-1].message) == 'The supplied dataset has observables not present in the simulation result. ' \
                                         'They will be ignored.'
        target = {'AB_complex'}
        self.assertSetEqual(mm._get_required_observables({'A', 'B'}, None), {'A', 'B'})
        self.assertSetEqual(test, target)

    def test_check_observables(self):
        mm = MeasurementModel(self.sim_result)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test = mm._check_observables({'AB_complex', 'Nonexistent_Obs'}, mm._default_observables)
            assert str(w[-1].message) == "Observables not present in the simulation result will be ignored."
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


class TestWesternBlotModel(TestSolverModel, unittest.TestCase):
    def test_check_measured_values_dict(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        WesternBlot(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex']})

    def test_check_that_dict_items_are_vector_like(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlot(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex']})
        test = wb._check_that_dict_items_are_vector_like({1: [1, 2], 2: 'a'})
        target = {1: [1, 2], 2: ['a']}
        self.assertDictEqual(test, target)

    def test_convert_measured_values_to_dict(self):
        with self.assertRaises(ValueError) as error:
            data = pd.DataFrame([[2, 0, 0, "WT", 1],
                                 [0, 4, 9, "WT", 1]],
                                columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
            ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
            wb = WesternBlot(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex']})
            wb._check_that_measured_values_is_dict([])
        self.assertTrue(
            error.exception.args[0] == "'measured_values' must be a dict."
        )

    def test_check_measured_values_dict_key_not_in_data(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlot(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex']})
        with self.assertRaises(ValueError) as error:
            wb._check_measured_values_dict({'PARP': ['A_free'], 'EXTRA': ['AB_complex']}, ds)
        self.assertTrue(
            error.exception.args[0] ==
            "'measured_values' contains a variable, 'EXTRA', not mentioned as an ordinal variable in the 'dataset'.")

    def test_check_measured_values_observables_updated(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlot(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex']}, observables=['B_free'])
        test = wb._check_measured_values_dict({'PARP': ['A_free'], 'cPARP': ['AB_complex']}, wb._dataset)[1]
        target = {'AB_complex', 'A_free', 'B_free'}
        self.assertSetEqual(test, target)

    def test_run(self):
        np.random.seed(10)
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlot(self.sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                         ['AB_complex'])
        test = wb.run()[['cPARP__0', 'cPARP__1', 'cPARP__2', 'cPARP__3']]
        target = pd.DataFrame([[0.999421,  0.000450,  0.000095,  0.000028],
                               [0.764923,  0.171357,  0.046156,  0.014758],
                               [0.329834,  0.359846,  0.204618,  0.087440],
                               [0.140177,  0.283851,  0.312991,  0.209810],
                               [0.069284,  0.182300,  0.309758,  0.329143],
                               [0.038948,  0.115747,  0.255914,  0.405116],
                               [0.038948,  0.115747,  0.255914,  0.405116],
                               [0.015943,  0.052228,  0.149662,  0.421105],
                               [0.008102,  0.027471,  0.087552,  0.348388]],
                              columns=['cPARP__0', 'cPARP__1', 'cPARP__2', 'cPARP__3'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_less_precise=1)

    def test_get_index_of_logistic_classifier_step(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlot(self.sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                         ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                               ['KO', 1]],
                                                                              columns=['condition', 'experiment']))
        wb.process.remove_step(0)
        wb.run(use_dataset=True)  # Train to data before trying out of sample predictions.

        test = wb.run(use_dataset=False)[['cPARP__0', 'cPARP__1', 'cPARP__2', 'cPARP__3', 'condition']]
        target = pd.DataFrame([[0.955810,  0.034055,  0.007452,  0.002259,        'WT'],
                               [0.038948,  0.115747,  0.255914,  0.405116,        'WT'],
                               [0.006092,  0.020842,  0.068392,  0.305703,        'WT'],
                               [0.955810,  0.034055,  0.007452,  0.002259,        'KO'],
                               [0.039151,  0.116252,  0.256514,  0.404618,        'KO'],
                               [0.006109,  0.020896,  0.068554,  0.306119,        'KO']],
                              columns=['cPARP__0',  'cPARP__1',  'cPARP__2',  'cPARP__3', 'condition'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_less_precise=1)

    def test_likelihood(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100, 'WT', 1,  0], [150, 'WT', 1,  1]],
                                        columns=['kbindAB', 'condition', 'experiment', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        wb = WesternBlot(sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                         ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                               ['KO', 1]],
                                                                              columns=['condition', 'experiment']))
        wb.process.remove_step(0)
        results = wb.likelihood()
        self.assertAlmostEqual(results, 12.9654678049418, 10)

    def test_likelihood_with_sample_average_step_added(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100, 'WT', 1,  0], [150, 'WT', 1,  1]],
                                        columns=['kbindAB', 'condition', 'experiment', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        wb = WesternBlot(sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                         ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                               ['KO', 1]],
                                                                              columns=['condition', 'experiment']))
        wb.process.set_params(sample_average__sample_size=50)
        results = wb.likelihood()
        self.assertAlmostEqual(results, 9.595265322480328, 3)

    def test_likelihood_with_sample_average_step_added_wo_apply_noise(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100, 'WT', 1,  0], [150, 'WT', 1,  1]],
                                        columns=['kbindAB', 'condition', 'experiment', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        wb = WesternBlot(sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                         ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                               ['KO', 1]],
                                                                              columns=['condition', 'experiment']))
        wb.process.add_step(('sample_average',
                             SampleAverage(columns=['AB_complex'], drop_columns='simulation',
                                          groupby=list(set(wb.experimental_conditions_df.columns)- {'simulation'}),
                                          apply_noise=True, variances=0.0, sample_size=4)), index=0)
        results = wb.likelihood()
        self.assertAlmostEqual(results, 12.072994981309265, 3)

    def test_likelihood_with_different_index_in_data(self):
        data = pd.DataFrame([[1, 0, 0, "KO", 1],
                             [2, 0, 1, "KO", 1],
                             [3, 0, 2, "KO", 1],
                             [2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data[data.condition=="WT"], {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100, 'WT', 1,  0], [150, 'WT', 1,  1]],
                                        columns=['kbindAB', 'condition', 'experiment', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        wb = WesternBlot(sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                         ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                               ['KO', 1]],
                                                                              columns=['condition', 'experiment']))
        wb.process.add_step(('sample_average',
                             SampleAverage(columns=['AB_complex'], drop_columns='simulation',
                                          groupby=list(set(wb.experimental_conditions_df.columns)- {'simulation'}),
                                          apply_noise=True, variances=0.0, sample_size=4)), index=0)
        results = wb.likelihood()
        self.assertAlmostEqual(results, 12.072994981309265, 3)


class TestFluorescence(TestSolverModel, unittest.TestCase):
    def test_observables(self):
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100,0], [150, 1]],
                                        columns=['kbindAB', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        fl = Fluorescence(sim_result)
        test = fl._get_process_observables(fl._dataset, None)[1]
        target = {'A_free', '__s0', '__s2', 'B_free', '__s1', 'AB_complex'}
        self.assertSetEqual(test, target)

    def test_observables_user_obs(self):
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100,0], [150, 1]],
                                        columns=['kbindAB', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        fl = Fluorescence(sim_result, observables=['AB_complex'])
        test = fl._get_process_observables(fl._dataset, None)[1]
        target = {'AB_complex'}
        self.assertSetEqual(test, target)

    def test_get_groupby_columns_no_cols(self):
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100,0], [150, 1]],
                                        columns=['kbindAB', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        fl = Fluorescence(sim_result, observables=['AB_complex'])
        test = fl._get_groupby_columns()
        target = ['simulation']
        self.assertListEqual(test, target)

    def test_get_groupby_column(self):
        fl = Fluorescence(self.sim_result, observables=['AB_complex'])
        test = set(fl._get_groupby_columns())
        target = {'experiment', 'condition'}
        self.assertSetEqual(test, target)

    def test_run_wo_dataset(self):
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[1e-3, 0], [1.5e-3, 1], [0.3e-3, 2]],
                                        columns=['kbindAB', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 1, 3))
        fl = Fluorescence(sim_result, observables=['A_free'])
        test = fl.run()
        target = pd.DataFrame([[0.000000,   0.0,           0],
                               [0.512295,   0.5,           0],
                               [1.000000,   1.0,           0],
                               [0.000000,   0.0,           1],
                               [0.518291,   0.5,           1],
                               [1.000000,   1.0,           1],
                               [0.000000,   0.0,           2],
                               [0.503731,   0.5,           2],
                               [1.000000,   1.0,           2]],
                              columns=['A_free',  'time',  'simulation'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])
        sim.param_values = pd.DataFrame([[0.1e-3, 0]], columns=['kbindAB', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 1, 3))
        fl.update_simulation_result(sim_result)
        test = fl.run(do_fit_transform=False)
        target = pd.DataFrame([[0.000000,   0.0,           0],
                               [0.516127,   0.5,           0],
                               [1.029683,   1.0,           0]],
                              columns=['A_free', 'time', 'simulation'])
        pd.testing.assert_frame_equal(test[test.columns].sort_values(by='simulation').reset_index(drop=True),
                                      target[test.columns].sort_values(by='simulation').reset_index(drop=True))

    def test_run_wo_dataset_groups(self):
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[1e-3,   'WT', 0],
                                         [1.5e-3, 'WT', 1],
                                         [0.3e-3, 'KO', 2]],
                                        columns=['kbindAB', 'exp', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 1, 3))
        fl = Fluorescence(sim_result, observables=['A_free'])
        test = fl.run(do_fit_transform=True)
        target = pd.DataFrame([[0.000000,   0.0, 'KO', 2],
                               [0.503731,   0.5, 'KO', 2],
                               [1.000000,   1.0, 'KO', 2],
                               [0.000000,   0.0, 'WT', 0],
                               [0.512295,   0.5, 'WT', 0],
                               [1.000000,   1.0, 'WT', 0],
                               [0.000000,   0.0, 'WT', 1],
                               [0.510195,   0.5, 'WT', 1],
                               [0.984379,   1.0, 'WT', 1]],
                              columns=['A_free',  'time',  'exp', 'simulation'])

        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_dtype=False)

        sim.param_values = pd.DataFrame([[0.1e-3, 'KO', 0]], columns=['kbindAB', 'exp','simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 1, 3))
        fl.update_simulation_result(sim_result)
        test = fl.run(do_fit_transform=False)
        target = pd.DataFrame([[0.000000,            0.0,   'KO', 0],
                               [0.504569630733837,   0.5,   'KO', 0],
                               [1.006626882968067,   1.0,   'KO', 0]],
                              columns=['A_free', 'time', 'exp', 'simulation'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_dtype=False)

        # pd.testing.assert_frame_equal(test[test.columns].sort_values(by='simulation').reset_index(drop=True),
        #                               target[test.columns].sort_values(by='simulation').reset_index(drop=True))

    def test_check_dataset_wo_measured_values(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        with self.assertRaises(ValueError) as error:
            Fluorescence(self.sim_result, ds)
        self.assertTrue(error.exception.args[0] == "You must provide 'measured_values'.")

    def test_check_dataset_bad_measured_values(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        with self.assertRaises(ValueError) as error:
            Fluorescence(self.sim_result, ds, {'PARP': ['A_free']})
        self.assertTrue(error.exception.args[0] ==
                        "'measured_values' contains a variable, 'PARP', not mentioned "
                        "as a 'quantitative' or 'semi-quantitative' variable in the 'dataset_fluorescence'.")

    def test_check_process_observables(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'quantitative', 'cPARP': 'semi-quantitative'})
        fl = Fluorescence(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP':['AB_complex']})
        test = fl._get_process_observables(fl._dataset, {'PARP': ['A_free'], 'cPARP':['AB_complex']})[0]
        target = {'PARP': ['A_free'], 'cPARP':['AB_complex']}
        self.assertDictEqual(test, target)

    def test_run_with_dataset(self):
        data = pd.DataFrame([[0.00, 0.00, 0, "WT", 1],
                             [0.90, 0.00, 1, "WT", 1],
                             [0.98, 0.00, 2, "WT", 1],
                             [0.99, 0.25, 3, "WT", 1],
                             [0.99, 0.50, 4, "WT", 1],
                             [0.99, 0.75, 5, "WT", 1],
                             [0.99, 0.79, 5, "WT", 1],
                             [0.52, 0.95, 7, "WT", 1],
                             [0.14, 0.99, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'quantitative', 'cPARP': 'semi-quantitative'},)
        fl = Fluorescence(self.sim_result, ds,
                          {'PARP': ['A_free'], 'cPARP': ['AB_complex']},
                          observables=['A_free'],
                          experimental_conditions=pd.DataFrame([[1, 'WT'],
                                                                [5, 'WT'],
                                                                [6, 'WT']],
                                                               columns=['time', 'condition']))
        test = fl.run()
        target = pd.DataFrame([[0.0,    0.000000,        'WT',   0.0,           1,           0],
                               [0.2,    0.111110,        'WT',   1.0,           1,           0],
                               [0.4,    0.222220,        'WT',   2.0,           1,           0],
                               [0.6,    0.333330,        'WT',   3.0,           1,           0],
                               [0.8,    0.444440,        'WT',   4.0,           1,           0],
                               [1.0,    0.555551,        'WT',   5.0,           1,           0],
                               [1.0,    0.777775,        'WT',   7.0,           1,           0],
                               [1.0,    1.000000,        'WT',   9.0,           1,           0]],
                              columns=['A_free',  'AB_complex', 'condition',  'time',  'experiment',  'simulation'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

        test = fl.run(use_dataset=False, do_fit_transform=False)
        target = pd.DataFrame([[ 0.2,    0.111110,        'WT',   1.0,           1,           0],
                               [ 1.0,    0.555551,        'WT',   5.0,           1,           0],
                               [ 1.0,    0.666663,        'WT',   6.0,           1,           0]],
                              columns=['A_free', 'AB_complex', 'condition', 'time', 'experiment', 'simulation'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_run_with_dataset_two_observables_for_one_value(self):
        data = pd.DataFrame([[0.00, 0.00, 0, "WT", 1],
                             [0.90, 0.00, 1, "WT", 1],
                             [0.98, 0.00, 2, "WT", 1],
                             [0.99, 0.25, 3, "WT", 1],
                             [0.99, 0.50, 4, "WT", 1],
                             [0.99, 0.75, 5, "WT", 1],
                             [0.99, 0.79, 5, "WT", 1],
                             [0.52, 0.95, 7, "WT", 1],
                             [0.14, 0.99, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'quantitative', 'cPARP': 'semi-quantitative'},)
        fl = Fluorescence(self.sim_result, ds,
                          {'PARP': ['A_free','AB_complex']},
                          observables=['A_free'],
                          experimental_conditions=pd.DataFrame([[1, 'WT'],
                                                                [5, 'WT'],
                                                                [6, 'WT']],
                                                               columns=['time', 'condition']))

        def add_cols(x):
            return pd.DataFrame(x['AB_complex']+x['A_free'], columns=['AB_complex+A_free'])
        fl.process.add_step(
            ('add_cols', Scale(columns=['AB_complex', 'A_free'], scale_fn=add_cols))
        )
        self.assertAlmostEquals(fl.likelihood(), 69.96190165459436, 5)

    def test_run_with_dataset_two_observables_for_one_value_raise_error(self):
        data = pd.DataFrame([[0.00, 0.00, 0, "WT", 1],
                             [0.90, 0.00, 1, "WT", 1],
                             [0.98, 0.00, 2, "WT", 1],
                             [0.99, 0.25, 3, "WT", 1],
                             [0.99, 0.50, 4, "WT", 1],
                             [0.99, 0.75, 5, "WT", 1],
                             [0.99, 0.79, 5, "WT", 1],
                             [0.52, 0.95, 7, "WT", 1],
                             [0.14, 0.99, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'quantitative', 'cPARP': 'semi-quantitative'}, )
        fl = Fluorescence(self.sim_result, ds,
                          {'PARP': ['A_free', 'AB_complex']},
                          observables=['A_free'],
                          experimental_conditions=pd.DataFrame([[1, 'WT'],
                                                                [5, 'WT'],
                                                                [6, 'WT']],
                                                               columns=['time', 'condition']))

        with self.assertRaises(ValueError) as error:
            fl.likelihood()
        self.assertTrue(error.exception.args[0] ==
                        "The measurement model result for PARP can only have one column. It had the following: 'AB_complex', and 'A_free'.")

    def test_check_simulation_result_non_opt2q(self):
        # If no experimental conditions are supplied, the results dataframe has no simulation column.
        result = Simulator(model=self.model, tspan=np.linspace(0, 0.1, 10)).run()
        data = pd.DataFrame([[0.00, 0.00, 0.00],
                             [0.90, 0.00, 0.01],
                             [0.98, 0.00, 0.02],
                             [0.99, 0.25, 0.03],
                             [0.99, 0.50, 0.04],
                             [0.99, 0.75, 0.05],
                             [0.99, 0.79, 0.06],
                             [0.52, 0.95, 0.07],
                             [0.14, 0.99, 0.09]],
                            columns=['PARP', 'cPARP', 'time'])
        ds = DataSet(data, {'PARP': 'quantitative', 'cPARP': 'semi-quantitative'}, )
        fl = Fluorescence(result,
                          dataset=ds,
                          measured_values={'PARP': ['A_free'], 'cPARP':['AB_complex']},
                          observables=['A_free', 'AB_complex'])
        test = fl.run()
        target = pd.DataFrame([[0.000000, 0.000000, 0.00, 0],
                               [0.724010, 0.034499, 0.01, 0],
                               [0.942321, 0.132210, 0.02, 0],
                               [0.991255, 0.251093, 0.03, 0],
                               [0.998815, 0.375148, 0.04, 0],
                               [0.999845, 0.500019, 0.05, 0],
                               [0.999980, 0.625003, 0.06, 0],
                               [0.999997, 0.750000, 0.07, 0],
                               [1.000000, 1.000000, 0.09, 0]],
                              columns=['A_free',  'AB_complex',  'time',  'simulation'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_less_precise=4)


class TestWesternBlotPTM(TestSolverModel, unittest.TestCase):
    def test_check_measured_values_dict(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        WesternBlotPTM(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex']})
        with self.assertRaises(ValueError) as error:
            WesternBlotPTM(self.sim_result, ds, {'PARP': ['A_free'], 'cPARP': ['AB_complex', 'A_free']})
        self.assertTrue(error.exception.args[0] == " All the lists in 'measured_variables' must have the same length.")

    def test_dataset_restructure(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlotPTM(self.sim_result, ds, {'cPARP': ['AB_complex'], 'PARP': ['A_free']})
        print(wb._restructured_ds)

    def test_x_restructure(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['PARP', 'cPARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlotPTM(self.sim_result, ds, {'PARP': ['AB_complex', 'B_free'], 'cPARP': ['A_free', 'B_free']},
                            ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                                  ['KO', 1]],
                                                                                 columns=['condition', 'experiment']))
        print(wb._restructure_x(wb.simulation_result_df[wb._results_cols]))

    def test_wb_run(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['cPARP', 'PARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlotPTM(self.sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['A_free']},
                            ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                                  ['KO', 1]],
                                                                                 columns=['condition', 'experiment']))
        wb.process.get_step('sample_average').set_params(**{'sample_size': 1})
        print(wb.run().columns)

    def test_wb_run_out_of_sample(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [0, 4, 9, "WT", 1]],
                            columns=['cPARP', 'PARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'})
        wb = WesternBlotPTM(self.sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['A_free']},
                            ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                                  ['KO', 1]],
                                                                                 columns=['condition', 'experiment']))
        wb.process.get_step('sample_average').set_params(**{'sample_size': 1})
        print(wb.run()[['time', 'PARP__1', 'cPARP__1', 'condition', 'experiment']])
        print(wb.run(use_dataset=False)[['time', 'PARP__1', 'cPARP__1', 'condition', 'experiment']])

    def test_likelihood(self):
        data = pd.DataFrame([[2, 0, 0, "WT", 1],
                             [2, 0, 1, "WT", 1],
                             [2, 0, 2, "WT", 1],
                             [2, 1, 3, "WT", 1],
                             [2, 2, 4, "WT", 1],
                             [2, 3, 5, "WT", 1],
                             [1, 3, 5, "WT", 1],
                             [1, 4, 7, "WT", 1],
                             [1, 4, 9, "WT", 1]],
                            columns=['cPARP', 'PARP', 'time', 'condition', 'experiment'])
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'ordinal'}, use_common_ordinal_classifier=True)
        sim = Simulator(self.model)
        sim.param_values = pd.DataFrame([[100, 'WT', 1, 0], [150, 'WT', 1, 1]],
                                        columns=['kbindAB', 'condition', 'experiment', 'simulation'])
        sim_result = sim.run(tspan=np.linspace(0, 10, 3))
        wb = WesternBlotPTM(sim_result, ds, {'PARP': ['AB_complex'], 'cPARP': ['AB_complex']},
                            ['AB_complex'], experimental_conditions=pd.DataFrame([['WT', 1],
                                                                                  ['KO', 1]],
                                                                                 columns=['condition', 'experiment']))
        wb.process.remove_step(0)
        print(wb.likelihood())
        # self.assertAlmostEqual(results, 12.9654678049418, 10)
