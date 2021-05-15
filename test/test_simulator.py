from pysb import Monomer, Parameter, Initial, Observable, Rule
from pysb.bng import generate_equations
from pysb.testing import *
from opt2q.simulator import Simulator
from opt2q.utils import IncompatibleFormatWarning, CupSodaNotInstalledWarning
import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import unittest
import warnings


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

    def tearDown(self):
        self.model=None
        self.mon=None
        self.test_non_opt2q_params = None
        self.test_non_opt2q_params_df = None


class TestSolver(TestSolverModel, unittest.TestCase):
    """test solver"""

    def test_get_solver_kwargs(self):
        sim = Simulator(self.model)
        test = sim._get_solver_kwargs({'a': 2})
        self.assertDictEqual({'a': 2}, test)

    def test_add_integrator_options_dict_none(self):
        sim = Simulator(self.model)
        self.assertDictEqual(sim.solver_kwargs, {'integrator_options': {}})  # when None return empty dict

    def test_custom_solver_options(self):
        sim = Simulator(self.model, solver_options={'integrator':'lsoda'}, integrator_options={'mxstep': 2**10})
        assert sim.sim.opts == {'mxstep': 1024}
        assert sim.sim._init_kwargs['integrator'] == 'lsoda'

    def test_warning_setting(self):
        sim = Simulator(self.model)
        assert sim._capture_warnings_setting is False

    def test_components_are_compatible_no_model_params(self):
        sim = Simulator(self.model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test1, test2, test3 = sim._is_compatible({'a', 'b', 'c'}, set([]))
            assert issubclass(w[-1].category, IncompatibleFormatWarning)
        target1 = False
        target2 = {'a', 'b', 'c'}
        target3 = set([])
        self.assertEqual(test1, target1)
        self.assertEqual(test2, target2)
        self.assertEqual(test3, target3)

    def test_components_are_compatible_no_added_params(self):
        sim = Simulator(self.model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test1, test2, test3 = sim._is_compatible({'a', 'b', 'c'}, {'a', 'b', 'c'})
            assert issubclass(w[-1].category, IncompatibleFormatWarning)
        target1 = False
        target2 = set([])
        target3 = {'a', 'b', 'c'}
        self.assertEqual(test1, target1)
        self.assertEqual(test2, target2)
        self.assertEqual(test3, target3)

    def test_components_are_compatible_yes(self):
        sim = Simulator(self.model)
        test1, test2, test3 = sim._is_compatible({'b', 'c'}, {'a', 'b'})
        target1 = True
        target2 = {'c'}
        target3 = {'b'}
        self.assertEqual(test1, target1)
        self.assertEqual(test2, target2)
        self.assertEqual(test3, target3)

    def test_add_simulation_column_case_1(self):
        sim = Simulator(self.model)
        var_in = pd.DataFrame(np.arange(5), columns=['simulation'])
        test = sim._add_simulations_column(var_in)
        target = var_in
        pd_testing.assert_frame_equal(test, target)

    def test_add_simulation_column_case_2(self):
        sim = Simulator(self.model)
        var_in = pd.DataFrame(np.arange(1,5), columns=['values'])
        test = sim._add_simulations_column(var_in)
        target = var_in
        target['simulation'] = np.arange(target.shape[0])
        pd_testing.assert_frame_equal(test, target)

    def test_params_for_run_none(self):
        sim = Simulator(self.model)
        test = sim._params_for_run(None, 'params_col', 'model_params_names', 'model_params_values')
        target = None
        self.assertEqual(test, target)

    def test_params_for_run_df(self):
        sim = Simulator(self.model)
        test = sim._params_for_run(pd.DataFrame(np.ones((3, 2)), columns=['c', 'd']), {'c', 'd'},
                                   ['a', 'b', 'c', 'd'], [2, 3, 4, 5])
        target = pd.DataFrame(np.ones((3, 2)), columns=['c', 'd'])
        target['a'] = 2
        target['b'] = 3
        np.testing.assert_allclose(test, target[['a', 'b', 'c', 'd']].values)

    def test_initials_for_run(self):
        sim = Simulator(self.model)
        test = sim._initials_for_run(pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']), ['b'])
        self.assertDictEqual(test, {'b':[2, 4]})

    def test_initials_for_run_none(self):
        sim = Simulator(self.model)
        test = sim._initials_for_run(None, ['b'])
        assert test is None

    def test_check_components_none_none(self):
        sim = Simulator(self.model)
        assert sim.param_values is None
        assert sim.initials is None
        assert sim._param_values_run is None
        assert sim._initials_run is None
        assert sim._params_are_compatible is False
        assert sim._initials_are_compatible is False

    def test_check_components_dict_none(self):
        sim = Simulator(self.model, param_values={'a': 42})
        self.assertDictEqual(sim.param_values, {'a': 42})
        assert sim.initials is None
        self.assertDictEqual(sim._param_values_run, {'a': 42})
        assert sim._initials_run is None
        assert sim._params_are_compatible is False
        assert sim._initials_are_compatible is False

    def test_check_components_none_dict(self):
        sim = Simulator(self.model, initials={'a': 42})
        assert sim.param_values is None
        self.assertDictEqual(sim.initials, {'a': 42})
        assert sim._param_values_run is None
        self.assertDictEqual(sim._initials_run, {'a': 42})
        assert sim._params_are_compatible is False
        assert sim._initials_are_compatible is False

    def test_check_components_array_dict(self):
        sim = Simulator(self.model, param_values=np.ones((3, 5)), initials={'a': 42})
        np.testing.assert_array_equal(sim.param_values, np.ones((3, 5)))
        self.assertDictEqual(sim.initials, {'a': 42})
        np.testing.assert_array_equal(sim._param_values_run, np.ones((3, 5)))
        self.assertDictEqual(sim._initials_run, {'a': 42})
        assert sim._params_are_compatible is False
        assert sim._initials_are_compatible is False

    def test_check_components_dict_non_o2_df(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        sim = Simulator(self.model, param_values={'a': 42}, initials=initials)
        self.assertDictEqual(sim.param_values, {'a': 42})
        self.assertDictEqual(sim._param_values_run, {'a': 42})
        assert sim._params_are_compatible is False
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is False

    def test_check_components_dict_o2_df(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        sim = Simulator(self.model, param_values={'a': 42}, initials=initials)
        self.assertDictEqual(sim.param_values, {'a': 42})
        self.assertDictEqual(sim._param_values_run, {'a': 42})
        assert sim._params_are_compatible is False
        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True

    def test_check_components_non_o2_df_o2_df(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        params_df = pd.DataFrame([[1, 2], [0, 1]], columns=['ksynthA', 'ksynthB'])
        sim = Simulator(self.model, param_values=params_df, initials=initials)

        pd_testing.assert_frame_equal(sim.param_values, params_df, check_dtype=False)
        params_df[['kbindAB', 'A_init', 'B_init']] = pd.DataFrame([[100, 0, 0],[100, 0, 0]])
        np.testing.assert_allclose(sim._param_values_run, params_df[['ksynthA', 'ksynthB', 'kbindAB', 'A_init', 'B_init']].values)
        assert sim._params_are_compatible is False
        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True

    def test_check_components_o2_df_o2_df(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        params_df = pd.DataFrame([[1, 2, 'a'], [0, 1, 'a']], columns=['ksynthA', 'ksynthB', 'ec'])
        sim = Simulator(self.model, param_values=params_df, initials=initials)
        params_df['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.param_values, params_df, check_dtype=False)
        params_df[['kbindAB', 'A_init', 'B_init']] = pd.DataFrame([[100, 0, 0], [100, 0, 0]])
        params_df = params_df.drop(columns=['simulation', 'ec'])
        np.testing.assert_allclose(sim._param_values_run, params_df[params_df.columns].values)
        assert sim._params_are_compatible is True
        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True

    def test_check_updates(self):
        sim = Simulator(self.model)
        self.assertEqual(sim._update_components_w_check, sim._update_components)
        sim.check_updates = False
        assert sim.check_updates is False
        self.assertEqual(sim._update_components_wo_check, sim._update_components)

    def test_param_update_o2_df_o2_df(self):
        params_df = pd.DataFrame([[1, 2, 'a'], [0, 1, 'a']], columns=['ksynthA', 'ksynthB', 'ec'])
        sim = Simulator(self.model, param_values=params_df)
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        initials['simulation'] = [0, 1]
        sim.initials = initials

        params_df['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.param_values, params_df, check_dtype=False)
        params_df[['kbindAB', 'A_init', 'B_init']] = pd.DataFrame([[100, 0, 0], [100, 0, 0]])
        params_df = params_df.drop(columns=['simulation', 'ec'])
        np.testing.assert_allclose(sim._param_values_run, params_df[params_df.columns].values)
        assert sim._params_are_compatible is True
        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True

    def test_params_wo_check(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        params_df = pd.DataFrame([[1, 2, 'a'], [0, 1, 'b']], columns=['ksynthA', 'ksynthB', 'ec'])
        sim = Simulator(self.model, param_values=params_df)
        sim.check_updates = False
        sim.initials = initials

        params_df['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.param_values, params_df, check_dtype=False)
        params_df[['kbindAB', 'A_init', 'B_init']] = pd.DataFrame([[100, 0, 0], [100, 0, 0]])
        params_df=params_df.drop(columns=['simulation', 'ec'])
        np.testing.assert_allclose(sim._param_values_run, params_df[params_df.columns].values)
        assert sim._params_are_compatible is True

        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is False

    def test_initials_wo_check(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        params_df = pd.DataFrame([[1, 2, 'a'], [0, 1, 'b']], columns=['ksynthA', 'ksynthB', 'ec'])

        sim = Simulator(self.model, initials=initials, check_updates=False)
        sim.param_values = params_df

        pd_testing.assert_frame_equal(sim.param_values, params_df, check_dtype=False)

        params_df[['kbindAB', 'A_init', 'B_init']] = pd.DataFrame([[100, 0, 0], [100, 0, 0]])
        params_df = params_df.drop(columns=['ec'])
        np.testing.assert_allclose(sim._param_values_run, params_df[params_df.columns].values)

        assert sim._params_are_compatible is False

        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True
        sim.check_updates = True
        assert sim._update_components == sim._update_components_w_check

    def test_initials_w_check_empty_df(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        sim = Simulator(self.model, initials=initials, check_updates=True)
        sim.param_values = pd.DataFrame()
        assert sim.param_values is None
        assert sim._params_are_compatible is False

        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True

    def test_initials_wo_check_other(self):
        initials = pd.DataFrame([[1, 2], [3, 4]], columns=self.model.species[0:2])
        initials['ec'] = 'a'
        sim = Simulator(self.model, initials=initials, check_updates=False)
        sim.param_values = 42
        assert sim.param_values == 42
        assert sim._param_values_run == 42
        assert sim._params_are_compatible is False

        initials['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.initials, initials, check_dtype=False)
        self.assertDictEqual(sim._initials_run, {self.model.species[0]: [1, 3],
                                                 self.model.species[1]: [2, 4]})
        assert sim._initials_are_compatible is True

    def test_params_wo_check_other(self):
        params_df = pd.DataFrame([[1, 2, 'a'], [0, 1, 'b']], columns=['ksynthA', 'ksynthB', 'ec'])
        sim = Simulator(self.model, param_values=params_df)
        sim.check_updates = False
        sim.initials = 42

        params_df['simulation'] = [0, 1]
        pd_testing.assert_frame_equal(sim.param_values, params_df, check_dtype=False)
        params_df[['kbindAB', 'A_init', 'B_init']] = pd.DataFrame([[100, 0, 0], [100, 0, 0]])
        params_df = params_df.drop(columns=['simulation', 'ec'])
        np.testing.assert_allclose(sim._param_values_run, params_df[params_df.columns].values)
        assert sim._params_are_compatible is True

        assert sim.initials == 42
        assert sim._initials_run == 42

    def test_base_run(self):
        target = pd.DataFrame(
            np.array([[0,     0,     0,       0,       0,           0],
                      [1,     1,   249,       1,       1,         249],
                      [1,     1,   499,       1,       1,         499],
                      [1,     1,   749,       1,       1,         749],
                      [1,     1,   999,       1,       1,         999]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model, solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2 ** 10})
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False)

    def test_non_opt2q_df_run(self):
        target = pd.DataFrame(
            np.array([[0,     0,     0,       0,       0,           0],
                      [1,     1,   249,       1,       1,         249],
                      [1,     1,   499,       1,       1,         499],
                      [1,     1,   749,       1,       1,         749],
                      [1,     1,   999,       1,       1,         999]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model, param_values=pd.DataFrame(columns=['simulation']),
                        solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2 ** 10})
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False)

    def test_non_opt2q_df_run_2(self):
        target = pd.DataFrame(
            np.array([[0,     0,     0,       0,       0,           0],
                      [1,     1,   249,       1,       1,         249],
                      [1,     1,   499,       1,       1,         499],
                      [1,     1,   749,       1,       1,         749],
                      [1,     1,   999,       1,       1,         999]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model,
                        param_values=pd.DataFrame(),
                        solver_options={'integrator': 'lsoda'},
                        integrator_options={'mxstep': 2 ** 10})
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False)

    def test_non_opt2q_initials_df_run(self):
        target = pd.DataFrame(
            np.array([[0,     0,     0,       0,       0,           0],
                      [1,     1,   249,       1,       1,         249],
                      [1,     1,   499,       1,       1,         499],
                      [1,     1,   749,       1,       1,         749],
                      [1,     1,   999,       1,       1,         999]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model,
                        param_values=pd.DataFrame(),
                        solver_options={'integrator': 'lsoda'},
                        integrator_options={'mxstep': 2 ** 10})
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False)

    def test_opt2q_run(self):
        target = pd.DataFrame(
            np.array([[0, 0, 0, 0, 0, 0],
                      [1, 1, 249, 1, 1, 249],
                      [1, 1, 499, 1, 1, 499],
                      [1, 1, 749, 1, 1, 749],
                      [1, 1, 999, 1, 1, 999]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model, solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2 ** 10})
        sim.initials = pd.DataFrame([[0, 0.0]], columns=['exp', self.model.species[-1]])
        assert sim._initials_are_compatible is True
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False)

    def test_opt2q_run_params_initials(self):
        target = pd.DataFrame(
            np.array([[       0,          0,          1,        0,          0,   1.000000],
                      [0.004000, 125.004000, 125.996000, 0.004000, 125.004000, 125.996000],
                      [0.002000, 250.002000, 250.998000, 0.002000, 250.002000, 250.998000],
                      [0.001333, 375.001333, 375.998667, 0.001333, 375.001333, 375.998667],
                      [0.001000, 500.001000, 500.999000, 0.001000, 500.001000, 500.999000]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model, initials=pd.DataFrame([[0, 1.0]], columns=['exp', self.model.species[-1]]),
                        param_values={'ksynthA': [50]})
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(pd.DataFrame([[0, 1.0, 0]], columns=['exp', self.model.species[-1], 'simulation'])
                                      , sim.initials)
        self.assertDictEqual({'ksynthA': [50]}, sim.param_values)
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False, check_less_precise=True)

    def test_opt2q_dataframe(self):
        target = pd.DataFrame(
            np.array([[     0.0,        0.0,        1.0,      0.0,        0.0,   1.000000, 0, 'WT'],
                      [0.004000, 125.004000, 125.996000, 0.004000, 125.004000, 125.996000, 0, 'WT'],
                      [0.002000, 250.002000, 250.998000, 0.002000, 250.002000, 250.998000, 0, 'WT'],
                      [0.001333, 375.001333, 375.998667, 0.001333, 375.001333, 375.998667, 0, 'WT'],
                      [0.001000, 500.001000, 500.999000, 0.001000, 500.001000, 500.999000, 0, 'WT']]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex', 'simulation', 'exp'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        dtypes={'__s0':float, '__s1':float, '__s2':float, 'A_free':float, 'B_free':float,'AB_complex':float,
                'simulation':int, 'exp':object}
        for col in dtypes.keys():
            target[col] = target[col].astype(dtypes[col])

        sim = Simulator(self.model, initials=pd.DataFrame([['WT', 1.0]], columns=['exp', self.model.species[-1]]))
        res = sim.run(tspan=np.linspace(0, 10, 5), param_values={'ksynthA': [50]})
        pd_testing.assert_frame_equal(res.opt2q_dataframe, target,
                                      check_dtype=False,
                                      check_less_precise=True,
                                      check_index_type=False,
                                      check_column_type=False)

    def test_opt2q_dataframe_params_too(self):
        target = pd.DataFrame(
            np.array([[     0.0,        0.0,        1.0,      0.0,        0.0,   1.000000, 0, 'WT'],
                      [0.004000, 125.004000, 125.996000, 0.004000, 125.004000, 125.996000, 0, 'WT'],
                      [0.002000, 250.002000, 250.998000, 0.002000, 250.002000, 250.998000, 0, 'WT'],
                      [0.001333, 375.001333, 375.998667, 0.001333, 375.001333, 375.998667, 0, 'WT'],
                      [0.001000, 500.001000, 500.999000, 0.001000, 500.001000, 500.999000, 0, 'WT']]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex', 'simulation', 'exp'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        dtypes={'__s0':float, '__s1':float, '__s2':float, 'A_free':float, 'B_free':float,'AB_complex':float,
                'simulation':int, 'exp':object}
        for col in dtypes.keys():
            target[col] = target[col].astype(dtypes[col])

        sim = Simulator(self.model, initials=pd.DataFrame([['WT', 1.0]], columns=['exp', self.model.species[-1]]))
        res = sim.run(tspan=np.linspace(0, 10, 5), param_values=pd.DataFrame({'ksynthA': [50], 'exp': ['WT']}))
        pd_testing.assert_frame_equal(res.opt2q_dataframe, target,
                                      check_dtype=False,
                                      check_less_precise=True,
                                      check_index_type=False,
                                      check_column_type=False)

    def test_opt2q_dataframe_multi_sims(self):
        target = pd.DataFrame(
            np.array([[0,  0.000000,  100.000000,    0.000000,  0.000000,  100.000000,   0.000000,  'WT'],
                      [0,  0.009999,  100.009999,  499.990001,  0.009999,  100.009999, 499.990001,  'WT'],
                      [0,  0.009999,  100.009999,  999.990001,  0.009999,  100.009999, 999.990001,  'WT'],
                      [1,  0.000000,    0.000000,    0.000000,  0.000000,    0.000000,   0.000000, 'KO'],
                      [1,  1.000000,    1.000000,  499.000000,  1.000000,    1.000000, 499.000000, 'KO'],
                      [1,  1.000000,    1.000000,  999.000000,  1.000000,    1.000000, 999.000000, 'KO']]),
            columns=[ 'simulation', u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex', 'exp'],
            index=pd.Index([0.0, 5.0, 10.0, 0.0, 5.0, 10.0], name='time'))
        dtypes = {'__s0': float, '__s1': float, '__s2': float, 'A_free': float, 'B_free': float, 'AB_complex': float,
                  'simulation': int, 'exp': object}
        for col in dtypes.keys():
            target[col] = target[col].astype(dtypes[col])

        sim = Simulator(self.model)
        res = sim.run(tspan=np.linspace(0, 10, 3),
                      initials=pd.DataFrame({self.model.species[1]: [100, 0], "exp": ['WT', 'KO']}))
        pd_testing.assert_frame_equal(res.opt2q_dataframe, target,
                                      check_dtype=False,
                                      check_less_precise=True,
                                      check_index_type=False,
                                      check_column_type=False)

    def test_check_solver_when_cupsoda_is_not_installed(self):
        try:
            from pysb.pathfinder import get_path
            # Path to cupSODA executable
            get_path('cupsoda')
            pass
        except Exception:
            with warnings.catch_warnings(record=True) as w:
                Simulator(self.model, solver='cupsoda')
                warnings.simplefilter("always")
                assert issubclass(w[-1].category, CupSodaNotInstalledWarning)

    def test_base_run_cupsoda_option(self):
        target = pd.DataFrame(
            np.array([[0,     0,     0,       0,       0,           0],
                      [1,     1,   249,       1,       1,         249],
                      [1,     1,   499,       1,       1,         499],
                      [1,     1,   749,       1,       1,         749],
                      [1,     1,   999,       1,       1,         999]]),
            columns=[u'__s0', u'__s1', u'__s2', u'A_free', u'B_free', u'AB_complex'],
            index=pd.Index([0.0, 2.5, 5.0, 7.5, 10.0], name='time'))
        sim = Simulator(self.model, solver='cupsoda', integrator_options={'max_steps': 2 ** 10})
        res = sim.run(tspan=np.linspace(0, 10, 5))
        pd_testing.assert_frame_equal(target, res.dataframe, check_dtype=False)

