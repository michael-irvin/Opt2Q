# MW Irvin -- Lopez Lab -- 2018-08-31
from opt2q.measurement.base import Interpolate, Pipeline
from opt2q.utils import _is_vector_like
import numpy as np
import pandas as pd
import unittest
import warnings


class TestInterpolate(unittest.TestCase):
    def test_check_new_values(self):
        interpolate = Interpolate('iv', 'dv', [0.1])
        test = interpolate.new_values
        target = pd.DataFrame([0.1], columns=['iv'])
        pd.testing.assert_frame_equal(test, target)

    def test_check_new_values_vector_like_df(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        interpolate.new_values = pd.DataFrame([3.5])
        test = interpolate.new_values
        target = pd.DataFrame([3.5], columns=['iv'])
        assert interpolate._transform == interpolate._transform_new_values_simple
        pd.testing.assert_frame_equal(test, target)

    def test_check_new_values_df(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        interpolate.new_values = pd.DataFrame([[3.5, 'a']], columns=['iv', 'ec'])
        test = interpolate.new_values
        target = pd.DataFrame([[3.5, 'a']], columns=['iv', 'ec'])
        assert interpolate._transform == interpolate._transform_new_values_extra_cols
        pd.testing.assert_frame_equal(test, target)

    def test_check_values_df_no_iv(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        with self.assertRaises(ValueError) as error:
            interpolate.new_values = pd.DataFrame([[3.5, 'a']], columns=['not_iv', 'ec'])
        self.assertTrue(error.exception.args[0] == "'new_values' must have an column named 'iv'.")

    def test_check_values_bad_input(self):
        with self.assertRaises(ValueError) as error:
            Interpolate('iv', 'dv', {0.0: 1})
        self.assertTrue(error.exception.args[0] == "'new_values' must be vector-like list of numbers or a "
                                                   "pd.DataFrame.")

    def test_interpolate_setup(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        assert 'iv' == interpolate._independent_variable_name
        self.assertListEqual(['dv'], interpolate._dependent_variable_name)
        assert 'cubic' == interpolate._interpolation_method_name
        assert interpolate._interpolate == interpolate._interpolation_not_in_groups

    def test_intersect_x_and_new_values_experimental_condition(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        nx = pd.DataFrame([[1, 0.5],
                           [1, 1.5],
                           [3, 0.5],
                           [3, 1.5],
                           [4, 1],
                           [4, 2]], columns=['ec', 'iv'])
        x = pd.DataFrame([[1, 0, 0],
                          [1, 1, 2],
                          [1, 2, 5],
                          [2, 0, 0],
                          [2, 1, 3],
                          [2, 2, 6],
                          [3, 0, 0],
                          [3, 1, 4],
                          [3, 2, 7],
                          [3, 3, 12]], columns=['ec', 'iv', 'dv'])
        interpolate.new_values = nx
        test = interpolate._intersect_x_and_new_values_experimental_condition(x, nx)
        target = pd.DataFrame([[1, 0, 0],
                          [1, 1, 2],
                          [1, 2, 5],
                          [3, 0, 0],
                          [3, 1, 4],
                          [3, 2, 7],
                          [3, 3, 12]], columns=['ec', 'iv', 'dv'])
        pd.testing.assert_frame_equal(test, target)

    def test_intersect_x_and_new_values_experimental_condition_extra_cols_in_x(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        nx = pd.DataFrame([[1, 0.5],
                           [1, 1.5],
                           [3, 0.5],
                           [3, 1.5],
                           [4, 1],
                           [4, 2]], columns=['ec', 'iv'])
        x = pd.DataFrame([[1, 0, 0,  1],
                          [1, 1, 2,  1],
                          [1, 2, 5,  1],
                          [2, 0, 0,  1],
                          [2, 1, 3,  1],
                          [2, 2, 6,  1],
                          [3, 0, 0,  1],
                          [3, 1, 4,  1],
                          [3, 2, 7,  1],
                          [3, 3, 12, 1]], columns=['ec', 'iv', 'dv', 'ec1'])
        interpolate.new_values = nx
        test = interpolate._intersect_x_and_new_values_experimental_condition(x, nx)
        target = pd.DataFrame([[1, 0, 0,  1],
                               [1, 1, 2,  1],
                               [1, 2, 5,  1],
                               [3, 0, 0,  1],
                               [3, 1, 4,  1],
                               [3, 2, 7,  1],
                               [3, 3, 12, 1]], columns=['ec', 'iv', 'dv', 'ec1'])
        pd.testing.assert_frame_equal(test[['ec', 'iv', 'dv', 'ec1']], target[['ec', 'iv', 'dv', 'ec1']])

    def test_intersect_x_and_new_values_experimental_condition_extra_cols_not_in_x(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        nx = pd.DataFrame([[1, 0.5],
                           [1, 1.5],
                           [3, 0.5],
                           [3, 1.5],
                           [4, 1],
                           [4, 2]], columns=['ec', 'iv'])
        x = pd.DataFrame([[1, 0, 0,  1],
                          [1, 1, 2,  1],
                          [1, 2, 5,  1],
                          [2, 0, 0,  1],
                          [2, 1, 3,  1],
                          [2, 2, 6,  1],
                          [3, 0, 0,  1],
                          [3, 1, 4,  1],
                          [3, 2, 7,  1],
                          [3, 3, 12, 1]], columns=['ec1', 'iv', 'dv', 'ec2'])
        interpolate.new_values = nx
        with self.assertRaises(KeyError) as error:
            interpolate._intersect_x_and_new_values_experimental_condition(x, nx)
        self.assertTrue(error.exception.args[0] == "'new_values' contains columns not present in x: 'ec'")

    def test_transform_simplest(self):
        interpolate = Interpolate('iv', 'dv', [0.0])
        nx = pd.DataFrame([[0.5],
                           [1.5]], columns=['iv'])
        x = pd.DataFrame([[0, 0],
                          [1, 2],
                          [2, 4]], columns=['iv', 'dv'])
        target = pd.DataFrame([[0.5, 1.0],
                               [1.5, 3.0]], columns=['iv', 'dv'])
        interpolate.new_values = nx
        test = interpolate.transform(x)
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_extra_cols_no_groups(self):
        # Automatically repeat the interpolate for each unique row of extra columns
        interpolate = Interpolate('iv', 'dv', [0.0])
        nx = pd.DataFrame([[1, 0.5],
                           [1, 1.5]], columns=['ec', 'iv'])
        x = pd.DataFrame([[1, 1, 0, 0],
                          [2, 1, 1, 2],
                          [1, 1, 2, 4]], columns=['ec1', 'ec', 'iv', 'dv'])
        target = pd.DataFrame([[1, 1, 0.5, 1.0],
                               [2, 1, 0.5, 1.0],
                               [1, 1, 1.5, 3.0],
                               [2, 1, 1.5, 3.0]], columns=['ec1', 'ec', 'iv', 'dv'])
        interpolate.new_values = nx
        test = interpolate.transform(x)
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_extra_cols_no_groups_vector_for_new_values(self):
        # Automatically repeat the interpolate for each unique row of extra columns
        # Builds a default pd.DataFrame for ``new_values`` from [0.5, 1.5].
        interpolate = Interpolate('iv', 'dv', [0.5, 1.5])
        x = pd.DataFrame([[1, 1, 0, 0],
                          [2, 1, 1, 2],
                          [1, 1, 2, 4]], columns=['ec1', 'ec', 'iv', 'dv'])
        target = pd.DataFrame([[1, 1, 0.5, 1.0],
                               [2, 1, 0.5, 1.0],
                               [1, 1, 1.5, 3.0],
                               [2, 1, 1.5, 3.0]], columns=['ec1', 'ec', 'iv', 'dv'])
        test = interpolate.transform(x)
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_extra_cols_w_groups_vector_for_new_values(self):
        # Automatically repeat the interpolate for each unique row of extra columns
        # Builds a default pd.DataFrame for ``new_values`` from [0.5, 1.5].
        interpolate = Interpolate('iv', 'dv', [0.5, 1.5], groupby='ec1')
        x = pd.DataFrame([[1, 1, 0, 0],
                          [1, 1, 1, 2],
                          [1, 1, 2, 4],
                          [2, 1, 0, 0],
                          [2, 1, 1, 3],
                          [2, 1, 2, 6]], columns=['ec1', 'ec', 'iv', 'dv'])
        target = pd.DataFrame([[1, 1, 0.5, 1.0],
                               [1, 1, 1.5, 3.0],
                               [2, 1, 0.5, 1.5],
                               [2, 1, 1.5, 4.5]], columns=['ec1', 'ec', 'iv', 'dv'])
        test = interpolate.transform(x)
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_interpolation_in_groups(self):
        x = pd.DataFrame([[0, 'WT', 0, 0],
                          [0, 'WT', 1, 2],
                          [0, 'WT', 2, 4],
                          [1, 'WT', 0, 1],
                          [1, 'WT', 1, 3],
                          [1, 'WT', 2, 5],
                          [2, 'KO', 0, 1],
                          [2, 'KO', 1, 2],
                          [2, 'KO', 2, 3],
                          [3, 'KO', 0, 0],
                          [3, 'KO', 1, 1],
                          [3, 'KO', 2, 2]], columns=['sim', 'ec', 'iv', 'dv'])
        new_x = pd.DataFrame([['WT', 0.5], ['KO', 1.5]], columns=['ec', 'iv'])
        interpolate = Interpolate('iv', 'dv', new_x, groupby='sim')
        test = interpolate.transform(x)
        target = pd.DataFrame([[0, 'WT', 0.5, 1.0],
                               [1, 'WT', 0.5, 2.0],
                               [2, 'KO', 1.5, 2.5],
                               [3, 'KO', 1.5, 1.5]], columns=['sim', 'ec', 'iv', 'dv'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_interpolation_get_params(self):
        interpolate = Interpolate('iv', 'dv', [0.0], interpolation_method_name='cubic')
        new_x = pd.DataFrame([['WT', 0.5], ['KO', 1.5]], columns=['ec', 'iv'])
        interpolate.new_values = new_x
        target = {'named_transform__new_values': new_x,
                  'named_transform__interpolation_method_name': 'cubic'}
        test = interpolate.get_params('named_transform')
        pd.testing.assert_frame_equal(target.pop('named_transform__new_values'),
                                      test.pop('named_transform__new_values'))
        self.assertDictEqual(test, target)
        assert interpolate.__repr__() == "Interpolate(independent_variable_name='iv', " \
                                         "dependent_variable_name=['dv'], " \
                                         "new_values='DataFrame(shape=(2, 2))', " \
                                         "options={'interpolation_method_name': 'cubic'})"
        interpolate.set_params(**{'interpolation_method_name': 'linear'})
        assert interpolate.interpolation_method_name == 'linear'


class TestPipeline(unittest.TestCase):
    def test_blank_input(self):
        pl = Pipeline()
        self.assertListEqual(pl.steps, [])
        assert pl.__repr__() == 'Pipeline(steps=[])'

    def test_duplicate_inputs(self):
        with self.assertRaises(ValueError) as error:
            Pipeline(steps=[('interpolate', Interpolate('iv', 'dv', [0.0])),
                            ('interpolate', Interpolate('iv', 'dv', [2.0]))
                            ])
        self.assertTrue(error.exception.args[0] ==
                        "Each steps must have a unique name. Duplicate steps are not allowed.")

    def test_not_a_transport_error(self):
        with self.assertRaises(ValueError) as error:
            Pipeline(steps=[('interpolate', Interpolate('iv', 'dv', [0.0])),
                            ('interpolate2', [1, 2, 3])
                            ])
        self.assertTrue(error.exception.args[0] ==
                        "Each step must be a Transform class instance. [1, 2, 3] is not.")

    def test_remove_step(self):
        process = Pipeline(steps=[('interpolate', Interpolate('iv', 'dv', [1, 2, 3]))])
        process.remove_step('interpolate')
        self.assertListEqual(process.steps, [])

    def test_remove_step_by_int(self):
        i1 = Interpolate('iv', 'dv', [1, 2, 3])
        i2 = Interpolate('iv', 'dv', [1, 2])
        i3 = Interpolate('iv', 'dv', [1, 4])
        process = Pipeline(steps=[('interpolate', i1),
                                  ('interpolate_2_', i2),
                                  ('interpolate_3_', i3)])
        process.remove_step(1)
        target_steps = [('interpolate', i1), ('interpolate_3_', i3)]
        self.assertListEqual(process.steps, target_steps)

    def test_add_step_by_default(self):
        i1 = Interpolate('iv', 'dv', [1, 2, 3])
        i2 = Interpolate('iv', 'dv', [1, 2])
        i3 = Interpolate('iv', 'dv', [1, 4])
        process = Pipeline(steps=[('interpolate', i1),
                                  ('interpolate_3_', i3)])
        process.add_step(('interpolate_2_', i2))
        target_steps = [('interpolate', i1), ('interpolate_3_', i3), ('interpolate_2_', i2)]
        self.assertListEqual(process.steps, target_steps)

    def test_add_step_by_str_name(self):
        i1 = Interpolate('iv', 'dv', [1, 2, 3])
        i2 = Interpolate('iv', 'dv', [1, 2])
        i3 = Interpolate('iv', 'dv', [1, 4])
        process = Pipeline(steps=[('interpolate', i1),
                                  ('interpolate_3_', i3)])
        process.add_step(('interpolate_2_', i2), 'interpolate')
        target_steps = [('interpolate', i1), ('interpolate_2_', i2), ('interpolate_3_', i3)]
        self.assertListEqual(process.steps, target_steps)

    def test_add_step_by_index(self):
        i1 = Interpolate('iv', 'dv', [1, 2, 3])
        i2 = Interpolate('iv', 'dv', [1, 2])
        i3 = Interpolate('iv', 'dv', [1, 4])
        process = Pipeline(steps=[('interpolate', i1),
                                  ('interpolate_3_', i3)])
        process.add_step(('interpolate_2_', i2), 10)
        target_steps = [('interpolate', i1), ('interpolate_3_', i3), ('interpolate_2_', i2)]
        self.assertListEqual(process.steps, target_steps)

    def test_add_step_by_index_zero(self):
        i1 = Interpolate('iv', 'dv', [1, 2, 3])
        i2 = Interpolate('iv', 'dv', [1, 2])
        i3 = Interpolate('iv', 'dv', [1, 4])
        process = Pipeline(steps=[('interpolate', i1),
                                  ('interpolate_3_', i3)])
        process.add_step(('interpolate_2_', i2), 0)
        target_steps = [('interpolate_2_', i2), ('interpolate', i1), ('interpolate_3_', i3)]
        self.assertListEqual(process.steps, target_steps)

    def test_add_step_example(self):
        process = Pipeline()
        process.add_step(('interpolate', Interpolate('iv', 'dv', [0.0])))
        assert str(process.steps) == \
            "[('interpolate', Interpolate(independent_variable_name='iv', dependent_variable_name=['dv'], " \
            "new_values='DataFrame(shape=(1, 1))', options={'interpolation_method_name': 'cubic'}))]"

    def test_get_params(self):
        process = Pipeline()
        process.add_step(('interpolate', Interpolate('iv', 'dv', [0.0])))
        test = process.get_params('pipeline')
        target = {
            'pipeline__interpolate__new_values': pd.DataFrame([0.0], columns=['iv']),
            'pipeline__interpolate__interpolation_method_name': 'cubic'
        }
        pd.testing.assert_frame_equal(target.pop('pipeline__interpolate__new_values'),
                                      test.pop('pipeline__interpolate__new_values'))
        self.assertDictEqual(test, target)

    def test_transform_method(self):
        process = Pipeline()
        process.add_step(('interpolate', Interpolate('iv', 'dv', [0.0])))
        x = pd.DataFrame([[0.0, 1],
                          [1.0, 2],
                          [1.4, 3]], columns=['iv', 'dv'])
        test = process.transform(x)
        target = pd.DataFrame([[0.0, 1]], columns=['iv', 'dv'])
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_scipy(self):
        import scipy
        print(scipy.__version__)
