# MW Irvin -- Lopez Lab -- 2018-08-31
from opt2q.measurement.base import Interpolate, Pipeline, SampleAverage, Scale, Standardize, \
    LogisticClassifier, CumulativeComputation
from opt2q.measurement.base.functions import log_scale, TransformFunction, polynomial_features
from opt2q.utils import _is_vector_like
from opt2q.data import DataSet
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

    def test_interpolate_new_values_pd_updates_group_by(self):
        interpolate = Interpolate('time', ['fluorescence', '1-fluorescence'], [0])
        interpolate.new_values = pd.DataFrame([[2, 'early'], [8, 'late']], columns=['time', 'observation'])
        x = pd.DataFrame([[1, 10, 0, 1, 'early'],
                          [3, 8, 2, 1, 'early'],
                          [5, 5, 5, 1, 'early'],
                          [7, 2, 8, 2, 'late'],
                          [9, 10, 0, 2, 'late']],
                         columns=['time', 'fluorescence', '1-fluorescence', 'sample', 'observation'])
        test = interpolate.transform(x)
        target = pd.DataFrame([['early',      1,     2,           9.0,             1.0],
                               ['late',       2,     8,           6.0,             4.0]],
                              columns=['observation', 'sample', 'time', 'fluorescence', '1-fluorescence'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_topic_guide_groupby(self):
        x = pd.DataFrame([[1, 10, 0, 1, 'early'],
                          [3, 8, 2, 1, 'early'],
                          [5, 5, 5, 1, 'early'],
                          [7, 2, 8, 2, 'late'],
                          [9, 10, 0, 2, 'late']],
                         columns=['time', 'fluorescence', '1-fluorescence', 'sample', 'observation'])
        interpolate = Interpolate('time', ['fluorescence', '1-fluorescence'], [0])
        interpolate.new_values = pd.DataFrame([[2, 'early'], [8, 'late']], columns=['time', 'observation'])
        test = interpolate.transform(x)
        target = pd.DataFrame([['early', 1, 2, 9.0, 1.0],
                               ['late', 2, 8, 6.0, 4.0]],
                              columns=['observation', 'sample', 'time', 'fluorescence', '1-fluorescence'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_topic_guide_groupby_scaled_terms(self):
        x = pd.DataFrame([[1, 10, 0, 1, 'early'],
                          [3, 8, 2, 1, 'early'],
                          [5, 5, 5, 1, 'early'],
                          [7, 2, 8, 2, 'late'],
                          [9, 10, 0, 2, 'late']],
                         columns=['time', 'fluorescence', '1-fluorescence', 'sample', 'observation'])
        interpolate = Interpolate('time', ['fluorescence'], [0])
        interpolate.new_values = pd.DataFrame([[2, 'early'], [8, 'late']], columns=['time', 'observation'])
        test = interpolate.transform(x)
        target = pd.DataFrame([['early', 1, 2, 9.0, 1.0],
                               ['late', 2, 8, 6.0, 4.0]],
                              columns=['observation', 'sample', 'time', 'fluorescence', '1-fluorescence'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_extra_cols_no_groups_vector_for_new_values_parse_column_names(self):
        # Automatically repeat the interpolate for each unique row of extra columns
        # Builds a default pd.DataFrame for ``new_values`` from [0.5, 1.5].
        interpolate = Interpolate('iv', 'dv', [0.5, 1.5])
        x = pd.DataFrame([[1, 1, 0, 0, 0],
                          [2, 1, 1, 2, 4],
                          [1, 1, 2, 4, 8]], columns=['ec1', 'ec', 'iv', 'dv', 'dv$*$2'])
        target = pd.DataFrame([[1, 1, 0.5, 1.0, 2.0],
                               [2, 1, 0.5, 1.0, 2.0],
                               [1, 1, 1.5, 3.0, 6.0],
                               [2, 1, 1.5, 3.0, 6.0]], columns=['ec1', 'ec', 'iv', 'dv', 'dv$*$2'])
        test = interpolate.transform(x)
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])


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
        process.add_step(('interpolate', Interpolate('iv', 'dv', np.array([0.0]))))
        x = pd.DataFrame([[0.0, 1],
                          [1.0, 2],
                          [1.4, 3]], columns=['iv', 'dv'])
        test = process.transform(x)
        target = pd.DataFrame([[0.0, 1]], columns=['iv', 'dv'])
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_set_params_missing_key(self):
        process = Pipeline()
        process.add_step(('interpolate', Interpolate('iv', 'dv', np.array([0.0]))))
        with warnings.catch_warnings(record=True) as w:
            process.set_params(**{'nonexistent_step__param': 42})
            warnings.simplefilter("always")
            print(str(w[-1].message) )
            assert str(w[-1].message) == "The process does not have the following step(s): 'nonexistent_step'"


class TestScale(unittest.TestCase):
    @staticmethod
    def f(x, a=2):
        return x ** a

    def test_check_columns_None(self):
        s = Scale()
        assert s._columns is None

    def test_check_columns_str(self):
        # what is column is not in x? (The scaling ignores it).
        s = Scale(columns='a', scale_fn=self.f)
        self.assertSetEqual({'a'}, s._columns_set)
        test = s.transform(pd.DataFrame([1, 2, 3], columns=['b']))
        target = pd.DataFrame([1, 2, 3], columns=['b'])
        pd.testing.assert_frame_equal(test, target)
        test = s.transform(pd.DataFrame([1, 2, 3], columns=['a']))
        target = pd.DataFrame([1, 4, 9], columns=['a'])
        pd.testing.assert_frame_equal(test, target)

    def test_check_columns_list(self):
        s = Scale(columns=['a', 'c'])
        s.set_params(**{'scale_fn': self.f, 'scale_fn_kwargs': {'a': 2}})
        test = s.transform(pd.DataFrame([[1, 2, 'a'], [2, 2, 'b'], [3, 2, 'c']], columns=['a', 'b', 'c']))
        target = pd.DataFrame([[1, 2, 'a'], [4, 2, 'b'], [9, 2, 'c']], columns=['a', 'b', 'c'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_dtype=False)

    def test_check_scale_fn_str_in_scale_functions(self):
        s = Scale()
        test1, test2 = s._check_scale_fn('log2')
        assert s.scale_functions['log2'][0] == test1
        self.assertDictEqual({'base':2, 'clip_zeros': True}, test2)

    def test_check_scale_fn_str_not_in_scale_functions(self):
        with self.assertRaises(ValueError) as error:
            Scale(scale_fn='non_existent_fn')
        self.assertTrue(error.exception.args[0] ==
                        "'scale_fn' must be in 'scale_functions'. 'non_existent_fn' is not.")

    def test_check_scale_fn_transform_function(self):
        s = Scale()
        target1 = log_scale
        test1, test2 = s._check_scale_fn(target1)
        assert target1 == test1
        self.assertDictEqual(dict(), test2)

    def test_check_scale_fn_regular_function(self):
        s = Scale()
        test1, test2 = s._check_scale_fn(self.f)
        target1 = 'f(x, a=2)'
        assert test1.__repr__() == target1  # True if the conversion to TransformFunction works
        self.assertDictEqual(dict(), test2)

    def test_scale_fn_kwargs(self):
        s = Scale(scale_fn=self.f, a=3)
        assert s.scale_fn.__repr__() == 'f(x, a=3)'
        assert s.get_params()['scale_fn_kwargs__a'] == 3
        assert s.get_params()['scale_fn'].__repr__() == 'f(x, a=3)'

    def test_set_params(self):
        s = Scale()
        s.set_params(**{'scale_fn': self.f, 'scale_fn_kwargs': {'a': 1}})
        assert s.get_params()['scale_fn'].__repr__() == 'f(x, a=1)'

    def test_transform(self):
        s = Scale()
        s.set_params(**{'scale_fn': self.f, 'scale_fn_kwargs': {'a': 2}})
        test = s.transform(pd.DataFrame([1, 2, 3]))
        target = pd.DataFrame([1, 4, 9])
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_transform_non_numeric_cols(self):
        s = Scale()
        s.set_params(**{'scale_fn': self.f, 'scale_fn_kwargs': {'a': 2}})
        test = s.transform(pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'c']]))
        target = pd.DataFrame([[1, 'a'], [4, 'b'], [9, 'c']])
        pd.testing.assert_frame_equal(test, target, check_dtype=False)

    def test_topic_guide_example(self):
        scale = Scale(columns=[1, 2], scale_fn='log10')
        test = scale.transform(pd.DataFrame([[0, 0.1, 'a'], [1, 1.0, 'b'], [2, 10., 'c']]))
        target = pd.DataFrame([[0, -1., 'a'], [1, 0., 'b'], [2, 1., 'c']])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

        def custom_f(x, a=1):
            return x ** a

        scale.scale_fn = custom_f
        scale.scale_fn_kwargs = {'a': 2}
        test = scale.transform(pd.DataFrame([[0, 0.1, 'a'], [1, 1.0, 'b'], [2, 10., 'c']]))
        target = pd.DataFrame([[0, 0.01, 'a'], [1, 1.0, 'b'], [2, 100., 'c']])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_get_params(self):
        scale = Scale(columns=[0, 1], scale_fn='log2')
        scale.set_params(keep_old_columns=True)
        target = {'scale_fn_kwargs__base': 2,
                  'scale_fn_kwargs__clip_zeros': True, 'keep_old_columns': True}
        test = scale.get_params()
        assert test.pop('scale_fn').__repr__() == 'log_scale(x, base=2, clip_zeros=True)'
        self.assertDictEqual(test, target)

    def test_rename_scaled_columns(self):
        scale = Scale(columns=[0, 1], scale_fn='log2')
        test = scale._rename_scaled_columns(pd.DataFrame(np.ones((3, 4))), scaled_columns_set={0, 1}, name='scale').columns
        target = {'0__scale',  '1__scale',    2,    3}
        self.assertSetEqual(set(test), target)

    def test_transform_keep_old_columns(self):
        scale = Scale(columns=[1, 2], scale_fn='log10', keep_old_columns=True)
        test = scale.transform(pd.DataFrame([[0, 0.1, 'a'], [1, 1.0, 'b'], [2, 10., 'c']]))
        target = pd.DataFrame([[0, -1., 'a', 0.1], [1, 0., 'b', 1.0], [2, 1., 'c', 10.]], columns=[0, '1__scale', 2, 1])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_two_columns_into_one(self):
        def x1x2(x):
            return pd.DataFrame(x[0]*x[1], columns=['x1x2'])  # this will only work when x has columns 0 and 1

        scale = Scale(columns=[0, 1], scale_fn=x1x2, keep_old_columns=True)
        test = scale.transform(pd.DataFrame([[0, 0.1, 'a'], [1, 1.0, 'b'], [2, 10., 'c']]))
        target = pd.DataFrame([[0, 0., 'a', 0.1], [1, 1., 'b', 1.0], [2, 20., 'c', 10.]], columns=[0, 'x1x2', 2, 1])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_in_groups(self):
        def t_where_x_is_max(x, iv='time', dv='A'):
            return x.where(x[dv] == x[dv].max()).bfill()[[iv, dv]]
        scale = Scale(scale_fn=t_where_x_is_max, groupby='C', keep_old_columns=True, dv='A', iv='time')
        x = pd.DataFrame([
            [0, 0, 0, 'a'],
            [1, 3, 1, 'a'],
            [0, 2, 2, 'a'],
            [2, 1, 3, 'a'],
            [3, 0, 0, 'b'],
            [2, 2, 1, 'b'],
            [5, 4, 2, 'b'],
            [4, 6, 3, 'b'],
            [5, 8, 4, 'b'],
        ], columns=['A', 'B', 'time', 'C'])
        test = scale.transform(x, name='tau')
        target = pd.DataFrame([
            [3.0,     2.0,  'a',  0,     0,  0],
            [3.0,     2.0,  'a',  3,     1,  1],
            [3.0,     2.0,  'a',  2,     2,  0],
            [3.0,     2.0,  'a',  1,     3,  2],
            [2.0,     5.0,  'b',  0,     0,  3],
            [2.0,     5.0,  'b',  2,     1,  2],
            [2.0,     5.0,  'b',  4,     2,  5],
            [4.0,     5.0,  'b',  6,     3,  4],
            [4.0,     5.0,  'b',  8,     4,  5]],
            columns=['time__tau',  'A__tau',  'C',  'B',  'time',  'A'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])


class TestStandardize(unittest.TestCase):
    def test_defaults(self):
        st = Standardize()
        assert st._columns is None
        self.assertSetEqual(st._columns_set, set([]))
        assert st._transform_get_scale_fn == st._scale_fn_w_fit

    def test_columns_str(self):
        st = Standardize(columns='col')
        self.assertListEqual(['col'], st._columns)
        self.assertSetEqual({'col'}, st._columns_set)

    def test_columns_list(self):
        st = Standardize(columns=['col1', 'col2'])
        self.assertListEqual(['col1', 'col2'], st._columns)
        self.assertSetEqual({'col1', 'col2'}, st._columns_set)

    def test_do_fit_transform(self):
        st = Standardize(do_fit_transform=False)
        assert st._transform_get_scale_fn == st._scale_fn_wo_fit
        test = st._transform_get_scale_fn(pd.DataFrame([1, 2, 3])).transform(pd.DataFrame([3]))
        target = np.array([[1.224745]])
        np.testing.assert_array_almost_equal(test, target, decimal=7)
        self.assertListEqual(['__default'], list(st._transform_scale_fn_dict.keys()))

    def test_transform_not_in_groups(self):
        st = Standardize()
        test = st._transform_not_in_groups(pd.DataFrame([[1, 1],
                                                         [2, 1],
                                                         [3, 1]]), {0})
        target = pd.DataFrame([[-1.224745, 1],
                               [ 0.000000, 1],
                               [ 1.224745, 1]])
        pd.testing.assert_frame_equal(test, target)

    def test_transform_in_groups(self):
        st = Standardize(groupby=1)
        test = st._transform_in_groups(pd.DataFrame([[1, 1],
                                                     [2, 1],
                                                     [3, 1],
                                                     [0, 2],
                                                     [1, 2]]), {0})
        target = pd.DataFrame([[-1.224745, 1],
                               [0.000000, 1],
                               [1.224745, 1],
                               [-1,       2],
                               [1.,       2]])
        pd.testing.assert_frame_equal(test, target)
        st.do_fit_transform = False
        test = st._transform_in_groups(pd.DataFrame([[1, 1],
                                                     [2, 1],
                                                     [3, 1],
                                                     [1, 2],
                                                     [2, 2]]), {0})
        target = pd.DataFrame([[-1.224745, 1],
                               [0.000000, 1],
                               [1.224745, 1],
                               [1,        2],
                               [3.,       2]])
        pd.testing.assert_frame_equal(test, target)

    def test_transform(self):
        st = Standardize(groupby=1)
        test = st.transform(pd.DataFrame([[1, 1],
                                          [2, 1],
                                          [3, 1],
                                          [0, 2],
                                          [1, 2]]))
        target = pd.DataFrame([[-1.224745, 1],
                               [0.000000, 1],
                               [1.224745, 1],
                               [-1, 2],
                               [1., 2]])
        pd.testing.assert_frame_equal(test, target)
        st.do_fit_transform = False
        test = st.transform(pd.DataFrame([[1, 1],
                                          [2, 1],
                                          [3, 1],
                                          [1, 2],
                                          [2, 2]]))
        target = pd.DataFrame([[-1.224745, 1],
                               [0.000000, 1],
                               [1.224745, 1],
                               [1, 2],
                               [3., 2]])
        pd.testing.assert_frame_equal(test, target)


class TestLogisticClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset = DataSet(pd.DataFrame(columns=['a', 'b', 'c']), measured_variables=['a', 'b', 'c'])
        self.dataset.measured_variables = {'a': 'default', 'b': 'ordinal', 'c': 'quantitative'}
        self.dataset.data = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])

    def test_check_columns_and_col_groups_both_not_None(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        with self.assertRaises(ValueError) as error:
            lc._check_columns(['cols'], {'group': ['cols']})
        self.assertTrue(error.exception.args[0] == "Use only 'columns' or 'column_groups'. Not both.")

    def test_check_columns_both_none(self):
        lc = LogisticClassifier(self.dataset)
        test = lc._check_columns(None, None)
        self.assertSetEqual(set(), test[0])
        self.assertDictEqual(dict(), test[1])

    def test_check_columns_bad_type_cols(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        with self.assertRaises(ValueError) as error:
            lc._check_columns({'cols': 3}, None)
        self.assertTrue(error.exception.args[0] == "columns can only a str or list of str.")

    def test_convert_columns_to_dict(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        test1, test2 = lc._convert_columns_to_dict(['test', 'columns'])
        target1 = ['test', 'columns']
        target2 = {'test': ['test'], 'columns': ['columns']}
        self.assertListEqual(test1, target1)
        self.assertDictEqual(test2, target2)

    def test_convert_columns_to_dict_int_cols(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        test1, test2 = lc._convert_columns_to_dict([1, 'columns'])
        target1 = [1, 'columns']
        target2 = {1: [1], 'columns': ['columns']}
        self.assertListEqual(test1, target1)
        self.assertDictEqual(test2, target2)

    def test_convert_columns_to_dict_str(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        test1, test2 = lc._convert_columns_to_dict('column')
        target1 = ['column']
        target2 = {'column': ['column']}
        self.assertListEqual(test1, target1)
        self.assertDictEqual(test2, target2)

    def test_convert_columns_to_dict_int(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        test1, test2 = lc._convert_columns_to_dict(1)
        target1 = [1]
        target2 = {1: [1]}
        self.assertListEqual(test1, target1)
        self.assertDictEqual(test2, target2)

    def test_get_columns_from_column_dict(self):
        lc = LogisticClassifier(self.dataset, columns=['b'])
        test, test_dict = lc._get_columns_from_column_dict({'1': [1]})
        self.assertSetEqual(test, {1})
        test, test_dict = lc._get_columns_from_column_dict({'1': [1],
                                                            '2': [1, 'a']})
        self.assertSetEqual(set(test), {1, 'a'})
        test, test_dict = lc._get_columns_from_column_dict({'1': [1],
                                                            '2': 'a'})
        self.assertSetEqual(set(test), {1, 'a'})
        self.assertDictEqual(test_dict, {'1': [1], '2': ['a']})

    def test_check_that_dataset_has_required_columns(self):
        lc = LogisticClassifier(self.dataset, column_groups={'a': [1, 2]})
        with self.assertRaises(ValueError) as error:
            lc._check_that_dataset_has_required_columns(['a', 'b', 'c'], {'cols': [1], 'a': [1, 2], 'col': [1, 2]})
        assert error.exception.args[0] == \
            "The 'dataset' must have the following nominal or ordinal measured-variables columns: 'col', and 'cols'" \
            or error.exception.args[0] == \
            "The 'dataset' must have the following nominal or ordinal measured-variables columns: 'cols', and 'col'"

    def test_check_dataset(self):
        lc = LogisticClassifier(pd.DataFrame())
        test = lc._check_dataset(pd.DataFrame([1, 2, 3]), {0: ['a']})
        pd.testing.assert_frame_equal(test, pd.DataFrame([1, 2, 3]))

    def test_check_dataset_dataset(self):
        lc = LogisticClassifier(pd.DataFrame())
        test = lc._check_dataset(self.dataset, {'b': ['a']})
        pd.testing.assert_frame_equal(test, pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c']))

    def test_check_dataset_dataset_missing_cols(self):
        lc = LogisticClassifier(pd.DataFrame())
        with self.assertRaises(ValueError) as error:
            lc._check_dataset(self.dataset, {'c': [1, 2, 3]})
        assert error.exception.args[0] == \
            "The 'dataset' must have the following nominal or ordinal measured-variables columns: 'c'"

    def test_group_features_error_no_group_name(self):
        with self.assertRaises(ValueError) as error:
            LogisticClassifier(pd.DataFrame([[1, 2, 3]]), columns=[1], group_features=True)
        assert error.exception.args[0] == "'group_name' must be a string."

    def test_group_features(self):
        lc = LogisticClassifier(pd.DataFrame([[1, 2, 3]]), columns=['a', 'b'], group_features=True, group_name=0)
        test = lc._columns_dict
        target = {0: ['a', 'b']}
        self.assertDictEqual(test, target)

    def test_do_fit_transform(self):
        lc = LogisticClassifier(pd.DataFrame([[1, 2, 3]]), columns=['a', 'b'], group_features=True, group_name=0)
        test = lc.get_params()
        target = {'do_fit_transform': True}
        self.assertDictEqual(test, target)
        lc.set_params(do_fit_transform=False)
        assert lc._get_transform == lc._get_transform_wo_fit

    def test_transform_get_columns_from_x(self):
        lc = LogisticClassifier(pd.DataFrame([0, 0, 0, 0, 1, 1, 1]))
        x = pd.DataFrame(np.sort(np.random.normal(size=7)))
        self.assertDictEqual({0: [0]}, lc._transform_get_columns_from_x(x)[1])

    def test_transform_get_columns_from_column_dict_missing_params(self):
        lc = LogisticClassifier(pd.DataFrame([0, 0, 0, 0, 1, 1, 1], columns=['a']), column_groups={'a': [1, 2]})
        np.random.seed(10)
        x = pd.DataFrame(np.sort(np.random.normal(size=7)))
        with self.assertRaises(ValueError) as error:
            lc._transform_get_columns_from_column_dict(x)
        assert error.exception.args[0] == "'x' is missing the following numeric columns: '1', and '2'"

    def test_transform_get_columns_from_column_dict(self):
        lc = LogisticClassifier(pd.DataFrame([0, 0, 0, 0, 1, 1, 1], columns=['a']), column_groups={'a': [1, 2]})
        np.random.seed(10)
        x = pd.DataFrame(np.sort(np.random.normal(size=(7, 3))))
        test = lc._transform_get_columns_from_column_dict(x)[1]
        target = {'a': [1, 2]}
        self.assertDictEqual(test, target)

    def test_get_scalable_columns(self):
        lc = LogisticClassifier(pd.DataFrame([0, 0, 0, 0, 1, 1, 1], columns=['a']), column_groups={'a': [1, 2]})
        with self.assertRaises(TypeError) as error:
            lc._get_scalable_columns({'Not a DataFrame'})
        assert error.exception.args[0] == "x must be a pandas.DataFrame"

    def test_transform_get_columns(self):
        # picks the correct method based on whether user defined columns/column groups or not.
        lc = LogisticClassifier(pd.DataFrame([0, 0, 0, 0, 1, 1, 1], columns=['a']), column_groups={'a': [1, 2]})
        x = pd.DataFrame(np.sort(np.random.normal(size=(7, 3))))
        test = lc._transform_get_columns(x)[1]
        target = {'a': [1, 2]}
        self.assertDictEqual(test, target)
        lc = LogisticClassifier(pd.DataFrame([0, 0, 0, 0, 1, 1, 1]))
        self.assertDictEqual({0: [0]}, lc._transform_get_columns(x)[1])

    def test_repeat_data_to_match_x_columns(self):
        np.random.seed(10)
        data = pd.DataFrame([
            [1, 1, 'WT', 0.0],
            [2, 2, 'WT', 0.0],
            [3, 1, 'WT', 0.0],
            [1, 3, 'WT', 1.0],
            [2, 3, 'WT', 1.0],
            [3, 3, 'WT', 1.0],
            [1, 1, 'KO', 1.0],
            [2, 2, 'KO', 1.0],
            [3, 2, 'KO', 1.0],
            [1, 1, 'DKO', 0.0],
            [2, 2, 'DKO', 0.0],
            [3, 1, 'DKO', 0.0],
            [1, 3, 'DKO', 1.0],
            [2, 3, 'DKO', 1.0],
            [3, 3, 'DKO', 1.0],
        ],
            columns=['trial', 'level', 'ec', 'time'])

        x = pd.DataFrame([
            [0, 10, 'WT', 0.0],
            [0, 20, 'WT', 1.0],
            [1, 11, 'WT', 0.0],
            [1, 21, 'WT', 1.0],
            [2, 10, 'KO', 0.0],
            [2, 11, 'KO', 1.0]],
            columns=['simulation', 'obs', 'ec', 'time'])

        combined_xy = pd.merge(x, data, how='outer').dropna()

        lc = LogisticClassifier(dataset=data, column_groups={'level': 'obs'}, classifier_type='ordinal_eoc',
                                do_fit_transform=False)
        for y_col, x_col in lc._columns_dict.items():
            lr = lc._transform_get_logistic_model(combined_xy[x_col], combined_xy[y_col].astype(int), y_col)
            test = lr.predict_proba(combined_xy[['obs']])
            target = [[7.04426051e-01, 2.92672144e-01, 2.90180501e-03],
                      [7.04426051e-01, 2.92672144e-01, 2.90180501e-03],
                      [7.04426051e-01, 2.92672144e-01, 2.90180501e-03],
                      [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                      [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                      [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                      [1.86427885e-04, 2.59936508e-02, 9.73819921e-01],
                      [1.86427885e-04, 2.59936508e-02, 9.73819921e-01],
                      [1.86427885e-04, 2.59936508e-02, 9.73819921e-01],
                      [7.24273585e-05, 1.02628619e-02, 9.89664711e-01],
                      [7.24273585e-05, 1.02628619e-02, 9.89664711e-01],
                      [7.24273585e-05, 1.02628619e-02, 9.89664711e-01],
                      [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                      [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                      [4.80732024e-01, 5.11831854e-01, 7.43612195e-03]]
            np.testing.assert_array_almost_equal(test, target)

        # make sure coef_ don't update when you call it again.
        lr = lc._transform_get_logistic_model(combined_xy['obs'], "do_fit_transform is False, No need for y", 'level')
        test = lr.predict_proba(combined_xy[['obs']])
        target = [[7.04426051e-01, 2.92672144e-01, 2.90180501e-03],
                  [7.04426051e-01, 2.92672144e-01, 2.90180501e-03],
                  [7.04426051e-01, 2.92672144e-01, 2.90180501e-03],
                  [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                  [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                  [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                  [1.86427885e-04, 2.59936508e-02, 9.73819921e-01],
                  [1.86427885e-04, 2.59936508e-02, 9.73819921e-01],
                  [1.86427885e-04, 2.59936508e-02, 9.73819921e-01],
                  [7.24273585e-05, 1.02628619e-02, 9.89664711e-01],
                  [7.24273585e-05, 1.02628619e-02, 9.89664711e-01],
                  [7.24273585e-05, 1.02628619e-02, 9.89664711e-01],
                  [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                  [4.80732024e-01, 5.11831854e-01, 7.43612195e-03],
                  [4.80732024e-01, 5.11831854e-01, 7.43612195e-03]]
        np.testing.assert_array_almost_equal(test, target)

    def test_check_classifier_type(self):
        # Raise error if not ordinal ordinal_eoc or nominal
        pass

    def test_prep_data(self):
        np.random.seed(10)
        data = pd.DataFrame([
            [1, 'alive', 'WT',  0.0],
            [2, 'alive', 'WT',  0.0],
            [3, 'dead',  'WT',  0.0],
            [1, 'alive', 'WT',  1.0],
            [2, 'dead',  'WT',  1.0],
            [3, 'dead',  'WT',  1.0],
            [1, 'alive', 'KO',  1.0],
            [2, 'alive', 'KO',  1.0],
            [3, 'alive', 'KO',  1.0],
            [1, 'alive', 'DKO', 0.0],
            [2, 'alive', 'DKO', 0.0],
            [3, 'alive', 'DKO', 0.0],
            [1, 'dead',  'DKO', 1.0],
            [2, 'dead',  'DKO', 1.0],
            [3, 'dead',  'DKO', 1.0],
        ],
            columns=['trial', 'result', 'ec', 'time'])

        x = pd.DataFrame([
            [0, 10, 'WT', 0.0],
            [0, 20, 'WT', 1.0],
            [1, 11, 'WT', 0.0],
            [1, 21, 'WT', 1.0],
            [2, 10, 'KO', 0.0],
            [2, 11, 'KO', 1.0]],
            columns=['simulation', 'obs', 'ec', 'time'])

        lc = LogisticClassifier(dataset=data, column_groups={'result': 'obs'}, classifier_type='nominal',
                                do_fit_transform=False)

        columns_set, columns_dict = lc._transform_get_columns(x)
        x_extra_columns = set(x.columns) - columns_set
        y_extra_columns = set(lc._data_df.columns) - set(lc._columns_dict.keys())
        y_cols = list(lc._columns_dict.keys())
        y = lc._data_df

        test = lc._prep_data(x, y, y_cols, x_extra_columns, y_extra_columns).sort_values('simulation').reset_index(drop=True)
        target =pd.DataFrame(
                [[ 0,   10,  'WT',   0.0,  'alive'],
                 [ 0,   10,  'WT',   0.0,  'alive'],
                 [ 0,   10,  'WT',   0.0,   'dead'],
                 [ 0,   20,  'WT',   1.0,  'alive'],
                 [ 0,   20,  'WT',   1.0,   'dead'],
                 [ 0,   20,  'WT',   1.0,   'dead'],
                 [ 1,   11,  'WT',   0.0,  'alive'],
                 [ 1,   11,  'WT',   0.0,  'alive'],
                 [ 1,   11,  'WT',   0.0,   'dead'],
                 [ 1,   21,  'WT',   1.0,  'alive'],
                 [ 1,   21,  'WT',   1.0,   'dead'],
                 [ 1,   21,  'WT',   1.0,   'dead'],
                 [ 2,   11,  'KO',   1.0,  'alive'],
                 [ 2,   11,  'KO',   1.0,  'alive'],
                 [ 2,   11,  'KO',   1.0,  'alive']], columns=['simulation', 'obs', 'ec', 'time', 'result'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform(self):
        np.random.seed(10)
        data = pd.DataFrame([
            [1, 'alive', 'WT', 0.0],
            [2, 'alive', 'WT', 0.0],
            [3, 'alive', 'WT', 0.0],
            [1, 'dead', 'WT', 1.0],
            [2, 'dead', 'WT', 1.0],
            [3, 'dead', 'WT', 1.0],
            [1, 'alive', 'KO', 1.0],
            [2, 'alive', 'KO', 1.0],
            [3, 'alive', 'KO', 1.0],
            [1, 'alive', 'DKO', 0.0],
            [2, 'alive', 'DKO', 0.0],
            [3, 'alive', 'DKO', 0.0],
            [1, 'dead', 'DKO', 1.0],
            [2, 'dead', 'DKO', 1.0],
            [3, 'dead', 'DKO', 1.0],
        ],
            columns=['trial', 'result', 'ec', 'time'])

        x = pd.DataFrame([
            [0, 10, 'WT', 0.0],
            [0, 20, 'WT', 1.0],
            [1, 10, 'WT', 0.0],
            [1, 21, 'WT', 1.0],
            [2, 10, 'KO', 0.0],
            [2, 11, 'KO', 1.0]],
            columns=['simulation', 'obs', 'ec', 'time'])

        lc = LogisticClassifier(dataset=data, column_groups={'result': 'obs'}, classifier_type='nominal')

        test = lc.transform(x).sort_values('simulation').reset_index(drop=True)
        target = pd.DataFrame(
            [[0.617041,      0.382959,  'WT',   0.0,           0],
             [0.617041,      0.382959,  'WT',   0.0,           0],
             [0.617041,      0.382959,  'WT',   0.0,           0],
             [0.312312,      0.687688,  'WT',   1.0,           0],
             [0.312312,      0.687688,  'WT',   1.0,           0],
             [0.312312,      0.687688,  'WT',   1.0,           0],
             [0.617041,      0.382959,  'WT',   0.0,           1],
             [0.617041,      0.382959,  'WT',   0.0,           1],
             [0.617041,      0.382959,  'WT',   0.0,           1],
             [0.285780,      0.714220,  'WT',   1.0,           1],
             [0.285780,      0.714220,  'WT',   1.0,           1],
             [0.285780,      0.714220,  'WT',   1.0,           1],
             [0.586708,      0.413292,  'KO',   1.0,           2],
             [0.586708,      0.413292,  'KO',   1.0,           2],
             [0.586708,      0.413292,  'KO',   1.0,           2]],
            columns=['result__alive',  'result__dead',  'ec',  'time',  'simulation'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_ordinal(self):
        data = pd.DataFrame([
            [1, 1, 'WT', 0.0],
            [2, 2, 'WT', 0.0],
            [3, 1, 'WT', 0.0],
            [1, 3, 'WT', 1.0],
            [2, 3, 'WT', 1.0],
            [3, 3, 'WT', 1.0],
            [1, 1, 'KO', 1.0],
            [2, 2, 'KO', 1.0],
            [3, 2, 'KO', 1.0]],
            columns=['trial', 'level', 'ec', 'time'])

        x = pd.DataFrame([
            [0, 10, 'WT', 0.0],
            [0, 20, 'WT', 1.0],
            [1, 11, 'WT', 0.0],
            [1, 21, 'WT', 1.0],
            [2, 10, 'KO', 0.0],
            [2, 11, 'KO', 1.0]],
            columns=['simulation', 'obs', 'ec', 'time'])

        lc = LogisticClassifier(dataset=data, column_groups={'level': 'obs'}, classifier_type='ordinal')
        test = lc.transform(x).sort_values('simulation').reset_index(drop=True)
        target = pd.DataFrame(
            [[0.704426, 0.292672, 0.002902, 0, 'WT', 0.0],
             [0.704426, 0.292672, 0.002902, 0, 'WT', 0.0],
             [0.704426, 0.292672, 0.002902, 0, 'WT', 0.0],
             [0.000186, 0.025994, 0.973820, 0, 'WT', 1.0],
             [0.000186, 0.025994, 0.973820, 0, 'WT', 1.0],
             [0.000186, 0.025994, 0.973820, 0, 'WT', 1.0],
             [0.480732, 0.511832, 0.007436, 1, 'WT', 0.0],
             [0.480732, 0.511832, 0.007436, 1, 'WT', 0.0],
             [0.480732, 0.511832, 0.007436, 1, 'WT', 0.0],
             [0.000072, 0.010263, 0.989665, 1, 'WT', 1.0],
             [0.000072, 0.010263, 0.989665, 1, 'WT', 1.0],
             [0.000072, 0.010263, 0.989665, 1, 'WT', 1.0],
             [0.480732, 0.511832, 0.007436, 2, 'KO', 1.0],
             [0.480732, 0.511832, 0.007436, 2, 'KO', 1.0],
             [0.480732, 0.511832, 0.007436, 2, 'KO', 1.0]],
            columns=['level__1', 'level__2', 'level__3', 'simulation', 'ec', 'time']
        )
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_less_precise=1)

    def test_transform_ordinal_reverse_order(self):
        np.random.seed(10)
        data = pd.DataFrame([
            [1, 1, 'WT', 0.0],
            [2, 2, 'WT', 0.0],
            [3, 1, 'WT', 0.0],
            [1, 3, 'WT', 1.0],
            [2, 3, 'WT', 1.0],
            [3, 3, 'WT', 1.0],
            [1, 1, 'KO', 1.0],
            [2, 2, 'KO', 1.0],
            [3, 2, 'KO', 1.0]],
            columns=['trial', 'level', 'ec', 'time'])

        x = pd.DataFrame([
            [0, 30, 'WT', 0.0],
            [0, 20, 'WT', 1.0],
            [1, 25, 'WT', 0.0],
            [1, 21, 'WT', 1.0],
            [2, 30, 'KO', 0.0],
            [2, 25, 'KO', 1.0]],
            columns=['simulation', 'obs', 'ec', 'time'])

        lc = LogisticClassifier(dataset=data, column_groups={'level': 'obs'}, classifier_type='ordinal')
        test = lc.transform(x).sort_values('simulation').reset_index(drop=True)
        target = pd.DataFrame(
            [[0.972899, 0.025845, 0.001256, 0, 'WT', 0.0],
             [0.972899, 0.025845, 0.001256, 0, 'WT', 0.0],
             [0.972899, 0.025845, 0.001256, 0, 'WT', 0.0],
             [0.005954, 0.111213, 0.882832, 0, 'WT', 1.0],
             [0.005954, 0.111213, 0.882832, 0, 'WT', 1.0],
             [0.005954, 0.111213, 0.882832, 0, 'WT', 1.0],
             [0.316811, 0.594493, 0.088696, 1, 'WT', 0.0],
             [0.316811, 0.594493, 0.088696, 1, 'WT', 0.0],
             [0.316811, 0.594493, 0.088696, 1, 'WT', 0.0],
             [0.014094, 0.226452, 0.759455, 1, 'WT', 1.0],
             [0.014094, 0.226452, 0.759455, 1, 'WT', 1.0],
             [0.014094, 0.226452, 0.759455, 1, 'WT', 1.0],
             [0.316811, 0.594493, 0.088696, 2, 'KO', 1.0],
             [0.316811, 0.594493, 0.088696, 2, 'KO', 1.0],
             [0.316811, 0.594493, 0.088696, 2, 'KO', 1.0]],
            columns=['level__1', 'level__2', 'level__3', 'simulation', 'ec', 'time']
        )
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_less_precise=1)

        lc = LogisticClassifier(dataset=data, column_groups={'level': 'obs'}, classifier_type='ordinal_eoc')
        test = lc.transform(x).sort_values('simulation').reset_index(drop=True)
        target = pd.DataFrame(
            [[0.185185, 0.574815, 0.24, 0, 'WT', 0.0],
             [0.185185, 0.574815, 0.24, 0, 'WT', 0.0],
             [0.185185, 0.574815, 0.24, 0, 'WT', 0.0],
             [0.185185, 0.574815, 0.24, 0, 'WT', 1.0],
             [0.185185, 0.574815, 0.24, 0, 'WT', 1.0],
             [0.185185, 0.574815, 0.24, 0, 'WT', 1.0],
             [0.185185, 0.574815, 0.24, 1, 'WT', 0.0],
             [0.185185, 0.574815, 0.24, 1, 'WT', 0.0],
             [0.185185, 0.574815, 0.24, 1, 'WT', 0.0],
             [0.185185, 0.574815, 0.24, 1, 'WT', 1.0],
             [0.185185, 0.574815, 0.24, 1, 'WT', 1.0],
             [0.185185, 0.574815, 0.24, 1, 'WT', 1.0],
             [0.185185, 0.574815, 0.24, 2, 'KO', 1.0],
             [0.185185, 0.574815, 0.24, 2, 'KO', 1.0],
             [0.185185, 0.574815, 0.24, 2, 'KO', 1.0]],
            columns=['level__1', 'level__2', 'level__3', 'simulation', 'ec', 'time']
        )
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], check_less_precise=1)

    def test_get_params(self):
        # before transform. Calling Coefficients produces calls transform.
        pass


class TestSampleAverage(unittest.TestCase):
    def test_defaults(self):
        sa = SampleAverage()
        assert sa._columns is None
        assert sa._group_by is None
        self.assertSetEqual(sa._columns_set, set([]))

    def test_columns_str(self):
        sa = SampleAverage(columns='col')
        self.assertListEqual(['col'], sa._columns)
        self.assertSetEqual({'col'}, sa._columns_set)

    def test_columns_list(self):
        st = SampleAverage(columns=['col1', 'col2'])
        self.assertListEqual(['col1', 'col2'], st._columns)
        self.assertSetEqual({'col1', 'col2'}, st._columns_set)

    def test_groupby_vector_like(self):
        st = SampleAverage(groupby=('col1', 'col2'))
        self.assertSetEqual({'col1', 'col2'}, set(st._group_by))

    def test_groupby_bad_group_types(self):
        with self.assertRaises(ValueError) as error:
            SampleAverage(groupby={'col1': 'col2'})
        self.assertTrue(error.exception.args[0] == "groupby must be a string or list of strings")

    def test_groupby_overlapping_cols(self):
        with self.assertRaises(ValueError) as error:
            SampleAverage(groupby={'col1', 'col2'}, columns='col1')
        self.assertTrue(error.exception.args[0] == "columns and groupby cannot be have any of the same column names.")

    def test_apply_noise(self):
        sa = SampleAverage()
        assert sa._apply_noise is False
        sa = SampleAverage(apply_noise=True)
        assert sa._apply_noise is True
        sa = SampleAverage(apply_noise="I didn't read the docs")
        assert sa._apply_noise is False

    def test_transform_wo_apply_noise_in_groups(self):
        df2 = pd.DataFrame({'X': ['B', 'B', 'A', 'A'], 'Y': [1, 2, 3, 4], 'Z': [1, 1, 2, 3]})
        sa = SampleAverage(columns=['Y'], groupby='X')
        test = sa._transform_wo_apply_noise_in_groups(df2, {'Y'})
        target = pd.DataFrame({'X': ['A', 'A', 'B'], 'Y': [3.5, 3.5, 1.5], 'Z': [2, 3, 1]})
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_wo_apply_noise_wo_groups(self):
        df2 = pd.DataFrame({'X': ['B', 'B', 'A', 'A'], 'Y': [1, 2, 3, 4], 'Z': [1, 1, 2, 3]})
        sa = SampleAverage(columns=['Y'])
        test = sa._transform_wo_apply_noise_wo_groups(df2, {'Y', 'Z'})
        target = pd.DataFrame({'X': ['B', 'A'], 'Y': [2.5, 2.5], 'Z': [1.75, 1.75]})
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_set_noise_term(self):
        sa = SampleAverage(columns=['Y'])
        self.assertDictEqual(sa._noise_term, {'default':0.0})
        test = sa._set_noise_term(5)
        self.assertDictEqual(test, {'default': 5})
        test = sa._set_noise_term({'Y':5})
        self.assertDictEqual(test, {'default': 0.0, 'Y': 5})

    def test_transform_w_apply_noise_wo_groups(self):
        np.random.seed(0)
        sa = SampleAverage(sample_size=4, variances={'Y': 10})
        df2 = pd.DataFrame({'X': ['B', 'B', 'A', 'A'], 'Y': [1, 2, 3, 4], 'Z': [1, 1, 2, 3]})
        test = sa._transform_w_apply_noise_wo_groups(df2, {'Y', 'Z'})

        target = pd.DataFrame([[22.381083,  2.594476,  'B'],
                               [-7.903609,  1.941561,  'B'],
                               [12.614164,  2.218535,  'B'],
                               [ 0.888727,  2.822746,  'B'],
                               [22.381083,  2.594476,  'A'],
                               [-7.903609,  1.941561,  'A'],
                               [12.614164,  2.218535,  'A'],
                               [ 0.888727,  2.822746,  'A']], columns=['Y', 'Z', 'X'])
        print(test)
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_w_apply_noise_w_groups(self):
        sa = SampleAverage(sample_size=4, apply_noise=True, variances={'Y': 10}, groupby='X', columns=['Y'])
        df2 = pd.DataFrame({'X': ['B', 'B', 'A', 'A'], 'Y': [1, 2, 3, 4], '2^Y': [1, 1, 2, 3]})
        np.random.seed(0)
        test = sa.transform(df2)
        target = pd.DataFrame([[23.109359, 3.382026, 'A'],
                               [-6.761418, 2.700079, 'A'],
                               [13.475928, 2.989369, 'A'],
                               [ 1.910749, 3.620447, 'A'],
                               [ 9.490896, 1.000000, 'B'],
                               [ 2.777588, 1.000000, 'B'],
                               [ 6.160564, 1.000000, 'B'],
                               [ 5.003580, 1.000000, 'B']], columns=['Y', '2^Y', 'X'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_drop_columns(self):
        np.random.seed(0)
        sa = SampleAverage(sample_size=4,
                           apply_noise=True,
                           groupby='X',
                           drop_columns=['Y', 'A'])
        sa.set_params(noise_term__Y=10)
        df2 = pd.DataFrame({'X': ['B', 'B', 'A', 'A'], 'Y': [1, 2, 3, 4], 'Z': [1, 1, 2, 3]})
        np.random.seed(0)
        test = sa.transform(df2)
        target = pd.DataFrame([[3.382026, 'A'],
                               [2.700079, 'A'],
                               [2.989369, 'A'],
                               [3.620447, 'A'],
                               [1.000000, 'B'],
                               [1.000000, 'B'],
                               [1.000000, 'B'],
                               [1.000000, 'B']],
                              columns=['Z', 'X'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_set_params_and_get_params(self):
        sa = SampleAverage()
        self.assertDictEqual(sa.get_params(), {'noise_term__default': 0.0, 'sample_size': 50})
        sa.set_params(noise_term=42)
        self.assertDictEqual(sa.get_params(), {'noise_term__default': 42, 'sample_size': 50})
        sa.set_params(noise_term__x=500)
        self.assertDictEqual(sa.get_params(), {'noise_term__default': 42, 'noise_term__x': 500, 'sample_size': 50})


class TestTransform(unittest.TestCase):
    def test_parse_column_names(self):
        cell_death_data = pd.DataFrame(np.arange(6).reshape(2, 3),
                                       columns=['p53', 'RIP1__sum', 'p53 KO'])
        scale = Scale(columns=['p53', 'RIP1__sum'], scale_fn=polynomial_features, degree=2)
        scaled_x = scale.transform(cell_death_data[['p53', 'RIP1__sum', 'p53 KO']])

        test = scale._parse_column_names(set(scaled_x.columns), {'p53', 'RIP1__sum'})
        target = {'RIP1__sum^2', 'p53', 'p53^2', 'RIP1__sum'}
        assert 'p53$RIP1__sum' in test or 'RIP1__sum$p53' in test
        test -= {'p53$RIP1__sum', 'RIP1__sum$p53'}
        self.assertSetEqual(test, target)

        test = scale._parse_column_names(set(scaled_x.columns), {'RIP1__sum'})
        target = {'RIP1__sum^2', 'RIP1__sum'}
        assert 'p53$RIP1__sum' in test or 'RIP1__sum$p53' in test
        test -= {'p53$RIP1__sum', 'RIP1__sum$p53'}

        self.assertSetEqual(test, target)

        test = scale._parse_column_names(set(scaled_x.columns), {'p53'})
        target = {'p53', 'p53^2'}
        assert 'p53$RIP1__sum' in test or 'RIP1__sum$p53' in test
        test -= {'p53$RIP1__sum', 'RIP1__sum$p53'}
        self.assertSetEqual(test, target)

        test = scale._parse_column_names(set(scaled_x.columns), {'RIP1'})
        assert 'p53$RIP1__sum' in test or 'RIP1__sum$p53' in test
        target = {'RIP1__sum^2', 'RIP1__sum', 'RIP1'}
        test -= {'p53$RIP1__sum', 'RIP1__sum$p53'}
        self.assertSetEqual(test, target)


class TestCumulativeComputation(unittest.TestCase):
    def test_check_operation(self):
        cum_comp = CumulativeComputation()
        test = cum_comp._check_operation('max')
        target = 'max'
        assert test == target
        with self.assertRaises(ValueError) as error:
            cum_comp._check_operation('unsupported_value')
        self.assertTrue(error.exception.args[0] == "'operation' must be one of the supported "
                                                   "operations: 'min', 'max', 'sum', and 'prod'.")

    def test_get_set_params(self):
        cum_comp = CumulativeComputation()
        cum_comp.set_params(operation='min', keep_old_columns=True)
        self.assertDictEqual(cum_comp.get_params(), {'operation':'min', 'keep_old_columns':True})

    def test_transform_not_in_groups(self):
        df = pd.DataFrame([[2.0, 1.0],
                           [3.0, np.nan],
                           [1.0, 0.0]],
                          columns=list('AB'))
        cumprod = CumulativeComputation(operation='prod', keep_old_columns=True)
        test = cumprod.transform(df)
        target = pd.DataFrame([
            [2.0,       1.0,        'a',    2.0,       1.0],
            [6.0,    np.NaN,        'a',    3.0,    np.NaN],
            [6.0,       0.0,        'a',    1.0,       0.0]],
            columns=['A__prod', 'B__prod', 'C', 'A', 'B'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_transform_in_groups(self):
        df = pd.DataFrame([[2.0, 1.0,    'a'],
                           [3.0, np.nan, 'a'],
                           [1.0, 0.0,    'a'],
                           [3.0, 1.0,    'b'],
                           [2.0, np.nan, 'b'],
                           [4.0, 0.0,    'b']],
                          columns=list('ABC'))
        test = CumulativeComputation(operation='max', keep_old_columns=True, groupby='C').transform(df)
        target = pd.DataFrame([
            [2.0,     1.0,  'a',  2.0,    1.0],
            [3.0,  np.NaN,  'a',  3.0, np.NaN],
            [3.0,     1.0,  'a',  1.0,    0.0],
            [3.0,     1.0,  'b',  3.0,    1.0],
            [3.0,  np.NaN,  'b',  2.0, np.NaN],
            [4.0,     1.0,  'b',  4.0,    0.0]],
            columns=['A__max', 'B__max', 'C', 'A', 'B'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

        test = CumulativeComputation(operation='sum', keep_old_columns=True, groupby='C').transform(df)
        target = pd.DataFrame([
            [2.0,   1.0,    'a', 2.0,    1.0],
            [5.0,   np.NaN, 'a', 3.0, np.NaN],
            [6.0,   1.0,    'a', 1.0,    0.0],
            [3.0,   1.0,    'b', 3.0,    1.0],
            [5.0,   np.NaN, 'b', 2.0, np.NaN],
            [9.0,   1.0,    'b', 4.0,    0.0]],
            columns=['A__sum', 'B__sum', 'C', 'A', 'B'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

