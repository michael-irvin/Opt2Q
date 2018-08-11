from pysb.examples.michment import model as pysb_model
from opt2q.noise import NoiseModel
from opt2q.utils import MissingParametersErrors
from nose.tools import *
import numpy as np
import pandas.util.testing as pd_testing
import pandas as pd
import unittest


class TestNoise(unittest.TestCase):
    def test_check_required_columns_none_in(self):
        target = pd.DataFrame()
        nm = NoiseModel()
        test = nm._check_required_columns(param_df=None)
        pd_testing.assert_almost_equal(test, target)

    def test_check_required_columns_bad_var_name(self):
        nm = NoiseModel()
        with self.assertRaises(KeyError) as error:
            nm._check_required_columns(param_df=pd.DataFrame(), var_name='unsupported dataframe')
        self.assertEqual(error.exception.args[0],
                         "'unsupported dataframe' is not supported")

    def test_check_required_columns_missing_cols(self):
        nm = NoiseModel()
        with self.assertRaises(ValueError) as error:
            nm._check_required_columns(param_df=pd.DataFrame([1]), var_name='param_covariance')
        self.assertTrue(
            error.exception.args[0] == "'param_covariance' must be a pd.DataFrame with the following column names: "
                                       "'param_i', 'param_j', and 'value'." or
            error.exception.args[0] == "'param_covariance' must be a pd.DataFrame with the following column names: "
                                       "'param_i', 'value', and 'param_j'." or
            error.exception.args[0] == "'param_covariance' must be a pd.DataFrame with the following column names: "
                                       "'param_j', 'param_i', and 'value'." or
            error.exception.args[0] == "'param_covariance' must be a pd.DataFrame with the following column names: "
                                       "'param_j', 'value', and 'param_i'." or
            error.exception.args[0] == "'param_covariance' must be a pd.DataFrame with the following column names: "
                                       "'value', 'param_j', and 'param_i'." or
            error.exception.args[0] == "'param_covariance' must be a pd.DataFrame with the following column names: "
                                       "'value', 'param_i', and 'param_j'.")

    def test_check_required_columns_mean(self):
        target = pd.DataFrame([[1, 2]], columns=['param', 'value'])
        nm = NoiseModel()
        test = nm._check_required_columns(param_df=target)
        pd_testing.assert_almost_equal(test, target)

    def test_check_required_columns_cov(self):
        target = pd.DataFrame([[1, 2, 3]], columns=['param_i', 'param_j', 'value'])
        nm = NoiseModel()
        test = nm._check_required_columns(param_df=target, var_name='param_covariance')
        pd_testing.assert_almost_equal(test, target)

    def test_add_apply_noise_col(self):
        input_arg = pd.DataFrame([[1, 2]], columns=['param', 'value'])
        target = pd.DataFrame([[1, 2, False]], columns=['param', 'value', 'apply_noise'])
        nm = NoiseModel()
        test = nm._add_apply_noise_col(input_arg)
        pd_testing.assert_almost_equal(test, target)

    def test_add_apply_noise_col_preexisting(self):
        input_arg = pd.concat([pd.DataFrame([[1, 2]],columns=['param', 'value']),
                               pd.DataFrame([[3, 4, True]], columns=['param', 'value', 'apply_noise'])],
                              ignore_index=True, sort=False)
        target = pd.DataFrame([[1, 2, False],[3, 4, True]], columns=['param', 'value', 'apply_noise'])
        nm = NoiseModel()
        test = nm._add_apply_noise_col(input_arg)
        pd_testing.assert_almost_equal(test[['param', 'value', 'apply_noise']],
                                       target[['param', 'value', 'apply_noise']],
                                       check_dtype=False)

    def test_add_apply_noise_col_preexisting_with_np_nan(self):
        input_arg = pd.DataFrame([[1, 2, np.NaN], [3, 4, True]], columns=['param', 'value', 'apply_noise'])
        target = pd.DataFrame([[1, 2, False], [3, 4, True]], columns=['param', 'value', 'apply_noise'])
        nm = NoiseModel()
        test = nm._add_apply_noise_col(input_arg)
        pd_testing.assert_almost_equal(test[['param', 'value', 'apply_noise']],
                                       target[['param', 'value', 'apply_noise']],
                                       check_dtype=False)

    def test_these_columns_cannot_annotate_exp_cons(self):
        nm = NoiseModel()
        nm.required_columns = {'a':{'a', 'b', 'c'}, 'b':{'e', 'f', 'g'}}
        nm.other_useful_columns = {'h', 'i'}
        target = {'a', 'b', 'c', 'e', 'f', 'g','h', 'i'}
        test = nm._these_columns_cannot_annotate_exp_cons()
        self.assertSetEqual(test, target)

    def test_copy_experimental_conditions_to_second_df(self):
        wo = pd.DataFrame([1, 2], columns=['a'])
        w = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'b']], columns=['a', 'b'])
        wo_new = pd.DataFrame([[1, 'a'], [2, 'a'], [1, 'b'], [2, 'b']], columns=['a', 'b'])
        target = wo_new
        nm = NoiseModel()
        test = nm._copy_experimental_conditions_to_second_df(wo, set([]), w, {'b'})
        pd_testing.assert_frame_equal(target, test[0], check_dtype=False, check_index_type=False, check_column_type=False)

    def test_copy_experimental_conditions_to_second_df_reversed_order(self):
        wo = pd.DataFrame([1, 2], columns=['a'])
        w = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'b']], columns=['a', 'b'])
        wo_new = pd.DataFrame([[1, 'a'], [2, 'a'], [1, 'b'], [2, 'b']], columns=['a', 'b'])
        target = wo_new
        nm = NoiseModel()
        test = nm._copy_experimental_conditions_to_second_df(w, {'b'}, wo, set([]),)
        pd_testing.assert_frame_equal(target, test[1], check_dtype=False, check_column_type=False)

    def test_copy_experimental_conditions_to_second_df_empty_df(self):
        wo = pd.DataFrame(columns=['a'])
        w = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'b']], columns=['a', 'b'])
        wo_new = pd.DataFrame(columns=['a', 'b'])
        target = wo_new
        nm = NoiseModel()
        test = nm._copy_experimental_conditions_to_second_df(w, {'b'}, wo, set([]),)
        pd_testing.assert_frame_equal(target, test[1],
                                      check_dtype=False,
                                      check_column_type=False,
                                      check_index_type=False)

    def test_test_copy_experimental_conditions_to_second_df_both_df(self):
        df1 = pd.DataFrame([[1, 'a'], [2, 'b'], [3, 'b']], columns=['a', 'b'])
        df2 = pd.DataFrame([[1, 'e'], [2, 'b'], [3, 'b']], columns=['a', 'b'])
        target = pd.DataFrame(['a', 'b', 'e'], columns=['b'])
        nm = NoiseModel()
        test = nm._copy_experimental_conditions_to_second_df(df1, {'b'}, df2, {'b'})
        pd_testing.assert_frame_equal(df1, test[0],
                                      check_dtype=False,
                                      check_column_type=False,
                                      check_index_type=False)
        pd_testing.assert_frame_equal(df2, test[1],
                                      check_dtype=False,
                                      check_column_type=False,
                                      check_index_type=False)
        pd_testing.assert_frame_equal(target, test[3],
                                      check_dtype=False,
                                      check_column_type=False,
                                      check_index_type=False)

    def test_combine_param_i_j(self):
        cov_ = pd.DataFrame([['a', 'a', 1],
                             ['b', 'c', 3],
                             ['c', 'd', 2]], columns=['param_i', 'param_j', 'value'])
        target = pd.DataFrame(['a',
                               'b',
                               'c',
                               'd'], columns=['param'])
        nm = NoiseModel()
        test = nm._combine_param_i_j(cov_)
        pd_testing.assert_frame_equal(test, target)

    def test_combine_param_i_j_w_ec(self):
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['a', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        target = pd.DataFrame([['a', 'ec1'],
                               ['b', 'ec1'],
                               ['a', 'ec2'],
                               ['c', 'ec1'],
                               ['d', 'ec2']], columns=['param', 'ec'])
        nm = NoiseModel()
        test = nm._combine_param_i_j(cov_)
        pd_testing.assert_frame_equal(test, target)

    def test_add_params_from_param_covariance_empty_mean_and_cov(self):
        mean = pd.DataFrame()
        cov_ = pd.DataFrame()
        target = pd.DataFrame()
        nm = NoiseModel()
        test = nm._add_params_from_param_covariance(mean, cov_)
        pd_testing.assert_frame_equal(target, test)

    def test_add_params_from_param_covariance_empty_mean(self):
        mean = pd.DataFrame()
        cov_ = pd.DataFrame([['a', 'a', 1],
                             ['b', 'c', 1],
                             ['c', 'd', 1]], columns=['param_i', 'param_j', 'value'])
        target = pd.DataFrame([['a', np.NaN, True],
                               ['b', np.NaN, True],
                               ['c', np.NaN, True],
                               ['d', np.NaN, True]], columns=['param', 'value', 'apply_noise'])
        nm = NoiseModel()
        test = nm._add_params_from_param_covariance(mean, cov_)
        cols = ['param', 'value', 'apply_noise']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    def test_add_params_from_param_covariance_empty_mean_ec_included(self):
        mean = pd.DataFrame()
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['a', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        target = pd.DataFrame([['a', np.NaN, True, 'ec1'],
                               ['b', np.NaN, True, 'ec1'],
                               ['c', np.NaN, True, 'ec1'],
                               ['a', np.NaN, True, 'ec2'],
                               ['d', np.NaN, True, 'ec2']], columns=['param', 'value', 'apply_noise', 'ec'])
        nm = NoiseModel()
        test = nm._add_params_from_param_covariance(mean, cov_)
        cols = ['param', 'value', 'apply_noise', 'ec']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    def test_add_params_from_param_covariance_param_mean(self):
        mean = pd.DataFrame([['a', 1],
                             ['b', 1],
                             ['c', 1]], columns=['param', 'value'])
        cov_ = pd.DataFrame([['a', 'a', 1],
                             ['b', 'c', 1],
                             ['c', 'd', 1]], columns=['param_i', 'param_j', 'value'])
        target = pd.DataFrame([['a', 1,      True],
                               ['b', 1,      True],
                               ['c', 1,      True],
                               ['d', np.NaN, True]], columns=['param', 'value', 'apply_noise'])
        nm = NoiseModel()
        test = nm._add_params_from_param_covariance(mean, cov_)
        cols = ['param', 'value', 'apply_noise']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    def test_add_params_from_param_covariance_param_mean_cov_ec(self):
        mean = pd.DataFrame([['a', 1],
                             ['b', 1],
                             ['c', 1]], columns=['param', 'value'])
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['c', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        target = pd.DataFrame([['a', 1,      True,   'ec1'],
                               ['b', 1,      True,   'ec1'],
                               ['c', 1,      True,   'ec1'],
                               ['a', 1,      np.NaN, 'ec2'],
                               ['b', 1,      np.NaN, 'ec2'],
                               ['c', 1,      True,   'ec2'],
                               ['d', np.NaN, True,   'ec2']], columns=['param', 'value', 'apply_noise', 'ec'])
        nm = NoiseModel()
        mean, c, d, e = nm._check_experimental_condition_cols(mean, cov_)
        test = nm._add_params_from_param_covariance(mean, cov_)
        cols = ['param', 'value', 'apply_noise', 'ec']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    def test_add_params_from_param_covariance_param_mean_ec_cov_ec(self):
        mean = pd.DataFrame([['a', 1, 'ec1'],
                             ['b', 1, 'ec1'],
                             ['c', 1, 'ec2']], columns=['param', 'value', 'ec'])
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['c', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        target = pd.DataFrame([['a', 1,      True,   'ec1'],
                               ['b', 1,      True,   'ec1'],
                               ['c', np.NaN, True,   'ec1'],
                               ['c', 1,      True,   'ec2'],
                               ['d', np.NaN, True,   'ec2']], columns=['param', 'value', 'apply_noise', 'ec'])
        nm = NoiseModel()
        test = nm._add_params_from_param_covariance(mean, cov_)
        cols = ['param', 'value', 'apply_noise', 'ec']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    def test_add_params_from_param_covariance_param_mean_ec_apply_noise_cov_ec_(self):
        mean = pd.DataFrame([['a', 1, 'ec1', False],
                             ['b', 1, 'ec1', False],
                             ['a', 1, 'ec2', False]], columns=['param', 'value', 'ec', 'apply_noise'])
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['c', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        target = pd.DataFrame([['a', 1,      True,   'ec1'],
                               ['b', 1,      True,   'ec1'],
                               ['c', np.NaN, True,   'ec1'],
                               ['a', 1,      False,  'ec2'],
                               ['c', np.NaN, True,   'ec2'],
                               ['d', np.NaN, True,   'ec2']], columns=['param', 'value', 'apply_noise', 'ec'])
        nm = NoiseModel()
        test = nm._add_params_from_param_covariance(mean, cov_)
        cols = ['param', 'value', 'apply_noise', 'ec']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    def test_add_params_from_param_covariance_add_apply_noise_col(self):
        mean = pd.DataFrame([['a', 1],
                             ['b', 1],
                             ['c', 1]], columns=['param', 'value'])
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['c', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        target = pd.DataFrame([['a', 1,      True,  'ec1'],
                               ['b', 1,      True,  'ec1'],
                               ['c', 1,      True,  'ec1'],
                               ['a', 1,      False, 'ec2'],
                               ['b', 1,      False, 'ec2'],
                               ['c', 1,      True,  'ec2'],
                               ['d', np.NaN, True,  'ec2']], columns=['param', 'value', 'apply_noise', 'ec'])
        nm = NoiseModel()
        mean, c, d, e = nm._check_experimental_condition_cols(mean, cov_)
        mean = nm._add_params_from_param_covariance(mean, cov_)
        test = nm._add_apply_noise_col(mean)
        cols = ['param', 'value', 'apply_noise', 'ec']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols])

    @raises(MissingParametersErrors)
    def test_missing_parameters_error_raised(self):
        mean = pd.DataFrame([['a', 1],
                             ['b', 1],
                             ['c', 1]], columns=['param', 'value'])
        cov_ = pd.DataFrame([['a', 'a', 1, 'ec1'],
                             ['b', 'c', 1, 'ec1'],
                             ['c', 'd', 1, 'ec2']], columns=['param_i', 'param_j', 'value', 'ec'])
        nm = NoiseModel(mean, cov_)  # d is missing

    def test_add_missing_params(self):
        mean_ = pd.DataFrame([['a', 1,      True, 'ec1'],
                              ['b', 1,      True, 'ec1'],
                              ['c', np.NaN, True, 'ec1'],
                              ['c', 1,      True, 'ec2'],
                              ['d', np.NaN, True, 'ec2']],
                             columns=['param', 'value', 'apply_noise', 'ec'])
        NoiseModel.default_param_values = {'c':10, 'd':13}
        nm = NoiseModel()
        test = nm._add_missing_param_values(mean_)
        target = pd.DataFrame([['a', 1,  True, 'ec1'],
                               ['b', 1,  True, 'ec1'],
                               ['c', 10, True, 'ec1'],
                               ['c', 1,  True, 'ec2'],
                               ['d', 13, True, 'ec2']],
                             columns=['param', 'value', 'apply_noise', 'ec'])
        cols = ['param', 'value', 'apply_noise', 'ec']
        NoiseModel.default_param_values = None
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols],
                                      check_dtype=False)

    def test_add_missing_params_from_model(self):
        mean_ = pd.DataFrame([['a', 1,      True, 'ec1'],
                              ['b', 1,      True, 'ec1'],
                              ['vol', np.NaN, True, 'ec1']],
                             columns=['param', 'value', 'apply_noise', 'ec'])
        nm = NoiseModel()
        test = nm._add_missing_param_values(mean_, model=pysb_model)
        target = pd.DataFrame([['a', 1,  True, 'ec1'],
                               ['b', 1,  True, 'ec1'],
                               ['vol', 10, True, 'ec1']],
                              columns=['param', 'value', 'apply_noise', 'ec'])
        cols = ['param', 'value', 'apply_noise', 'ec']
        pd_testing.assert_frame_equal(target.sort_values(by=cols).reset_index(drop=True)[cols],
                                      test.sort_values(by=cols).reset_index(drop=True)[cols],
                                      check_dtype=False)