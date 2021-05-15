from pysb.examples.michment import model as pysb_model
from opt2q.noise import NoiseModel
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
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
                                      test.sort_values(by=cols).reset_index(drop=True)[cols],
                                      check_dtype=False)

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
                                      test.sort_values(by=cols).reset_index(drop=True)[cols],
                                      check_dtype=False)

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

    def test_add_num_sims_col_to_experimental_conditions_df_w_num_sims_exp_c(self):
        in0 = pd.DataFrame([[1,      'a', 'a', 0.0, False],
                            [2,      'a', 'a', 0.0, False],
                            [3,      'b', 'a', 0.0, False],
                            [3,      'b', 'a', 0.0, False],
                            [1,      'b', 'c', 0.0, False],
                            [np.NaN, 'b', 'c', 0.0, False]],
                           columns=['num_sims', 'ec1', 'ec2', 'values', 'apply_noise'])
        in1 = pd.DataFrame([['a', 'a'],
                            ['b', 'c'],
                            ['b', 'a']], columns=['ec1', 'ec2'])
        in2 = {'ec1', 'ec2'}
        target = pd.DataFrame([['a', 'a', 2],
                               ['b', 'c', 1],
                               ['b', 'a', 3]], columns=['ec1', 'ec2', 'num_sims'])
        nm=NoiseModel()
        test = nm._add_num_sims_col_to_experimental_conditions_df(in0, in1, in2)
        pd_testing.assert_frame_equal(test, target, check_dtype=False)

    def test_add_num_sims_col_to_experimental_conditions_df_w_num_sims_no_exp_c(self):
        in0 = pd.DataFrame([[1, 0.0, False],
                            [2, 0.0, True],
                            [3, 0.0, False],
                            [3, 0.0, False],
                            [1, 0.0, False],
                            [np.NaN, 0.0, True]],
                           columns=['num_sims', 'values', 'apply_noise'])
        in1 = pd.DataFrame()
        in2 = set([])
        target = pd.DataFrame([3], columns=['num_sims'])
        nm = NoiseModel()
        test = nm._add_num_sims_col_to_experimental_conditions_df(in0, in1, in2)
        pd_testing.assert_frame_equal(test, target, check_dtype=False)

    def test_add_num_sims_col_to_experimental_conditions_df_no_num_sims_exp_c(self):
        in0 = pd.DataFrame([['a', 'a', 0.0, True],
                            ['a', 'a', 0.0, False],
                            ['b', 'a', 0.0, False],
                            ['b', 'a', 0.0, True],
                            ['b', 'c', 0.0, False],
                            ['b', 'c', 0.0, False]],
                           columns=['ec1', 'ec2', 'values', 'apply_noise'])
        in1 = pd.DataFrame([['a', 'a'],
                            ['b', 'c'],
                            ['b', 'a']], columns=['ec1', 'ec2'])
        in2 = {'ec1', 'ec2'}
        target = pd.DataFrame([['a', 'a', NoiseModel.default_sample_size],
                               ['b', 'c', 1],
                               ['b', 'a', NoiseModel.default_sample_size]],
                              columns=['ec1', 'ec2', 'num_sims'])
        nm = NoiseModel()
        test = nm._add_num_sims_col_to_experimental_conditions_df(in0, in1, in2)
        pd_testing.assert_frame_equal(test, target, check_dtype=False)

    def test_add_num_sims_col_to_experimental_conditions_df_no_num_sims_no_exp_c(self):
        in0 = pd.DataFrame([[0.0, False],
                            [0.0, True],
                            [0.0, False],
                            [0.0, False],
                            [0.0, False],
                            [0.0, True]],
                           columns=['values', 'apply_noise'])
        in1 = pd.DataFrame()
        in2 = set([])
        target = pd.DataFrame([NoiseModel.default_sample_size], columns=['num_sims'])
        nm = NoiseModel()
        test = nm._add_num_sims_col_to_experimental_conditions_df(in0, in1, in2)
        pd_testing.assert_frame_equal(test, target, check_dtype=False)

    def test_init_case1(self):
        mean_values = pd.DataFrame([['A', 1.0, 'KO'], ['B', 1.0, 'WT'], ['A', 1.0, 'WT']],
                                   columns=['param', 'value', 'ec'])
        noise_model = NoiseModel(param_mean=mean_values)
        target = pd.DataFrame([['A', 1.0, 'KO', False], ['B', 1.0, 'WT',False], ['A', 1.0, 'WT',False]],
                              columns=['param', 'value', 'ec', 'apply_noise'])
        test = noise_model.param_mean
        pd_testing.assert_frame_equal(test, target, check_dtype=False)

    def test_param_mean_update(self):
        mean_values = pd.DataFrame([['A', 1.0, 'KO'], ['B', 1.0, 'WT'], ['A', 1.0, 'WT']],
                                   columns=['param', 'value', 'ec'])
        noise_model = NoiseModel(param_mean=mean_values)
        noise_model.update_values(param_mean=pd.DataFrame([['A', 2]], columns=['param', 'value']))
        test = noise_model.param_mean
        target = pd.DataFrame([['A', 2.0, 'KO', False, 1], ['A', 2.0, 'WT', False, 1], ['B', 1.0, 'WT', False, 1]],
                              columns=['param', 'value', 'ec', 'apply_noise', 'num_sims'])
        pd_testing.assert_frame_equal(test[test.columns], target[test.columns], check_dtype=False)

    def test_param_mean_update_with_num_sims(self):
        mean_values = pd.DataFrame([['A', 1.0, 3],
                                    ['B', 1.0, 1]],
                                   columns=['param', 'value', 'num_sims'])
        noise_model = NoiseModel(param_mean=mean_values)
        noise_model.update_values(param_mean=pd.DataFrame([['B', 2, 10]], columns=['param', 'value', 'num_sims']))
        target_mean = pd.DataFrame([['A', 1.0, 10,  False],
                                    ['B', 2.0, 10, False]],
                                   columns=['param', 'value', 'num_sims', 'apply_noise'])
        target_exp_cols = pd.DataFrame([10], columns=['num_sims'])
        pertinent_cols = ['param', 'value', 'apply_noise']
        test_mean = noise_model.param_mean
        test_exp_cos = noise_model._exp_cols_df
        pd_testing.assert_frame_equal(test_mean[pertinent_cols], target_mean[pertinent_cols], check_dtype=False)
        pd_testing.assert_frame_equal(test_exp_cos[test_exp_cos.columns], target_exp_cols[test_exp_cos.columns],
                                      check_dtype=False)

    def test_param_mean_update_num_sims_num_sims_not_in_initial_pass(self):
        mean_values = pd.DataFrame([['A', 1.0],
                                    ['B', 1.0]],
                                   columns=['param', 'value'])
        noise_model = NoiseModel(param_mean=mean_values)
        noise_model.update_values(param_mean=pd.DataFrame([['B', 2, 10]], columns=['param', 'value', 'num_sims']))
        target_mean = pd.DataFrame([['A', 1.0, 10,  False],
                                    ['B', 2.0, 10, False]],
                                   columns=['param', 'value', 'num_sims', 'apply_noise'])
        target_exp_cols = pd.DataFrame([10], columns=['num_sims'])
        pertinent_cols = ['param', 'value', 'apply_noise']
        test_mean = noise_model.param_mean
        test_exp_cos = noise_model._exp_cols_df
        pd_testing.assert_frame_equal(test_mean[pertinent_cols], target_mean[pertinent_cols], check_dtype=False)
        pd_testing.assert_frame_equal(test_exp_cos[test_exp_cos.columns], target_exp_cols[test_exp_cos.columns],
                                      check_dtype=False)

    def test_param_mean_update_with_num_sims_experiments(self):
        mean_values = pd.DataFrame([['A', 1.0, 3, 'KO'],
                                    ['B', 1.0, 1, 'WT'],
                                    ['A', 1.0, 1, 'WT']],
                                   columns=['param', 'value', 'num_sims', 'ec'])
        noise_model = NoiseModel(param_mean=mean_values)
        noise_model.update_values(param_mean=pd.DataFrame([['B', 2, 10]], columns=['param', 'value', 'num_sims']))
        target_mean = pd.DataFrame([['A', 1.0,  3, False, 'KO'],
                                    ['A', 1.0,  1, False, 'WT'],
                                    ['B', 2.0, 10, False, 'WT']],
                                   columns=['param', 'value', 'num_sims', 'apply_noise', 'ec'])
        pertinent_cols = ['param', 'value', 'apply_noise', 'ec']
        target_exp_cols = pd.DataFrame([[3,'KO'],[10, 'WT']], columns=['num_sims', 'ec'])
        test_mean = noise_model.param_mean
        test_exp_cos = noise_model._exp_cols_df
        pd_testing.assert_frame_equal(test_mean[pertinent_cols], target_mean[pertinent_cols], check_dtype=False)
        pd_testing.assert_frame_equal(test_exp_cos[test_exp_cos.columns], target_exp_cols[test_exp_cos.columns],
                                      check_dtype=False)

    def test_param_covariance_update(self):
        mean = pd.DataFrame([['a', 1.0], ['b', 1.0], ['c', 1.0]], columns=['param', 'value'])
        cov = pd.DataFrame([['a', 'b', 0.01],
                            ['c', 'b', 0.01]], columns=['param_i', 'param_j', 'value'])
        nm = NoiseModel(param_mean=mean, param_covariance=cov)
        nm.update_values(param_covariance=pd.DataFrame([['a', 0.1]], columns=['param_j', 'value']))
        test = nm.param_covariance
        target = pd.DataFrame([['a', 'b', 0.1],
                               ['c', 'b', 0.01]], columns=['param_i', 'param_j', 'value'])
        pd_testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_param_covariance_update_case2(self):
        mean = pd.DataFrame([['a', 1.0], ['b', 1.0], ['c', 1.0]], columns=['param', 'value'])
        cov = pd.DataFrame([['a', 'b', 0.01],
                            ['c', 'b', 0.01]], columns=['param_i', 'param_j', 'value'])
        nm = NoiseModel(param_mean=mean, param_covariance=cov)
        nm.update_values(param_covariance=pd.DataFrame([['a', 'b', 0.1]], columns=['param_j', 'param_i','value']))
        test = nm.param_covariance
        target = pd.DataFrame([['a', 'b', 0.1],
                               ['c', 'b', 0.01]], columns=['param_i', 'param_j', 'value'])
        pd_testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_param_mean_num_sims_only(self):
        param_mean = pd.DataFrame([['vol', 10, 'wild_type', False],
                                   ['kr', 100, 'high_affinity', np.NaN],
                                   ['kcat', 100, 'high_affinity', np.NaN],
                                   ['vol', 10, 'pt_mutation', True],
                                   ['kr', 1000, 'pt_mutation', False],
                                   ['kcat', 10, 'pt_mutation', True]],
                                  columns=['param', 'value', 'exp_condition', 'apply_noise'])
        param_cov = pd.DataFrame([['kr', 'kcat', 0.1, 'high_affinity']],
                                 columns=['param_i', 'param_j', 'value', 'exp_condition'])
        noise_model_1 = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        noise_model_1.update_values(param_mean=pd.DataFrame([['pt_mutation', 19]],
                                                            columns=['exp_condition', 'num_sims']))
        target_exp = pd.DataFrame([['high_affinity', 50],
                                   ['pt_mutation',   19],
                                   ['wild_type',      1]],
                                  columns=['exp_condition',  'num_sims'])
        test_exp = noise_model_1.experimental_conditions_dataframe
        pd_testing.assert_frame_equal(test_exp[test_exp.columns], target_exp[test_exp.columns])

    def test_simulate_exp_empty_param_mean_no_cov(self):
        nm = NoiseModel()
        test = nm._simulate(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        target = pd.DataFrame()
        pd_testing.assert_frame_equal(test, target)

    def test_simulate_exp_empty_param_mean_w_conditions(self):
        nm = NoiseModel(param_mean=pd.DataFrame([[1, 2, 3]], columns=['param', 'value', 'exp_condition']))
        test = nm._simulate(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        target = pd.DataFrame(columns=['exp_condition'])
        pd_testing.assert_frame_equal(test, target)

    def test_simulate_exp_empty_param_mean_w_numeric_conditions(self):
        nm = NoiseModel(param_mean=pd.DataFrame([[1, 2, 3]], columns=['param', 'value', 1]))
        test = nm._simulate(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        target = pd.DataFrame(columns=[1])
        pd_testing.assert_frame_equal(test, target)

    def test_add_fixed_values(self):
        param_mean = pd.DataFrame([['a', 2.0, 3, True],
                                   ['b', 1.0, 1, False]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        nm = NoiseModel(param_mean=param_mean)
        test=pd.DataFrame()
        fix_params_df = nm._add_fixed_values(param_mean, nm.experimental_conditions_dataframe)
        test[fix_params_df.columns] = fix_params_df
        target = pd.DataFrame([1.0, 1.0, 1.0], columns=['b'])
        pd_testing.assert_frame_equal(test, target)

    def test_add_noisy_values_no_noise_applied(self):
        param_mean = pd.DataFrame([['b', 1.0, 1, False]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        nm = NoiseModel(param_mean=param_mean)
        test = nm._add_noisy_values(param_mean, pd.DataFrame(), nm.experimental_conditions_dataframe)
        target = pd.DataFrame()
        pd_testing.assert_frame_equal(test, target)

    def test_add_noisy_values_create_covariance_matrix(self):
        param_mean = pd.DataFrame([['a', 2.0, 3, True],
                                   ['b', 1.0, 1, False]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        param_cov = pd.DataFrame([['a', 'c', 0.1]], columns=['param_i', 'param_j', 'value'])
        NoiseModel.default_param_values = {'c': 3.0}
        nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        varied_terms = nm.param_mean[nm.param_mean['apply_noise'] == True]
        test = nm._create_covariance_matrix(varied_terms, nm.param_covariance)
        target = pd.DataFrame([[0.16, 0.1], [0.1, 0.36]], columns=['a', 'c'], index=['a', 'c'])
        pd_testing.assert_frame_equal(test, target)

    def test_add_noisy_values_create_covariance_matrix_case2(self):
        param_mean = pd.DataFrame([['a', 2.0, 3, True],
                                   ['b', 1.0, 1, True]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        param_cov = pd.DataFrame([['a', 'c', 0.1]], columns=['param_i', 'param_j', 'value'])
        NoiseModel.default_param_values = {'c': 3.0}
        nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        varied_terms = nm.param_mean[nm.param_mean['apply_noise'] == True]
        test = nm._create_covariance_matrix(varied_terms, nm.param_covariance)
        target = pd.DataFrame([[0.16, 0.1, 0.0],
                               [0.1, 0.36, 0.0],
                               [0.0, 0.0, 0.04]], columns=['a', 'c', 'b'], index=['a', 'c', 'b'])
        pd_testing.assert_frame_equal(test, target)

    def test_mv_log_normal_distribution_fn(self):
        # Average and cov of the resulting distribution should be app. what you started with
        n = 100000
        param_mean = pd.DataFrame([['a', 2.0, n, True],
                                   ['b', 0.0, 1, True]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        param_cov = pd.DataFrame([['a', 'c', 0.1]], columns=['param_i', 'param_j', 'value'])
        NoiseModel.default_param_values = {'c': 3.0}
        nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        test = nm._add_noisy_values(nm.param_mean, nm.param_covariance, nm.experimental_conditions_dataframe)
        target_mean = np.array([2, 3, 2.0e-2])
        target_cov = np.array([[0.16,   0.1,    0.0],
                               [ 0.1,  0.36,    0.0],
                               [ 0.0,   0.0, 1.0e-2]])
        test_mean = test.mean()
        test_cov = test.cov()
        assert_array_almost_equal(test_cov.values, target_cov, 2)
        assert_array_almost_equal(test_mean.values, target_mean, 2)

    def test_simulate_exp(self):
        target = pd.DataFrame([[1.0, 2.541798, 1.455033],
                               [1.0, 3.804277, 2.540719],
                               [1.0, 2.456074, 1.908849],
                               [1.0, 2.846912, 1.854556]], columns=['b', 'c', 'a'])
        np.random.seed(10)

        n = 4
        param_mean = pd.DataFrame([['a', 2.0, n, True],
                                   ['b', 1.0, 1, False]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        param_cov = pd.DataFrame([['a', 'c', 0.1]], columns=['param_i', 'param_j', 'value'])
        NoiseModel.default_param_values = {'c': 3.0}
        nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        test = nm._simulate(nm.param_mean, nm.param_covariance, nm.experimental_conditions_dataframe)
        pd_testing.assert_frame_equal(test[['b', 'c', 'a']], target[['b', 'c', 'a']])

    def test_noise_run(self):
        target = pd.DataFrame([[0, 1.0, 2.541798,  1.455033],
                               [1, 1.0, 3.804277,  2.540719],
                               [2, 1.0, 2.456074,  1.908849],
                               [3, 1.0, 2.846912,  1.854556]], columns=['simulation','b', 'c', 'a'])
        np.random.seed(10)

        n = 4
        param_mean = pd.DataFrame([['a', 2.0, n, True],
                                   ['b', 1.0, 1, False]],
                                  columns=['param', 'value', 'num_sims', 'apply_noise'])
        param_cov = pd.DataFrame([['a', 'c', 0.1]], columns=['param_i', 'param_j', 'value'])
        NoiseModel.default_param_values = {'c': 3.0}
        nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        test = nm.run()
        pd_testing.assert_frame_equal(test[['b', 'c', 'a']], target[['b', 'c', 'a']])

    def test_noise_run_multiple_exp(self):
        target = pd.DataFrame([[ 0,  1.455033,  2.541798,        'ec1', np.NaN],
                               [ 1,  2.540719,  3.804277,        'ec1', np.NaN],
                               [ 2,  1.908849,  2.456074,        'ec1', np.NaN],
                               [ 3,  1.854556,  2.846912,        'ec1', np.NaN],
                               [ 4,    np.NaN,    np.NaN,        'ec2',    1.0]],
                              columns=['simulation', 'a', 'c', 'experiment', 'b'])
        np.random.seed(10)
        n = 4
        param_mean = pd.DataFrame([['a', 2.0, n, True, 'ec1'],
                                   ['b', 1.0, 1, False, 'ec2']],
                                  columns=['param', 'value', 'num_sims', 'apply_noise', 'experiment'])
        param_cov = pd.DataFrame([['a', 'c', 0.1, 'ec1']], columns=['param_i', 'param_j', 'value', 'experiment'])
        NoiseModel.default_param_values = {'c': 3.0}
        nm = NoiseModel(param_mean=param_mean, param_covariance=param_cov)
        test = nm.run()
        pd_testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_topic_guide_modeling_experiment(self):
        experimental_treatments = NoiseModel(pd.DataFrame([['kcat', 500, 'high_activity'],
                                                           ['kcat', 100, 'low_activity']],
                                                          columns = ['param', 'value', 'experimental_treatment']))
        test = experimental_treatments.run()
        target = pd.DataFrame([[0,   500, 'high_activity'],
                               [1,   100,  'low_activity']], columns=['simulation', 'kcat', 'experimental_treatment'])
        pd_testing.assert_frame_equal(test[test.columns], target[target.columns])


