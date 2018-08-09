from opt2q.noise import NoiseModel

import numpy as np
import pandas.util.testing as pd_testing
import  pandas as pd
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
            nm._check_required_columns(param_df=pd.DataFrame(), var_name='param_covariance')
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


