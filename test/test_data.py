import unittest
import pandas as pd
import numpy as np
from opt2q.data import DataSet


class TestData(object):
    def setUp(self):
        self.data = pd.DataFrame([[2, 0, 0, "WB"],
                                  [2, 0, 1, "WB"],
                                  [2, 0, 2, "WB"],
                                  [2, 1, 3, "WB"],
                                  [2, 2, 4, "WB"],
                                  [2, 3, 5, "WB"],
                                  [2, 3, 5, "WB"],
                                  [1, 4, 7, "WB"],
                                  [0, 4, 9, "WB"]],
                                 columns=['PARP', 'cPARP', 'time', 'assay'])


class TestDataSet(TestData, unittest.TestCase):
    def test_empty_dataset(self):
        ds = DataSet(pd.DataFrame(), [])
        test = ds.data
        target = pd.DataFrame().reset_index(drop=True)
        pd.testing.assert_frame_equal(test, target)
        test = ds.experimental_conditions
        target = pd.DataFrame().reset_index(drop=True)
        pd.testing.assert_frame_equal(test, target)
        test = ds.measured_variables
        target = dict()
        self.assertDictEqual(test, target)
        test = ds.observables
        target = []
        self.assertListEqual(test, target)

    def test_convert_measured_variables_to_dict(self):
        ds = DataSet(pd.DataFrame(), [])
        test = ds._convert_measured_variables_to_dict({'valid_dict': 'default'})
        target = {'valid_dict': 'default'}
        self.assertDictEqual(test, target)

    def test_convert_measured_variables_to_dict_bad_type_name(self):
        ds = DataSet(pd.DataFrame(), [])
        with self.assertRaises(ValueError) as error:
            ds._convert_measured_variables_to_dict({'invalid_dict': 'this wrong type'})
        self.assertTrue(error.exception.args[0] ==
                        "'measured_variables' can only be 'quantitative', 'semi-quantitative', 'ordinal' or 'nominal'. "
                        "Not 'this wrong type'.")

    def test_convert_measured_variables_to_dict_from_list(self):
        ds = DataSet(pd.DataFrame(), [])
        test = ds._convert_measured_variables_to_dict(['col1', 'col2'])
        target = {'col1': 'default', 'col2': 'default'}
        self.assertDictEqual(test, target)

    def test_check_data_not_dataframe(self):
        with self.assertRaises(ValueError) as error:
            DataSet("Not a DataFrame", [])
        self.assertTrue(error.exception.args[0] == "'data' can only be a pandas DataFrame. Not a str")

    def test_check_data_missing_variables(self):
        with self.assertRaises(ValueError) as error:
            DataSet(self.data, ['missing_variable'])
        self.assertTrue(error.exception.args[0] ==
                        "'measured_variables' mentioned these variables not present in the data: 'missing_variable'.")

    def test_experimental_conditions(self):
        ds = DataSet(self.data, ['cPARP', 'PARP'])
        test = ds.experimental_conditions
        target = self.data[['time', 'assay']]
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_measured_variables_bad_type(self):
        with self.assertRaises(ValueError) as error:
            DataSet(pd.DataFrame(), "Bad Type")
        self.assertTrue(error.exception.args[0] == "measured_variables must be either dict or list. Not str.")

    def test_check_measurement_error_bad_input(self):
        with self.assertRaises(ValueError) as error:
            DataSet(pd.DataFrame(), [], measurement_error='Bad Type')
        self.assertTrue(error.exception.args[0] == 'measurement_error must be a float or dict with float item')

    def test_check_measurement_error_float_input(self):
        ds = DataSet(pd.DataFrame(columns=['a', 'b', 'c']), ['a', 'c'], measurement_error=0.15)
        self.assertDictEqual(ds._measurement_error, {'a': 0.15, 'c': 0.15})

    def test_check_measurement_error_dict_input(self):
        ds = DataSet(pd.DataFrame(columns=['a', 'b', 'c']), {'a': 'quantitative', 'b': 'quantitative', 'c': 'ordinal'},
                     measurement_error={'a':0.15})
        self.assertDictEqual(ds._measurement_error, {'a': 0.15, 'b': 0.20, 'c': 0.05})

    def test_make_default_ordinal_errors_matrix(self):
        ds = DataSet(self.data, ['cPARP', 'PARP'])
        target = {'cPARP': np.array([[0.8, 0.2, 0. , 0. , 0. ],
                                     [0.1, 0.8, 0.1, 0. , 0. ],
                                     [0. , 0.1, 0.8, 0.1, 0. ],
                                     [0. , 0. , 0.1, 0.8, 0.1],
                                     [0. , 0. , 0. , 0.2, 0.8]]),
                  'PARP': np.array([[0.8, 0.2, 0. ],
                                    [0.1, 0.8, 0.1],
                                    [0. , 0.2, 0.8]])}

        for k, v in ds._get_ordinal_errors_matrices(dict(), ['cPARP', 'PARP']).items():
            np.testing.assert_array_almost_equal(target[k], v)

    def test_prep_data_for_likelihood(self):
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
        ds = DataSet(data, {'PARP': 'ordinal', 'cPARP': 'quantitative'})

        test = ds._one_hot_transform_of_data(ordinal_variables_names=['PARP'])

        target = pd.DataFrame([[0,        'WT',           1,        0,        0,        1],
                               [1,        'WT',           1,        0,        0,        1],
                               [2,        'WT',           1,        0,        0,        1],
                               [3,        'WT',           1,        0,        0,        1],
                               [4,        'WT',           1,        0,        0,        1],
                               [5,        'WT',           1,        0,        0,        1],
                               [5,        'WT',           1,        0,        0,        1],
                               [7,        'WT',           1,        0,        1,        0],
                               [9,        'WT',           1,        1,        0,        0]],
                              columns=['time', 'condition', 'experiment', 'PARP__0', 'PARP__1', 'PARP__2'])

        pd.testing.assert_frame_equal(test[target.columns], target[target.columns], check_dtype=False)
        target = pd.DataFrame([[0,        'WT',           1,    0.000,     0.05,    0.950],
                               [1,        'WT',           1,    0.000,     0.05,    0.950],
                               [2,        'WT',           1,    0.000,     0.05,    0.950],
                               [3,        'WT',           1,    0.000,     0.05,    0.950],
                               [4,        'WT',           1,    0.000,     0.05,    0.950],
                               [5,        'WT',           1,    0.000,     0.05,    0.950],
                               [5,        'WT',           1,    0.000,     0.05,    0.950],
                               [7,        'WT',           1,    0.025,     0.95,    0.025],
                               [9,        'WT',           1,    0.950,     0.05,    0.000]],
                              columns=['time', 'condition', 'experiment', 'PARP__0',  'PARP__1',  'PARP__2'])
        test = ds.ordinal_errors_df
        pd.testing.assert_frame_equal(test[target.columns], target[target.columns])

# TODO: add a way to change the _ordinal_errors_matrices and it update the _ordinal_errors_df automatically.

