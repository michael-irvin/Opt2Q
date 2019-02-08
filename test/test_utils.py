import opt2q.utils as ut
import pandas as pd
import numpy as np
import unittest
import warnings


class TesOpt2QUtils(unittest.TestCase):
    def test_error_message_list_len_1(self):
        error_list = ['a']
        target = "'a'"
        test = ut._list_the_errors(error_list)
        self.assertEqual(test, target)

    def test_error_message_list_len_2(self):
        error_list = ['a', 'b']
        target = "'a', and 'b'"
        test = ut._list_the_errors(error_list)
        self.assertEqual(test, target)

    def test_error_message_list_len_3(self):
        error_list = ['a', 'b', 'c']
        target = "'a', 'b', and 'c'"
        test = ut._list_the_errors(error_list)
        self.assertEqual(test, target)

    @staticmethod
    def test_incompatible_format_warming():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ut.incompatible_format_warning({'ksynthA': 10})
            assert issubclass(w[-1].category, ut.IncompatibleFormatWarning)

    @staticmethod
    def test_is_vector_like():
        assert ut._is_vector_like(None) is False
        assert ut._is_vector_like((1, 2, 3)) is True
        assert ut._is_vector_like(pd.DataFrame([1])) is True

    def _convert_vector_like_to_list(self):
        target = [1, 2, 3]
        test = ut._convert_vector_like_to_list({1, 2, 3})
        assert isinstance(test, list)
        test = ut._convert_vector_like_to_list((1, 2, 3))
        self.assertListEqual(test, target)
        test = ut._convert_vector_like_to_list(np.array([[1, 2, 3]]))
        self.assertListEqual(test, target)
        test = ut._convert_vector_like_to_list(pd.DataFrame([1, 2, 3]))
        self.assertListEqual(test, target)

    def test_convert_vector_like_to_set(self):
        target = {1, 2, 3}
        test = ut._convert_vector_like_to_set([1, 2, 3])
        self.assertSetEqual(test, target)
        test = ut._convert_vector_like_to_set((1, 2, 3))
        self.assertSetEqual(test, target)
        test = ut._convert_vector_like_to_set(np.array([[1, 2, 3]]))
        self.assertSetEqual(test, target)
        test = ut._convert_vector_like_to_set(pd.DataFrame([1, 2, 3]))
        self.assertSetEqual(test, target)

    def test_parse_column_names(self):
        test = ut.parse_column_names({'A_free+AB_complex'}, {'A_free'})
        target = {'A_free+AB_complex', 'A_free'}
        self.assertSetEqual(test, target)


