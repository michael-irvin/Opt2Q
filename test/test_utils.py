import opt2q.utils as ut
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
