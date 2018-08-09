import opt2q.utils as ut
import unittest


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
