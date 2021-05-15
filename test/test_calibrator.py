import unittest
from numpy.testing import assert_array_equal

import numpy as np
from opt2q.calibrator import objective_function


class TestNoise(unittest.TestCase):
    def test_objective_function_repr(self):
        @objective_function(h=2)
        def my_obj(x):
            return

        target = 'my_obj(x)'
        test = my_obj.__repr__()
        self.assertEqual(test, target)

    def test_objective_function_access_to_decorator_obj(self):
        @objective_function(h=2)
        def my_obj(x):
            return np.array(x) + my_obj.h

        target = [2, 3, 4]
        test = my_obj([0, 1, 2])
        assert_array_equal(test, target)
