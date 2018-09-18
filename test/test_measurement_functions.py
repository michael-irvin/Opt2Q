# MW Irvin -- Lopez Lab -- 2018-09-07
from opt2q.measurement.base.functions import transform_function, log_scale
import numpy as np
import pandas as pd
import unittest


class TestTransformFunction(unittest.TestCase):
    def test_transform_function_repr(self):
        @transform_function
        def f(x, k=3):
            return x + k
        assert f.__repr__() == 'f(x, k=3)'
        f.signature(k=5)
        assert f.__repr__() == 'f(x, k=5)'
        f.signature(x=5)
        assert f.__repr__() == 'f(x=5, k=3)'
        assert f(2) == 5

    def test_clip_zeros(self):
        @transform_function
        def f(x, k=3):
            return x + k
        test = f.clip_zeros([0, 1, 2])
        target = pd.DataFrame([0.1, 2, 3])
        pd.testing.assert_frame_equal(test, target)

    def test_log_scale(self):
        test = log_scale(pd.DataFrame([[0, 1, 2],
                                       [4, 8, 16]],
                                      columns=['a', 'b', 'c']), base=2, clip_zeros=False)
        target = pd.DataFrame([[-np.inf, 0.0, 1.0],
                               [2.0,     3.0, 4.0]],
                              columns=['a', 'b', 'c'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

        test = log_scale(pd.DataFrame([[0, 1, 2],
                                       [4, 8, 16]],
                                      columns=['a', 'b', 'c']), base=2, clip_zeros=True)
        target = pd.DataFrame([[-1.321928, 0.0, 1.0],
                               [2.0, 3.0, 4.0]],
                              columns=['a', 'b', 'c'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

