# MW Irvin -- Lopez Lab -- 2018-09-07
from opt2q.measurement.base.functions import transform_function, log_scale, polynomial_features, derivative
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
        target = pd.DataFrame([0.1, 1, 2])
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

    def test_polynomial_features(self):
        test = polynomial_features(pd.DataFrame([[0, 1, 2],[1, 2, 3]],columns=['a ', 'b', 'c']), degree=2)
        target=pd.DataFrame([
            [0.0, 1.0, 2.0,  0.0,  0.0,  0.0, 1.0, 2.0, 4.0],
            [1.0, 2.0, 3.0,  1.0,  2.0,  3.0, 4.0, 6.0, 9.0]],
            columns=['a ', 'b', 'c', 'a$^2', 'a$$b', 'a$$c',  'b^2',  'b$c',  'c^2'])

        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])

    def test_derivative(self):
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
        test = derivative(x[['A', 'B']])

        target = pd.DataFrame([
            [1.0,  3.0],
            [0.0,  1.0],
            [0.5, -1.0],
            [1.5, -1.0],
            [0.0,  0.5],
            [1.0,  2.0],
            [1.0,  2.0],
            [0.0,  2.0],
            [1.0,  2.0],

        ], columns=['A', 'B'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns])