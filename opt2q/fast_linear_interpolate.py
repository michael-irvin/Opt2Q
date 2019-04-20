# MW Irvin -- Lopez Lab -- 2019-03-05

"""
Rewriting interpolate to speed it up.
"""

from numba import jit, double
import numpy as np
import pandas as pd

x = np.linspace(0.01, 0.1, 10)

df = pd.DataFrame({'a': x,
                   'b': x**2,
                   'c': np.sin(x),
                   'x': 'x'})

new_values = pd.DataFrame([0.011, 0.012, 0.013, 0.014, 0.015, 0.061], columns=['a']).astype(float)
prepped_x = df.merge(new_values, on='a', how='outer', sort=True)

print(np.argwhere(np.isnan(prepped_x.select_dtypes(include='number').values)))


#@jit(nopython=True)
def fast_linear_interpolate_fillna(values, indices):
    # result = np.zeros_like(values, dtype=np.float32)
    result = values

    for idx in range(indices.shape[0]):
        x = indices[idx, 0]
        y = indices[idx, 1]

        value = values[x, y]
        if x == 0:
            new_val = value
        elif x == len(values[:, 0]) - 1:
            new_val = value
        elif np.isnan(value):  # interpolate
            lid = 0
            while True:
                lid += 1
                left = values[x - lid, y]
                if not np.isnan(left):
                    break
            rid = 0
            while True:
                rid += 1
                right = values[x + rid, y]
                if not np.isnan(right):
                    break

            new_val = left + (right - left) * values[x, 0] / (values[x + rid, 0] - values[x - lid, 0])

        else:
            new_val = value

        result[x, y] = new_val
    return result


fast_linear_interpolate_fillna = jit(double[:, :](double[:,:], double[:,:]))(fast_linear_interpolate_fillna)


# @jit(nopython=True)
def numba_lin_interpolate(values):
    result = values

    for y in range(1, values.shape[1]):
        # slice along x axis
        for x in range(values.shape[0]):
            value = values[x, y]
            if x == 0:
                new_val = value
            elif x == len(values[:, 0])-1:
                new_val = value
            elif np.isnan(value):  # interpolate
                lid = 0
                while True:
                    lid += 1
                    left = values[x-lid, y]
                    if not np.isnan(left):
                        break
                rid = 0
                while True:
                    rid += 1
                    right = values[x+rid, y]
                    if not np.isnan(right):
                        break

                new_val = left + (right - left) * values[x, 0] / (values[x+rid, 0] - values[x-lid, 0])

            else:
                new_val = value
            result[x, y] = new_val
        result[:, 0] = values[:, 0]
    return result


numba_lin_interpolate = jit(double[:](double[:]))(numba_lin_interpolate)

print(fast_linear_interpolate_fillna(prepped_x.select_dtypes(include='number').values, np.argwhere(np.isnan(prepped_x.select_dtypes(include='number').values))))

# print(prepped_x.select_dtypes(include='number').transform(lambda v: numba_lin_interpolate(np.array(v))))
