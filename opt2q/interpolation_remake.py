"""
Remaking (and hopefully speeding up) the interpolation transform.

interpolation variables:
iv,
dv,
new_values,
group_by,
x,

-------- Workflow -------
x and new_values -> nx: intersection of x and new_values
mentioned_columns == union 'iv', 'dv', 'ec' and 'groupby' cols
group_by columns

"""
import pandas as pd
import numpy as np
from opt2q.measurement.base import Interpolate

# ======= Simple Case ========
nx = pd.DataFrame([[0.5],
                   [1.5]], columns=['iv'])
x = pd.DataFrame([[0, 0],
                  [1, 2],
                  [2, 4]], columns=['iv', 'dv'])

target = pd.DataFrame([[0.5, 1.0],
                       [1.5, 3.0]], columns=['iv', 'dv'])

prepped_x = nx.merge(x, how='outer').sort_values(['iv']).set_index('iv')
x_interpolated = prepped_x.transform(pd.DataFrame.interpolate, **{'method': 'values'}).reset_index()
x_out = x_interpolated.merge(nx)

print("case 1")
pd.testing.assert_frame_equal(x_out, target)

interpolate = Interpolate('iv', 'dv', [0.0])
nx = pd.DataFrame([[0.5],
                   [1.5]], columns=['iv'])
x = pd.DataFrame([[0, 0],
                  [1, 2],
                  [2, 4]], columns=['iv', 'dv'])
target = pd.DataFrame([[0.5, 1.0],
                       [1.5, 3.0]], columns=['iv', 'dv'])
interpolate.new_values = nx
test = interpolate.transform(x)
print(test)
print(x_out)

# ======= Annotating Columns no groupby column ========
nx = pd.DataFrame([[1, 0.5],
                   [1, 1.5]], columns=['ec', 'iv'])
x = pd.DataFrame([[1, 1, 0, 0],
                  [2, 1, 1, 2],
                  [1, 1, 2, 4]], columns=['ec1', 'ec', 'iv', 'dv'])
target = pd.DataFrame([[1, 1, 0.5, 1.0],
                       [2, 1, 0.5, 1.0],
                       [1, 1, 1.5, 3.0],
                       [2, 1, 1.5, 3.0]], columns=['ec1', 'ec', 'iv', 'dv'])


def repeat_new_values_rows_for_every_unique_row_in_x_extra_cols(new_values, x_):
    unmentioned_cols_in_x = set(x_.columns) - {'ec', 'sim', 'iv', 'dv'}  # sim is the groupby
    if len(unmentioned_cols_in_x) != 0:
        x_unmentioned_cols = x_[list(unmentioned_cols_in_x)].drop_duplicates().reset_index(drop=True)
        len_x = len(x_unmentioned_cols)
        len_nv = len(new_values)

        nv_repeated = x_unmentioned_cols.iloc[np.tile(range(len_x), len_nv)].reset_index(drop=True)
        nv_repeated[new_values.columns] = new_values.iloc[np.repeat(range(len_nv), len_x)].reset_index(drop=True)
        return nv_repeated
    else:
        return new_values


groupby_cols = []
nx_ = repeat_new_values_rows_for_every_unique_row_in_x_extra_cols(nx, x)
prepped_x = nx_.merge(x, how='outer').sort_values(['iv'] + groupby_cols).set_index('iv')

x_interpolated = prepped_x.transform(pd.DataFrame.interpolate, **{'method': 'values'}).reset_index()
x_out = x_interpolated.merge(nx_)

print("case 2")
# print(repeat_new_values_rows_for_every_unique_row_in_x_extra_cols(nx, x))
# print(prepped_x)

interpolate = Interpolate('iv', 'dv', [0.0])
nx = pd.DataFrame([[1, 0.5],
                   [1, 1.5]], columns=['ec', 'iv'])
x = pd.DataFrame([[1, 1, 0, 0],
                  [2, 1, 1, 2],
                  [1, 1, 2, 4]], columns=['ec1', 'ec', 'iv', 'dv'])
target = pd.DataFrame([[1, 1, 0.5, 1.0],
                       [2, 1, 0.5, 1.0],
                       [1, 1, 1.5, 3.0],
                       [2, 1, 1.5, 3.0]], columns=['ec1', 'ec', 'iv', 'dv'])
interpolate.new_values = nx
test = interpolate.transform(x)
print(x_out)
print(test)

# ======= Groupby Columns =======
print("case 3")
x = pd.DataFrame([[0, 'WT', 0, 0],
                  [0, 'WT', 1, 2],
                  [0, 'WT', 2, 4],
                  [1, 'WT', 0, 1],
                  [1, 'WT', 1, 3],
                  [1, 'WT', 2, 5],
                  [2, 'KO', 0, 1],
                  [2, 'KO', 1, 2],
                  [2, 'KO', 2, 3],
                  [3, 'KO', 0, 0],
                  [3, 'KO', 1, 1],
                  [3, 'KO', 2, 2]], columns=['sim', 'ec', 'iv', 'dv'])
new_val = pd.DataFrame([['WT', 0.5],
                        ['KO', 1.5]],
                       columns=['ec', 'iv'])

interpolate = Interpolate('iv', 'dv', new_val, groupby='sim')
groupby_cols = interpolate._group_by

new_x = interpolate._intersect_x_and_new_values_experimental_condition(x, new_val)

new_val_repeats = repeat_new_values_rows_for_every_unique_row_in_x_extra_cols(new_val, new_x)
new_vals_w_extra_cols_from_x = new_val_repeats.merge(new_x[['ec']+groupby_cols].drop_duplicates().reset_index(drop=True))
prepped_x = new_vals_w_extra_cols_from_x.merge(new_x, how='outer').sort_values(groupby_cols + ['iv']).set_index('iv')

x_interpolated = prepped_x.groupby(groupby_cols).transform(pd.DataFrame.interpolate, **{'method': 'values'})
x_interpolated[groupby_cols] = prepped_x[groupby_cols]
x_out = new_vals_w_extra_cols_from_x.merge(x_interpolated.reset_index())


test = interpolate.transform(x)
target = pd.DataFrame([[0, 'WT', 0.5, 1.0],
                       [1, 'WT', 0.5, 2.0],
                       [2, 'KO', 1.5, 2.5],
                       [3, 'KO', 1.5, 1.5]], columns=['sim', 'ec', 'iv', 'dv'])
print(test)
print(x_out)

# ======= Recognize derivative columns ========
print("case 4")