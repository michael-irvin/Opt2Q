==========================================
Measurement Models and Processes in Detail
==========================================

Opt2Q has a suite of measurement models. The following lists them and details their use.

Measurement Models
==================
Opt2Q measurement models

Western Blot Model
------------------
Western Blot Model

Measurement Processes
=====================
The measurement process uses various :class:`~opt2q.measurement.base.Transform` classes that transform a :class:`~pandas.DataFrame`
(e.g. the :class:`~pysb.simulator.SimulationResult`'s ``opt2q_dataframe``) of numeric values to a form that mimics a particular
type of measurement. Each :class:`~opt2q.measurement.base.Transform` class has a :meth:`~opt2q.measurement.base.Transform.transform`
method that carries out these changes.

Scale
-----
The :class:`~opt2q.measurement.base.Scale` scales numeric values of in a :class:`~pandas.DataFrame`. Use the ``columns`` and
``scale_fn`` arguments to define which columns of the :class:`~pandas.DataFrame` to scale and the scaling function.

>>> import pandas as pd
>>> from opt2q.measurement.base import Scale
>>> scale = Scale(columns=[1, 2], scale_fn='log10')

These scale settings get applied to any :class:`~pandas.DataFrame` passed to the :meth:`~opt2q.measurement.base.Scale.transform`
method.

>>> print(scale.transform(pd.DataFrame([[0, 0.1, 'a'], [1, 1.0, 'b'], [2, 10., 'c']])))
   0     1  2
0  0  -1.0  a
1  1   0.0  b
2  2   1.0  c

Notice that column ``2`` in the dataframe remained unchanged. The :meth:`~opt2q.measurement.base.Scale.transform` method only affects
the numeric columns named in :class:`~opt2q.measurement.base.Scale`'s ``columns`` argument.

The :class:`~opt2q.measurement.base.Scale` contains a various pre-encoded functions (i.e.
:attr:`~opt2q.measurement.base.Scale.scale_functions`). The :class:`~opt2q.measurement.base.Scale` function also accepts custom scale
functions. These functions must accept for its first argument, ``x`` , a :class:`~pandas.DataFrame`; and if supplied a
:class:`~pandas.DataFrame`, it must return a :class:`~pandas.DataFrame`.

>>> def custom_f(x, a=1):
...     return x**a
>>> scale.scale_fn = custom_f
>>> scale.scale_fn_kwargs = {'a':2}
>>> print(scale.transform(pd.DataFrame([[0, 0.1, 'a'], [1, 1.0, 'b'], [2, 10., 'c']])))
   0       1  2
0  0    0.01  a
1  1    1.00  b
2  2  100.00  c

Interpolate
-----------
The :class:`~opt2q.measurement.base.Interpolate` transform does 1-dimensional interpolation on
columns of a :class:`~pandas.DataFrame`. One-dimensional interpolation creates an interpolant (i.e.
a function relating dependent variables to values of an independent variable), and uses it to
estimate values of the dependent variables at new values of the independent variable.

The independent (``'time'``) and dependent (``['fluorescence', '1-fluorescence']``) variables, and
new values ``[3, 4, 5, 7, 9]`` are defined via arguments passed to the
:class:`~opt2q.measurement.base.Interpolate` class, as in the following:

>>> import pandas as pd
>>> from opt2q.measurement.base import Interpolate
>>> interpolate = Interpolate('time', ['fluorescence', '1-fluorescence'], [3, 4, 5, 7, 9])

The :meth:`~opt2q.measurement.base.Interpolate.transform` applies this interpolation to
:class:`~pandas.DataFrames`.

>>> x = pd.DataFrame([[1, 10, 0], [3, 8, 2],[5, 5, 5],[7, 2, 8],[9, 10, 0]],
...                  columns=['time', 'fluorescence', '1-fluorescence'])
>>> print(interpolate.transform(x))
   time  fluorescence  1-fluorescence
0     3       8.00000         2.00000
1     4       6.71875         3.28125
2     5       5.00000         5.00000
3     7       2.00000         8.00000
4     9      10.00000         0.00000

.. note::
    ``x`` must have columns named in the ``independent_variable`` and ``dependent_variable`` arguments.

.. note::
    The values mentioned in ``new_values`` cannot exceed the range of values in the independent variable column
    of ``x``. (Do not attempt extrapolation).

The :class:`~opt2q.measurement.base.Interpolate` transform also accepts a :class:`~pandas.DataFrame`. The
columns must contain the independent variable. Any additional columns annotate experimental conditions etc.

>>> interpolate.new_values = pd.DataFrame([[2, 'early'], [8, 'late']], columns=['time', 'observation'])

The :meth:`~opt2q.measurement.base.Interpolate.transform` now performs a separate interpolation for each unique
group in the ```observation``` column of ``x``.

>>> x = pd.DataFrame([[1, 10, 0, 1, 'early'],
...                   [3,  8, 2, 1, 'early'],
...                   [5,  5, 5, 1, 'early'],
...                   [7,  2, 8, 2, 'late'],
...                   [9, 10, 0, 2, 'late']],
...                 columns=['time', 'fluorescence', '1-fluorescence', 'sample', 'observation'])
>>> print(interpolate.transform(x))
   sample observation  time  fluorescence  1-fluorescence
0       1       early     2           9.0             1.0
1       2        late     8           6.0             4.0

You can also perform separate interpolations per group using the ``group_by`` arguement.

>>> x = pd.DataFrame([[1, 10, 0, 1, 'early'],
...                   [3,  8, 2, 1, 'early'],
...                   [5,  6, 4, 1, 'early'],
...                   [5,  5, 5, 2, 'early'],
...                   [7,  2, 8, 2, 'late'],
...                   [9, 10, 0, 2, 'late']],
...                 columns=['time', 'fluorescence', '1-fluorescence', 'sample', 'observation'])
>>> Interpolate('time', ['fluorescence', '1-fluorescence'], [5], groupby='sample').transform(x)
   sample observation  time  fluorescence  1-fluorescence
0       1       early     5           6.0             4.0
1       2       early     5           5.0             5.0
2       2        late     5           5.0             5.0

Notice in the above example, sample 2 has 'early' and 'late' values for the 'observation' column. The interpolation
is repeated for both.

Logistic Classifier
-------------------
The logistic classifier maps values in a :class:`~pandas.DataFrame` to ordinal or nominal categories.
This is useful for modeling categorical measurements or observations.

The :meth:`~opt2q.measurement.base.LogisticClassifier.transform` method accepts the :class:`~pandas.DataFrame`
(``x``) whose values will be mapped into categories.

Since logistic regression is a supervised learning method, the :class:`~opt2q.measurement.base.LogisticClassifier` requires
a :class:`~opt2q.data.DataSet` or a :class:`~pandas.DataFrame` of empirical data (i.e. targets).

>>> import numpy as np
>>> import pandas as pd
>>> np.random.seed(10)
>>> targets = pd.DataFrame(np.sort(np.random.normal(size=(10, 3)), axis=0)))

The ``columns`` argument defines which columns in the dataset to regard as targets. These columns names must also be in ``x``.

>>> dataset_fluorescence

Often the multiple columns in ``x`` serve as features for a target in the dataset, or the feature column in ``x`` has a different name than the
target column in the dataset. Use the ``column_groups`` argument to assign feature and corresponding target columns.

>>> column_groups = {'column_as_named_in_dataset': ['corresponding column(s) in x']}  # Cell fate example

>>> # group features, group name is another approach but limited to a single group.

Use the ``classifier_type`` argument to specify which of the supported logistic classifiers should carryout the transformation.The
:attr:`~opt2q.measurement.base.LogisticClassifier.classifiers` names the supported ordinal and nominal logistic classifiers.

Ordinal Classification Example
..............................
The western blot assigns an ordinal metric to the amount of protein in a sample. Albeck and Sorger *link* illustrates this by describing a
relationship between the western blot to fluorescent (semi-quantitative) measurements protein abundance.
The :class:`~opt2q.measurement.base.LogisticClassifier` can similarly model this relationship.

>>> dataset_fluorescence
>>> #  x = C3 activity (proxy for cPARP)
>>> #  transform
>>> #  plot simulated western

Nominal Classification Example
..............................
Use the other Apoptosis paper example

