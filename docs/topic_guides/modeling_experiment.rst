============
Opt2Q Models
============

Opt2Q models dynamics. It also models experimental treatments, noise sources, measurement processes, etc. (i.e. factors
that influence our observations of the biological system's dynamics). All together, these models constitute what I
define as a an *experiment* model.

The following details methods and steps involved in building an experiment model:

1. Models of experimental treatments and extrinsic noise sources
2. `PySB`_ models of the dynamics
3. Models of the measurement.

.. _PySB: http://pysb.org

Modeling Experimental Treatments
================================
Modelers represent various experimental treatments as variations in the dynamical model's parameters. The Opt2Q
simulator accepts a :class:`~pandas.DataFrame` of parameters and their values.

>>> # Example of supplying parameters directly Opt2Q simulator

However, the Opt2Q :class:`~opt2q.noise.NoiseModel` helps users this :class:`~pandas.DataFrame` of parameters and their
values, and apply extrinsic noise to them.

>>> from opt2q.noise import NoiseModel
>>> experimental_treatments = NoiseModel(pd.DataFrame([['kcat', 500, 'high_activity'],
...                                                    ['kcat', 100, 'low_activity']],
...                                                   columns=['param', 'value', 'experimental_treatment']))
>>> experimental_treatments.run()  # Method is not established yet.

If parameters in `param_covariance` do not also appear in `param_mean`, the Opt2Q noise model will look for them in its
``default_param_values`` (dict) or in the PySB model, if supplied.

>>> #example using ``default_param_values`` to get missing params

>>> #example using PySB model to get missing params

Modeling Dynamics with PySB
===========================

.. note:: Do not use double underscores in your PySB model parameter names. This interferes with the Opt2Q calibrator.
