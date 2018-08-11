============
Opt2Q Models
============

Opt2Q models dynamics. It also models experimental treatments, noise sources, and measurement process, etc. (i.e.
factors that influence our observations of the biological system's dynamics). All together these models constitute what
I define as a an *experiment* model.

The following details methods and steps involved in building an experiment model. It includes the following:

1. Models of experimental treatments and extrinsic noise sources
2. `PySB`_ models of the dynamics
3. Models of the measurement.

.. _PySB: http://pysb.org

Modeling Experimental Treatments
================================

If parameters in `param_covariance` do not also appear in `param_mean`, the Opt2Q noise model will look for them in its
``default_param_values`` (dict) or in the PySB model, if supplied.

>>> #example using ``default_param_values`` to get missing params

>>> #example using PySB model to get missing params

Modeling Dynamics with PySB
===========================

.. note:: Do not use double underscores in your PySB model parameter names. This interferes with the Opt2Q calibrator.
