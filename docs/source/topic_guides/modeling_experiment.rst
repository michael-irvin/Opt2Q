=====================
Building Opt2Q Models
=====================

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
:class:`~opt2q.simulator.Simulator` accepts a :class:`~pandas.DataFrame` of parameters and their values.

>>> import pandas as pd
>>> parameters = pd.DataFrame([[0, 500, 'high_activity'],
...                           [1, 100, 'low_activity']],
...                          columns=['simulation', 'kcat', 'experimental_treatment'])
>>> print(parameters)
   simulation  kcat experimental_treatment
0           0   500          high_activity
1           1   100           low_activity

>>> # Example of supplying parameters directly Opt2Q simulator

Notice the :class:`~pandas.DataFrame` can have additional columns annotating the different experimental conditions.

Noise Models
------------
The Opt2Q :class:`~opt2q.noise.NoiseModel` lets you create the above :class:`~pandas.DataFrame` of parameters, and apply
extrinsic noise to it.

>>> from opt2q.noise import NoiseModel
>>> experimental_treatments = NoiseModel(pd.DataFrame([['kcat', 500, 'high_activity'],
...                                                    ['kcat', 100, 'low_activity']],
...                                                   columns=['param', 'value', 'experimental_treatment']))
>>> parameters = experimental_treatments.run()
   simulation  kcat experimental_treatment
0           0   500          high_activity
1           1   100           low_activity

To apply default (i.e. log-normal distributed with a coefficient of variation of 0.2) extrinsic noise to your
parameters, include an 'apply_noise' column.

>>> experimental_treatments = NoiseModel(
...                            pd.DataFrame([['kcat', 500, 'high_activity', False],
...                                          ['kcat', 100, 'low_activity' , False],
...                                          ['vol',   10, 'high_activity', True],
...                                          ['vol',   10, 'low_activity' , True]],
...                                         columns=['param', 'value', 'experimental_treatment', 'apply_noise']))
>>> parameters = experimental_treatments.run()
   simulation  kcat        vol experimental_treatment
0           0   500  10.624326          high_activity
1           1   500  12.854892          high_activity
2           2   500   9.453784          high_activity
3           3   500   9.969232          high_activity
4           4   500   9.517106          high_activity

In the above case, :class:`~opt2q.noise.NoiseModel` returns a noisy sample (of size 50) of parameter values. You can
change the sample size via the class variable, :attr:`~opt2q.noise.NoiseModel.default_sample_size`

>>> NoiseModel.default_sample_size = 200

Or you can specify it via a 'num_sims' column in your ``param_means`` argument. Note: the sample size should be
consistent for unique experimental condition.

>>> mean = pd.DataFrame([['kcat', 200, 'high_activity', 200],
...                      ['kcat', 100, 'low_activity' , 200],
...                      ['vol',   10, 'high_activity', 100],
...                      ['vol',   10, 'low_activity' , 100]],
...                     columns=['param', 'value', 'experimental_treatment', 'num_sims'])
>>> cov = pd.DataFrame([['vol', 'kcat', 1.0], ['vol', 'vol', 3.0]], columns=['param_i', 'param_j', 'value'])
>>> experimental_treatments = NoiseModel(param_mean=mean, param_covariance=cov)

As also shown above, you can set variance and covariance using a :class:`~opt2q.noise.NoiseModel`'s
``param_convariance`` argument. You only need to assign values to parameters with non-zero covariance using
'param_i' and 'param_j' columns, as shown above. Use the same parameter name for both 'param_i' and 'param_j' to assign
variance terms.

Notice the lack of experimental treatment columns in the ``param_covariance``. They are optional. The
:class:`~opt2q.noise.NoiseModel` interprets their absence to mean the covariance settings apply to *all* the
experimental treatments. Using the same parameter name for both 'param_i' and 'param_j'.

.. code-block:: python

    parameters = experimental_treatments.run()

    # plot
    cm = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,6))
    for i, (label, df) in enumerate(parameters.groupby('experimental_treatment')):
       df.plot.scatter(x='kcat', y='vol', ax=ax, label=label, color=cm.colors[i])
    plt.legend()
    plt.show()


.. image:: /auto_examples/images/sphx_glr_plot_simple_noise_model_001.png
    :class: sphx-glr-single-img

Noise parameters in ``param_covariance`` must also appear in ``param_mean``. If not, the Opt2Q noise model will look
for them in its ``default_param_values`` (dict) or in a PySB model, if either is supplied.

>>> NoiseModel.default_param_values = {'vol':10}  # missing parameter 'vol' is retrieved from ``default_param_values``
>>> mean = pd.DataFrame([['kcat', 200, 'high_activity', 200],
...                      ['kcat', 100, 'low_activity' , 200]]
...                     columns=['param', 'value', 'experimental_treatment', 'num_sims'])
>>> cov = pd.DataFrame([['vol', 'kcat', 1.0], ['vol', 'vol', 3.0]], columns=['param_i', 'param_j', 'value'])
>>> experimental_treatments = NoiseModel(param_mean=mean, param_covariance=cov)

Retrieve missing parameters from :class:`~pysb.core.Model`. Note: this approach will take to instantiate the model.

>>> from pysb.examples.michment import model
>>> experimental_treatments = NoiseModel(model=model, param_mean=mean, param_covariance=cov)

.. note:: You can only use either the :class:`~pysb.core.Model` or ``default_param_values`` dict. Not both.


Modeling Dynamics with PySB
===========================

.. note:: Do not use double underscores in your PySB model parameter names. This interferes with the Opt2Q calibrator.
