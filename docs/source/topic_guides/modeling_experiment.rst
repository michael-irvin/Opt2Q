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

Modeling Experimental Treatments
================================
Modelers represent various experimental treatments as variations in the dynamical model's parameters. The Opt2Q
:class:`~opt2q.simulator.Simulator` accepts a :class:`~pandas.DataFrame` of parameters and their values (as formatted
below).

>>> import pandas as pd
>>> parameters = pd.DataFrame([[0, 500, 'high_activity'],
...                           [1, 100, 'low_activity']],
...                          columns=['simulation', 'kcat', 'experimental_treatment'])
>>> print(parameters)
   simulation  kcat experimental_treatment
0           0   500          high_activity
1           1   100           low_activity

Notice the :class:`~pandas.DataFrame` can have additional columns annotating the different experimental treatment.

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
...                      ['kcat', 100, 'low_activity' , 100],
...                      ['vol',   10, 'high_activity', 200],
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
The Opt2Q :class:`~opt2q.simulator.Simulator` class uses PySB simulators (e.g.
:class:`~pysb.simulator.ScipyOdeSimulator`) to simulate the dynamics of a biological system. The Opt2Q simulator
behaves much like PySB simulators: e.g. it accepts the same kinds of objects for its ``param_values`` and ``initials``
arguments and likewise returns a `PySB` :class:`~pysb.simulator.SimulationResult`. The PySB simulators are discussed
`here <https://pysb.readthedocs.io/en/stable/modules/simulator.html>`_.

The Opt2Q :class:`~opt2q.simulator.Simulator` also accepts :class:`DataFrames <pandas.DataFrame>` for its ``param_values``
and ``initials`` arguments. The column names are the `PySB` model's :class:`~pysb.core.Parameter` names (for
``param_values``) and the PySB ``model.species`` (for ``initials``). Additional columns can designate experimental
treatments, conditions, etc.

>>> import numpy as np
>>> import pandas as pd
>>> from matplotlib import pyplot as plt
>>> from opt2q.simulator import Simulator
>>> from pysb.examples.michment import model
>>> new_params = pd.DataFrame([[np.nan, 'normal', 1],
...                            [10.0,   'slow',   1],
...                            [1.0e3,  'fast',   1]],
...                           columns=['kcat', 'condition', 'experiment'])
>>> sim = Simulator(model=model, param_values=new_params)
>>> results = sim.run(np.linspace(0, 50, 50))

The Opt2Q :class:`~opt2q.simulator.Simulator` returns the `PySB` :class:`~pysb.simulator.SimulationResult` that retains the
additional ``opt2q_dataframe`` that annotates the simulation results by experimental treatment.

.. code-block:: python

    results_df = results.opt2q_dataframe

    #plot
    cm = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,6))
    for i, (label, df) in enumerate(results_df.groupby(['experiment', 'condition'])):
        df.plot.line(y='Product', ax=ax, label=label, color=cm.colors[i])
    plt.legend()
    plt.show()


.. image:: /auto_examples/images/sphx_glr_plot_simple_dynamics_simulation_001.png
    :class: sphx-glr-single-img

The Opt2Q :class:`~opt2q.simulator.Simulator` only accepts PySB :class:`models <pysb.core.Model>`.

.. note:: Do not use double underscores in your PySB model parameter names. This interferes with the Opt2Q calibrator.

.. _PySB: http://pysb.org

Modeling Measurement Process
============================
Opt2Q has a suite of :class:`measurement models <~opt2q.measurement.base.MeasurementModel>` that mimic qualities of the
measurements researchers conduct on biological systems.

The :class:`~opt2q.measurement.WesternBlot` model, for instance, mimics the ordinal quality of western blot measurements by
mapping quantities in the Opt2Q simulation :class:`~pysb.simulator.SimulationResult` (described above) to ordinal categories.

The following describes steps to set-up and use measurement models.

Set-up the Measurement Model
----------------------------
The :class:`measurement model <~opt2q.measurement.base.MeasurementModel>` receives a :class:`~pysb.simulator.SimulationResult`
for its first argument.

In addition, the measurement model needs to know which observables, experimental conditions, and time-points are pertinent to
the process. This information exists in Opt2Q :class:`DataSets <opt2q.data.DataSet>`. I recommend supplying a
:class:`~opt2.data.DataSet` to your measurement model.

>>> dataset_fluorescence

They can also be specified manually using the ``observables``, ``experimental_conditions`` and ``time_points`` arguments.

>>> # Example:

Important info: The experimental conditions defines a subset of the experiments (in the simulation result) to which to apply the
measurement. Each experimental conditions can have a unique set of time-points but the observables must be the same for all of
them.

.. note::

    Time-points supplied to the measurement model should be in the range of the simulation result's time axis.
    Avoid extrapolation which is less accurate than the ODE solver.

The Measurement Process
-----------------------
The measurement process conducts a series of transformations (e.g. log-scaling, normalization, interpolation, classification etc)
on the :class:`~pysb.simulator.SimulationResult`. Each transformation is carried out by an Opt2Q
:class:`~opt2q.measurement.base.Transform` class that possess methods for getting and setting parameters, and conducting the
transformation.

>>> # Get_params, set_params and Run a transformation for a measurement model #Plot results

Measurement Likelihood Metric
-----------------------------
Simulate extrinsic noise

>>> params_m = pd.DataFrame([['kc3', 1.0, '-', True],
...                          ['kc3', 0.3, '+', True],
...                          ['kc4', 1.0, '-', True],
...                          ['kc4', 0.3, '+', True]],
...                         columns=['param', 'value', 'inhibitor', 'apply_noise'])
>>>
>>> param_cov = pd.DataFrame([['kc4', 'kc3', 0.01,   '-'],
...                           ['kc3', 'kc3', 0.009,  '+'],
...                           ['kc4', 'kc4', 0.009,  '+'],
...                           ['kc4', 'kc3', 0.001,  '+']],
...                          columns=['param_i', 'param_j', 'value', 'inhibitor'])
>>>
>>> NoiseModel.default_sample_size = 500
>>> noise = NoiseModel(param_mean=params_m, param_covariance=param_cov)
>>> parameters = noise.run()

Simulate Dynamics

>>> sim = Simulator(model=model, param_values=parameters)
>>> results = sim.run(np.linspace(0, 5000, 100))
>>> results_df = results.opt2q_dataframe

Annotate Data

>>> western_blot = pd.read_csv('Albeck_Sorger_WB.csv')
>>> western_blot['time'] = western_blot['time'].apply(lambda x: x*500)
>>> western_blot['inhibitor'] = '-'
>>> dataset = DataSet(data=western_blot, measured_variables=['cPARP', 'PARP'])

Simulate Measurement

>>> ec = pd.DataFrame(['-', '+'], columns=['inhibitor'])
>>> wb = WesternBlot(simulation_result=results,
...                  dataset=dataset,
...                  measured_values={'PARP': ['PARP_obs'], 'cPARP': ['cPARP_obs']},
...                  observables=['PARP_obs', 'cPARP_obs'],
...                  experimental_conditions=pd.DataFrame(['-', '+'], columns=['inhibitor']),
...                  time_points=[1500, 2000, 2500, 3500, 4500])
>>>
>>> western_blot_results = wb.run(use_dataset=False)  # runs dataset first to get coefficients, then predicts the rest.

Calibrating Measurement

>>> @objective_function(noise_model=noise, dynamics_sim=sim, western_blot = wb)
>>> def my_objective_func(x):
...     new_params_m=pd.DataFrame([['kc3', x[0], '-'],
...                                ['kc3', x[1], '+'],
...                                ['kc4', x[0], '-'],
...                                ['kc4', x[1], '+']], columns=['param', 'value', 'inhibitor'])
...     my_objective_func.noise_model.update_values(param_mean=new_params_m)
...     my_objective_func.western_blot.process.set_params(
...         {'classifier__coefficients__PARP__coef_`': np.array([x[2]]),
...          'classifier__coefficients__PARP__theta_`': np.array([x[3], x[4]]),
...          'classifier__coefficients__cPARP__coef_`': np.array([x[5]]),
...          'classifier__coefficients__cPARP__theta_`': np.array([x[6], x[7],  x[8], x[9]])})
...     sim_results = my_objective_func.sim.run(np.linspace(0, 5000, 100))
...     return my_objective_func.western_blot.likelihood()

