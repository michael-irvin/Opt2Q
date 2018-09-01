========================
Calibrating Opt2Q Models
========================

Opt2Q models consist of a dynamical model and accompanying noise and measurement models. These models can contain many
free parameters who values are tune to optimize an objective function.

Creating a DataSet
==================
The DataSet houses the measured values from the experiment along with annotations about the experimental conditions. It
also specifies which observables and/or species of the PySB :class:`pysb.core.Model` are involved in the measurement.

Assembling an Objective Function
================================
The objective function can be constructed in two ways: You can write your own function that runs the components of the
Opt2Q. The first argument is a list ``x`` of floats passed by the optimizer. Subsequent arguments are the components of
the Opt2Q model. This approach will work for most optimizers.

>>> def obj_f(x, noise, sim, measurement1, measurement2):
...     # This code does not compile. It only provides an idea of the approach.
...     noise.param_mean.update_value(row, col, val=x[0])
...     measurements['name'].update_values({'param_name':np.array(["values from x"])})
...     etc.

Some optimizers (e.g. `PSO`_) prohibit the additional arguments. The Opt2Q :class:`~opt2q.calibrator.objective_function`
decorator provides a work-around by accepting these arguments for use within the objective function before presenting to
the optimizer.

.. _PSO: https://github.com/LoLab-VU/ParticleSwarmOptimization

Lets start by importing an example opt2q model:

>>> from opt2q.examples import opt2q_model
>>> noise_models = {'model1':opt2q_model.noise_model,
...                 'model2':opt2q_model.noise_model_2}
>>> dynamics_simulator = opt2q_model.simulator

Set up objective function with Opt2Q :class:`~opt2q.calibrator.objective_function` decorator.

>>> from opt2q.calibrator import objective_function
>>> @objective_function(noise=noise_models, sim=dynamics_simulator)
>>> def obj_f(x):
...     # noise model
...     obj_f.noise['model1'].update_values(pd.DataFrame([['vol', x[0]]], columns=['param', 'value']))
...     obj_f.noise['model2'].update_values(pd.DataFrame([['vol', x[1]]], columns=['param', 'value']))
...     params = pd.concat([obj_f.noise['model1'].run(), obj_f.noise['model2'].run())
...     # simulate dynamics
...     obj_f.sim.param_values = params
...     sim_res = obj_f.sim.run(np.linspace(0, 1, 100))
...     # measurement model
...     obj_f.measurement.simulation_result = sim_res
...     obj_f.measurement.set_params({"!logistic_regression__coef_": np.array([[x[2], x[3]],[x[4], x[3]]])})
...     return obj_f.measurement.run()

Updating parameters without checking them: The simulator will expect the updates to be similar to what
was present at instantiation of the simulator. Print sim.param_values for a template.

As a dataframe it should include a 'simulation' column.

>>> print(sim.param_values)

.. note::
    Caution: Your objective should not try to update structural attributes e.g. experimental treatment names etc. It
    could cause your model to fail.