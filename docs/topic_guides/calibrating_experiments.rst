========================
Calibrating Opt2Q Models
========================

Opt2Q models consist of a dynamical model and accompanying noise and measurement models. These models can contain many
free parameters who values are tune to optimize an objective function.


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
decorator provides a work-around by accepting these arguments before presenting to the optimizer.

.. _PSO: https://github.com/LoLab-VU/ParticleSwarmOptimization

Lets start by importing an example opt2q model:

>>> from opt2q.examples import opt2q_model
>>> noise_models = {'model1':opt2q_model.noise_model,
...                 'model2':opt2q_model.noise_model_2}

Set up objective function with Opt2Q objective_function decorator

>>> from opt2q.calibrator import objective_function
>>> @objective_function(noise=noise_models)
>>> def obj_f(x):
...     obj_f.noise['model1'].update_values(pd.DataFrame())
...     obj_f.noise['model2'].update_values(pd.DataFrame())
