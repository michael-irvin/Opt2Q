.. Opt2Q documentation master file, created by
   sphinx-quickstart on Tue Aug  7 03:57:51 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Opt2Q documentation
===================

Opt2Q
-----
Opt2Q uses quantitative and non-quantitative data to parameterize dynamical models of complex systems.

Scientists model complex dynamics with differential equations and accompanying free parameters. In many instances, there
exist uncertainty in values of the free parameters. Model parametrization tunes these free parameters so that the model
predictions agree with observations.

Model predictions are quantitative; and conventional parametrization methods (for dynamical models) require quantitative
data. Complexity and technical limitations impeded precise quantitative measurements, and can even render measurements
non-quantitative.

Opt2Q accomplishes model calibration to non-quantitative data by also modeling factors (e.g. noise sources and
measurement processes) that help render measurements non-quantitative. These additions to the dynamical model
facilitate more direct comparision between model predictions and measurements.

Opt2Q support modeling efforts in systems biology; where non-quantitative measurements outnumber quantitative ones.
Opt2Q can, however, support the calibration of dynamical models to non-quantitative measurements/data, in general.

Documentation
-------------
This documentation supports your use of Opt2Q to parameterize models of complex dynamical biological systems.

Opt2Q models represent biological systems via three interacting elements:

- Noise Model: models extrinsic biological noise by adding random variability to models' free parameters.
- Dynamical Model: rules describing the interactions amongst components that comprise a complex system, and differential equations describing their dynamics. Opt2Q uses `PySB`_ to for these models.
- Measurement Model: describes aspects of the experimental measurement processes that engenders the data.

Opt2Q builds an objective function to optimize the parameters in these three models. The three Opt2Q model elements, and
objective function are explained in the following documentation sections:

- *Tutorials* to help get started building and running the models.
- *Topic guides* provide more detail of Opt2Q -- with overviews, examples, how-tos and guides.
- *Module References* provide indexed/searchable documentation the modules, functions, etc.

.. _PySB: http://pysb.org
.. _Pandas: https://pandas.pydata.org

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial_docs/index
   topic_guides/index
   module_refs/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
