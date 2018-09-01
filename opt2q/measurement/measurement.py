# MW Irvin -- Lopez Lab -- 2018-08-24
"""
Suite of Measurement Models
"""
from opt2q.measurement.base.base import MeasurementModel
from opt2q.measurement.base.transforms import Interpolate


class WesternBlot(MeasurementModel):
    """
    Simulates a Western Blot Measurement

    Conducts a series of transformations on the :class:`~pysb.simulator.SimulationResult` to represent attributes of the
    Western Blot Measurement.

    Parameters
    ----------
    simulation_result: :class:`~pysb.simulator.SimulationResult`
        Results of a `PySB` Model simulation. Produced by the run methods in the `PySB`
        (:meth:`~pysb.simulator.ScipyOdeSimulator.run`) and `Opt2Q` (:meth:`~opt2q.simulator.Simulator.run`)
        simulators.

    dataset: :class:`~opt2q.data.DataSet`, optional
        Measured values and associated attributes, e.g. 'observables', 'experimental_conditions', that dictate what
        rows of the simulation_result should apply to the measurement model.

        The :meth:`opt2q.measurement.base.likelihood` method *requires" the :class:`~opt2q.data.DataSet`, and will refer
        to the DataSet's attributes, even when such attributes were defined manually via ``time_points``,
        ``observables``, ``experimental_conditions`` arguments.

        .. note::
            The DataSet's experimental_conditions columns must be the same as those in the
            :class:`~pysb.simulator.SimulationResult`

        .. note:: The :class:`~opt2q.measurement.WesternBlot` requires a :class:`~opt2q.data.DataSet`.

    observables: vector-like, optional
        Lists the names (str) of the PySB PySB :class:`pysb.core.Model` observables and/or species involved in the
        measurement.

        These observables apply to all the experimental conditions involved in the measurement. Observables not mentioned
        in the ``simulation_result`` and/or ``dataset`` (if supplied) are ignored.

    time_points: vector-like, optional
        Lists the time-points involved in the measurement. Defaults to the time points in the
        :class:`~pysb.simulator.SimulationResult`. You can also assign time-points using a 'time' in the
        ``experimental_conditions`` argument (as follows).

    experimental_conditions: :class:`~pandas.DataFrame`, optional
        The experimental conditions involved in the measurement model. Defaults to experimental conditions in
        the ``dataset`` (if present) or the ``simulation_result``.

        You can add a 'time' column to specify time-points that are specific to the individual experimental conditions.
        NaNs in this column will be replace by the ``time_points`` values or the time-points mentioned in the
        :class:`~pysb.simulator.SimulationResult`
    """
    def __init__(self, simulation_result, dataset=None, observables=None, time_points=None, experimental_conditions=None):
        super(WesternBlot).__init__(simulation_result,
                                    dataset=dataset,
                                    observables=observables,
                                    time_points=time_points,
                                    experimental_conditions=experimental_conditions)
        # Is there a time-axis for the measurement?
        #   If yes, the values of the observables at each or the time-points mentioned in the experimental conditions
        #   will be individually used as features of the ordinal regression step.

        #   If no, then a dimension reduction or feature-extraction pre-processing step is required.
        self.time_axis_dimension_reduction = "We need a function to do this!"

        # The western-blot will have observables and each observable is processed individually.
        for obs in observables:
            "Do ordinal-logistic reguression. {}".format(obs)
            pass

        # Other pre-processing:
        #   Put this in the pipeline:
        #       Log-standardize

        # The measurement Process has to have a way to manipulate parameters etc from the outside. Set the process to
        #  be a pipeline of transformations.
        interpolate = Interpolate('independent_variable_name', 'dependent_variable_name', 'new_values')