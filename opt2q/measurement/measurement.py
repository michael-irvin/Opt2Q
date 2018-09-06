# MW Irvin -- Lopez Lab -- 2018-08-24
"""
Suite of Measurement Models
"""
from opt2q.measurement.base.base import MeasurementModel
from opt2q.measurement.base.transforms import Interpolate, Pipeline


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

    time_dependent: bool, optional
        If False, the time-axis is eliminated by averaging and/or other operations so that the simulated measurement is
        independent of time. If True a measurement is reported for every time-point in the series. Defaults to True.

    Attributes
    ----------
    process: :class:`~opt2q.measurement.base.transforms.Transform`
        Series of transformations that model the measurement process.

    """
    def __init__(self, simulation_result, dataset=None, observables=None, time_points=None,
                 experimental_conditions=None, time_dependent=True):
        super(WesternBlot).__init__(simulation_result,
                                    dataset=dataset,
                                    observables=observables,
                                    time_points=time_points,
                                    experimental_conditions=experimental_conditions)
        self.process = Pipeline()

    def run(self):
        return self.process.run()

        # time-dependent WB
        #   False by default
        #   Trim dataframe to include just the observables in ``observables``
        #   For each observable, do logistic regression on the values
        #       one-feature per observable
        #       how-many classes? As many as are in the dataset for that observable
        #           get n_classes per dataset_observable
        #           otherwise - default to min(4, len(data)) classes
        #           (I did a calculation showing the effective resolution to be 6 categories)
        #   Same calculation; i.e same observables, same coefficients (i.e. number of classes) for
        #   ALL time-points and experimental conditions.