# MW Irvin -- Lopez Lab -- 2018-08-23
import pandas as pd


class DataSet(object):
    """
    Formats Data for use in Opt2Q models

    observables:
        provides a way for the users to customize what columns of the data he/she wants to use.

    Attributes
    ----------
    observables: list
        Names of the observables in the data. They can be directly measured or indirect markers of the measured data.

    """
    def __init__(self, observables=None):
        self.observables = {1, 2, 3}  # Todo: Measurement model requires vector-like attr, 'observables'.
        self.experimental_conditions = pd.DataFrame()  # This will also contain time-points (if necessary).

    def _get_observables(self):
        """
        Returns list of observables' names mentioned in the data or from user input.

        :return: list
            observables' names
        """
        pass

    def _get_time_points(self):
        """
        Get time-points mentioned in the data set or from user input.

        :return: list
            Observable names
        """
        pass