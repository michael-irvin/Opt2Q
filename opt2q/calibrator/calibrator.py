"""
Tools for calibrating an Opt2Q Model
"""


# MW Irvin -- Lopez Lab -- 2018-08-07

class ObjectiveFunction(object):
    """
    Behaves like a function but has additional features that support its implementation as an objective function in
    conventional optimizers.

    Parameters
    ----------
    f: Not sure yet what this will be. But it has to contain the noise, sim and measurement models.

    Examples
    --------
    Create lookup tables
    >>> # obj_fun.create_lookup['noise_model_name'](param_mean=pd.DataFrame)

    >>> # obj_fun.create_lookup['all_measurements'](pd.DataFrame)

    Update lookup tables
    >>> # obj_fun.update_lookup['noise_model_name'](param_mean=pd.DataFrame)

    Display lookup tables
    >>> # obj_fun.lookup --> Dictionary or DataFrames
    >>> # obj_fun.lookup['noise_model_name'] --> param_mean and (if present) param_cov lookups
    >>> # obj_fun.lookup['measurements'] --> DataFrame of measurement_models | attr_name | xid

    """

    def __init__(self, f):
        # self.noise_model(s) = Get noise model from f
        #   ..note:: multiple noise models could be used to construct a mixture models
        #   |-- name the Noise Models
        # Get Simulator from f
        # Get all Measurement Model(s) from f
        #   |-- name the Measurement models
        pass

    def __call__(self, x):
        """
        Assign values to parameters to the Opt2Q noise and measurement models. Next, run the models and sum their
        reported likelihood values.

        Parameters
        ----------
        x: list
            List of values (floats) of the parameters in the Opt2Q models.

        Return
        ------
        float: Likelihood of the data provided the models.
        """
        # run noise
        # run simulator
        # for m in measurement model run measurement and get likelihood.
        # return sum
        pass

    def _create_lookups(self):
        """
        Creates a dictionaries of lookup tables. The dictionaries' keys are the names fo the noise models, and within is
        a dictionary with 'param_mean' and 'param_cov' (from the noise model) and 'measurement_models' (for *all* the
        measurement models).

        Creates self.lookup with a dictionary of DataFrames that contain an additional 'xid' column.
        Creates self._lookup with a the same DataFrames mentioned above, for 'param_mean' and 'param_cov'. And a
        dictionary for the measurement model.

        View using ``obj_fn.lookup['noise_model']['param_mean']``
        Access via ``obj_fn._lookup['noise_model']['param_mean']``
        """
        pass

    def update_lookups(self, lookup_table_name, new_df):
        """
        Updates self.look_up['lookup_table_name'] with the new DataFrame provide.


        Parameter
        ---------
        lookup_table_name:
        new_df: :class:`~pandas.DataFrame`
            Must have xid column.
        """

        # For param_mean and param_cov:
        # Find the row(s) in the original df (e.g. self.noise_model.param_means) corresponding to those in the new_df.
        #   Note: *All* rows of original df matching that of the new_df are added/updated in the self.lookup.
        #   Completed DataFrame
        # Set their 'calibrate_value' to True.
        # Update self.look_up and self._lookup with the Completed DataFrame

        # For measurement_models: (possibly use a default-dict)
        # Make sure the new_df's measurement_model_name column only names present in the measurement_models.keys()
        # Combine new_df with self.lookup
        # Group self.lookup by measurement_model_name
        # For each group update the self._lookup['measurement_model_name']
        #   with {param_name:xid}.
        pass

    def _assign_values(self, x):
        """
        Assign values of x to the parameters in the noise model and measurement models.
        """

        # merge self._lookup['noise_model_name'][param_mean or param_cov] with DF(|'values'=x[xid] | xid |)
        #   updates values
        # update noise.param_mean with the updated self._lookup

        # for measurement_model_name in self._lookup['measurement_models'].iteritems():
        #   measurement_models['measurement_model_name'].set_params(
        #       x[self._lookup['measurement_models']['measurement_model_name']]
        #                                       ^^^^^^^^^ xid.
        pass


class objective_function(object):
    """
    Decorator that creates an objective function using an Opt2Q noise and simulation model, and Opt2Q measurement
    model(s)

    Parameters:
    -----------
    func: not sure yet

    Returns
    -------
    :class:`~opt2q.calibrator.ObjectiveFunction` instance

    Example
    -------
    >>> from opt2q.calibrator import objective_function


    """
    def __init__(self, noise=None):
        # Process Decorator arguments
        pass

    def __call__(self, f):
        # presents decorated function
        return f
