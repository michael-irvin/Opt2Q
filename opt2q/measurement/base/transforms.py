# MW Irvin -- Lopez Lab -- 2018-08-24


class Transform(object):
    """
    Base class for measurement processes, i.e. transformations to the simulation results that constitute a simulation of
    the measurement process.

    This class can change the number rows and/or columns; as the underlying calculations can apply to individual columns
    and rows, or to groups of columns and/or rows. Some transforms do not support certain grouped calculations while
    others require it.

    This class can also depend on free-parameters.
    """
    def __init__(self):
        pass

    def _groupby(self, columns=None, rows=None, **kwargs):
        """
        Group the transformation.

        Transformation applied to a group of columns (or rows) will change the number of columns (or rows). E.g.
        averaging all the rows in each simulation drops the number of rows in the :class:`~pandas.DataFrame` to 1 per
        simulation.

        Group columns by observable, observable_groups, all. Group row by unique values in list of columns.

        Parameters
        ----------
        columns: str
            'column': apply the calculation to each individual column.
            'observable': apply the calculation to each group a column pertaining to an individual observable.
            'all': apply the calculation to all the columns together.

        Example
        -------
        >>> # Average values grouped by simulation

        >>> # Max values grouped by experiment

        >>> # Logistic Regression by column groups

        >>> # PCA on all columns and Data

        """
        pass

    def _get_params(self, free_params_only=True):
        """
        Return parameters in the model.

        Parameters
        ----------
        free_params_only: bool
            When True, only returns parameters that have a numeric value or set of numeric values. This is useful to the
        calibrator.
        """
        pass

    def _set_params(self, **kwargs):
        """
        Sets parameter values of the measurement process.
        """
        # These should be in Pipeline and Transform classes. The user accesses the measurement model directly, and can
        # those parameters by name. The Pipeline and Transform classes create a set of parameters that are harder to
        # anticipate. This makes it easier.
        pass


class Interpolate(Transform):
    """
    Interpolates values in numeric (dependent variable) columns of a :class:`~pandas.DataFrame` at new values of an
    independent-variable.

    Represents dependent variable columns of a :class:`~pandas.DataFrame` as a function of an independent-variable
    column, and uses it to find new values of the dependent variables. The DataFrame is passed to this class's
    :meth:`~opt2q.measurement.base.transform.Interpolate.transform` method. But, names of the independent and dependent
    variables, and the new values are supplied at instantiation.

    Parameter
    ---------
    independent_variable_name: str
        The column name of the independent variable.

    dependent_variable_name: str or list of strings
        The column name(s) of the dependent variable(s)

    new_values: vector-like or :class:`~pandas.DataFrame`
        The values of the independent variable in the interpolation. Additional columns annotate experimental
        conditions, or etc.

    options: dict
         Dictionary of keyword arguments:

        ``interpolate_times_method`` (str): interpolation method that gets passed to the `SciPy`
        :class:`~scipy.interpolate.interp1d` class. Defaults to 'cubic'.

    groupby: str, optional
        The name of the column by which to group the operation by. Each unique value in this column denotes a
        separate group. Defaults to None and the interpolation is carried out on all rows of the DataFrame together.
    """
    def __init__(self, independent_variable_name, dependent_variable_name, new_values, groupby=None, **options):
        super(Interpolate).__init__()


        self._annotating_df = new_values
        # The independent variable column must be in the new-values
        # only unique values of the independent variable are allowed (per group) in the x.
        # take the intersection of the numeric columns
        pass

    def transform(self, x):
        """
        Interpolates, in a DataFrame, x, the values in numeric column(s) relative to new values of an independent
        variable.

        Parameter
        ---------
        x: :class:`~pandas.DataFrame`
            The independent and dependent variables utilized to generate the interpolant.

            The independent and dependent variable column names (specified at instantiation of this class) must be in x.
            The independent variable cannot have repeat values within the same group.


        :class:`~opt2q.measurement.base.transforms.Interpolate` class.
        :param x:
        """
        # trim x to only have the experimental conditions mentioned in the values.
        pass


class Pipeline(object):
    pass