import pandas as pd
import numpy as np
from scipy.stats import norm


# categorical likelihoods
def categorical_dist_likelihood(predicted, dataset):
    """Gives the likelihood of the data and simulation using probability of class-membership
    The predicted df gives probability of class membership. The dataset ordinal_errors_df contains probabilities of
    class-membership (i.e. a categorical distribution).
    FOr each data-point the marginal probability of the data and prediction is given by summing the probablity of the
    data and prediction for each categorical value"""

    # likelihood of categorical outcomes where results and data are in distribution form.
    # e.g. y_data = probabilities of each categorical value

    ordinal_errors = dataset.ordinal_errors_df
    category_names = ordinal_errors.filter(regex='__').columns

    # Probability of the predictions and the data referencing the same ordinal category
    likelihood_per_category_and_simulation = pd.DataFrame(
        predicted[category_names].values * ordinal_errors[category_names].values,
        columns=category_names)

    # Marginal probability of the categories for each measurement of each measured variable
    transposed_likelihoods = likelihood_per_category_and_simulation.transpose().reset_index()
    transposed_likelihoods['index'] = [this.split("__")[0] for this in transposed_likelihoods['index']]
    likelihoods = transposed_likelihoods.groupby('index').sum(numeric_only=True).transpose()

    # Sum neg-log likelihood
    return np.sum(-np.log(np.array(
        likelihoods[
            likelihoods.columns.difference(list(dataset.experimental_conditions.columns))].values)
                          .astype(float)
                          ))


# quantitative likelihood
def normal_pdf_empirical_var_likelihood(predicted, dataset, measured_values):
    """Gives the likelihood of the data and simulation using normal pdf.
    Variance of the pdf is assumed to be that of the data."""

    # combine measured data values and error (i.e. variance) into a single dataframe
    error_cols_list = [k + '__error' for k in measured_values.keys()]
    y_ = dataset.data
    y_[error_cols_list] = dataset.measurement_error_df[error_cols_list]
    exp_conditions_cols = list(dataset.experimental_conditions.columns)

    # combine data and predictions into a single dataframe
    y = y_.merge(predicted, how='outer', on=exp_conditions_cols).drop_duplicates().reset_index(drop=True)

    # sum log-likelihood of each value
    log_likelihood = 0
    for data_obs, sim_obs in measured_values.items():
        y_sim = y[sim_obs[0]].values
        y_data = y[data_obs].values
        y_error = np.sqrt(y[data_obs + '__error'].values)  # convert non-zero variances values to st.dev

        log_likelihood += np.sum(-norm.logpdf(y_sim, loc=y_data, scale=y_error))
    return log_likelihood
