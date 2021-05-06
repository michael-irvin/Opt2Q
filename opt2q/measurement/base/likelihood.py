import pandas as pd
import numpy as np


# ordinal likelihoods
def categorical_dist_likelihood(predicted, dataset):
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
