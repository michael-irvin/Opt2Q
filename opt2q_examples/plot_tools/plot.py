# MW Irvin -- Lopez Lab -- 2020-09-07
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from opt2q_examples.plot_tools import calc, utils

cm = plt.get_cmap('tab10')


# Convergence Metrics
def gelman_rubin_values(ax, param_names, gr_vals, labels=True):
    ax.barh(y=param_names, width=gr_vals, color='k', alpha=0.5)
    ax.axvline(x=1.2, linestyle='--', color='k', alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlim(0.95 * min(gr_vals), max(1.25, 1.05 * max(gr_vals)))
    if labels:
        ax.set_ylabel('Parameter Name')
        ax.set_xlabel('Gelman Rubin Diagnostic')


def gelman_rubin_histogram(ax, gr_vals, labels=True):
    ax.hist(gr_vals, color='k', alpha=0.6, edgecolor='k')
    ax.axvline(x=1.2, linestyle='--', color='k', alpha=0.5)
    if labels:
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Gelman Rubin Diagnostic')


def parameter_traces_gr(ax, traces, param_idx=0, burnin=50000, thin=None, labels=True):
    if thin is None:
        thin = int(max((len(traces[0])-burnin)/100, 1))  # thin so that the calculation runs 100 times
    grt = calc.gelman_rubin_trace(traces, param_idx, burnin, thin)

    ax.plot(range(burnin, len(traces[0]), thin), grt, color='k')
    ax.axhline(y=1.2, linestyle='--', color='k', alpha=0.5)
    if labels:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gelman-Rubin')


def parameter_traces(ax, traces, param_idx=0, burnin=0, labels=True):
    x = np.array(list(range(len(traces[0])))) + burnin
    for i, trace in enumerate(traces):
        ax.plot(x, trace[:, param_idx],
                color=cm.colors[i], label=f'trace {i}')
    if labels:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')


def parameter_traces_kde(ax, traces, param_idx=0, labels=True):
    for i, trace in enumerate(traces):
        sns.kdeplot(np.array(trace[:, param_idx]),
                    ax=ax, color=cm.colors[i],label=f'trace {i}')
    if labels:
        ax.set_xlabel('Parameter Value')


def parameter_traces_histogram(ax, traces, param_idx=0, bins=10, labels=True):
    for i, trace in enumerate(traces):
        ax.hist(np.array(trace[:, param_idx]), bins=bins,
                color=cm.colors[i], alpha=0.35, label=f'trace {i}')
    if labels:
        ax.set_xlabel('Parameter Value')


def parameter_traces_acf(ax, traces, param_idx=0, burnin=0, labels=True):
    rho_t = calc.multi_trace_autocorrelation(traces, param_idx)

    thin = int(max(len(rho_t) / 30, 1))

    y_stem = rho_t[::thin]
    x_stem = np.array(list(range(len(rho_t)))[::thin]) + burnin

    ax.stem(x_stem, y_stem, 'k', markerfmt='ko')
    ax.axhline(y=0.15, linestyle='--', color='k', alpha=0.5)
    ax.axhline(y=-0.15, linestyle='--', color='k', alpha=0.5)

    if labels:
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')


# Parameter Values
def model_param_estimates_relative_to_prior(ax, traces, priors, labels=True):
    post_m, post_s = calc.parameter_posterior_estimates(traces)
    prior_m = priors[0].dist.mean()
    prior_s = priors[0].dist.std()

    post_m_scaled = post_m[:len(prior_m)] - prior_m
    post_s_scaled = post_s[:len(prior_m)]/prior_s

    y = range(len(post_m_scaled))
    x = post_m_scaled
    xerr = post_s_scaled

    ax.errorbar(x, y, xerr=xerr, fmt='ko')
    ax.axvline(x=1.0, linestyle='--', color='k', alpha=0.5)
    ax.axvline(x=-1.0, linestyle='--', color='k', alpha=0.5)
    ax.axvline(x=0, linestyle='-', color='k', alpha=1.0)

    if labels:
        ax.set_xlabel('Difference from Prior Value')
        ax.set_ylabel('Parameter')


def plot_simulation_results(ax, sim_res, obs, **plot_kwargs):
    if isinstance(sim_res, pd.DataFrame):
        res_df = sim_res
    else:
        res_df = sim_res.opt2q_dataframe

    if 'time' not in res_df.columns:
        res_df.reset_index(inplace=True)
        res_df = res_df.rename(columns={'index': 'time'})

    if 'simulation' in res_df.columns:
        for name, df_ in res_df.groupby('simulation'):
            ax.plot(df_['time'].values, df_[obs].values, **plot_kwargs)
    else:
        ax.plot(res_df['time'].values, res_df[obs].values, **plot_kwargs)


def plot_simulation_results_quantile_fill_between(ax, sim_res, obs, **kwargs):
    upper_quantile = kwargs.pop('upper_quantile', 0.95)
    lower_quantile = kwargs.pop('lower_quantile', 0.05)

    sim_res_groupby = kwargs.pop('sim_groupby', 'time')
    if hasattr(sim_res, 'opt2q_dataframe'):
        upper_sim = calc.simulation_results_quantile(sim_res.opt2q_dataframe, upper_quantile, sim_res_groupby)
        lower_sim = calc.simulation_results_quantile(sim_res.opt2q_dataframe, lower_quantile, sim_res_groupby)
    else:
        upper_sim = calc.simulation_results_quantile(sim_res, upper_quantile, sim_res_groupby)
        lower_sim = calc.simulation_results_quantile(sim_res, lower_quantile, sim_res_groupby)

    y1 = lower_sim[obs].values
    y2 = upper_sim[obs].values
    x = upper_sim['time'].values
    ax.fill_between(x, y1, y2, **kwargs)


def plot_extrinsic_noise_on_parameter(ax, param_name, param_sample, population_param_sample=None, **kwargs):
    if population_param_sample is not None:  # Only use this when the population term is not already present.
        param_sample = utils.add_population_param_to_sample(param_sample, population_param_sample)

    m = utils.noisy_params[param_name]

    if len(param_sample.shape) == 1:
        if param_name is 'kc0':
            m = 10**param_sample[2]

        s2 = ((abs(param_sample[34]) ** -0.5) * m) ** 2  # 34 is the index of the population parameter.
        lp = calc.extrinsic_noise_distribution(m, s2)

        # Plot the extrinsic noise pdf
        x = np.linspace(lp.ppf(0.01),
                        lp.ppf(0.99), 1000)
        y = lp.pdf(x)/max(lp.pdf(x))

        ax.plot(x, y, **kwargs)
    else:
        lps = []
        x_min = []
        x_max = []
        for i in range(param_sample.shape[0]):
            if param_name is 'kc0':
                m = 10**param_sample[i, 2]

            s2 = ((abs(param_sample[i, 34]) ** -0.5) * m) ** 2  # 34 is the index of the population parameter.

            lps.append(calc.extrinsic_noise_distribution(m, s2))
            x_min.append(lps[i].ppf(0.01))
            x_max.append(lps[i].ppf(0.99))

        x = np.linspace(min(x_min), max(x_max), 1000)
        [ax.plot(x, lp.pdf(x)/max(lp.pdf(x)), **kwargs) for lp in lps]


def kde_of_parameter(ax, params, **kwargs):
    sns.kdeplot(np.array(params), ax=ax, **kwargs)


def kde_of_features(ax, features_df, feature, **kwargs):
    if 'population' in features_df:
        for n, g in features_df.groupby('population'):
            sns.kdeplot(g[feature], ax=ax, **kwargs)
    else:
        sns.kdeplot(features_df[feature], ax=ax, **kwargs)


def bivariate_kde_of_features(ax, features_df, x_feature, y_feature, **kwargs):
    sns.kdeplot(features_df[x_feature], features_df[y_feature], ax=ax, **kwargs)


def population_kde_of_features(ax, features_df, x_feature, y_feature, **kwargs):
    for n, g in features_df.groupby('population'):
        bivariate_kde_of_features(ax, g, x_feature, y_feature, **kwargs)


def measurement_model_sample(ax, params_array, x_feature, y_feature, probability, x, **kwargs):
    try:
        with np.errstate(divide='raise'):
            y = calc.cell_death_measurement_model_sample(params_array, x_feature, y_feature, probability, x)
            for i in range(y.shape[0]):
                ax.plot(x, y[i], **kwargs)
    except FloatingPointError:
        yt = calc.cell_death_measurement_model_sample(params_array, y_feature, x_feature, probability, x)
        for i in range(yt.shape[0]):
            ax.plot(yt[i], x, **kwargs)


def measurement_model_quantile_fill_between(ax, params_array, x_feature, y_feature, probability, x, **kwargs):
    upper_quantile = kwargs.pop('upper_quantile', 0.95)
    lower_quantile = kwargs.pop('lower_quantile', 0.05)

    y = calc.cell_death_measurement_model_sample(params_array, x_feature, y_feature, probability, x)

    upper_y = np.quantile(y, upper_quantile, axis=0)
    lower_y = np.quantile(y, lower_quantile, axis=0)

    ax.fill_between(x, lower_y, upper_y, **kwargs)


def measurement_model_quantile(ax, params_array, x_feature, y_feature, probability, x, **kwargs):
    quantile = kwargs.pop('quantile', 0.5)
    y = calc.cell_death_measurement_model_sample(params_array, x_feature, y_feature, probability, x)
    y1 = np.quantile(y, quantile, axis=0)

    ax.plot(x, y1, **kwargs)


def ordinal_measurement_model_quantile(ax, params_array, measurement_model):
    plot_domain = pd.DataFrame({'tBID_obs': np.linspace(0, 1, 100), 'cPARP_obs': np.linspace(0, 1, 100)})

    tBID_results_array = np.empty((100, 5, 100))

    for row_id in range(len(params_array)):
        row = params_array[row_id]
        c0 = row[0]
        t1 = row[1]
        t2 = t1 + row[2]
        t3 = t2 + row[3]
        t4 = t3 + row[4]

        c5 = row[5]
        t6 = row[6]
        t7 = t6 + row[7]
        t8 = t7 + row[8]

        measurement_model.process.get_step('classifier').set_params(
            **{'coefficients__tBID_blot__coef_': np.array([c0]),
               'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
               'coefficients__cPARP_blot__coef_': np.array([c5]),
               'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5})

        lc_results = measurement_model.process.get_step('classifier').transform(plot_domain)
        cPARP_results = lc_results.filter(regex='cPARP_blot')
        tBID_results = lc_results.filter(regex='tBID_blot')

        tBID_results_array[:, :, row_id] = tBID_results[
                ['tBID_blot__0', 'tBID_blot__1', 'tBID_blot__2', 'tBID_blot__3', 'tBID_blot__4']].values

    for xi in range(100):
        y05