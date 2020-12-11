# MW Irvin -- Lopez Lab -- 2020-10-13
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft, ifftshift
from pydream.convergence import Gelman_Rubin
from scipy.stats import lognorm
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import shift_and_scale_heterogeneous_population_to_new_params as simulate_population
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup import pre_processing


def gelman_rubin_trace(traces, param_idx, burn_in, thin):
    return [Gelman_Rubin([trace[:i] for trace in traces])[param_idx]
            for i in range(burn_in, len(traces[0]), thin)]


def autocorrelation(x):
    xp = ifftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)


def multi_trace_autocorrelation(traces, param_idx=0):
    # compute combined auto-correlation based on
    # https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
    w_list = [np.var(np.array(trace[:, param_idx])) for trace in traces]  # within-chain variances
    w = 1.0/len(w_list)*np.sum(w_list)  # within-chain variance for all chains

    pm_list = [np.average(np.array(trace[:, param_idx])) for trace in traces]  # within-chain means
    pm = 1.0/len(pm_list)*np.sum(pm_list)

    n = traces[0].shape[0]  # number of iterations
    m = len(pm_list)
    var_est = (n-1.0)/n*w + 1/(m-1.0) * np.average((pm_list - pm)**2)  # variance estimator

    rho_tm = [np.var(np.array(trace[:, param_idx])) * autocorrelation(trace[:, param_idx]) for trace in traces]
    avr_rho_tm = np.average(rho_tm, axis=0)

    rho_t = 1 - (w - avr_rho_tm)/var_est

    return rho_t


def effective_sample_size(rho_t, n_chains):
    try:  # max lag size with autocorrelation greater than 0
        t_ = np.argwhere(np.array(rho_t) <= 0)[0][0]-1
    except IndexError:
        t_ = len(rho_t) - 1

    d_rho_t = np.roll(rho_t, -1)[:-1] - rho_t[:-1]

    try:  # max lag size with monotonic autocorrelation
        t_ = np.argwhere(np.array(d_rho_t[:t_]) >= 0)[0][0] -1
    except IndexError:
        pass

    return n_chains*len(rho_t)/(1+2*sum(rho_t[:t_]))


def param_mean_and_std(traces, param_idx=0):
    rho_t = multi_trace_autocorrelation(traces, param_idx=0)
    n_eff = effective_sample_size(rho_t, len(traces))
    p = np.concatenate(traces, axis=0)
    p_avr = np.average(p[:, param_idx])
    p_std = np.std(p[:, param_idx])/np.sqrt(n_eff)

    return p_avr, p_std


def parameter_posterior_estimates(traces):
    param_posterior_mean = []
    param_posterior_std = []

    for param_id in range(traces[0].shape[1]):
        p_avr, p_std = param_mean_and_std(traces, param_idx=param_id)
        param_posterior_mean.append(p_avr)
        param_posterior_std.append(p_std)

    return param_posterior_mean, param_posterior_std


def simulation_results_quantile(sim_results_df, quantile, groupby='time'):
    if groupby is not None:
        quantile_results = sim_results_df.groupby(groupby).quantile(quantile)
    else:
        quantile_results = sim_results_df.groupby(groupby).quantile(quantile)
    quantile_results.reset_index(inplace=True)

    return quantile_results.rename(columns={'index': groupby})


def simulation_results_quantiles_list(sim_res, quantiles_list, groupby='time'):
    if isinstance(sim_res, pd.DataFrame):
        res_df = sim_res
    else:
        res_df = sim_res.opt2q_dataframe

    return (simulation_results_quantile(res_df, q, groupby) for q in quantiles_list)


def extrinsic_noise_distribution(m, v):
    mu = np.log(m) - 0.5*np.log(1+v/m**2)
    sig2 = np.log(1+v/m**2)

    return lognorm(s=np.sqrt(sig2), scale=np.exp(mu))


def simulate_population_multi_params(params):
    if len(params.shape) == 1:
        return simulate_population(params)

    else:
        _n = range(params.shape[0])
        pops = []
        pop0_df = simulate_population(params[0, :])
        cols = pop0_df.columns
        pops.append(pop0_df)

        for i in _n[1:]:
            df = simulate_population(params[i, :])
            pops.append(df[cols].values)
        pops_ = np.vstack(pops)
        dfs = pd.DataFrame(pops_, columns=cols)
        ids = pd.DataFrame(np.repeat(_n, 400), columns=['population'])
        all_pops = pd.concat([dfs, ids], axis=1)
        all_pops.rename({'simulation': 'simulation_'})
        all_pops['simulation'] = np.arange(all_pops.shape[0])
        return all_pops


def pre_process_simulation(sim_res):
    if isinstance(sim_res, pd.DataFrame):
        res_df = sim_res
    else:
        res_df = sim_res.opt2q_dataframe

    if 'time' not in res_df.columns:
        res_df.reset_index(inplace=True)
        res_df = res_df.rename(columns={'index': 'time'})
    if 'simulation' not in res_df.columns:
        res_df['simulation'] = 0

    if 'population' in res_df.columns:
        gn = res_df.groupby('population')
        pp = []
        pp0 = pre_processing(gn.get_group(0))
        cols = pp0.columns
        pp_array = pp0[cols].values
        pp_array = np.concatenate([pp_array, np.full((pp_array.shape[0], 1), 0)], axis=1)
        pp.append(pp_array)

        for n, g in gn.groups.items():
            if n == 0:
                pass
            else:
                pp_array = pre_processing(gn.get_group(n))[cols].values
                pp_array = np.concatenate([pp_array, np.full((pp_array.shape[0], 1), n)], axis=1)
                pp.append(pp_array)

        pp_df = pd.DataFrame(np.vstack(pp), columns=cols.append(pd.Index(['population'])))

        return pp_df
    else:
        return pre_processing(res_df)


def cell_death_measurement_model(param, x_feature, y_feature, prob=0.5):
    feature_idx_dict = {'Unrelated_Signal': 2, 'tBID_obs': 3, 'time': 4}
    slope = param[0]
    intercept = slope * param[1]
    y_coef = slope * param[feature_idx_dict[y_feature]]
    x_coef = slope * param[feature_idx_dict[x_feature]]

    def measurement_model(x, probability=prob):
        n = np.log(probability/(1-probability))
        return (1/y_coef) * (n - intercept - (x_coef * x))

    return measurement_model


def cell_death_measurement_model_sample(params_array, x_feature, y_feature, probability, x):
    y_list = []
    for p_row in range(params_array.shape[0]):
        mm = cell_death_measurement_model(params_array[p_row, :], x_feature, y_feature, probability)
        y_list.append(mm(x))
    return np.vstack(y_list)


def feature_values(params_array, feature):
    feature_idx_dict = {'Unrelated_Signal': -3, 'tBID_obs': -2, 'time': -1}
    slope = params_array[:, -5]
    return params_array[:, feature_idx_dict[feature]]*slope


