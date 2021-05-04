import os
import re
import numpy as np
import pandas as pd
from pydream.parameters import SampledParam
from scipy.stats import norm, invgamma
from opt2q.simulator import Simulator
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import shift_and_scale_heterogeneous_population_to_new_params as sim_population, true_params, pre_processing

# MW Irvin -- Lopez Lab -- 2020-09-07


# Frequently used plotting code for the Opt2Q project
def load_cell_death_data(script_dir, file_name, measurement_type='cell_death_data'):
    file_path = os.path.join(script_dir, file_name)
    if measurement_type == 'cell_death_data':
        synth_data = pd.read_csv(file_path).iloc[::2].reset_index(
            drop=True)  # Remove half dataset to relieve load on solvers
        synth_data['simulation'] = range(len(synth_data))
        return synth_data
    elif measurement_type == 'fluorescence':
        raw_fluorescence_data = pd.read_csv(file_path)
        fluorescence_data = raw_fluorescence_data[
            ['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']] \
            .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
        fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')
        return fluorescence_data


def load_parameter_and_log_p_traces(script_dir, cal_folder, cal_date, cal_tag, include_extra_reactions=False):
    log_p_file_paths_ = sorted([os.path.join(script_dir, cal_folder, f) for f in
                                os.listdir(os.path.join(script_dir, cal_folder))
                                if cal_date in f and 'log_p' in f and cal_tag in f])

    parameter_file_paths_ = sorted([os.path.join(script_dir, cal_folder, f) for f in
                                    os.listdir(os.path.join(script_dir, cal_folder))
                                    if cal_date in f and 'parameters' in f and cal_tag in f])

    # reorder traces to be in numerical order (i.e. 1000 before 10000).
    file_order = sorted(list(set(int(re.findall(r'\d+', file_name)[-1]) for file_name in parameter_file_paths_)))

    log_p_file_paths = []
    parameter_file_paths = []
    for file_num in file_order:
        log_p_file_paths += [f for f in log_p_file_paths_ if f'_{file_num}_' in f]
        parameter_file_paths += [g for g in parameter_file_paths_ if f'_{file_num}_' in g]

    # get parameters
    log_p_traces = []
    parameter_samples = []

    traces = sorted(list(set(int(re.findall(r'\d+', file_name)[-2]) for file_name in parameter_file_paths_)))
    for trace_num in traces:
        print('Trace: ', trace_num)
        log_p_trace = np.concatenate([np.load(os.path.join(script_dir, cal_folder, lp))
                                      for lp in log_p_file_paths if f'_{trace_num}_' in lp])
        log_p_traces.append(log_p_trace)

        parameter_sample = np.concatenate([np.load(os.path.join(script_dir, cal_folder, pp))
                                           for pp in parameter_file_paths if f'_{trace_num}_' in pp])

        parameter_samples.append(parameter_sample)

    if include_extra_reactions:
        return parameter_samples, log_p_traces
    else:
        return [np.delete(ps, range(28, 34), axis=1) for ps in parameter_samples], log_p_traces


def get_parameter_sample(param_traces, log_p_traces, sample_size=100):
    p = np.concatenate(param_traces, axis=0)
    lp = np.concatenate(log_p_traces, axis=0)
    idx = np.random.randint(0, len(p), sample_size)
    return p[idx], lp[idx]


def get_max_posterior_parameters(param_traces, log_p_traces, sample_size=100):
    p = np.concatenate(param_traces, axis=0)
    lp = np.concatenate(log_p_traces, axis=0)
    all_idx = np.unique(lp, return_index=True)
    idx = all_idx[1][-sample_size:]
    return p[idx], lp[idx], idx


def thin_traces(param_traces, log_p_traces, thin=1, burn_in=0):
    lp_traces = []
    parameter_samples = []

    for param_sample in param_traces:
        parameter_samples.append(param_sample[burn_in::thin])

    for log_p_trace in log_p_traces:
        lp_traces.append(log_p_trace[burn_in::thin])

    return parameter_samples, lp_traces


def load_chain_history(script_dir, cal_folder, cal_date, cal_tag):
    chain_history_file = [os.path.join(script_dir, cal_folder, f) for f in
                          os.listdir(os.path.join(script_dir, cal_folder))
                          if cal_date in f and 'chain_history' in f and cal_tag in f][0]
    return np.load(chain_history_file)


def load_gelman_rubin_values(script_dir, cal_folder, cal_date, cal_tag):
    # reorder traces to be in numerical order (i.e. 1000 before 10000).
    parameter_file_paths_ = sorted([os.path.join(script_dir, cal_folder, f) for f in
                                    os.listdir(os.path.join(script_dir, cal_folder))
                                    if cal_date in f and 'parameters' in f and cal_tag in f])
    file_order = sorted(list(set(int(re.findall(r'\d+', file_name)[-1]) for file_name in parameter_file_paths_)))
    print(sorted([os.path.join(script_dir, cal_folder, f) for f in
                      os.listdir(os.path.join(script_dir, cal_folder))
                      if cal_date in f and str(file_order[-1]) in f and '.txt' in f and cal_tag in f]))
    gr_file = sorted([os.path.join(script_dir, cal_folder, f) for f in
                      os.listdir(os.path.join(script_dir, cal_folder))
                      if cal_date in f and str(file_order[-1]) in f and '.txt' in f and cal_tag in f])[0]

    return np.loadtxt(gr_file)


def get_model_param_names(model, include_extra_reactions=False):
    if include_extra_reactions:
        return [p.name for p in model.parameters_rules()]
    else:
        return [p.name for p in model.parameters_rules()][:-6]


def get_measurement_param_names(measurement_type):
    if measurement_type == 'fluorescence':
        return []
    elif measurement_type == 'immunoblot':
        return ['coefficients__tBID_blot__coef_', 'coefficients__tBID_blot__theta_1 ',
                'coefficients__tBID_blot__theta_2 ', 'coefficients__tBID_blot__theta_3 ',
                'coefficients__tBID_blot__theta_4 ', 'coefficients__cPARP_blot__coef_',
                'coefficients__cPARP_blot__theta_1', 'coefficients__cPARP_blot__theta_2',
                'coefficients__cPARP_blot__theta_3', 'coefficients__cPARP_blot__theta_4']
    elif measurement_type == 'cell_death_data':
        return ['slope', 'intercept', 'unrelated signal coef', 'tBID obs coef', 'time coef']
    elif measurement_type == 'immunoblot_disc':
        return ['coefficients__IC_DISC_localization__coef_',
                'coefficients__IC_DISC_localization__theta_1',
                'coefficients__IC_DISC_localization__theta_2',
                'coefficients__IC_DISC_localization__theta_3']
    else:
        raise ValueError("measurement_type can only be 'fluorescence', 'immunoblot', 'immunoblot_disc', "
                         "or 'cell_death_data'")


def get_population_param(measurement_type):
    if measurement_type == 'cell_death_data':
        return ['Population covariance term']
    else:
        return []


def get_model_param_true(include_extra_reactions=False):
    script_dir = os.path.dirname(__file__)
    if include_extra_reactions:
        return np.load(os.path.join(script_dir, 'true_params.npy'))
    else:
        return np.load(os.path.join(script_dir, 'true_params.npy'))[:-6]


def get_model_param_start(model, include_extra_reactions=False):
    if include_extra_reactions:
        return [p.value for p in model.parameters_rules()]
    else:
        return [p.value for p in model.parameters_rules()][:-6]


def get_population_param_start(measurement_type='cell_death_data'):
    if measurement_type == 'cell_death_data':
        return [25.0]
    else:
        return []


def get_measurement_model_params(measurement_type='cell_death_data', cal_tag=None, params=None):
    if measurement_type == 'cell_death_data':
        if 'opt2q' in cal_tag:
            if params is None:
                raise TypeError("params cannot be None")
            return params[-5:]


def get_measurement_model_true_params(measurement_type='cell_death_data'):
    if measurement_type == 'cell_death_data':
        return np.array([4, -0.25, 0.00, 0.25, -1])
    if measurement_type == 'immunoblot':
        return np.array([50, 0.03, 0.37, 0.42, 0.15, 50, 0.03, 0.17, 0.77])
    if measurement_type == 'immunoblot_disc':
        return np.array([25, 0.05, 0.35, 0.45])


def get_classifier_params(x, measurement_type='immunoblot'):
    if measurement_type == 'immunoblot':
        return get_classifier_params_immunoblot(x)
    else:
        raise ValueError('try immunoblot')


def get_classifier_params_immunoblot(x):
    x = abs(x)
    c0 = x[0]
    t1 = x[1]
    t2 = t1 + x[2]
    t3 = t2 + x[3]
    t4 = t3 + x[4]

    c5 = x[5]
    t6 = x[6]
    t7 = t6 + x[7]
    t8 = t7 + x[8]
    return {'coefficients__tBID_blot__coef_': np.array([c0]),
            'coefficients__tBID_blot__theta_': np.array([t1, t2, t3, t4]) * c0,
            'coefficients__cPARP_blot__coef_': np.array([c5]),
            'coefficients__cPARP_blot__theta_': np.array([t6, t7, t8]) * c5}


def get_model_param_priors(include_extra_reactions=False):
    p = get_model_param_true(include_extra_reactions=include_extra_reactions)
    return [SampledParam(norm, loc=p, scale=1.5)]


def sample_model_param_priors(priors, size):
    return np.column_stack([priors[0].dist.rvs() for i in range(size)]).T


def get_population_param_priors(measurement_type='cell_death_data'):
    if measurement_type == 'cell_death_data':
        nu = 100
        noisy_param_stdev = 0.20
        alpha = int(np.ceil(nu / 2.0))
        beta = alpha / noisy_param_stdev ** 2
        return[SampledParam(invgamma, *[alpha], scale=beta)]
    else:
        return []


def set_up_simulator(measurement_type, model):
    parameters = pd.DataFrame([[p.value for p in model.parameters_rules()]],
                              columns=[p.name for p in model.parameters_rules()])

    if measurement_type == 'cell_death_data':
        sim = Simulator(model=model, param_values=parameters, solver='scipyode',
                        solver_options={'integrator': 'lsoda'}, tspan=np.linspace(0, 20160, 100),
                        integrator_options={'rtol': 1e-12, 'atol': 1e-12, 'mxstep': 2**20})
        sim.run()
        return sim

    if measurement_type == 'fluorescence':
        sim = Simulator(model=model, param_values=parameters, solver='scipyode',
                        solver_options={'integrator': 'lsoda'}, tspan=np.linspace(0, 20160, 100),
                        integrator_options={'rtol': 1e-3, 'atol': 1e-2})
        sim.run()
        return sim

    if measurement_type == 'immunoblot':
        sim = Simulator(model=model, param_values=parameters, solver='scipyode', tspan=np.linspace(0, 20160, 100),
                        solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2**20})
        return sim

    else:
        raise ValueError(f"Measurement_type can only be 'fluorescence', 'immunoblot', 'cell_death_data'.")


def apply_extrinsic_noise_to_parameter_sample(param_sample, population_param_terms=None):
    if population_param_terms is not None:  # Only use this when the population term is not already present.
        param_sample = add_population_param_to_sample(param_sample, population_param_terms)

    for i in range(param_sample.shape[0]):
        df = sim_population(param_sample[i, :])
        df['param_sample'] = i


def add_population_param_to_sample(model_param_sample, population_param_sample):
    if len(model_param_sample.shape) == 1:
        return np.concatenate((model_param_sample, population_param_sample))
    else:
        return np.concatenate((model_param_sample, population_param_sample[:, None]), axis=1)


noisy_params = {'MOMP_sig_0': 100000, 'USM1_0': 1000, 'USM2_0': 1000, 'USM3_0': 1000, 'kc0': 10**true_params[2]}


def sample_model_priors_for_feature_processing(model_param_priors, pop_param_prior, sim_, size):
    n = 0
    m = 0
    x = list(sample_model_param_priors(model_param_priors, 1)[0]) + list(pop_param_prior[0].dist.rvs(1))
    pr_array_list = []
    sim_res_array_list = []

    pr_df = sim_population(x)
    pr_cols = pr_df.columns

    sim_.param_values = pr_df
    sim_res_df = sim_.run().opt2q_dataframe.reset_index()
    sim_res_cols = sim_res_df.columns

    if pre_processing(sim_res_df) is not None:
        pr_array_list.append(pr_df[pr_cols].values)
        sim_res_array_list.append(sim_res_df[sim_res_cols].values)
        print(f'accepted {n} out of 10')
        n += 1
    else:
        m += 1
        print(f'{m} unsuccessful tries')

    while n < size:
        x = list(sample_model_param_priors(model_param_priors, 1)[0]) + list(pop_param_prior[0].dist.rvs(1))
        pr_df = sim_population(x)
        sim_.param_values = pr_df
        sim_res_df = sim_.run().opt2q_dataframe.reset_index()
        if pre_processing(sim_res_df) is not None:
            pr_array_list.append(pr_df[pr_cols].values)
            sim_res_array_list.append(sim_res_df[sim_res_cols].values)
            print(n)
            n += 1
        else:
            m += 1
            print(f'{m} unsuccessful tries')

    all_pop = pd.DataFrame(np.vstack(pr_array_list), columns=pr_cols)
    all_pop = all_pop.drop(['simulation'], axis=1)
    all_pop['simulation'] = range(len(all_pop))
    all_pop['population'] = np.repeat(range(size), 400)

    all_pop_sim_res = pd.DataFrame(np.vstack(sim_res_array_list), columns=sim_res_cols)
    all_pop_sim_res['simulation'] = np.repeat(range(len(all_pop)), 100)
    all_pop_sim_res['population'] = np.repeat(range(size), 40000)

    return all_pop, all_pop_sim_res
