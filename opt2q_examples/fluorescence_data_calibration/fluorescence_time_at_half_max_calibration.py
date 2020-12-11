import os
import pandas as pd
import numpy as np
import datetime as dt
from opt2q.calibrator import objective_function
from multiprocessing import current_process
from opt2q_examples.fluorescence_data_calibration.generate_time_at_half_max_data import \
    set_up_simulator, pre_processing
from opt2q_examples.cell_death_data_calibration.cell_death_data_calibration_setup \
    import shift_and_scale_heterogeneous_population_to_new_params as sim_population
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from scipy.stats import norm, invgamma

script_dir = os.path.dirname(__file__)

# Model Name
now = dt.datetime.now()
model_name = f'apoptosis_model_tbid_time_at_half_max_{now.year}{now.month}{now.day}'


# ------- Initial Conditions -------
file_path = os.path.join(script_dir, '../cell_death_data_calibration/true_params_extrinsic_noise_large.csv')
extrinsic_noise_params = pd.read_csv(file_path).iloc[::2].reset_index(drop=True)  # Remove half parameters as with data
extrinsic_noise_params['simulation'] = range(len(extrinsic_noise_params))

# ------- Data --------
file_path = os.path.join(script_dir, 'fluorescence_time_at_half_max_dataset.csv')
data = pd.read_csv(file_path)

# ------- Priors --------
true_params = np.load('true_params.npy')

nu = 100
noisy_param_stdev = 0.20

alpha = int(np.ceil(nu / 2.0))
beta = alpha / noisy_param_stdev ** 2

alpha_tau = 1
beta_tau = alpha_tau / 240.0 ** 2

sampled_params_0 = [SampledParam(norm, loc=true_params, scale=1.5),
                    SampledParam(invgamma, *[alpha], scale=beta),
                    SampledParam(invgamma, *[alpha_tau], scale=beta_tau)]

# ------- PyDREAM Parameters -------
n_chains = 4
n_iterations = 100000  # iterations per file-save
burn_in_len = 100000   # number of iterations during burn-in
max_iterations = 120000

# ------- Simulations -------
sim = set_up_simulator('scipyode')

# ------- Likelihood --------
# The likelihood function comes from the following:
# Bronstein, L., Zechner, C., & Koeppl, H. (2015). Bayesian inference of reaction kinetics from single-cell recordings
# across a heterogeneous cell population. Methods, 85, 22-35.

# log(\mathscr{L}(y|\theta)) = M \sum_{g=1}^G \hat{\pi}_g log(\hat{\pi}_g)
#                               - M \sum_{g=1}^G \hat{\pi}_g log(\frac{\hat{\pi}_g}{\pi_g})

bins_10ng_ml = sorted(data[data.TRAIL_conc == '10ng/mL']['time'].unique())
bins_50ng_ml = sorted(data[data.TRAIL_conc == '50ng/mL']['time'].unique())

empirical_dist_10ng_mL = data[data.TRAIL_conc == '10ng/mL'].groupby('time')['simulation'].nunique()
empirical_dist_50ng_mL = data[data.TRAIL_conc == '50ng/mL'].groupby('time')['simulation'].nunique()

m_10ng_ml = len(data[data.TRAIL_conc == '10ng/mL'])
entropy_term_10ng_ml = sum([i*np.log(i/m_10ng_ml) for i in empirical_dist_10ng_mL])

m_50ng_ml = len(data[data.TRAIL_conc == '50ng/mL'])
entropy_term_50ng_ml = sum([i*np.log(i/m_10ng_ml) for i in empirical_dist_50ng_mL])


# likelihood function
@objective_function(gen_param_df=sim_population, sim=sim, pre_processing=pre_processing,
                    dataset=data, return_results=False, evals=0)
def likelihood(x):
    params_df = likelihood.gen_param_df(x)  # simulate heterogeneous population around new param values
    params_df = params_df.iloc[::4].reset_index(drop=True)  # Simulate a smaller sample from the population
    params_df['simulation'] = range(len(params_df))

    likelihood.sim.param_values = params_df

    try:
        if hasattr(likelihood.sim.sim, 'gpu'):
            process_id = current_process().ident % 4
            likelihood.sim.sim.gpu = [process_id]

            # likelihood.sim.sim.gpu = [1]
        new_results = likelihood.sim.run().opt2q_dataframe.reset_index()

        # run pre-processing
        likelihood.prediction = likelihood.pre_processing(new_results)

        # Model predicted distribution as mixture of Gaussian functions
        tau = abs(x[35]) ** -0.5  # index 0-33 are the model params. 34 is the extrinsic noise term. 35 is this.
        pp = likelihood.prediction

        # get the modeled probability of each bin 10ng/mL
        pi_g_0 = 1/50.0*sum([norm.cdf(bins_10ng_ml[1:] - np.full_like(bins_10ng_ml[1:], 90.0), loc=k, scale=tau)
                             for k in pp[pp.TRAIL_conc == '10ng/mL']['time']])
        pi_g_0 = np.hstack(([0], pi_g_0))
        pi_g_1 = 1/50.0*sum([norm.cdf(bins_10ng_ml[:-1] + np.full_like(bins_10ng_ml[:-1], 90.0), loc=k, scale=tau)
                             for k in pp[pp.TRAIL_conc == '10ng/mL']['time']])
        pi_g_1 = np.hstack((pi_g_1, [1]))
        pi_g = np.clip(pi_g_1 - pi_g_0, 1e-15, np.inf)

        # calculate the entropy term
        ll = entropy_term_10ng_ml - sum([v*np.log(v/(m_10ng_ml*pi_g[i]))
                                         for i, v in enumerate(empirical_dist_10ng_mL)])

        # get the modeled probability of each bin 50ng/mL
        pi_g_0 = 1 / 50.0 * sum([norm.cdf(bins_10ng_ml[1:] - np.full_like(bins_10ng_ml[1:], 90.0), loc=k, scale=tau)
                                 for k in pp[pp.TRAIL_conc == '50ng/mL']['time']])
        pi_g_0 = np.hstack(([0], pi_g_0))
        pi_g_1 = 1 / 50.0 * sum([norm.cdf(bins_10ng_ml[:-1] + np.full_like(bins_10ng_ml[:-1], 90.0), loc=k, scale=tau)
                                 for k in pp[pp.TRAIL_conc == '50ng/mL']['time']])
        pi_g_1 = np.hstack((pi_g_1, [1]))
        pi_g = np.clip(pi_g_1 - pi_g_0, 1e-15, np.inf)

        ll += entropy_term_50ng_ml - sum([v*np.log(v/(m_50ng_ml*pi_g[i]))
                                          for i, v in enumerate(empirical_dist_50ng_mL)])
        return ll

    except (ValueError, ZeroDivisionError, TypeError):
        return -1e10


# -------- Calibration -------
# Model Inference via PyDREAM
if __name__ == '__main__':
    ncr = 25
    gamma_levels = 8
    p_gamma_unity = 0.1
    print(ncr, gamma_levels, p_gamma_unity)

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       nCR=ncr,
                                       gamma_levels=gamma_levels,
                                       adapt_gamma=True,
                                       p_gamma_unity=p_gamma_unity,
                                       history_thin=1,
                                       model_name=model_name,
                                       verbose=True,
                                       crossover_burnin=min(n_iterations, burn_in_len),
                                       )

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters', sampled_params[chain])
        np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    burn_in_len = max(burn_in_len - n_iterations, 0)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')

    np.savetxt(model_name + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.isnan(GR).any() or np.any(GR > 1.2):
        # append sample with a re-run of the pyDream algorithm
        while not converged or (total_iterations < max_iterations):
            starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

            total_iterations += n_iterations
            sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                               likelihood=likelihood,
                                               niterations=n_iterations,
                                               nchains=n_chains,
                                               multitry=False,
                                               nCR=ncr,
                                               gamma_levels=gamma_levels,
                                               adapt_gamma=True,
                                               p_gamma_unity=p_gamma_unity,
                                               history_thin=1,
                                               model_name=model_name,
                                               verbose=True,
                                               restart=True,  # restart at the last sampled position
                                               start=starts,
                                               crossover_burnin=min(n_iterations, burn_in_len))

            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'parameters',
                        sampled_params[chain])
                np.save(model_name + '_' + str(chain) + '_' + str(total_iterations) + '_' + 'log_p', log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(n_chains)]
            GR = Gelman_Rubin(old_samples)
            burn_in_len = max(burn_in_len - n_iterations, 0)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')

            np.savetxt(model_name + str(total_iterations) + '.txt', GR)

            if np.all(GR < 1.2):
                converged = True


make_plot = False
if make_plot:
    from matplotlib import pyplot as plt
    from scipy.stats import norm

    test_params = np.hstack((true_params, [25, 1.736111111111111e-05]))
    l1 = likelihood(test_params)

    pp = likelihood.prediction
    x = np.linspace(min(bins_50ng_ml), max(bins_10ng_ml), 100)
    bins10 = int(
        (data[data.TRAIL_conc == '10ng/mL']['time'].max() - data[data.TRAIL_conc == '10ng/mL']['time'].min()) / 180.0)
    bins50 = int(
        (data[data.TRAIL_conc == '50ng/mL']['time'].max() - data[data.TRAIL_conc == '50ng/mL']['time'].min()) / 180.0)
    y10 = sum([1 / 50.0 * norm.pdf(x, k, 240.0) for k in pp[pp.TRAIL_conc == '10ng/mL']['time']])
    y50 = sum([1 / 50.0 * norm.pdf(x, k, 240.0) for k in pp[pp.TRAIL_conc == '50ng/mL']['time']])
    cm = plt.get_cmap('tab10')
    plt.hist(data[data.TRAIL_conc == '10ng/mL']['time'], alpha=0.5, color=cm.colors[7], label='10 ng/mL TRAIL Data',
             bins=bins10, density=True)
    plt.hist(data[data.TRAIL_conc == '50ng/mL']['time'], alpha=0.5, color=cm.colors[1], label='50 ng/mL TRAIL Data',
             bins=bins50, density=True)
    plt.plot(x, y10, color=cm.colors[7], label='10 ng/mL TRAIL Modeled')
    plt.plot(x, y50, color=cm.colors[1], label='50 ng/mL TRAIL Modeled')
    plt.title(f'Time at half-max BID truncation, likelihood = {l1}')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    test_params[2] += 0.15
    l2 = likelihood(test_params)

    pp = likelihood.prediction
    x = np.linspace(min(bins_50ng_ml), max(bins_10ng_ml), 100)
    bins10 = int(
        (data[data.TRAIL_conc == '10ng/mL']['time'].max() - data[data.TRAIL_conc == '10ng/mL']['time'].min()) / 180.0)
    bins50 = int(
        (data[data.TRAIL_conc == '50ng/mL']['time'].max() - data[data.TRAIL_conc == '50ng/mL']['time'].min()) / 180.0)
    y10 = sum([1 / 50.0 * norm.pdf(x, k, 240.0) for k in pp[pp.TRAIL_conc == '10ng/mL']['time']])
    y50 = sum([1 / 50.0 * norm.pdf(x, k, 240.0) for k in pp[pp.TRAIL_conc == '50ng/mL']['time']])
    cm = plt.get_cmap('tab10')
    plt.hist(data[data.TRAIL_conc == '10ng/mL']['time'], alpha=0.5, color=cm.colors[7], label='10 ng/mL TRAIL Data',
             bins=bins10, density=True)
    plt.hist(data[data.TRAIL_conc == '50ng/mL']['time'], alpha=0.5, color=cm.colors[1], label='50 ng/mL TRAIL Data',
             bins=bins50, density=True)
    plt.plot(x, y10, color=cm.colors[7], label='10 ng/mL TRAIL Modeled')
    plt.plot(x, y50, color=cm.colors[1], label='50 ng/mL TRAIL Modeled')
    plt.title(f'Time at half-max BID truncation, likelihood = {l2}')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()

    test_params[2] += 0.05
    l2 = likelihood(test_params)

    pp = likelihood.prediction
    x = np.linspace(min(bins_50ng_ml), max(bins_10ng_ml), 100)
    bins10 = int(
        (data[data.TRAIL_conc == '10ng/mL']['time'].max() - data[data.TRAIL_conc == '10ng/mL']['time'].min()) / 180.0)
    bins50 = int(
        (data[data.TRAIL_conc == '50ng/mL']['time'].max() - data[data.TRAIL_conc == '50ng/mL']['time'].min()) / 180.0)
    y10 = sum([1 / 50.0 * norm.pdf(x, k, 240.0) for k in pp[pp.TRAIL_conc == '10ng/mL']['time']])
    y50 = sum([1 / 50.0 * norm.pdf(x, k, 240.0) for k in pp[pp.TRAIL_conc == '50ng/mL']['time']])
    cm = plt.get_cmap('tab10')
    plt.hist(data[data.TRAIL_conc == '10ng/mL']['time'], alpha=0.5, color=cm.colors[7], label='10 ng/mL TRAIL Data',
             bins=bins10, density=True)
    plt.hist(data[data.TRAIL_conc == '50ng/mL']['time'], alpha=0.5, color=cm.colors[1], label='50 ng/mL TRAIL Data',
             bins=bins50, density=True)
    plt.plot(x, y10, color=cm.colors[7], label='10 ng/mL TRAIL Modeled')
    plt.plot(x, y50, color=cm.colors[1], label='50 ng/mL TRAIL Modeled')
    plt.title(f'Time at half-max BID truncation, likelihood = {l2}')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()
