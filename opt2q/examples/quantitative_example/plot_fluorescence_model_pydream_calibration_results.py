# MW Irvin -- Lopez Lab -- 2019-06-24

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from opt2q.examples.quantitative_example.fluorescence_likelihood_fn import fluorescence_data, likelihood_fn
from opt2q.examples.quantitative_example.calibrating_fluorescence_model_pydream import sampled_params_0, n_chains

chain_history = np.load('PyDREAM_Fluorescence_20190506_DREAM_chain_history.npy')
chain_log_post_0 = np.load('PyDREAM_Fluorescence_201905060_10000.npy')
chain_log_post_1 = np.load('PyDREAM_Fluorescence_201905061_10000.npy')
chain_log_post_2 = np.load('PyDREAM_Fluorescence_201905062_10000.npy')
chain_log_post_3 = np.load('PyDREAM_Fluorescence_201905063_10000.npy')
chain_log_post_4 = np.load('PyDREAM_Fluorescence_201905064_10000.npy')


n_params = len(sampled_params_0)
param_names = ['kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7']
parameters = chain_history.reshape(int(len(chain_history)/n_params), n_params)

cm = plt.get_cmap('tab10')

# Posterior Values Per Chain Convergence
plt.plot(chain_log_post_0, alpha=0.8, label='chain 0')
plt.plot(chain_log_post_1, alpha=0.8, label='chain 1')
plt.plot(chain_log_post_2, alpha=0.8, label='chain 2')
plt.plot(chain_log_post_3, alpha=0.8, label='chain 3')
plt.plot(chain_log_post_4, alpha=0.8, label='chain 4')
plt.title("Log Posterior")
plt.xlabel('Iteration')
plt.legend()
plt.show()

plt.plot(range(2000, 10000), chain_log_post_0[2000:], alpha=0.8, label='chain 0')
plt.plot(range(2000, 10000), chain_log_post_1[2000:], alpha=0.8, label='chain 1')
plt.plot(range(2000, 10000), chain_log_post_2[2000:], alpha=0.8, label='chain 2')
plt.plot(range(2000, 10000), chain_log_post_3[2000:], alpha=0.8, label='chain 3')
plt.plot(range(2000, 10000), chain_log_post_4[2000:], alpha=0.8, label='chain 4')
plt.title("Log Posterior")
plt.xlabel('Iteration')
plt.legend()
plt.show()

# Convergence of Parameter Values Per Chain
for param_num in range(n_params):
    for chain_num in range(n_chains):
        plt.plot(parameters[chain_num::n_chains, param_num], label='chain %s' % chain_num, alpha=0.3)
    plt.legend()
    plt.title('Parameter %s; %s' % (param_num, param_names[param_num]))
    plt.ylabel('parameter value')
    plt.xlabel('iteration')
    plt.show()

# Parameter Values
for param_num in range(n_params):
    init_dist = sampled_params_0[param_num].dist
    x = np.linspace(init_dist.ppf(0.01), init_dist.ppf(0.99), 100)
    plt.plot(x, init_dist.pdf(x), '--', color=cm.colors[0], label='initial')

    param_samples = parameters[2000*n_chains:, param_num]
    xs = np.linspace(np.min(param_samples), np.max(param_samples), 100)
    density = gaussian_kde(param_samples)

    plt.hist(parameters[2000*n_chains:, param_num], normed=True, bins=30, alpha=0.5, color=cm.colors[1])
    plt.plot(xs, density(xs), label='final', color=cm.colors[0])
    plt.legend()
    plt.xlabel('parameter value')
    plt.title('Marginal distribution of parameter %s' % param_names[param_num])
    plt.show()

# Joint Parameter Value Distributions
params_df = pd.DataFrame(parameters[2000*n_chains:, :], columns=param_names)
g = sns.pairplot(params_df, diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, color=cm.colors[0], edgecolor=cm.colors[0], linewidth=1, alpha=0.01),
                 diag_kws=dict(shade=True, color=cm.colors[0]))
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)
plt.show()


