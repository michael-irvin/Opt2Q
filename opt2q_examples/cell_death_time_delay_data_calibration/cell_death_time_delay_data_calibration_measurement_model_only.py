# MW Irvin -- Lopez Lab -- 2018-11-18

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import invgamma, truncnorm
from matplotlib import pyplot as plt
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
from pydream.parameters import SampledParam
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import Interpolate, ScaleToMinMax
from opt2q.calibrator import objective_function
from opt2q_examples.apoptosis_model import model

# ------- Synthetic Data ----
script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, 'synthetic_cPARP_dependent_apoptosis_data_noisy_threshold_model.csv')
synth_data = pd.read_csv(file_path)

file_path = os.path.join(script_dir, 'true_params_extrinsic_noise.csv')
extrinsic_noise_params = pd.read_csv(file_path)

# ------- Simulations -------
# fluorescence data as reference
file_path = os.path.join(script_dir, '../fluorescence_data_calibration/fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
fluorescence_data = raw_fluorescence_data[['# Time', 'norm_IC-RP', 'nrm_var_IC-RP', 'norm_EC-RP', 'nrm_var_EC-RP']]\
    .rename(columns={'# Time': 'time_min'})  # Remove unnecessary whitespace in column name
fluorescence_data = fluorescence_data.assign(time=fluorescence_data.time_min * 60).drop(columns='time_min')

sim = Simulator(model=model, param_values=extrinsic_noise_params, solver='cupsoda')
sim_results = sim.run(np.linspace(0, fluorescence_data.time.max(), 100))
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})


