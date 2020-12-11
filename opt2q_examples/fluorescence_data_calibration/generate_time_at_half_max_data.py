import os
import pandas as pd
import numpy as np
from opt2q.simulator import Simulator
from opt2q_examples.apoptosis_model import model
from opt2q.measurement.base.transforms import ScaleToMinMax, Interpolate

script_dir = os.path.dirname(__file__)
overwrite_synthetic_dataset = True

# ------- Initial Conditions -------
file_path = os.path.join(script_dir, '../cell_death_data_calibration/true_params_extrinsic_noise_large.csv')
extrinsic_noise_params = pd.read_csv(file_path).iloc[::2].reset_index(drop=True)  # Remove half parameters as with data
extrinsic_noise_params['simulation'] = range(len(extrinsic_noise_params))

# ------- Simulations -------
# fluorescence data as reference
file_path = os.path.join(script_dir, 'fluorescence_data.csv')
raw_fluorescence_data = pd.read_csv(file_path)
time_axis = np.linspace(0, raw_fluorescence_data['# Time'].max()*60, 100)

sim = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver='cupsoda',
                integrator_options={'vol': 4.0e-15, 'max_steps': 2**20})


# ======= Import this from cell_death_data.calibration_setup ============
def set_up_simulator(solver_name):
    # 'cupsoda', dae_solver' and 'scipydoe' are valid solver names
    if solver_name == 'cupsoda':
        integrator_options = {'vol': 4.0e-15, 'max_steps': 2**20}
        solver_options = dict()
        if 'timeout' in Simulator.supported_solvers['cupsoda']._integrator_options_allowed:
            solver_options.update({'timeout': 60})
    elif solver_name == 'scipyode':
        solver_options = {'integrator': 'lsoda'}
        integrator_options = {'mxstep': 2**20}
    else:
        solver_options = {'atol': 1e-12}
        integrator_options = {}
    sim_ = Simulator(model=model, param_values=extrinsic_noise_params, tspan=time_axis, solver=solver_name,
                     solver_options=solver_options, integrator_options=integrator_options)
    sim_.run()

    return sim_


sim_results = sim.run()
results = sim_results.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})


def pre_processing(sim_res):
    res_scaled = ScaleToMinMax(feature_range=(0, 1), columns=['tBID_obs'],
                               groupby='simulation', do_fit_transform=True).\
        transform(sim_res[['tBID_obs', 'time', 'simulation', 'TRAIL_conc']])
    t_at_half_max = Interpolate(independent_variable_name='tBID_obs',
                                dependent_variable_name='time', new_values=[0.5],
                                groupby='simulation').\
        transform(res_scaled[['tBID_obs', 'time', 'simulation', 'TRAIL_conc']])
    return t_at_half_max


pp = pre_processing(results)

pp['time'] = 180.0*np.round(pp['time'].values/180.0)

if __name__ == '__main__':
    bins10 = int((pp[pp.TRAIL_conc == '10ng/mL']['time'].max() - pp[pp.TRAIL_conc == '10ng/mL']['time'].min())/180.0)
    bins50 = int((pp[pp.TRAIL_conc == '50ng/mL']['time'].max() - pp[pp.TRAIL_conc == '50ng/mL']['time'].min())/180.0)

    from matplotlib import pyplot as plt
    cm = plt.get_cmap('tab10')
    plt.hist(pp[pp.TRAIL_conc == '10ng/mL']['time'], alpha=0.5, color=cm.colors[7], label='10 ng/mL TRAIL', bins=bins10)
    plt.hist(pp[pp.TRAIL_conc == '50ng/mL']['time'], alpha=0.5, color=cm.colors[1], label='50 ng/mL TRAIL', bins=bins50)
    plt.title('Time at half-max BID truncation')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if overwrite_synthetic_dataset:
        pp.to_csv('fluorescence_time_at_half_max_dataset.csv')
