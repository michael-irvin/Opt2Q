# MW Irvin -- Lopez Lab -- 2019-07-31

import numpy as np
import pandas as pd
from opt2q.noise import NoiseModel
from opt2q.simulator import Simulator
from opt2q.measurement import *
from opt2q.measurement.base import CumulativeComputation
from apoptosis_model_calibrations.apoptosis_model import model
from apoptosis_model_calibrations.compile_measurement_models import compile_cparp_dependent_fractional_killing_model
from apoptosis_model_calibrations.compile_apoptosis_data import convert_cv_params_to_covariance_values
from opt2q.calibrator import objective_function
from multiprocessing import current_process
from pydream.parameters import SampledParam
from scipy.stats import norm, laplace, uniform


param_priors = [SampledParam(norm, loc=-5, scale=1.0),           # x0  float  kc0 -- 95% bounded in (-7,  -3)
                SampledParam(norm, loc=-2, scale=1.5),           # x1  float  kc2 -- 95% bounded in (-5,   1)
                SampledParam(norm, loc=-8.5, scale=1.25),        # x2  float  kf3 -- 95% bounded in (-11, -6)
                SampledParam(norm, loc=-2, scale=1.5),           # x3  float  kc3 -- 95% bounded in (-5,   1)
                SampledParam(norm, loc=-7, scale=1.5),           # x4  float  kf4 -- 95% bounded in (-10, -4)
                SampledParam(norm, loc=-2, scale=3.0),           # x5  float  kr7 -- 95% bounded in (-8,   4)
                SampledParam(norm, loc=-6, scale=1.5),           # x6  float  kc8 -- 95% bounded in (-9,  -3)
                SampledParam(uniform, loc=[0], scale=[0.5]),     # x7  float  kc2_cv --  bounded in [0,  0.5]
                SampledParam(uniform, loc=[0], scale=[0.5]),     # x8  float  kc3_cv --  bounded in [0,  0.5]
                SampledParam(uniform, loc=[-1.0], scale=[2.0]),  # x9  float  kc2_kc3 -- bounded in [-1.0,  1.0]
                SampledParam(uniform, loc=[-4.0], scale=[2.0]),  # x10 float  kc2_zVAD -- bounded in [-4.0, -2.0]
                SampledParam(uniform, loc=[-4.0], scale=[2.0]),  # x11 float  kc4_zVAD -- bounded in [-4.0, -2.0]
                SampledParam(uniform, loc=[-4.0], scale=[2.0]),  # x12 float  kc3_cas3_inh -- bounded in [-4.0,-2.0]
                SampledParam(uniform, loc=[-4.0], scale=[2.0]),  # x13 float  kc7_cas3_inh -- bounded in [-4.0,-2.0]
                SampledParam(uniform, loc=[0.0], scale=[3.0]),   # x14 float  kc0_siCHIP -- bounded in [0.0,  3.0]
                SampledParam(uniform, loc=[0.0], scale=[0.5]),   # x15 float  Bid_0_BidKD -- bounded in [-1.0,  1.0]
                SampledParam(laplace, loc=0.0, scale=10.0),      # x16 float  LR coef
                SampledParam(laplace, loc=0.0, scale=10.0),      # x17 float  LR coef
                SampledParam(laplace, loc=0.0, scale=10.0),      # x18 float  LR coef
                SampledParam(laplace, loc=0.0, scale=10.0),      # x19 float  LR coef
                SampledParam(laplace, loc=0.0, scale=1.0)]       # x20 float  LR intercept


def generate_likelihood_fn(compiled_data, n_sims, n_timepoints):
    """
    Generate a likelihood fn of the apoptosis model given the data in the compiled dataset.
    """
    cd = compiled_data

    # --------- Noise Model ----------
    # Noise Model and Parameters
    # n_sims = 2
    NoiseModel.default_sample_size = n_sims
    noise = NoiseModel(param_mean=cd.default_model_parameter_means,
                       param_covariance=cd.default_model_parameter_covariances)
    noise.update_values(param_mean=cd.experimental_conditions)
    parameters = noise.run()

    # ------- Dynamical Model -------
    # linear range: 0 to max time in dataset
    t_span = np.linspace(0, cd.dataset_experimental_conditions.time.max(), n_timepoints)
    sim = Simulator(model=model, param_values=parameters, solver='cupsoda',
                    integrator_options={'n_blocks': 64, 'memory_usage': 'global', 'vol': 4e-15})

    results = sim.run(t_span)

    # ------- Measurement Models ---------
    measurement_model_classes = {'Fluorescence': Fluorescence,
                                 'FractionalKilling': FractionalKilling,
                                 'WesternBlot': WesternBlot,
                                 'WesternBlotPTM': WesternBlotPTM}

    print("Calibrating to data from the following experiments: ")
    measurement_models_list = []
    for k, v in cd.data_set_dict.items():
        experiment = (cd.data['Figure'] == k[0]) & \
                     (cd.data['Measurement Name'] == k[1]) & \
                     (cd.data['Measurement Model'] == k[2])  # relevant rows of the cd.data

        measurement_model = measurement_model_classes[k[2]](simulation_result=results, **v)
        if k[2] == 'FractionalKilling':
            measurement_model = compile_cparp_dependent_fractional_killing_model(measurement_model)
            measurement_model.interpolate_first = False

        if 'Western' in k[2]:
            measurement_model.process.set_params(**{'classifier__do_fit_transform': True,
                                                    'sample_average__sample_size': n_sims})

        if 'Hellwig_Rehm_2008' in k[0] and 'Caspase 8 FRET Emission' in k[1]:
            # Fluorescent EITD-ase data in Hellwig et al. 2008 does not model Bid concentration in BidKD cells.
            # Model it as the integral w.r.t. time of active Caspase-8 concentration.
            measurement_model.process.add_step(('auc', CumulativeComputation(
                groupby='simulation', columns=['C8_active_obs'], operation='sum', )), index=0)

        print(k[0], k[1], k[2])
        getattr(measurement_model, 'setup', measurement_model.run)()
        measurement_models_list.append(measurement_model)

    # -------- Likelihood Function --------
    exp_cons = cd.experimental_conditions.copy()

    @objective_function(noise_model=noise, simulator=sim, measurements=measurement_models_list,
                        experiments=exp_cons, evals=0)
    def likelihood_fun(x):
        """
        Likelihood Function

        x: list of floats; length
            [(-7, -3),     # float  kc0            0: kinetic parameter uniform or normal prior
             (-5,   1),    # float  kc2            1: kinetic parameter uniform or normal prior
             (-11, -6),    # float  kf3            2: kinetic parameter uniform or normal prior
             (-5,   1),    # float  kc3            3: kinetic parameter uniform or normal prior
             (-10, -4),    # float  kf4            4: kinetic parameter uniform or normal prior
             (-8,   4),    # float  kr7            5: kinetic parameter uniform or normal prior
             (-9,  -3),    # float  kc8            6: kinetic parameter uniform or normal prior
             (0,  0.5),    # float  kc2_cv         7: kc2 coefficient of variation uniform prior
             (0,  0.5),    # float  kc3_cv         8: kc3 coefficient of variation uniform prior
             (-1,   1),    # float  kc2_kc3,       9: kc2, kc3 covariance term uniform prior
             (-4,  -2),    # float  kc2_zVAD      10: zVAD effect on kc2 uniform prior
             (-4,  -2),    # float  kc4_zVAD      11: zVAD effect on kc4 uniform prior
             (-4,  -2),    # float  kc3_zVAD      12: Caspase 3 Inh as decreased kc3 uniform prior
             (-4,  -2),    # float  kc3_zVAD      13: Caspase 3 Inh as decreased kc3 uniform prior
             (0,    3),    # float  kc0_siCHIP    14: siCHIP effect on kc0 uniform prior
             (0,  0.5),    # float  Bid_0_BidKD   15: BidKD effect on Bid_0 uniform prior
             (-10, 10),    # float  coef_         16: Cell Viability measurement parameter
             (-10, 10),    # float  coef_         17: Cell Viability measurement parameter
             (-10, 10),    # float  coef_         18: Cell Viability measurement parameter
             (-10, 10),    # float  coef_         19: Cell Viability measurement parameter
             (-10, 10)]    # float  intercept_    20: Cell Viability measurement parameter

        """
        # --- Background Parameters ---
        param_names = ('kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc8')
        param_means = pd.DataFrame({'param': param_names, 'value': [10 ** p for p in x[:7]]})
        kc2, kc3 = 10 ** x[1], 10 ** x[3]
        kc2_cv, kc3_cv, kc2_kc3_cor = x[7:10]
        kc2_var, kc3_var, kc2_kc3_cov = (
            (kc2 * kc2_cv) ** 2, (kc3 * kc3_cv) ** 2, kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor)
        param_variances = pd.DataFrame([['kc2', 'kc2', kc2_var],
                                        ['kc3', 'kc3', kc3_var],
                                        ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance
                                       columns=['param_i', 'param_j', 'value'])

        if likelihood_fun.noise_model.param_covariance.shape[0] > 0:
            likelihood_fun.noise_model.update_values(param_mean=param_means,
                                                     param_covariance=param_variances)
        else:  # fluorescence data omits noise term
            likelihood_fun.noise_model.update_values(param_mean=cd.experimental_conditions)

        # --- Experimental Conditions ---
        ec = likelihood_fun.experiments
        ec_cols = list(set(ec.columns) - {'value', 'param', 'apply_noise'})
        experimental_condition_cv_params = pd.DataFrame()
        # zVAD inhibitor will nearly completely oblate caspase activity.
        # Model this by decreasing caspase activity parameters, kc2 and kc4, typically to 0.001 x the endogenous value.
        ec.loc[ec.Condition.str.contains('zVAD') & ec.param.str.match('^kc2$'), 'value'] = 10 ** (x[10] + x[2])
        ec.loc[ec.Condition.str.contains('zVAD') & ec.param.str.match('^kc4$'), 'value'] = 10 ** x[11]

        experiment_rows = ec[ec.Condition.str.contains('zVAD')][ec_cols].drop_duplicates()  # update kc2 coef. var.
        cv = convert_cv_params_to_covariance_values(x[10] + x[2], x[3], x[7], x[8], x[9], experiment=experiment_rows)
        experimental_condition_cv_params = pd.concat([experimental_condition_cv_params, cv], ignore_index=True,
                                                     sort=False)

        # Caspase 3 inhibition is modeled similarly as a decrease in caspase 3 activity parameters, kc3, kc7
        ec.loc[ec.Condition.str.contains('Caspase 3 Inh') & ec.param.str.match('^kc3$'), 'value'] = 10 ** (x[12] + x[3])
        ec.loc[ec.Condition.str.contains('Caspase 3 Inh') & ec.param.str.match('^kc7$'), 'value'] = 10 ** (x[13] + x[5])

        # siCHIP stabilizes FADD, thereby enhancing DISC formation.
        # Model this by increasing DISC formation parameter, kc0, typically to 10 x the endogenous value.
        ec.loc[ec.Condition.str.contains('siCHIP') & ec.param.str.match('^kc0$'), 'value'] = 10 ** (x[14] + x[0])

        # BidKD is accomplished via stable transfection of anti-bid shRNA, reducing expression by roughly 70%.
        # Model this by decreasing the initial Bid concentration parameter, Bid_0.
        ec.loc[ec.Condition.str.contains('BidKD') & ec.param.str.match('^Bid_0$'), 'value'] = 4.0e4 * x[15]

        # Bcl-2 overexpression blocks effector caspase activation and apoptosis.
        # Model this by setting effector caspase activation parameter, kc6 to 0.
        # * This is already an experimental conditions preset *
        if likelihood_fun.noise_model.param_covariance.shape[0] > 0:
            likelihood_fun.ec = experimental_condition_cv_params
            likelihood_fun.noise_model.update_values(param_mean=cd.experimental_conditions,
                                                     param_covariance=experimental_condition_cv_params)
        else:
            likelihood_fun.noise_model.update_values(param_mean=cd.experimental_conditions)  # no noise term

        simulation_parameters = likelihood_fun.noise_model.run()

        # --- Dynamics Simulation ---
        process_id = current_process().ident % 4
        likelihood_fun.simulator.sim.gpu = [process_id]
        likelihood_fun.simulator.param_values = simulation_parameters
        sim_results = likelihood_fun.simulator.run()

        # --- Measurement Model ---
        viability_coef = np.array([[x[16],  # :  (-100, 100),   float
                                    x[17],  # :  (-100, 100),   float
                                    x[18],  # :  (-100, 100),   float
                                    x[19]]])  # :  (-100, 100),   float
        viability_intercept = np.array([x[20]])  # :  (-10, 10)]     float

        measurement_model_params = {'classifier__coefficients__Fraction Killed__coef_': viability_coef,
                                    'classifier__coefficients__Fraction Killed__intercept_': viability_intercept}

        ll = 0
        for mm in likelihood_fun.measurements:
            mm.update_simulation_result(sim_results)
            if isinstance(mm, FractionalKilling):
                mm.process.set_params(**measurement_model_params)
            ll += mm.likelihood()

        likelihood_fun.evals += 1
        print(likelihood_fun.evals)
        print(x)
        print(ll)
        return ll

    return likelihood_fun
