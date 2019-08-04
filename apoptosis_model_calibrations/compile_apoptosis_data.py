# MWIrvin -- Lopez Lab -- 2019-07-14

import os
import pandas as pd
import numpy as np
from opt2q.data import DataSet
from apoptosis_model_calibrations.apoptosis_model import model


# apoptosis model free parameters
kc0, kc2, kf3, kc3, kf4, kr7, kc8 = (1.0e-05, 1.0e-02, 3.0e-08, 1.0e-02, 1.0e-06, 1.0e-02, 1.0e-6)

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'apoptosis_data.xlsx')
data_file = pd.read_excel(file_path, sheet_name='Data Sets')
exp_con_file = pd.read_excel(file_path, sheet_name='Experimental Conditions')


class CompileData(object):
    measurement_scales = {'WesternBlotPTM': 'ordinal',
                          'WesternBlot': 'ordinal',
                          'FractionalKilling': 'nominal',
                          'Fluorescence': 'semi-quantitative'}
    annotating_columns = {'Experiment Name', 'Condition', 'TRAIL_conc ng/mL', 'Figure', 'Measurement Name',
                          'Measurement Model', 'time_min'}
    model_observables = {
        'tBid': ['tBID_obs'],
        'Full Length Caspase 3': ['C3_inactive_obs'],
        'IP Caspase-8 FADD': ['C8_DISC_recruitment_obs'],
        'DVED FRET Indicator': ['cPARP_obs'],  # Accumulation of DVED-ase dependent indicator is a proxy for cPARP
        'Fraction Killed': ['cPARP_obs'],  # cPARP is a common marker of apoptotic cell death
        'EITD Fret Indicator': ['tBID_obs'],  # Accumulation of EITD-ase dependent indicator is a proxy for tBid
        'EITD Fret Indicator Hellwig_Rehm': ['C8_active_obs'],  # EITD-ase indicator cannot model tBid in BIDKD cells
        'IP TRAIL DR4': ['TRAIL_receptor_obs'],
        'Bid': ['BID_obs'],
        'Cleaved Caspase 3': ['C3_active_obs'],
        'cPARP': ['cPARP_obs'],
        'IP TRAIL FADD': ['DISC_obs'],
        'IP TRAIL Caspase-8': ['C8_DISC_recruitment_obs'],
        'Cleaved Caspase 8': ['C8_active_obs'],
        'Full Length Caspase 8': ['C8_inactive_obs'],
        'PARP': ['PARP_obs'],
        'DEVD Cleavage Rate AU/2hr': ['C3_active_obs'],  # DEVD cleavage rate (i.e. dDEVD/dt = k[Caspase-3])
        'DEVD Cleavage Rate AU/1hr': ['C3_active_obs']   # DEVD cleavage rate (i.e. dDEVD/dt = k[Caspase-3])
        }

    def __init__(self):
        self.all_data = data_file
        self._data_df = self.all_data.drop(columns='Notes')
        self._data = self._convert_time_min_to_sec(self._data_df)

        self.all_experimental_conditions = exp_con_file
        self._ec_df = self.all_experimental_conditions.drop(columns='Notes')
        self._data_df_ec = self._data[['Experiment Name', 'Condition', 'TRAIL_conc ng/mL', 'Figure',
                                       'Measurement Name', 'Measurement Model']]\
            .drop_duplicates().reset_index(drop=True)

        # Experimental conditions argument for the NoiseModel
        self._experimental_conditions = self._ec_df.merge(
            self._data_df_ec, on=['Experiment Name', 'Condition', 'TRAIL_conc ng/mL']
        )  # give data_df and experimental_conditions_df the same annotating columns.

        # Experimental conditions argument for the DataSet
        self._ec_df_wo_params = self._ec_df.drop(columns=['param', 'value', 'apply_noise'])
        self._data_df_ec_t = self._data[['Experiment Name', 'Condition', 'TRAIL_conc ng/mL', 'Figure',
                                         'Measurement Name', 'Measurement Model', 'time']] \
            .drop_duplicates().reset_index(drop=True)
        self._experimental_conditions_dataset = self._ec_df_wo_params.merge(
            self._data_df_ec_t, on=['Experiment Name', 'Condition', 'TRAIL_conc ng/mL']
        )

        # Compile Data Sets
        self._data_set_dict = self._compile_dataset_dict(self._data)

        # Compile model parameters
        self._model_parameter_means, self._model_parameter_covariances = self._compile_model_parameters()

    @staticmethod
    def _convert_time_min_to_sec(df):
        return df.assign(time=df.time_min*60).drop(columns='time_min')

    def _compile_dataset_dict(self, data_df):
        ds_dict = {}

        if 'time_min' in data_df.columns:
            data_df = self._convert_time_min_to_sec(data_df)
        if 'Notes' in data_df.columns:
            data_df = data_df.drop(columns='Notes')

        for name, group in data_df.groupby(['Figure', 'Measurement Name', 'Measurement Model']):
            data_set = self._compile_dataset_from_group(group)
            measured_values = {mv: self.model_observables[mv] for mv in data_set.measured_variables.keys()}
            observables = self._compile_observables_list_(measured_values)
            experimental_conditions = group[list({'time'} | self.annotating_columns - {'time_min'})].\
                drop_duplicates().reset_index(drop=True)
            ds_dict.update({name: {'dataset': data_set,
                                   'measured_values': measured_values,
                                   'observables': observables,
                                   'experimental_conditions': experimental_conditions}
                            })  # kwargs for the measurement model
        return ds_dict

    def _compile_dataset_from_group(self, data_group):
        """Create a DataSet object for data_group"""
        data = data_group[data_group.columns[~data_group.isnull().all()]]
        measurement_scale = self.measurement_scales[data['Measurement Model'].unique()[-1]]

        measured_variables = set(data.columns) - self.annotating_columns
        error_columns = set(data.columns[data.columns.str.contains('stdev')])

        if len(error_columns) > 0:
            measured_variables -= error_columns
            data_ = data[list(measured_variables | self.annotating_columns - {'time_min'})]
            measurement_variables_dict = {m_var: measurement_scale for m_var in
                                          set(data_.columns) - self.annotating_columns - {'time'}}
            ds = DataSet(data_, measured_variables=measurement_variables_dict)

            rename_cols = {old_name: f'{old_name.split(" stdev")[0]}__error' for old_name in error_columns}
            ds.measurement_error_df = data[list((error_columns | self.annotating_columns | {'time'}) - {'time_min'})]\
                .rename(columns=rename_cols).reset_index(drop=True)

        else:
            data_ = data
            measurement_variables_dict = {m_var: measurement_scale for m_var in
                                          set(data_.columns) - self.annotating_columns - {'time'}}
            ds = DataSet(data_, measured_variables=measurement_variables_dict)
        return ds

    @staticmethod
    def _compile_observables_list_(obs_dict):
        obs_set = set()
        for obs_list in obs_dict.values():  # obs_dict is a dict of lists.
            obs_set |= set(obs_list)
        return list(obs_set)

    def _check_new_df_columns(self, data_df):
        if set(data_df.columns) - set(self.all_data.columns) != set():
            raise ValueError("The dataframe can only have mentioned 'apoptosis_data.xlsx'")
        if {'Figure', 'Measurement Name', 'Measurement Model'} - set(data_df.columns):
            raise ValueError("The data must have 'Figure', 'Measurement Name', 'Measurement Model' columns")

    def _trim_experimental_conditions_to_match_new_data(self, data_df):
        # merge all_experimental_conditions with new data_df ec.
        _data_df_ec = data_df[['Experiment Name', 'Condition', 'TRAIL_conc ng/mL',
                               'Figure', 'Measurement Name', 'Measurement Model']].drop_duplicates()
        return self._ec_df.merge(_data_df_ec, on=['Experiment Name', 'Condition', 'TRAIL_conc ng/mL'])

    def _compile_model_parameters(self):
        # population averaged measurements (e.g. Fractional Killing and Western Blot) will have extrinsic noise applied,
        # while single cell measurements (e.g. Fluorescence) will not. Note: data are for a single representative cell.
        experiments = self.experimental_conditions[self.annotating_columns - {'time_min'}]
        experiments_parameters = set(self.experimental_conditions['param'].unique()) - \
            {'kc0', 'kc2', 'kf3', 'kc3', 'kf4', 'kr7', 'kc8'}  # experimental conditions parameters
        default_parameters = pd.DataFrame([[p.name, p.value, False]
                                           for p in model.parameters if p.name in experiments_parameters],
                                          columns=['param', 'value', 'apply_noise'])

        bulk_measurement_parameters_ = pd.DataFrame([['kc0', kc0, True],
                                                     ['kc2', kc2, True],  # co-vary with kc3
                                                     ['kf3', kf3, False],
                                                     ['kc3', kc3, True],
                                                     ['kf4', kf4, False],
                                                     ['kr7', kr7, False],
                                                     ['kc8', kc8, False], ],
                                                    columns=['param', 'value', 'apply_noise'])
        bulk_measurement_parameters_ = pd.concat([bulk_measurement_parameters_, default_parameters],
                                                 axis=0, sort=False)

        bulk_measurement_experiments_ = experiments[experiments['Measurement Model'].str.match(r'^WesternBlot') |
                                                    experiments['Measurement Model'].str.match(r'^FractionalKilling$')]
        bulk_measurement_parameters = add_experiments_columns(bulk_measurement_parameters_,
                                                              bulk_measurement_experiments_)
        individual_measurement_parameters_ = pd.DataFrame([['kc0', kc0, False],
                                                           ['kc2', kc2, False],  # co-vary with kc3
                                                           ['kf3', kf3, False],
                                                           ['kc3', kc3, False],
                                                           ['kf4', kf4, False],
                                                           ['kr7', kr7, False],
                                                           ['kc8', kc8, False]],
                                                          columns=['param', 'value', 'apply_noise'])
        individual_measurement_parameters_ = pd.concat([individual_measurement_parameters_, default_parameters],
                                                       axis=0, sort=False)

        individual_measurement_experiments_ = experiments[experiments['Measurement Model'].str.match(r'^Fluorescence$')]
        individual_measurement_parameters = add_experiments_columns(individual_measurement_parameters_,
                                                                    individual_measurement_experiments_)

        background_parameters = pd.concat([bulk_measurement_parameters, individual_measurement_parameters],
                                          axis=0, sort=False).drop_duplicates().reset_index(drop=True)

        kc2_cv, kc3_cv, kc2_kc3_cor = (0.2, 0.2, 0.25)
        kc2_var, kc3_var, kc2_kc3_cov = (
        (kc2 * kc2_cv) ** 2, (kc3 * kc3_cv) ** 2, kc2 * kc2_cv * kc3 * kc3_cv * kc2_kc3_cor)
        param_variances = pd.DataFrame([['kc2', 'kc2', kc2_var],
                                        ['kc3', 'kc3', kc3_var],
                                        ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance between 'kc2' and kc3'
                                       columns=['param_i', 'param_j', 'value'])
        param_variances = add_experiments_columns(param_variances,
                                                  bulk_measurement_experiments_[list(self.annotating_columns
                                                                                     - {'time_min'})].
                                                  drop_duplicates().reset_index(drop=True))
        return background_parameters, param_variances

    @property
    def experimental_conditions(self):
        return self._experimental_conditions

    @property
    def dataset_experimental_conditions(self):
        return self._experimental_conditions_dataset

    @property
    def data_set_dict(self):
        return self._data_set_dict

    @property
    def default_model_parameter_means(self):
        """
        These are preset values of the parameters and do not contain changes defined by the experimental conditions
        """
        return self._model_parameter_means

    @property
    def default_model_parameter_covariances(self):
        """
        These are preset values of the parameters and do not contain changes defined by the experimental conditions
        """
        return self._model_parameter_covariances

    @property
    def data(self):
        return self._data

    def run(self, data_df):
        """
        Builds the DataSet and related NoiseModel and MeasurementModel kwarg for calibrating to data in data_df

        :param data_df: pd.DataFrame
            Contains the data that will be used in the calibration.
            This dataframe can only contain columns present in 'apoptosis_data.xlsx', and must contain
            'Figure', 'Measurement Name', 'Measurement Model' columns.
        """
        self._check_new_df_columns(data_df)
        self._experimental_conditions = self._trim_experimental_conditions_to_match_new_data(data_df)
        self._data_set_dict = self._compile_dataset_dict(data_df)
        self._model_parameter_means, self._model_parameter_covariances = self._compile_model_parameters()
        self._data = data_df


def add_experiments_columns(parameters, experiments):
    """repeats parameters for all experiments. Commonly needed function in pre-processing in calibrations"""
    len_ec = len(experiments)
    len_p = len(parameters)
    _parameters = parameters.reset_index(drop=True)
    _experiments = experiments.reset_index(drop=True)
    _params = _parameters.iloc[np.repeat(_parameters.index, len_ec)].reset_index(drop=True)
    _exp = experiments.iloc[np.tile(_experiments.index, len_p)].reset_index(drop=True)
    return pd.concat([_exp, _params], axis=1, sort=False)


def convert_cv_params_to_covariance_values(*arg_list, **kwargs):
    """convert new variation-coefficient and correlation parameters to variance and covariance"""
    _kc2, _kc3, _kc2_cv, _kc3_cv, _kc2_kc3_cor = arg_list
    _kc2, _kc3 = 10 ** _kc2, 10 ** _kc3
    kc2_var, kc3_var, kc2_kc3_cov = ((_kc2 * _kc2_cv) ** 2, (_kc3 * _kc3_cv) ** 2,
                                     _kc2 * _kc2_cv * _kc3 * _kc3_cv * _kc2_kc3_cor)

    if 'experiment' not in kwargs:
        return pd.DataFrame([['kc2', 'kc2', kc2_var],
                             ['kc3', 'kc3', kc3_var],
                             ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance
                            columns=['param_i', 'param_j', 'value'])
    else:
        return add_experiments_columns(pd.DataFrame([['kc2', 'kc2', kc2_var],
                                                     ['kc3', 'kc3', kc3_var],
                                                     ['kc2', 'kc3', kc2_kc3_cov]],  # Covariance
                                                    columns=['param_i', 'param_j', 'value']),
                                       kwargs['experiment'])
