from datetime import datetime

import numpy as np
import pandas as pd

import preprocessing_dataset
import cross_validation
import os

path = os.path.dirname(os.getcwd())
np.random.seed(0)

############################################
# Configure model here.
############################################

# Set grid size for model. That determines the data set and the hyperparameters which are loaded.
# Available grid sizes are: 1x1, 2x1, 1x3, 4x3, 6x3, 10x3, 6x6, 10x6, 10x10, 14x6, 14x10.
grid = '4x3'

# Set evaluation mode. Set 'cv' for 5-fold cross-validation, '2018' for evaluation on the year 2018, or 'sc25' for an
# evaluation on solar cycle 2025, i.e., from 2020 onwards.
eval_mode = 'cv'

# Set target metric. Set 'rmse' to load the hyperparameters optimizing the RMSE of the model or 'peak_rmse' to load the
# ones optimizing the RMSE at the peaks of HSSs.
target_metric = 'rmse'

# Set prediction model algorithm.
# prediction_model options: polynomial, linear, ch_area_baseline, sw_persistence_baseline, average_baseline
options = {'prediction_model': 'polynomial'}

# Set Lasso regularization parameters. Either specify an array-like [gamma_fs, gamma_pr] for feature selection and
# polynomial regression, or set to 'optimal', which then uses hyperparameters as optimized for the paper and
# specified in the file hyperparameters.ods.
lasso_regularization_parameters = 'optimal' #[1e-3, 1e-5]

############################################
# End of configuration.
############################################


# Load hyperparameters for particular grid and target metric.
if type(lasso_regularization_parameters) == str and lasso_regularization_parameters == 'optimal':
    try:
        hyperparameters = pd.read_excel(path + '/data/hyperparameters.ods', sheet_name=target_metric, index_col=0)
        hyperparameters = hyperparameters.loc[grid, :].to_numpy()

    except:
        raise ValueError('Hyperparameters for the given grid size are not optimized. Please specify them directly.')
else:
    hyperparameters = lasso_regularization_parameters


# Load lists of CME disturbances and HSSs.
cme_list, hss_list, enhancement_list = preprocessing_dataset.read_cme_hss_list(path)


# Load data set.
X, Y = preprocessing_dataset.read_ml_dataset(path, grid)

# Compute train-test split.
if eval_mode == 'cv':
    cv_range = Y.index < datetime(2020,1,1)
    Y = Y.iloc[cv_range]
    X = X.iloc[cv_range,:]
    data_split = cross_validation.cross_validation_split(Y)
elif eval_mode == '2018':
    cv_range = Y.index < datetime(2020, 1, 1)
    Y = Y.iloc[cv_range]
    X = X.iloc[cv_range, :]
    data_split = cross_validation.train_test_split_single_interval(Y,test_interval=eval_mode,discard_buffer_data=True)
elif eval_mode == 'sc25':
    train_bin = Y.index <= datetime(2015, 12, 31, 23)
    calibration_bin = (Y.index >= datetime(2020,1,1)) & (Y.index <= datetime(2020,12,31,23))
    train_bin = train_bin | calibration_bin

    test_bin = (Y.index >= datetime(2021,1,1)) & (Y.index <= datetime(2023,12,31,23))

    train_idx = Y.index[train_bin]
    test_idx = Y.index[test_bin]
    calibration_idx = Y.index[calibration_bin]
    data_split = [[train_idx],[test_idx],calibration_idx]

# Delete CMEs from train-test split and save both versions without and with CMEs.
data_split_without_cmes = cross_validation.delete_cmes_from_data_split(data_split, cme_list)

'''All data structures containing the input data, target output or information about them are organized as pandas 
DataFrames, where the index differentiates between the data with CMEs removed ('no_cme') and the original data with 
CMEs ('with_cme'), and the columns differentiate between training data ('train') and test data ('test'). Each entry 
of the DataFrame then contains the corresponding data, i.e., time series (e.g. target Y) or DataFrame (e.g. input X), 
or list of dates (e.g. cross-validation split).'''

data_split = [[data_split_without_cmes[0], data_split_without_cmes[1]],
              [data_split[0], data_split[1]]]
data_split = pd.DataFrame(data_split, dtype=object, index=['no_cme', 'with_cme'], columns=['train', 'test'])

print('Running model with {} grid.'.format(grid))

# Train and evaluate models with cross-validation scheme. If the evaluation mode is set to '2018', the cross-validation
# is simplified to just one train-test data split.
cross_validation.cross_validation(hyperparameters, X, Y, data_split, cme_list, enhancement_list, grid, options)