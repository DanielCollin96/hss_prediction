import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from datetime import datetime, timedelta
import evaluation
from copy import copy, deepcopy
import prediction_models
import preprocessing_dataset
import feature_importance


def cross_validation_split(Y, n_splits=5):
    """
    Split time series data into training and test sets for cross-validation.

    Parameters
    ----------
    Y : pandas.Series
        Target variable time series.
    n_splits : int, optional
        Number of splits for cross-validation. Default is 5.

    Returns
    -------
    cv_train : list
        The list contains n_splits lists of dates of the training set of each of the CV folds.
    cv_test : list
        The list contains n_splits lists of dates of the test set of each of the CV folds.
    """

    dates = Y.index
    discard_len = 180
    # distribute the discarding half to training and half to test data
    discard_len = int(discard_len / 2)
    chunk_len = len(dates) // n_splits
    chunk_remainder = len(dates) % n_splits

    # split data into equally big chunks
    data_split = np.zeros((len(dates), n_splits), dtype=bool)
    for i in range(n_splits):
        data_split[i * chunk_len:(i + 1) * chunk_len, i] = True
    # add remainder to last chunk
    if chunk_remainder > 0:
        data_split[-chunk_remainder:, n_splits - 1] = True

    # Assign one chunk as test data and the rest as training data.
    # Discard 6 months of data between training and test data.
    # To do so, discard 3 months of both training and test data.

    cv_train = []
    cv_test = []
    for i in range(n_splits):
        # assign chunk i as test set and all other chunks as training set
        test_data = np.copy(data_split[:, i])
        train_data = np.any(np.concatenate([data_split[:, :i], data_split[:, i + 1:]], axis=1), axis=1)
        if i == 0:
            # discard data at end of test chunk
            test_data[chunk_len - discard_len * 24:chunk_len] = False
            # discard data at start of training data
            train_data[chunk_len: chunk_len + discard_len * 24] = False
        elif i == (n_splits - 1):
            # discard data at start of test chunk
            test_data[chunk_len * i:chunk_len * i + discard_len * 24] = False
            # discard data at end of training data
            train_data[chunk_len * i - discard_len * 24: chunk_len * i] = False
        else:
            # discard data at start and end of test chunk
            test_data[chunk_len * i: chunk_len * i + discard_len * 24] = False
            test_data[chunk_len * (i + 1) - discard_len * 24:chunk_len * (i + 1)] = False
            # discard data at start and end of training data
            train_data[chunk_len * i - discard_len * 24: chunk_len * i] = False
            train_data[chunk_len * (i + 1): chunk_len * (i + 1) + discard_len * 24] = False

        cv_train.append(dates[train_data])
        cv_test.append(dates[test_data])

    return cv_train, cv_test

def train_test_split_single_interval(Y, test_interval='2018', discard_buffer_data=True):
    """
    Splits the data into a test set, being the year 2018, and the rest of the data as a training set.

    Parameters
    ----------
    Y : pandas.Series
        Target variable time series.

    Returns
    -------
    cv_train : list
        The list contains one list of dates of the training set.
    cv_test : list
        The list contains one list of dates of the test set.
    """
    if test_interval == '2018':
        test_interval_dates = [datetime(2018, 1, 1, 0), datetime(2018, 12, 31, 23)]
        train_interval_dates_1 = [datetime(2010, 6, 1, 0), datetime(2017, 12, 31, 23)]
        train_interval_dates_2 = [datetime(2019, 1, 1, 0), datetime(2019, 12, 31, 23)]
        if discard_buffer_data:
            train_interval_dates_1[1] = train_interval_dates_1[1] - timedelta(days=180)
            train_interval_dates_2[0] = train_interval_dates_2[0] + timedelta(days=180)
    elif test_interval == 'sc25':
        test_interval_dates = [datetime(2020, 1, 1, 0), datetime(2024, 6, 30, 23)]
        train_interval_dates_1 = [datetime(2010, 6, 1, 0), datetime(2019, 12, 31, 23)]
        train_interval_dates_2 = None
        if discard_buffer_data:
            train_interval_dates_1[1] = train_interval_dates_1[1] - timedelta(days=180)
    else:
        raise ValueError('Invalid test interval option. Test interval options are 2018 and sc25.')

    #dates = Y.index
    dates_test = Y[test_interval_dates[0]:test_interval_dates[1]].index
    dates_train = Y[train_interval_dates_1[0]:train_interval_dates_1[1]].index
    if train_interval_dates_2 is not None:
        dates_train = np.concatenate([dates_train,Y[train_interval_dates_2[0]:train_interval_dates_2[1]].index])

    data_split = [[dates_train], [dates_test]]

    #binary_test = np.concatenate([np.zeros(len(Y[:datetime(2017, 12, 31, 23)]), dtype=bool),
    #                              np.ones(len(dates_test), dtype=bool),
    #                              np.zeros(len(Y[datetime(2019, 1, 1, 0):]), dtype=bool)])
    #binary_train = np.concatenate([np.ones(len(Y[:datetime(2017, 12, 31, 23) - timedelta(days=180)]), dtype=bool),
    #                               np.zeros(len(dates_test) + 360 * 24, dtype=bool),
    #                               np.ones(len(Y[datetime(2019, 1, 1, 0) + timedelta(days=180):]), dtype=bool)])
    #data_split = [[dates[binary_train]], [dates[binary_test]]]
    return data_split


def delete_cmes_from_data_split(data_split, cme_list):
    """
    Remove dates affected by CMEs from train and test data splits.

    Parameters
    ----------
    data_split : list
        A list containing two elements: train and test splits, each containing lists of dates.
    cme_list : pandas.DataFrame
        DataFrame containing information about CMEs, including start and end dates.

    Returns
    -------
    list
        A list containing two elements: updated train and test splits with CME-affected dates removed.
    """

    # Deep copy the data split to avoid modifying the original data
    train_split = deepcopy(data_split[0])
    test_split = deepcopy(data_split[1])

    # Determine the number of folds
    n_folds = len(train_split)

    # Iterate over each fold
    for i in range(n_folds):
        dates_train = train_split[i]
        dates_test = test_split[i]

        # Create boolean arrays to mark dates affected by CMEs
        binary_cme_train = np.zeros(len(dates_train), dtype=bool)
        binary_cme_test = np.zeros(len(dates_test), dtype=bool)

        # Iterate over each CME in the list
        for j in range(cme_list.shape[0]):
            # Mark dates within the CME start and end dates (with some buffer)
            binary_cme_train |= ((cme_list.loc[j, 'start'] <= dates_train) & (cme_list.loc[j, 'end'] >= dates_train))
            binary_cme_train |= (((cme_list.loc[j, 'start'] + timedelta(days=26)) <= dates_train) &
                                 ((cme_list.loc[j, 'end'] + timedelta(days=28)) >= dates_train))
            binary_cme_test |= ((cme_list.loc[j, 'start'] <= dates_test) & (cme_list.loc[j, 'end'] >= dates_test))
            binary_cme_test |= (((cme_list.loc[j, 'start'] + timedelta(days=26)) <= dates_test) &
                                ((cme_list.loc[j, 'end'] + timedelta(days=28)) >= dates_test))

        # Invert the boolean arrays to mark dates not affected by CMEs
        binary_cme_train = np.invert(binary_cme_train)
        binary_cme_test = np.invert(binary_cme_test)

        # Filter out dates affected by CMEs from train and test splits
        train_split[i] = dates_train[binary_cme_train]
        test_split[i] = dates_test[binary_cme_test]

    # Return the updated data split
    return [train_split, test_split]


def delete_cmes(ts,cme_list):
    """
    Remove dates affected by CMEs from a time series.

    Parameters
    ----------
    ts : pandas Series
        A time series indexed with dates.
    cme_list : pandas.DataFrame
        DataFrame containing information about CMEs, including start and end dates.

    Returns
    -------
    ts : pandas Series
        A time series with CME-affected dates removed.
    """
    # Create boolean arrays to mark dates affected by CMEs
    idx = ts.index
    binary_cme = np.zeros(len(idx), dtype=bool)

    # Iterate over each CME in the list
    for j in range(cme_list.shape[0]):
        # Mark dates within the CME start and end dates (with some buffer)
        binary_cme |= ((cme_list.loc[j, 'start'] <= idx) & (cme_list.loc[j, 'end'] >= idx))
        binary_cme |= (((cme_list.loc[j, 'start'] + timedelta(days=26)) <= idx) &
                             ((cme_list.loc[j, 'end'] + timedelta(days=28)) >= idx))

    # Invert the boolean arrays to mark dates not affected by CMEs
    binary_cme = np.invert(binary_cme)

    # Filter out dates affected by CMEs from train and test splits
    ts = ts[binary_cme]
    return ts


def assign_fold_data(i, data_split, X, Y):
    """
    Assign data to train-test splits for a given fold index.

    Parameters
    ----------
    i : int
        The index of the fold.
    data_split : DataFrame
        The DataFrame containing information about data splits.
    X : DataFrame
        The feature data.
    Y : array_like
        The target data.

    Returns
    -------
    X_fold : DataFrame
        The feature data for each split and each class.
    Y_fold : DataFrame
        The target data for each split and each class.
    """
    # Extract train-test splits for both classes (no_cme and with_cme)
    train_split = data_split.loc['no_cme', 'train']
    test_split = data_split.loc['no_cme', 'test']
    train_split_cme = data_split.loc['with_cme', 'train']
    test_split_cme = data_split.loc['with_cme', 'test']

    # Select data for the current fold for each split
    X_train = X.loc[train_split[i], :]
    X_test = X.loc[test_split[i], :]
    Y_train = Y[train_split[i]]
    Y_test = Y[test_split[i]]

    X_train_cme = X.loc[train_split_cme[i], :]
    X_test_cme = X.loc[test_split_cme[i], :]
    Y_train_cme = Y[train_split_cme[i]]
    Y_test_cme = Y[test_split_cme[i]]

    # Combine data into arrays and then into DataFrames for both classes
    X_fold = np.array([[X_train, X_test],
                       [X_train_cme, X_test_cme]], dtype=object)
    X_fold = pd.DataFrame(X_fold, index=['no_cme', 'with_cme'],
                          columns=['train', 'test'])

    Y_fold = np.array([[Y_train, Y_test],
                       [Y_train_cme, Y_test_cme]], dtype=object)
    Y_fold = pd.DataFrame(Y_fold, index=['no_cme', 'with_cme'],
                          columns=['train', 'test'])

    return X_fold, Y_fold


def scale_data(X, Y, options=None):
    """
    Scale input features and output data to the interval [0,1].

    Parameters
    ----------
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    options : dict, optional
        Options for scaling.

    Returns
    -------
    X_scaled : DataFrame
        The scaled feature data for each split and each class.
    Y_scaled : DataFrame
        The scaled target data for each split and each class.
    scaler_output : scaler
        The scaler used for output scaling.
    """
    # Extract relevant data
    dates_train = X.loc['no_cme', 'train'].index
    dates_test = X.loc['no_cme', 'test'].index
    dates_train_cme = X.loc['with_cme', 'train'].index
    dates_test_cme = X.loc['with_cme', 'test'].index
    feature_names = X.loc['no_cme', 'train'].columns

    # Select appropriate scaler based on options. The persistence model and the average model do not need to be scaled.
    if options is not None and (options['prediction_model'] == 'sw_persistence_baseline' or options['prediction_model'] == 'average_baseline'):
        scaler_input = prediction_models.IdentityTransformation()
        scaler_output = prediction_models.IdentityTransformation()
    else:
        scaler_input = MinMaxScaler()
        scaler_output = MinMaxScaler()

    # Scale the data
    X_train_scaled = pd.DataFrame(scaler_input.fit_transform(X.loc['no_cme', 'train'].values), index=dates_train,
                                  columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler_input.transform(X.loc['no_cme', 'test'].values), index=dates_test,
                                 columns=feature_names)

    X_train_cme_scaled = pd.DataFrame(scaler_input.transform(X.loc['with_cme', 'train'].values), index=dates_train_cme,
                                      columns=feature_names)
    X_test_cme_scaled = pd.DataFrame(scaler_input.transform(X.loc['with_cme', 'test'].values), index=dates_test_cme,
                                     columns=feature_names)

    Y_train_scaled = scaler_output.fit_transform(Y.loc['no_cme', 'train'].values.reshape((-1, 1))).reshape(-1)
    Y_test_scaled = scaler_output.transform(Y.loc['no_cme', 'test'].values.reshape((-1, 1))).reshape(-1)

    Y_train_cme_scaled = scaler_output.transform(Y.loc['with_cme', 'train'].values.reshape((-1, 1))).reshape(-1)
    Y_test_cme_scaled = scaler_output.transform(Y.loc['with_cme', 'test'].values.reshape((-1, 1))).reshape(-1)

    # Combine scaled data into arrays and then into DataFrames for both classes
    X_scaled = np.array([[X_train_scaled, X_test_scaled],
                         [X_train_cme_scaled, X_test_cme_scaled]], dtype=object)
    X_scaled = pd.DataFrame(X_scaled, index=['no_cme', 'with_cme'],
                            columns=['train', 'test'])

    Y_scaled = np.array([[Y_train_scaled, Y_test_scaled],
                         [Y_train_cme_scaled, Y_test_cme_scaled]], dtype=object)
    Y_scaled = pd.DataFrame(Y_scaled, index=['no_cme', 'with_cme'],
                            columns=['train', 'test'])

    return X_scaled, Y_scaled, scaler_output




def merge_data(results):
    """
    Merge predictions, observation, and HSS peak data of all cross-validation sets.

    Parameters
    ----------
    results : list
        List of ModelResults objects.

    Returns
    -------
    results_cv : ModelResults
        Merged ModelResults object.
    """
    n_folds = len(results)

    # Merge prediction data for all folds
    pred_data_all_folds = []
    for idx in results[0].pred_data.index:
        for col in results[0].pred_data.columns:
            merge_sets = []
            for cv_iter in range(n_folds):
                merge_sets.append(results[cv_iter].pred_data.loc[idx, col])
            pred_data_all_folds.append(pd.concat(merge_sets, axis=0))

    pred_data_all_folds = np.array(pred_data_all_folds, dtype=object).reshape((8, 2))
    pred_data_all_folds = pd.DataFrame(pred_data_all_folds, index=results[0].pred_data.index, columns=results[0].pred_data.columns)

    # Merge observation data for all folds
    obs_data_all_folds = []
    for idx in results[0].obs_data.index:
        for col in results[0].obs_data.columns:
            merge_sets = []
            for cv_iter in range(n_folds):
                merge_sets.append(results[cv_iter].obs_data.loc[idx, col])
            obs_data_all_folds.append(pd.concat(merge_sets, axis=0))

    obs_data_all_folds = np.array(obs_data_all_folds, dtype=object).reshape((2, 2))
    obs_data_all_folds = pd.DataFrame(obs_data_all_folds, index=results[0].obs_data.index, columns=results[0].obs_data.columns)

    # Merge peak data for all folds
    peak_data_all_folds = []
    for idx in results[0].peak_data.index:
        for col in results[0].peak_data.columns:
            merge_sets = []
            for cv_iter in range(n_folds):
                merge_sets.append(results[cv_iter].peak_data.loc[idx, col])
            peak_data_all_folds.append(pd.concat(merge_sets, axis=0))

    peak_data_all_folds = np.array(peak_data_all_folds, dtype=object).reshape((2, 2))
    peak_data_all_folds = pd.DataFrame(peak_data_all_folds, index=results[0].peak_data.index, columns=results[0].peak_data.columns)

    # Create ModelResults object for merged data
    results_cv = evaluation.ModelResults(pred_data_all_folds, obs_data_all_folds, peak_data_all_folds)

    # Store individual fold data in the merged ModelResults object
    pred_data_single_folds = []
    obs_data_single_folds = []
    peak_data_single_folds = []
    for i in range(n_folds):
        pred_data_single_folds.append(results[i].pred_data)
        obs_data_single_folds.append(results[i].obs_data)
        peak_data_single_folds.append(results[i].peak_data)

    results_cv.pred_data_fold = pred_data_single_folds
    results_cv.obs_data_fold = obs_data_single_folds
    results_cv.peak_data_fold = peak_data_single_folds

    return results_cv




def cross_validation(hyperparameters, X, Y, data_split, cme_list, enhancement_list, grid, options):
    """
    Perform cross-validation to evaluate the prediction model.

    Parameters
    ----------
    hyperparameters : list
        List of hyperparameters (Lasso regularization parameter of feature selection and polynomial regression).
    X : DataFrame
        Input features.
    Y : DataFrame
        Target variable.
    data_split : DataFrame
        Data split (dates) for cross-validation.
    cme_list : list
        List of coronal mass ejections.
    enhancement_list : list
        List of all solar wind speed enhancements.
    grid : string
        A string of the format m x n specifying the used coronal hole grid.
    options : dict
        Options for the prediction model.

    Returns
    -------
    None
    """

    results = []
    n_folds = len(data_split.loc['no_cme', 'train'])

    # If the prediction algorithm is a baseline model, extract baseline features
    if options['prediction_model'][-8:] == 'baseline':
        X = preprocessing_dataset.extract_baseline_features(X,grid)

    # Iterate through each fold
    for i in range(n_folds):

        # Assign data of current cross-validation fold
        X_fold, Y_fold = assign_fold_data(i, data_split, X, Y)

        # Scale input and output data to interval [0,1]
        X_fold_scaled, Y_fold_scaled, scaler_output = scale_data(X_fold, Y_fold, options)

        # Perform feature selection
        X_fs, coef_fs, feature_names_fs = prediction_models.feature_selection(X_fold_scaled, Y_fold_scaled, hyperparameters[0], options)
        if X_fs.loc['no_cme','train'].shape[1] < 1:
            raise ValueError('Zero features remaining after feature selection. Decrease regularization.')


        # Fit the prediction model
        model, feature_transformer, coef_pred_model = prediction_models.fit_prediction_model(X_fs, Y_fold_scaled, hyperparameters[1], options)

        # Predict and scale back
        pred = prediction_models.predict_sw_speed(model, deepcopy(X_fs), feature_transformer, scaler_output, np.min(Y_fold.loc['with_cme', 'train']))

        # Compute feature importance
        feat_imp = feature_importance.feature_importance(model, X_fs, X_fold, Y_fold, pred.loc['with_cme','test'], cme_list, enhancement_list, feature_transformer,
                                                         scaler_output, options=options)

        # Apply distribution transformation
        if len(data_split) == 3:
            calibration_idx = data_split[2]
        else:
            calibration_idx = None
        pred_trans = prediction_models.apply_distribution_transformation(pred, Y_fold, np.min(Y_fold.loc['with_cme', 'train']), options,calibration_idx)

        # Compute baseline predictions
        pred_27, pred_base = prediction_models.compute_baseline_predictions(X_fold, Y_fold_scaled, grid, options, scaler_output)

        # Concatenate all predictions
        pred_data = pd.concat([pred, pred_trans, pred_base, pred_27], axis=0)
        pred_data.index = ['no_cme', 'with_cme', 'trans', 'trans_with_cme', 'base', 'base_with_cme', '27', '27_with_cme']

        # Create ModelResults object for the current fold
        results_fold = evaluation.ModelResults(pred_data, Y_fold)
        results_fold.coef_fs = coef_fs
        results_fold.coef_pred_model = coef_pred_model
        results_fold.feature_importance_fold = feat_imp

        # Detect peaks and evaluate predictions
        results_fold.detect_peaks(enhancement_list, cme_list)
        results_fold.evaluate()

        print('Results of cross-validation fold {}:'.format(i + 1))
        results_fold.print()

        results.append(results_fold)

    # Merge predictions and evaluate
    results_cv = merge_data(results)
    results_cv.evaluate()

    print('Results of full cross-validation averaged over all {} folds:'.format(n_folds))
    results_cv.print()

    # Assemble model coefficients and feature importance over all cross-validation folds
    results_cv = feature_importance.merge_feature_importance_and_coefficients(results, results_cv)

    # Save results
    results_cv.save(grid, options)

    return 
