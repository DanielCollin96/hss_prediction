import pickle
import numpy as np
import pandas as pd
import prediction_models
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import evaluation as eval
import os
from copy import copy, deepcopy
from sklearn.metrics import mean_squared_error



def get_score_after_feature_permutation(model, X, Y, X_cme, Y_cme, dates_pred_peak, dates_obs_peak, curr_feat, feature_transformer=None, output_scaler=None):
    """
    Return the RMSE and HSS peak RMSE of the model when a feature is permuted.

    Parameters
    ----------
    model : model object
        The trained model.
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    curr_feat : str
        The name of the feature to be permuted.
    feature_transformer : transformer object, optional
        Transformer for feature data.
    output_scaler : scaler object, optional
        Scaler for output data.

    Returns
    -------
    permuted_rmse : float
        The RMSE and HSS peak RMSE after permuting the feature.
    """
    X_permuted = X.copy()
    X_cme_permuted = X_cme.copy()


    # Permute the values of the specified column
    col_idx = list(X.columns).index(curr_feat)
    X_permuted.iloc[:, col_idx] = np.random.permutation(
        X_permuted[curr_feat].values
    )
    col_idx = list(X_cme.columns).index(curr_feat)
    X_cme_permuted.iloc[:, col_idx] = np.random.permutation(
        X_cme_permuted[curr_feat].values
    )

    # Apply feature transformer if available
    if feature_transformer is not None:
        X_permuted = feature_transformer.transform(X_permuted)
        X_cme_permuted = feature_transformer.transform(X_cme_permuted)

    # Predict using the model with the permuted feature
    permuted_prediction = model.predict(X_permuted)
    permuted_prediction_cme = model.predict(X_cme_permuted)


    # Inverse transform the output if output scaler is available
    if output_scaler is not None:
        permuted_prediction = output_scaler.inverse_transform(permuted_prediction.reshape((-1, 1))).reshape(-1)
        permuted_prediction_cme = output_scaler.inverse_transform(permuted_prediction_cme.reshape((-1, 1))).reshape(-1)


    permuted_prediction = pd.Series(permuted_prediction, index=X.index)
    permuted_prediction_cme = pd.Series(permuted_prediction_cme, index=X_cme.index)


    # Compute RMSE
    permuted_rmse = np.sqrt(mean_squared_error(permuted_prediction, Y))
    permuted_peak_rmse = np.sqrt(mean_squared_error(permuted_prediction_cme[dates_pred_peak], Y_cme[dates_obs_peak]))


    return permuted_rmse, permuted_peak_rmse


def compute_rmse_feature_permutation_importance(model, X, Y, X_cme, Y_cme, dates_pred_peak, dates_obs_peak, curr_feat, feature_transformer=None, output_scaler=None):
    """
    Compare the timeline and HSS peak RMSE when the current feature is permuted.

    Parameters
    ----------
    model : model object
        The trained model.
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    curr_feat : str
        The name of the feature to evaluate.
    feature_transformer : transformer object, optional
        Transformer for feature data.
    output_scaler : scaler object, optional
        Scaler for output data.

    Returns
    -------
    feature_rmse_impact : float
        The impact of the feature on the timeline and HSS peak RMSE.
    """
    # Transform features if feature transformer is available
    if feature_transformer is not None:
        X_transformed = feature_transformer.transform(X)
        X_cme_transformed = feature_transformer.transform(X_cme)
    else:
        X_transformed = copy(X)
        X_cme_transformed = copy(X_cme)


    # Predict baseline and transform back if output scaler is available
    baseline_prediction = model.predict(X_transformed)
    baseline_prediction_cme = model.predict(X_cme_transformed)
    if output_scaler is not None:
        baseline_prediction = output_scaler.inverse_transform(baseline_prediction.reshape((-1, 1))).reshape(-1)
        baseline_prediction_cme = output_scaler.inverse_transform(baseline_prediction_cme.reshape((-1, 1))).reshape(-1)
    baseline_prediction = pd.Series(baseline_prediction, index=X.index)
    baseline_prediction_cme = pd.Series(baseline_prediction_cme, index=X_cme.index)
    baseline_rmse = np.sqrt(mean_squared_error(baseline_prediction, Y))
    baseline_peak_rmse = np.sqrt(mean_squared_error(baseline_prediction_cme[dates_pred_peak], Y_cme[dates_obs_peak]))

    # Compute RMSE after feature permutation
    permuted_rmse, permuted_peak_rmse = get_score_after_feature_permutation(model, X, Y, X_cme, Y_cme, dates_pred_peak, dates_obs_peak, curr_feat, feature_transformer, output_scaler)

    # Feature importance is the difference between the two scores
    feature_rmse_impact = permuted_rmse - baseline_rmse
    feature_peak_rmse_impact = permuted_peak_rmse - baseline_peak_rmse

    return feature_rmse_impact, feature_peak_rmse_impact


def feature_importance(model, X_fs, X_unscaled, Y, baseline_prediction_cme, cme_list, enhancement_list, feature_transformer=None,
                       output_scaler=None,
                       n_repeats=10, options=None):
    """
    Calculate feature importance score for each feature.

    Parameters
    ----------
    model : model object
        The trained model.
    X_fs : DataFrame
        The feature data after feature selection.
    X_unscaled : DataFrame
        The unscaled feature data.
    Y : DataFrame
        The target data.
    cme_list : DataFrame
        List of CME events.
    enhancement_list : DataFrame
        List of enhancements.
    feature_transformer : transformer object, optional
        Transformer for feature data.
    output_scaler : scaler object, optional
        Scaler for output data.
    n_repeats : int, optional
        Number of repeats for computing feature importance.
    options : dict, optional
        Options of prediction model.

    Returns
    -------
    importances : DataFrame
        DataFrame containing importance scores for each feature.
    """
    # Extract the features selected by the feature selection step
    feature_names_fs = X_fs.loc['no_cme', 'train'].columns
    X_test_fs_unscaled = X_unscaled.loc['no_cme', 'test'].loc[:, feature_names_fs]
    X_test_cme_fs_unscaled = X_unscaled.loc['with_cme', 'test'].loc[:, feature_names_fs]

    # Calculate peak positions.
    pred_peaks, obs_peaks = eval.associate_enhancements(baseline_prediction_cme, enhancement_list, cme_list)
    hits = pred_peaks.loc[:, 'hit'].to_numpy()
    associated_peaks = pred_peaks.loc[hits, 'associated'].to_numpy()
    pred_peak_dates = pred_peaks.loc[hits, :].index
    obs_peak_dates = obs_peaks.loc[associated_peaks, :].index

    # Calculate peak impacts and RMSE impacts for each feature
    peak_permutation_impacts = []
    rmse_permutation_impacts = []
    for curr_feat in X_fs.loc['no_cme', 'test'].columns:

        # Compute permutation feature importance n_round times for different random permutations
        list_feature_rmse_permutation_impact = []
        list_feature_peak_rmse_permutation_impact = []
        for n_round in range(n_repeats):
            feature_rmse_permutation_impact, feature_peak_rmse_permutation_impact = compute_rmse_feature_permutation_importance(model, X_fs.loc['no_cme', 'test'],
                                                                  Y.loc['no_cme', 'test'], X_fs.loc['with_cme', 'test'], Y.loc['with_cme', 'test'], pred_peak_dates, obs_peak_dates,
                                                                  curr_feat, feature_transformer, output_scaler)
            list_feature_rmse_permutation_impact.append(feature_rmse_permutation_impact)
            list_feature_peak_rmse_permutation_impact.append(feature_peak_rmse_permutation_impact)

        rmse_permutation_impacts.append(list_feature_rmse_permutation_impact)
        peak_permutation_impacts.append(list_feature_peak_rmse_permutation_impact)


    # Convert to arrays and compute mean impacts of random permuations
    rmse_permutation_impacts = np.array(rmse_permutation_impacts)
    rmse_permutation_impacts_mean = np.mean(rmse_permutation_impacts, axis=1)
    peak_permutation_impacts = np.array(peak_permutation_impacts)
    peak_permutation_impacts_mean = np.mean(peak_permutation_impacts, axis=1)


    # Create DataFrame of importances
    importances = pd.DataFrame(np.stack([rmse_permutation_impacts_mean, peak_permutation_impacts_mean], axis=1),
                               index=X_fs.loc['no_cme', 'test'].columns, columns=['rmse_permutation', 'peak_rmse_permutation'])

    return importances


def merge_feature_importance_and_coefficients(results, results_cv):
    """
    Merge feature importance and coefficients from individual cross-validation folds into a single ModelResults object.

    Parameters
    ----------
    results : list
        List of ModelResults objects from individual cross-validation folds.
    results_cv : ModelResults
        Merged ModelResults object.

    Returns
    -------
    results_cv : ModelResults
        Merged ModelResults object with feature importance and coefficients.
    """
    n_folds = len(results)

    # Initialize lists to collect feature names
    features_fs = []
    features_pred_model = []
    features_imp = []
    col_names = []

    # Collect feature names from individual cross-validation folds
    for i in range(n_folds):
        features_fs.extend(results[i].coef_fs.index)
        features_pred_model.extend(results[i].coef_pred_model.index)
        features_imp.extend(results[i].feature_importance_fold.index)
        col_names.append('fold_{}'.format(str(i + 1)))

    col_names.append('cv_average')
    col_names.append('importance')

    # Remove duplicate feature names
    features_fs = list(set(features_fs))
    features_pred_model = list(set(features_pred_model))
    features_imp = list(set(features_imp))

    # Create DataFrames to store coefficients and feature importance
    coef_fs = pd.DataFrame(np.zeros((len(features_fs), n_folds + 2), dtype=float), index=features_fs, columns=col_names)
    coef_pred_model = pd.DataFrame(np.zeros((len(features_pred_model), n_folds + 2), dtype=float), index=features_pred_model, columns=col_names)
    imp_rmse_permutation = pd.DataFrame(np.zeros((len(features_imp), n_folds + 1), dtype=float), index=features_imp,
                            columns=col_names[:-1])
    imp_peak_rmse_permutation = pd.DataFrame(np.zeros((len(features_imp), n_folds + 1), dtype=float), index=features_imp, columns=col_names[:-1])


    # Populate DataFrames with coefficients and feature importance from individual folds
    for i in range(n_folds):
        coef_fs.loc[results[i].coef_fs.index, 'fold_{}'.format(i + 1)] = results[i].coef_fs
        coef_pred_model.loc[results[i].coef_pred_model.index, 'fold_{}'.format(i + 1)] = results[i].coef_pred_model
        imp_rmse_permutation.loc[results[i].feature_importance_fold.index, 'fold_{}'.format(i + 1)] = results[i].feature_importance_fold.loc[:, 'rmse_permutation']
        imp_peak_rmse_permutation.loc[results[i].feature_importance_fold.index, 'fold_{}'.format(i + 1)] = results[i].feature_importance_fold.loc[:, 'peak_rmse_permutation']


    # Compute CV averages
    coef_fs.loc[:, 'cv_average'] = coef_fs.iloc[:, :n_folds].mean(axis=1)
    coef_pred_model.loc[:, 'cv_average'] = coef_pred_model.iloc[:, :n_folds].mean(axis=1)
    imp_rmse_permutation.loc[:, 'cv_average'] = imp_rmse_permutation.iloc[:, :n_folds].mean(axis=1)
    imp_peak_rmse_permutation.loc[:, 'cv_average'] = imp_peak_rmse_permutation.iloc[:, :n_folds].mean(axis=1)

    coef_fs.loc[:, 'importance'] = np.abs(coef_fs.loc[:, 'cv_average'])
    coef_pred_model.loc[:, 'importance'] = np.abs(coef_pred_model.loc[:, 'cv_average'])

    # Sort DataFrames by importance
    coef_fs = coef_fs.sort_values('importance', ascending=False)
    coef_pred_model = coef_pred_model.sort_values('importance', ascending=False)
    imp_rmse_permutation = imp_rmse_permutation.sort_values('cv_average', ascending=False)
    imp_peak_rmse_permutation = imp_peak_rmse_permutation.sort_values('cv_average', ascending=False)


    # Update results_cv with merged feature importance and coefficients
    results_cv.coef_fs = coef_fs
    results_cv.coef_pred_model = coef_pred_model
    results_cv.feature_importance_rmse_permutation = imp_rmse_permutation
    results_cv.feature_importance_peak_rmse_permutation = imp_peak_rmse_permutation


    return results_cv
