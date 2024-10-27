import numpy as np
import pandas as pd
import preprocessing_dataset
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from copy import copy, deepcopy
from scipy import stats, special
import pickle



class DistributionTransformation:
    """A class for computing the distribution transformation."""

    def __init__(self):
        """Initialize parameters."""
        self.lambda_x = None
        self.lambda_y = None
        self.mu_x = None
        self.mu_y = None
        self.sigma_x = None
        self.sigma_y = None

    def fit(self, initial_distribution, target_distribution):
        """
        Fit the transformation parameters to map initial_distribution to target_distribution.

        Parameters
        ----------
        initial_distribution : array_like
            The initial distribution.
        target_distribution : array_like
            The target distribution.
        """
        # Box Cox only works for positive values
        initial_distribution[initial_distribution <= 0] = 1e-8
        target_distribution[target_distribution <= 0] = 1e-8

        # Fit the lambda parameter of the Box-Cox transformation for initial and target distributions
        boxcox_fit_initial = stats.boxcox(initial_distribution)
        boxcox_fit_target = stats.boxcox(target_distribution)

        # Compute standardization parameters
        self.lambda_x = boxcox_fit_initial[1]
        self.lambda_y = boxcox_fit_target[1]
        initial_distribution_transformed = boxcox_fit_initial[0]
        target_distribution_transformed = boxcox_fit_target[0]
        self.mu_x = np.mean(initial_distribution_transformed)
        self.mu_y = np.mean(target_distribution_transformed)
        self.sigma_x = np.std(initial_distribution_transformed)
        self.sigma_y = np.std(target_distribution_transformed)

    def transform(self, input_values):
        """
        Transform the input distribution based on the fitted parameters.

        Parameters
        ----------
        input_values : array_like
            The values to be transformed.

        Returns
        -------
        transformed_values : array_like
            The transformed distribution.
        """
        # Box Cox only works for positive values
        input_values[input_values <= 0] = 1e-8

        # Apply fitted Box-Cox transformation
        input_bc_transformed = stats.boxcox(input_values, self.lambda_x)

        # Standardize the transformed distribution
        input_bc_standardized = (input_bc_transformed - self.mu_x) / self.sigma_x

        # De-standardize to mean and std of target and appply inverse Box-Cox transformation
        transformed_values = special.inv_boxcox(input_bc_standardized * self.sigma_y + self.mu_y, self.lambda_y)

        return transformed_values


class SW_Persistence:
    """A class implementing the 27-day solar wind speed persistence model."""

    def __init__(self):
        """Initialize parameters."""
        self.column_position = None
        self.input_dimension = None
        self.coef = None

    def fit(self, X, column_position=None):
        """
        Fit the persistence model to the data.

        Parameters
        ----------
        X : array_like or DataFrame
            The input data to fit the model.
        column_position : int or None, optional
            The position of the column to use as the predictor variable.
            If None, the column with name 'speed(-27)' will be used.
        """
        if np.ndim(X) != 2:
            raise ValueError('Input array must be two-dimensional and in the format n_data_points * n_features, but {} dimensions were found.'.format(np.ndim(X)))
        else:
            self.input_dimension = X.shape[1]

        if column_position is None and isinstance(X, pd.core.frame.DataFrame):
            self.column_position = np.where(X.columns == 'speed(-27)')
        elif column_position is not None:
            self.column_position = column_position
        else:
            raise ValueError('Either provide a DataFrame with a column name matching "speed(-27)" or provide a column position.')

        self.coef = np.zeros((self.input_dimension, 1))
        self.coef[self.column_position, 0] = 1

    def predict(self, X):
        """
        Predict using the fitted persistence model.

        Parameters
        ----------
        X : array_like or DataFrame
            The input data for prediction.

        Returns
        -------
        array_like
            The predicted values.
        """
        if np.ndim(X) != 2:
            raise ValueError('Input array must be two-dimensional and in the format n_data_points * n_features, but {} dimensions were found.'.format(np.ndim(X)))

        if X.shape[1] != self.input_dimension:
            raise ValueError('Input array must have {} columns, but {} columns were found.'.format(self.input_dimension, X.shape[1]))

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.to_numpy()
        return X @ self.coef


class AveragePrediction:
    """A class implementing the average solar wind speed prediction model."""

    def __init__(self):
        """Initialize parameters."""
        self.coef = None
        self.avg_value = 0.0

    def fit(self, X, Y):
        """
        Fit the average model to the data.

        Parameters
        ----------
        X : array_like
            The input data.
        Y : array_like
            The target values.
        """
        if np.ndim(X) != 2:
            raise ValueError('Input array must be two-dimensional and in the format n_data_points * n_features, but {} dimensions were found.'.format(np.ndim(X)))
        self.coef = np.zeros(X.shape[1])

        if np.ndim(Y) != 1:
            raise ValueError('Output array must be one-dimensional, but {} dimensions were found.'.format(np.ndim(Y)))
        self.avg_value = np.mean(Y)

    def predict(self, X):
        """
        Predict using the fitted average model.

        Parameters
        ----------
        X : array_like
            The input data for prediction.

        Returns
        -------
        array_like
            The predicted values.
        """
        if np.ndim(X) != 2:
            raise ValueError('Input array must be two-dimensional and in the format n_data_points * n_features, but {} dimensions were found.'.format(np.ndim(X)))

        return np.ones(X.shape[0]) * self.avg_value


class IdentityTransformation:
    """A class implementing an identity feature transformation."""

    def __init__(self):
        """Initialize parameters."""
        self.input_dimension = None

    def fit(self, X):
        """
        Fit the identity transformation to the data.

        Parameters
        ----------
        X : array_like
            The input data.
        """
        if np.ndim(X) != 2:
            raise ValueError('Input array must be two-dimensional and in the format n_data_points * n_features, but {} dimensions were found.'.format(np.ndim(X)))
        else:
            self.input_dimension = X.shape[1]

    def fit_transform(self, X):
        """
        Fit and transform the data using the identity transformation.

        Parameters
        ----------
        X : array_like
            The input data.

        Returns
        -------
        array_like
            The transformed data (which is the same as input).
        """
        return X

    def transform(self, X):
        """
        Transform the data using the identity transformation.

        Parameters
        ----------
        X : array_like
            The input data.

        Returns
        -------
        array_like
            The transformed data (which is the same as input).
        """
        return X

    def inverse_transform(self, X):
        """
        Perform inverse transformation.

        Parameters
        ----------
        X : array_like
            The input data.

        Returns
        -------
        array_like
            The inverse transformed data (which is the same as input).
        """
        return X


def fit_polynomial_regression(X, Y, gamma, order=3):
    """
    Perform polynomial regression.

    Parameters
    ----------
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    gamma : float
        Regularization parameter.
    order : int, optional
        The degree of the polynomial.

    Returns
    -------
    pr_model : Lasso
        The polynomial regression model.
    pr_feature_transformer : PolynomialFeatures
        The feature transformer for polynomial regression.
    coef : Series
        The coefficients obtained from polynomial regression.
    """
    # Polynomial feature transformation
    pr_feature_transformer = PolynomialFeatures(order)

    # Lasso regression model for polynomial regression
    pr_model = linear_model.Lasso(alpha=gamma, max_iter=10000, selection='random', random_state=42)
    pr_model.fit(pr_feature_transformer.fit_transform(X.loc['no_cme', 'train']),
                 Y.loc['no_cme', 'train'].reshape(-1, 1))

    # Get feature names for polynomial features
    pr_feature_names = pr_feature_transformer.get_feature_names_out(input_features=X.loc['no_cme', 'train'].columns)

    # Save coefficients and compute feature importance.
    coef = pd.Series(pr_model.coef_.reshape(-1), index=pr_feature_names)

    return pr_model, pr_feature_transformer, coef



def fit_linear_regression(X, Y, gamma=None):
    """
    Perform linear regression.

    Parameters
    ----------
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    gamma : float, optional
        Regularization parameter for Ridge regression. If None, performs Linear Regression.

    Returns
    -------
    lr_model : LinearRegression or Ridge
        The linear regression model.
    None : None
        Placeholder for feature transformer
    coef : Series
        The coefficients obtained from linear regression.
    """
    # Select appropriate linear regression model based on gamma
    if gamma is None:
        lr_model = linear_model.LinearRegression()
    else:
        lr_model = linear_model.Ridge(alpha=gamma)

    # Fit the model
    lr_model.fit(X.loc['no_cme', 'train'], Y.loc['no_cme', 'train'].reshape(-1, 1))

    # Save coefficients
    coef = pd.Series(lr_model.coef_.reshape(-1), index=X.loc['no_cme', 'train'].columns)

    return lr_model, None, coef


def fit_sw_persistence_model(X):
    """
    Fit a 27-day solar wind speed persistence model.

    Parameters
    ----------
    X : DataFrame
        The feature data.

    Returns
    -------
    model : SW_Persistence
        The trained persistence model.
    coef : Series
        The coefficients obtained from the persistence model.
    """
    # Initialize the model
    model = SW_Persistence()

    # Fit the model
    model.fit(X.loc['no_cme', 'train'])

    # Save coefficients
    coef = pd.Series(model.coef.reshape(-1), index=X.loc['no_cme', 'train'].columns)

    return model, None, coef


def fit_average_model(X, Y):
    """
    Fit an average model.

    Parameters
    ----------
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.

    Returns
    -------
    model : AveragePrediction
        The trained average model.
    coef : Series
        The coefficients obtained from the average model.
    """
    # Initialize the model
    model = AveragePrediction()

    # Fit the model
    model.fit(X.loc['no_cme', 'train'], Y.loc['no_cme', 'train'])

    # Save coefficients
    coef = pd.Series(model.coef, index=X.loc['no_cme', 'train'].columns)

    return model, None, coef


def predict_sw_speed(model, X, feature_transformer=None, output_scaler=None, min_train=None):
    """
    Predict solar wind speed using a given model.

    Parameters
    ----------
    model : model object
        The trained model.
    X : DataFrame
        The feature data.
    feature_transformer : transformer object, optional
        Transformer for feature data.
    output_scaler : scaler object, optional
        Scaler for output data.
    min_train : float, optional
        Minimum threshold for prediction.

    Returns
    -------
    predictions : DataFrame
        Predicted solar wind speed.
    """
    predictions = []
    for idx in X.index:
        for col in X.columns:
            # Get dates
            dates = X.loc[idx, col].index

            # Transform features if feature transformer is available
            if feature_transformer is not None:
                X.loc[idx, col] = feature_transformer.transform(X.loc[idx, col])

            # Predict using the model
            pred = model.predict(X.loc[idx, col])

            # Transform back if output scaler is available
            if output_scaler is not None:
                pred = output_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

            # Apply minimum threshold if specified
            if min_train is not None:
                pred[pred < min_train] = min_train

            # Convert prediction to Series with dates as index
            pred = pd.Series(pred, index=dates)
            predictions.append(pred)

    # Put predictions for different categories into DataFrame
    predictions = np.array(predictions, dtype=object).reshape(len(X.index), len(X.columns))
    predictions = pd.DataFrame(predictions, index=X.index, columns=X.columns, dtype=object)
    return predictions

def compute_baseline_predictions(X, Y, grid,options, output_scaler=None):
    """
    Compute baseline predictions for the prediction efficiency metrics.

    Parameters
    ----------
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    output_scaler : scaler object, optional
        Scaler for output data.

    Returns
    -------
    pred_27 : DataFrame
        Baseline predictions using solar wind speed 27 days ago.
    pred_base : DataFrame
        Baseline predictions using linear regression on baseline features.
    """
    # Baseline predictions using solar wind speed 27 days ago
    pred_27 = []
    for idx in X.index:
        for col in X.columns:
            pred = X.loc[idx, col].loc[:, 'speed(-27)']
            pred_27.append(pred)
    pred_27 = np.array(pred_27, dtype=object).reshape(len(X.index), len(X.columns))
    pred_27 = pd.DataFrame(pred_27, index=X.index, columns=X.columns, dtype=object)

    # Baseline predictions using linear regression on baseline features
    base_model = linear_model.LinearRegression()
    MinMax_X_base = MinMaxScaler()

    pred_base = []
    for idx in X.index:
        for col in X.columns:
            dates = X.loc[idx, col].index

            # Extract baseline features from input dataset unless the prediction model itself is already a baseline model
            if options['prediction_model'][-8:] == 'baseline':
                X_base = copy(X.loc[idx, col])
            else:
                X_base = preprocessing_dataset.extract_baseline_features(X.loc[idx, col], grid)

            # Fit MinMax Scaler and model on training data
            if idx == 'no_cme' and col == 'train':
                MinMax_X_base.fit(X_base)
            X_base = MinMax_X_base.transform(X_base)

            if idx == 'no_cme' and col == 'train':
                base_model.fit(X_base, Y.loc[idx, col])

            # Compute baseline predictions
            pred = base_model.predict(X_base)

            # Scale to original speed range
            if output_scaler is not None:
                pred = output_scaler.inverse_transform(pred.reshape((-1, 1))).reshape(-1)

            pred = pd.Series(pred, index=dates)
            pred_base.append(pred)

    pred_base = np.array(pred_base, dtype=object).reshape(len(X.index), len(X.columns))
    pred_base = pd.DataFrame(pred_base, index=X.index, columns=X.columns, dtype=object)

    return pred_27, pred_base


def apply_distribution_transformation(pred, obs, min_train=None, options=None, calibration_idx=None):
    """
    Apply distribution transformation to predictions.

    Parameters
    ----------
    pred : DataFrame
        Predicted values.
    obs : DataFrame
        Observed values.
    min_train : float, optional
        Minimum threshold for prediction.
    options : dict, optional
        Additional options.

    Returns
    -------
    predictions_transformed : DataFrame
        Transformed predictions.
    """
    if options is not None and options['prediction_model'] == 'average_baseline':
        # If using average baseline, return deep copy of predictions
        predictions_transformed = deepcopy(pred)
    else:
        # Apply distribution transformation
        predictions_transformed = []
        dist_trans = DistributionTransformation()
        # If a calibration index is given, use this for fitting instead of the full training data.
        if calibration_idx is not None:
            dist_trans.fit(pred.loc['no_cme', 'train'].loc[calibration_idx], obs.loc['no_cme', 'train'].loc[calibration_idx])
        else:
            dist_trans.fit(pred.loc['no_cme', 'train'], obs.loc['no_cme', 'train'])


        for idx in pred.index:
            for col in pred.columns:
                # Transform predictions
                pred_trans = dist_trans.transform(pred.loc[idx, col])
                pred_trans = pd.Series(pred_trans,index=pred.loc[idx, col].index)

                # Apply minimum threshold if specified
                if min_train is not None:
                    pred_trans[pred_trans < min_train] = min_train

                predictions_transformed.append(pred_trans)

        # Reshape and convert to DataFrame
        predictions_transformed = np.array(predictions_transformed, dtype=object).reshape(len(pred.index),
                                                                                          len(pred.columns))
        predictions_transformed = pd.DataFrame(predictions_transformed, index=pred.index, columns=pred.columns,
                                               dtype=object)

    return predictions_transformed


def fit_prediction_model(X, Y, gamma, options):
    """
    Fit the prediction model based on the specified options.

    Parameters
    ----------
    X : DataFrame
        Input features.
    Y : DataFrame
        Target variable.
    gamma : float
        Regularization parameter.
    options : dict
        Options for selecting the prediction model.

    Returns
    -------
    model : object
        Fitted prediction model.
    feature_transformer : object
        Fitted feature transformer (if applicable).
    coef_pred_model : DataFrame
        Coefficients of the prediction model.
    """
    # Check the prediction model option and fit the corresponding model
    if options['prediction_model'] == 'polynomial':
        model, feature_transformer, coef_pred_model = fit_polynomial_regression(X, Y, gamma)
    elif options['prediction_model'] == 'linear':
        model, feature_transformer, coef_pred_model = fit_linear_regression(X, Y, gamma=519.80405454)
    elif options['prediction_model'] == 'ch_area_baseline':
        model, feature_transformer, coef_pred_model = fit_linear_regression(X, Y)
    elif options['prediction_model'] == 'sw_persistence_baseline':
        model, feature_transformer, coef_pred_model = fit_sw_persistence_model(X)
    elif options['prediction_model'] == 'average_baseline':
        model, feature_transformer, coef_pred_model = fit_average_model(X, Y)

    return model, feature_transformer, coef_pred_model


def feature_selection(X, Y, gamma, options=None):
    """
    Perform feature selection.

    Parameters
    ----------
    X : DataFrame
        The feature data.
    Y : DataFrame
        The target data.
    gamma : float
        Regularization parameter.
    options : dict, optional
        Options for feature selection.

    Returns
    -------
    X_fs : DataFrame
        The feature-selected data for each split and each class.
    coef : Series
        The coefficients obtained from feature selection.
    feature_names_fs : Index
        The selected feature names.
    """
    if options is not None and options['prediction_model'][-8:] == 'baseline':
        # If the prediction model is a baseline, use all features without selection
        X_fs = deepcopy(X)
        feature_names_fs = X.loc['no_cme', 'train'].columns
        coef = pd.Series(np.zeros(X.shape[1]), index=feature_names_fs)
    else:
        # Perform Lasso regression for feature selection
        feature_names = X.loc['no_cme', 'train'].columns
        fs_model = linear_model.Lasso(alpha=gamma, max_iter=10000).fit(X.loc['no_cme', 'train'].values,
                                                                       Y.loc['no_cme', 'train'])
        coef = pd.Series(fs_model.coef_.reshape(-1), index=feature_names)

        # Keep features with non-zero coefficients > 1e-4
        coef_abs = np.abs(coef.values)
        nonzero_binary = coef_abs > 1e-4
        feature_names_fs = feature_names[nonzero_binary]

        # Select relevant features for train and test data
        X_train_fs = X.loc['no_cme', 'train'].loc[:, feature_names_fs]
        X_test_fs = X.loc['no_cme', 'test'].loc[:, feature_names_fs]
        X_train_cme_fs = X.loc['with_cme', 'train'].loc[:, feature_names_fs]
        X_test_cme_fs = X.loc['with_cme', 'test'].loc[:, feature_names_fs]

        # Combine selected features into arrays and then into DataFrames for both classes
        X_fs = np.array([[X_train_fs, X_test_fs],
                         [X_train_cme_fs, X_test_cme_fs]], dtype=object)
        X_fs = pd.DataFrame(X_fs, index=['no_cme', 'with_cme'],
                            columns=['train', 'test'])

    return X_fs, coef, feature_names_fs
