import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from datetime import datetime, timedelta
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelResults:
    """
    A class to store model results, including predicted and observed data, and all sorts of evaluation results.

    Parameters
    ----------
    pred_data : array-like
        Predicted data.
    obs_data : array-like
        Observed data.
    peak_data : array-like, optional
        Peak data.

    Attributes
    ----------
    pred_data : pandas.DataFrame
        Model predictions.
    obs_data : pandas.DataFrame
        Observational data data.
    peak_data : pandas.DataFrame, optional
        HSS peak data.
    coef_fs : pandas.DataFrame
        Coefficients of feature selection model.
    coef_pred_model : pandas.DataFrame
        Coefficients of prediction model.
    feature_importance_fold : pandas.DataFrame
        Feature importance of each fold.
    feature_importance_rmse_permutation : pandas.DataFrame
        Permutation feature importance w.r.t. timeline RMSE.
    feature_importance_peak_rmse_permutation : pandas.DataFrame
        Permutation feature importance w.r.t. HSS peak RMSE.
    pred_data_fold : list of pandas.DataFrame
        Model prediction of each fold.
    obs_data_fold : list of pandas.DataFrame
        Observations of each fold.
    peak_data_fold : list of pandas.DataFrame
        HSS peak data of each fold.
    cont_metrics : list
        List of continuous metrics.
    event_metrics : list
        List of event-based metrics.
    general : pandas.DataFrame
        General timeline evaluations.
    peaks : pandas.DataFrame
        HSS peak evaluations.
    events : pandas.DataFrame
        Event-based evaluations.
    cv_sets : list
        List of cross-validation modes ('test', 'train').
    eval_data_modes : list
        List of evaluation data modes ('no_cme', 'trans', 'with_cme', 'trans_with_cme').
    """
    def __init__(self, pred_data, obs_data, peak_data=None):
        self.pred_data = pred_data
        self.obs_data = obs_data
        self.peak_data = peak_data
        self.coef_fs = None
        self.coef_pred_model = None
        self.feature_importance_fold = None
        self.feature_importance_rmse_permutation = None
        self.feature_importance_peak_rmse_permutation = None
        self.pred_data_fold = None
        self.obs_data_fold = None
        self.peak_data_fold = None

        # Initialize metric and evaluation data mode lists
        metrics = ['rmse', 'mae', 'me', 'cc', 'pe_av', 'pe_27', 'pe_base']
        cv_set = ['test', 'train']
        eval_data_mode = ['no_cme', 'trans', 'with_cme', 'trans_with_cme']

        # Create MultiIndex for continuous metrics DataFrame
        multi_row = pd.MultiIndex.from_product([cv_set, eval_data_mode], names=['cv_set', 'eval_data_mode'])

        # Initialize DataFrames for metrics
        continuous = pd.DataFrame(np.zeros((len(multi_row), len(metrics)), dtype=float), index=multi_row, columns=metrics)
        self.cont_metrics = metrics

        event_metrics = ['pod', 'far', 'ts', 'bs']
        event = pd.DataFrame(np.zeros((len(multi_row), len(event_metrics)), dtype=float), index=multi_row, columns=event_metrics)
        self.event_metrics = event_metrics

        # Initialize DataFrames for evaluation metrics
        self.general = continuous.copy()
        self.peaks = continuous.copy()
        self.events = event

        self.cv_sets = cv_set
        self.eval_data_modes = eval_data_mode

    def detect_peaks(self, enhancement_list, cme_list):
        """
        Detect peaks in geomagnetic storm data.

        Parameters
        ----------
        enhancement_list : pandas series
            List of enhancements.
        cme_list : pandas series
            List of CME data.
        """

        pred_peaks, obs_peaks = associate_enhancements(self.pred_data.loc['with_cme', 'test'], enhancement_list, cme_list)
        pred_peaks_trans, obs_peaks_trans = associate_enhancements(self.pred_data.loc['trans_with_cme', 'test'], enhancement_list, cme_list)

        arr = np.zeros((2, 2), dtype=object)
        arr[:, :] = [[pred_peaks, obs_peaks], [pred_peaks_trans, obs_peaks_trans]]

        peak_data = pd.DataFrame(arr, index=['no_cme', 'trans'], columns=['pred', 'obs'])
        self.peak_data = peak_data

    def evaluate(self):
        """
        Evaluate the model.
        """
        pred = self.pred_data
        obs = self.obs_data
        peaks = self.peak_data

        # Compute HSS-related metrics.
        if peaks is not None:
            # Extract detected peaks in predictions before and after distribution transformation.
            pred_peaks, obs_peaks = peaks.loc['no_cme', :].to_numpy()
            hits = pred_peaks.loc[:, 'hit'].to_numpy()
            associated_peaks = pred_peaks.loc[hits, 'associated'].to_numpy()

            pred_peaks_trans, obs_peaks_trans = peaks.loc['trans', :].to_numpy()
            hits_trans = pred_peaks_trans.loc[:, 'hit'].to_numpy()
            associated_peaks_trans = pred_peaks_trans.loc[hits_trans, 'associated'].to_numpy()

            self.peak_data = peaks

            # Compute peak velocity metrics.
            for metric in self.cont_metrics[:-2]:
                self.peaks.loc[('test', 'no_cme'), metric] = compute_metric(pred_peaks.loc[hits, 'peak_value'],
                                                                        obs_peaks.loc[associated_peaks, 'peak_value'], metric)
                self.peaks.loc[('test', 'trans'), metric] = compute_metric(pred_peaks_trans.loc[hits_trans, 'peak_value'],
                                                                       obs_peaks_trans.loc[associated_peaks_trans, 'peak_value'], metric)
            # Compute event metrics.
            for eval_data_mode in ['no_cme', 'trans']:
                for metric in self.event_metrics:
                    self.events.loc[('test', eval_data_mode), metric] = compute_metric(peaks.loc[eval_data_mode, 'pred'], peaks.loc[eval_data_mode, 'obs'], metric)

        # Save observational data in same format as predictions.
        obs = pd.concat([obs.iloc[0, :], obs.iloc[0, :], obs.iloc[1, :], obs.iloc[1, :]], axis=1).transpose()
        obs.index = ['no_cme', 'trans', 'with_cme', 'trans_with_cme']

        base = pred.loc[['base', 'base_with_cme'], :]
        pers_27 = pred.loc[['27', '27_with_cme'], :]

        # Compute general time series metrics.
        for cv_set in self.cv_sets:
            for eval_data_mode in self.eval_data_modes:
                for metric in self.cont_metrics:
                    if metric == 'pe_27':
                        if eval_data_mode[-8:] == 'with_cme':
                            self.general.loc[(cv_set, eval_data_mode), metric] = compute_metric(pred.loc[eval_data_mode, cv_set], obs.loc[eval_data_mode, cv_set], metric,
                                                                          pers_27.loc['27_with_cme', cv_set])
                        else:
                            self.general.loc[(cv_set, eval_data_mode), metric] = compute_metric(pred.loc[eval_data_mode, cv_set], obs.loc[eval_data_mode, cv_set], metric,
                                                                          pers_27.loc['27', cv_set])
                    elif metric == 'pe_base':
                        if eval_data_mode[-8:] == 'with_cme':
                            self.general.loc[(cv_set, eval_data_mode), metric] = compute_metric(pred.loc[eval_data_mode, cv_set], obs.loc[eval_data_mode, cv_set], metric,
                                                                          base.loc['base_with_cme', cv_set])
                        else:
                            self.general.loc[(cv_set, eval_data_mode), metric] = compute_metric(pred.loc[eval_data_mode, cv_set], obs.loc[eval_data_mode, cv_set], metric,
                                                                          base.loc['base', cv_set])
                    else:
                        self.general.loc[(cv_set, eval_data_mode), metric] = compute_metric(pred.loc[eval_data_mode, cv_set], obs.loc[eval_data_mode, cv_set], metric)

        self.pred_data = pred
        self.obs_date = obs

    def print(self):
        """
        Print evaluation results.
        """
        print('#########################################################################')
        print('Metrics for continuous time series:')
        print(self.general.loc[:, ['rmse', 'mae', 'me', 'cc']])
        print('#########################################################################')
        print('Prediction efficiencies for continuous time series:')
        print(self.general.loc[:, ['pe_av', 'pe_27', 'pe_base']])
        print('#########################################################################')
        print('Metrics for associated HSS peaks:')
        print(self.peaks.loc[('test', ['no_cme', 'trans']), ['rmse', 'mae', 'me', 'cc']])
        print('#########################################################################')
        print('Metrics for HSS detection:')
        print(self.events.loc[('test', ['no_cme', 'trans']), :])
        print('#########################################################################')

    def save(self, grid, options):
        """
        Save evaluation results.

        Parameters
        ----------
        grid : str
            Grid resolution.
        options : dict
            Model options.
        """
        if options['prediction_model'][-8:] == 'baseline':
            grid = 'no_grid'
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        now = datetime.now()
        dir_path = os.path.dirname(os.getcwd())
        path = dir_path + '/results'
        if not os.path.exists(path):
            os.mkdir(path)
        path = dir_path + '/results/model_eval'
        if not os.path.exists(path):
            os.mkdir(path)
        path = dir_path + '/results/model_eval/{}'.format(grid)
        if not os.path.exists(path):
            os.mkdir(path)
        file = path + '/{}_{}.xlsx'.format(options['prediction_model'], datetime.strftime(now, '%y%m%d-%H%M'))
        writer = pd.ExcelWriter(file, engine="xlsxwriter")
        self.general.to_excel(writer, sheet_name="general_cv")
        self.peaks.loc[('test', ['no_cme', 'trans']), ['rmse', 'mae', 'me', 'cc']].to_excel(writer,
                                                                                               sheet_name="peaks_cv")
        self.events.loc[('test', ['no_cme', 'trans']), :].to_excel(writer, sheet_name="events_cv")
        self.coef_fs.to_excel(writer, sheet_name="coefficients_fs")
        self.coef_pred_model.to_excel(writer, sheet_name="coefficients_pred_model")
        self.feature_importance_rmse_permutation.to_excel(writer, sheet_name="permutation_importance_rmse")
        self.feature_importance_peak_rmse_permutation.to_excel(writer, sheet_name="permutation_importance_peak")
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()

        path = dir_path + '/results/model_pred'
        if not os.path.exists(path):
            os.mkdir(path)
        path = dir_path + '/results/model_pred/{}'.format(grid)
        if not os.path.exists(path):
            os.mkdir(path)
        file = path + '/{}_{}.pickle'.format(options['prediction_model'], datetime.strftime(now, '%y%m%d-%H%M'))

        with open(file, 'wb') as f:
            pickle.dump(self, f)

        return



def compute_metric(pred, obs, metric, base=None):
    """
    Compute various evaluation metrics based on predictions and observations.

    Parameters
    ----------
    pred : array-like
        Predicted values.
    obs : array-like
        Observed values.
    metric : str
        The metric to compute. Available options are:
        - 'rmse': Root Mean Squared Error.
        - 'mae': Mean Absolute Error.
        - 'me': Mean Error.
        - 'cc': Pearson Correlation Coefficient.
        - 'pe_av': Prediction efficiency w.r.t. average baseline model.
        - 'pe_27': Prediction efficiency w.r.t. 27-day persistence baseline model.
        - 'pe_base': Prediction efficiency w.r.t. coronal hole baseline model.
        - 'pod': Probability of Detection.
        - 'fnr': False Negative Rate.
        - 'ppv': Positive Predictive Value.
        - 'far': False Alarm Rate.
        - 'ts': Threat Score.
        - 'bs': Bias.
    base : array-like, optional
        Baseline predictions used for prediction efficiencies.

    Returns
    -------
    float
        The computed metric value.
    """
    # Check if predictions or observations are None
    if pred is None or obs is None:
        return None

    # Check if predictions or observations are empty
    if len(pred) == 0 or len(obs) == 0:
        return np.inf

    # Compute metrics based on the chosen metric type
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(obs, pred))
    elif metric == 'mae':
        return mean_absolute_error(obs, pred)
    elif metric == 'me':
        return np.mean(pred) - np.mean(obs)
    elif metric == 'cc':
        # Check if the standard deviation of predictions is close to zero
        if np.std(pred) < 1e-8:
            return np.nan
        return np.corrcoef(obs.to_numpy(), pred.to_numpy())[0, 1]
    elif metric == 'pe_av':
        return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
    elif metric == 'pe_27':
        return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs - base) ** 2))
    elif metric == 'pe_base':
        return 1 - (np.sum((pred - obs) ** 2) / np.sum((obs - base) ** 2))
    else:
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = obs.loc[:, 'hit'].sum()
        FP = pred.loc[:, 'false_alarm'].sum()
        FN = obs.loc[:, 'miss'].sum()

        # Compute metrics based on contingency table
        if metric == 'pod':
            return TP / (TP + FN)
        elif metric == 'fnr':
            return FN / (TP + FN)
        elif metric == 'ppv':
            return TP / (TP + FP)
        elif metric == 'far':
            return FP / (TP + FP)
        elif metric == 'ts':
            return TP / (TP + FP + FN)
        elif metric == 'bs':
            return (TP + FP) / (TP + FN)



def find_peaks_in_time_series(time_series, exclude_unpredictable_cme_peaks=False, cme_list=None):
    """
    Find peaks in a time series and identify enhancement intervals.

    Parameters
    ----------
    time_series : pandas.Series
        Time series.
    exclude_unpredictable_cme_peaks : bool, optional
        Whether to exclude peaks affected by a CMEs. Default is False.
    cme_list : pandas.DataFrame, optional
        DataFrame containing information about CMEs, including start and end dates. Default is None.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing information about detected peaks and enhancement intervals.
    """

    # Parameters for peak detection
    smooth_window = 24
    distance = 4 * 24
    threshold_speed = 390
    prominence = 35

    # Smooth the time series and find peaks in it.
    ts_smoothed = pd.Series(gaussian_filter1d(time_series, smooth_window), index=time_series.index)
    peaks_smoothed_idx, _ = find_peaks(ts_smoothed, height=threshold_speed, distance=distance, prominence=prominence)
    smoothed_peaks = pd.to_datetime(pd.Series(time_series.index[peaks_smoothed_idx]))

    # Calculate start and end of enhancements, i.e., widths of peaks
    widths = peak_widths(ts_smoothed.values, peaks_smoothed_idx, rel_height=0.6)
    enhancement_intervals = np.concatenate([widths[2].reshape((-1, 1)), widths[3].reshape((-1, 1))], axis=1)
    enhancement_intervals[:, 0] = np.floor(enhancement_intervals[:, 0])
    enhancement_intervals[:, 1] = np.ceil(enhancement_intervals[:, 1])
    enhancement_intervals = enhancement_intervals.astype(int)

    # Remove intersecting intervals by truncating the longer enhancement
    for i in range(enhancement_intervals.shape[0] - 1):
        if enhancement_intervals[i, 1] > enhancement_intervals[i + 1, 0]:
            len_int_1 = enhancement_intervals[i, 1] - enhancement_intervals[i, 0]
            len_int_2 = enhancement_intervals[i + 1, 1] - enhancement_intervals[i + 1, 0]
            if len_int_1 > len_int_2:
                enhancement_intervals[i, 1] = enhancement_intervals[i + 1, 0] - 1
            else:
                enhancement_intervals[i + 1, 0] = enhancement_intervals[i, 1] + 1

    # Convert indices into dates
    enhancement_dates = np.zeros(enhancement_intervals.shape, dtype=datetime)
    enhancement_dates[:, 0] = time_series.index[enhancement_intervals[:, 0]]
    enhancement_dates[:, 1] = time_series.index[enhancement_intervals[:, 1]]
    enhancement_list = pd.DataFrame(enhancement_dates, columns=['start', 'end'])

    # Calculate peak speed value and date and check if they are affected by CMEs
    peak_date = np.zeros(enhancement_list.shape[0], dtype=datetime)
    peak_value = np.zeros(enhancement_list.shape[0], dtype=float)
    cme_flag = np.zeros(enhancement_list.shape[0], dtype=bool)
    for i in range(enhancement_list.shape[0]):
        enhancement_interval = time_series[enhancement_list.loc[i, 'start']:enhancement_list.loc[i, 'end']]
        peak_value[i] = np.max(enhancement_interval)
        peak_date[i] = enhancement_interval.index[np.argmax(enhancement_interval)]

        if exclude_unpredictable_cme_peaks and cme_list is not None:
            # Find enhancements that overlap with CMEs.
            cme_bin = (cme_list.loc[:, 'start'] <= peak_date[i]) & (
                    cme_list.loc[:, 'end'] >= peak_date[i])
            # Find enhancements that happen one solar rotation after a CME.
            cme_27_bin = (cme_list.loc[:, 'start'] <= (peak_date[i] - timedelta(days=26))) & (
                    cme_list.loc[:, 'end'] >= (peak_date[i] - timedelta(days=28)))
            cme_bin = cme_bin | cme_27_bin
            # Mark enhancement as affected by CME if there is at least one CME during or one solar rotation before an
            # enhancement.
            if np.sum(cme_bin.to_numpy()) > 0:
                cme_flag[i] = True

    # Create DataFrame with enhancement information
    peak_date = pd.to_datetime(pd.Series(peak_date))
    peak_value = pd.Series(peak_value, dtype=float)
    enhancement_list = pd.concat([peak_date, peak_value, enhancement_list, smoothed_peaks], axis=1)
    enhancement_list.columns = ['peak_date', 'peak_value', 'start', 'end', 'smoothed_peak_date']
    if exclude_unpredictable_cme_peaks:
        cme_flag = pd.DataFrame(cme_flag, columns=['cme_flag'], dtype=bool)
        enhancement_list = pd.concat([enhancement_list, cme_flag], axis=1)

    enhancement_list['start'] = pd.to_datetime(enhancement_list['start'])
    enhancement_list['end'] = pd.to_datetime(enhancement_list['end'])

    return enhancement_list


def associate_pred_obs_peaks(pred_enhancements, obs_enhancements, n_days=3):
    """
    Associate predicted and observed enhancement peaks.

    Parameters
    ----------
    pred_enhancements : numpy.array
        Predicted enhancement peak dates.
    obs_enhancements : numpy.array
        Observed enhancement peak dates.
    n_days : int, optional
        Number of days for peak association window. Default is 3.

    Returns
    -------
    tuple
        A tuple containing DataFrames indicating associations, hits, misses, and false alarms for predicted and observed enhancements.
    """

    # Initialize arrays to store associated peak dates and boolean flags for hits, misses, and false alarms
    obs_associated = np.zeros(len(obs_enhancements), dtype=datetime)
    pred_associated = np.zeros(len(pred_enhancements), dtype=datetime)
    hit_obs = np.zeros(len(obs_enhancements), dtype=bool)
    miss = np.zeros(len(obs_enhancements), dtype=bool)
    hit_pred = np.zeros(len(pred_enhancements), dtype=bool)
    false_alarm = np.zeros(len(pred_enhancements), dtype=bool)

    # Iterate through lists of predicted and observed enhancements to find new associations
    new_associations = 1
    while new_associations > 0:
        new_associations = 0
        # Associate predicted enhancements with observed enhancements
        for i in range(len(pred_enhancements)):
            if not hit_pred[i]:
                # Find observed enhancements within the window around the predicted enhancement
                binary = (obs_enhancements >= (pd.to_datetime(pred_enhancements[i]) - timedelta(days=n_days))) & (
                        obs_enhancements <= (pd.to_datetime(pred_enhancements[i]) + timedelta(days=n_days)))
                obs_temp = obs_enhancements[binary]
                obs_temp = obs_temp[~hit_obs[binary]]
                if len(obs_temp) > 0:
                    # Calculate distances and find the closest observed enhancement
                    dist = np.abs(obs_temp - pred_enhancements[i])
                    pred_associated[i] = obs_temp[np.argmin(dist)]
                else:
                    pred_associated[i] = None

        # Associate observed enhancements with predicted enhancements
        for i in range(len(obs_enhancements)):
            if not hit_obs[i]:
                # Find predicted enhancements within the window around the observed enhancement
                binary = ((pd.to_datetime(obs_enhancements[i]) - timedelta(days=n_days)) <= pred_enhancements) & (
                        (pd.to_datetime(obs_enhancements[i]) + timedelta(days=n_days)) >= pred_enhancements)
                pred_temp = pred_enhancements[binary]
                pred_temp = pred_temp[~hit_pred[binary]]
                if len(pred_temp) > 0:
                    # Calculate distances and find the closest predicted enhancement
                    dist = np.abs(pred_temp - obs_enhancements[i])
                    obs_associated[i] = pred_temp[np.argmin(dist)]
                else:
                    obs_associated[i] = None

        # Check if two peaks have been associated to each other
        for i in range(len(obs_enhancements)):
            if not hit_obs[i]:
                if obs_associated[i] is not None:
                    # get associated predicted index
                    idx_associated = np.where(pred_enhancements == obs_associated[i])[0][0]
                    # get the associated observed index of the associated predicted index
                    associated_date_associated = pred_associated[idx_associated]
                    if associated_date_associated == obs_enhancements[i]:
                        # if each of the peaks is associated with the other one, fix the association and mark them as hit.
                        hit_obs[i] = True
                        hit_pred[idx_associated] = True
                        new_associations += 1

    # Mark all remaining peaks as false alarms and misses
    for i in range(len(pred_enhancements)):
        if not hit_pred[i]:
            false_alarm[i] = True
    for i in range(len(obs_enhancements)):
        if not hit_obs[i]:
            miss[i] = True

    # Convert arrays to pandas Series and create DataFrames for the results
    pred_associated = pd.to_datetime(pd.Series(pred_associated))
    hit_pred = pd.Series(hit_pred, dtype=bool)
    false_alarm = pd.Series(false_alarm, dtype=bool)
    matching_pred = pd.concat([pred_associated, hit_pred, false_alarm], axis=1)
    matching_pred.columns = ['associated', 'hit', 'false_alarm']

    obs_associated = pd.to_datetime(pd.Series(obs_associated))
    hit_obs = pd.Series(hit_obs, dtype=bool)
    miss = pd.Series(miss, dtype=bool)
    matching_obs = pd.concat([obs_associated, hit_obs, miss], axis=1)
    matching_obs.columns = ['associated', 'hit', 'miss']

    return matching_pred, matching_obs



def delete_cme_peaks(pred_enhancements, obs_enhancements):
    """
    Delete predicted enhancement peaks associated with CME peaks.

    Parameters
    ----------
    pred_enhancements : pandas.DataFrame
        DataFrame containing predicted enhancement peaks.
    obs_enhancements : pandas.DataFrame
        DataFrame containing observed enhancement peaks.

    Returns
    -------
    tuple
        A tuple containing DataFrames with predicted and observed enhancement peaks after deletion.
    """


    delete_pred = []
    # Create a mask for observed enhancements that are not associated with CME peaks
    not_cme_obs = ~obs_enhancements['cme_flag']

    # Iterate over the predicted enhancements
    for i in range(len(pred_enhancements)):

        # Get the associated observed smoothed peak date for the current predicted enhancement
        associated_obs_smoothed_peak_date = pred_enhancements.at[i, 'associated']

        # Delete predicted peaks that are associated to an observed CME peak.
        if not pd.isnull(associated_obs_smoothed_peak_date):
            # Find the index of the associated observed peak
            associated_obs_idx = obs_enhancements.index[obs_enhancements['smoothed_peak_date'] == associated_obs_smoothed_peak_date][0]
            associated_obs_peak_date = obs_enhancements.at[associated_obs_idx, 'peak_date']
            # overwrite the associated dates with the actual peak dates, because currently the associated dates are
            # the one of the associated smoothed peaks
            pred_enhancements.at[i, 'associated'] = associated_obs_peak_date
            obs_enhancements.at[associated_obs_idx, 'associated'] = pred_enhancements.at[i, 'peak_date']
            if obs_enhancements.at[associated_obs_idx, 'cme_flag']:
                delete_pred.append(i)
        else:
            # Delete predicted peaks that are influenced by a CME peak.
            if pred_enhancements.at[i, 'cme_flag']:
                delete_pred.append(i)

    # Drop the predicted enhancements marked for deletion
    pred_enhancements = pred_enhancements.drop(index=delete_pred)

    # Filter out the observed enhancements associated with CME peaks
    obs_enhancements = obs_enhancements[not_cme_obs]

    return pred_enhancements, obs_enhancements


def associate_enhancements(pred_ts, obs_enhancement_list, cme_list):
    """
    Associate enhancement predictions with observations.

    Parameters
    ----------
    pred_ts : pandas.Series
        Predicted time series.
    obs_enhancement_list : pandas.DataFrame
        DataFrame containing observed enhancements.
    cme_list : pandas.DataFrame
        DataFrame containing CME data.

    Returns
    -------
    tuple
        A tuple containing DataFrames with predicted and observed enhancements after association.
    """
    # Filter observed enhancements to include only those within the time range of predicted time series
    binary = (obs_enhancement_list['start'] >= pred_ts.index[0]) & (obs_enhancement_list['end'] <= pred_ts.index[-1])
    obs_enhancement_list = obs_enhancement_list[binary].reset_index(drop=True)

    # Find peaks in the predicted time series, excluding those influenced by CMEs
    pred_enhancement_list = find_peaks_in_time_series(pred_ts, exclude_unpredictable_cme_peaks=True, cme_list=cme_list)

    # Convert smoothed peak dates to numpy arrays for association
    pred_smoothed_peaks = pred_enhancement_list['smoothed_peak_date'].to_numpy()
    obs_smoothed_peaks = obs_enhancement_list['smoothed_peak_date'].to_numpy()

    # Associate predicted enhancements with observed peaks
    matching_pred, matching_obs = associate_pred_obs_peaks(pred_smoothed_peaks, obs_smoothed_peaks)

    pred_enhancement_list = pd.concat([pred_enhancement_list, matching_pred], axis=1)
    obs_enhancement_list = pd.concat([obs_enhancement_list, matching_obs], axis=1)

    # Delete CME-influenced enhancements
    pred_enhancement_list, obs_enhancement_list = delete_cme_peaks(pred_enhancement_list, obs_enhancement_list)

    # Set peak dates as the index for both enhancement lists
    pred_enhancement_list.index = pred_enhancement_list['peak_date']
    obs_enhancement_list.index = obs_enhancement_list['peak_date']

    pred_enhancement_list = pred_enhancement_list.drop(columns=['peak_date'])
    obs_enhancement_list = obs_enhancement_list.drop(columns=['peak_date'])

    return pred_enhancement_list, obs_enhancement_list
