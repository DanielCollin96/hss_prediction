import numpy as np
import preprocessing_segmentation_maps
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle
import evaluation
import math




def create_input_output(start, end, area, alpha=None, sw=None, ssn_monthly=None, only_area=False,forecast_horizon=96, area_hist=72, area_freq=24, sw_hist=48, sw_freq=24):
    """
    Creates input features and target output data for a forecasting model.

    Parameters
    ----------
    start : datetime.datetime
        Start date of the dataset.
    end : datetime.datetime
        End date of the dataset.
    area : pandas.DataFrame
        Coronal hole area data.
    alpha : pandas.DataFrame, optional
        Latitudinal angle between solar equatorial plane and Earth.
    sw : pandas.DataFrame, optional
        Solar wind observations.
    ssn_monthly : pandas.Series, optional
        Monthly sunspot number data.
    only_area : bool, optional
        True if only coronal hole area features need to be computed, by default False.
    forecast_horizon : int, optional
        Forecast horizon in hours, by default 96.
    area_hist : int, optional
        Time span of coronal hole area history added to the dataset in hours, by default 72.
    area_freq : int, optional
        Frequency of coronal hole area features in hours, by default 24.
    sw_hist : int, optional
        Time span of solar wind history added to the dataset in hours, by default 48.
    sw_freq : int, optional
        Frequency of the solar wind frequency in hours, by default 24.

    Returns
    -------
    pandas.DataFrame, pandas.Series
        Feature DataFrame and target Series.
    """


    feature_names = []
    feature_cols = []

    if not only_area:
        if sw is None or ssn_monthly is None:
            raise ValueError('If only_area is False, solar wind data and monthly sunspot numbers need to be given.')

        # Extract heliospheric latitudinal angle alpha
        sw_alpha = sw['alpha']

        # Extract solar wind speed data within the specified time range as the target output
        output = sw['speed'][(sw.index >= start) & (sw.index <= end)]



        # Add alpha to feature columns
        feature_cols.append(sw.loc[output.index, 'alpha'].to_numpy().reshape((-1, 1)))
        feature_names.append('alpha')
        sw = sw.drop(columns=['alpha'], axis=1)

        # Calculate number of steps for solar wind history
        n_steps = int(sw_hist / sw_freq) + 1

        # Iterate over each column in solar wind data and add features and time difference to forecasted time point (in days)
        for col in sw.columns:
            idx_shift = int(- 27 * 24 - (sw_hist / 2))  # SW history symmetrically scattered around -27 days
            for i in range(n_steps):
                feature_cols.append(sw.loc[output.index + timedelta(hours=idx_shift), col].to_numpy().reshape((-1, 1)))
                feature_names.append('{}({})'.format(col, str(int(idx_shift / 24))))
                idx_shift = idx_shift + sw_freq

        # Compute hourly sunspot data
        ssn_hourly, ssn_change = compute_hourly_ssn(ssn_monthly)

        ssn_hourly = ssn_hourly[output.index[0]:output.index[-1]]
        ssn_change = ssn_change[output.index[0]:output.index[-1]]

        # Add sunspot number and change to feature columns
        feature_cols.append(ssn_hourly.values.reshape((-1, 1)))
        feature_names.append('sunspot_number')

        feature_cols.append(ssn_change.values.reshape((-1, 1)))
        feature_names.append('sunspot_number_change')
    else:
        if alpha is None:
            raise ValueError('If only_area is True, alpha needs to be given.')
        sw_alpha = alpha
        output = alpha[(alpha.index >= start) & (alpha.index <= end)]
        output.iloc[:] = np.nan


    # Calculate number of steps for area history
    n_steps = int(area_hist / area_freq) + 1
    area_grid = area.columns[-1][2:-1].split(",")
    area_grid = (int(area_grid[0]), int(area_grid[1]))
    grid_to_series = np.arange(area_grid[0] * area_grid[1]).reshape(area_grid)

    # Iterate over each column in area data and add features and time difference to forecasted time point (in days)
    for col in area.columns[:int(math.ceil(area.shape[1] / 2))]:
        idx_shift = - forecast_horizon - area_hist

        # calculate current grid cell
        grid_point = col[2:-1].split(',')
        grid_point = np.array([int(grid_point[0]), int(grid_point[1])])

        # calculate grid cell symmetric to equator
        sym_point = np.array([area_grid[0] - grid_point[0] + 1, grid_point[1]])
        sym_col = area.columns[grid_to_series[sym_point[0] - 1, sym_point[1] - 1]]

        # calculate S and D features by summing or subtracting symmetric grid cells
        for i in range(n_steps):
            col_sum = area.loc[output.index + timedelta(hours=idx_shift), col].to_numpy() + area.loc[
                output.index + timedelta(hours=idx_shift), sym_col].to_numpy()
            col_dif = area.loc[output.index + timedelta(hours=idx_shift), col].to_numpy() - area.loc[
                output.index + timedelta(hours=idx_shift), sym_col].to_numpy()
            col_dif = col_dif * sw_alpha[output.index + timedelta(hours=idx_shift)].to_numpy()
            feature_cols.append(col_sum.reshape((-1, 1)))
            feature_cols.append(col_dif.reshape((-1, 1)))

            feature_names.append('S[{},{}]({})'.format(grid_point[0], grid_point[1],
                                                          str(int(idx_shift / 24))))
            feature_names.append('D[{},{}]({})'.format(grid_point[0], grid_point[1],
                                                                str(int(idx_shift / 24))))

            idx_shift = idx_shift + area_freq



    # Create input DataFrame
    input_data = pd.DataFrame(np.concatenate(feature_cols, axis=1), index=output.index, columns=feature_names)

    return input_data, output



def compute_hourly_ssn(ssn_monthly_noisy,window_len=48,horizon=3,idx_to_use=-6):
    """
    Compute hourly smoothed sunspot number and sunspot number change based on monthly noisy sunspot data by extrapolating.

    Parameters
    ----------
    ssn_monthly_noisy : pandas.Series
        Monthly sunspot number (noisy).
    window_len : int, optional
        Length of the window for polynomial fitting. Default is 48 data points.
    horizon : int, optional
        Number of months ahead for extrapolation. Default is 3.
    idx_to_use : int, optional
        Index within the fitted window to use for slope computation. Default is -6.

    Returns
    -------
    tuple
        A tuple containing two pandas.Series:
        - ssn_hourly: Extrapolated hourly sunspot numbers.
        - ssn_hourly_change: Hourly change in sunspot numbers.
    """

    ssn_extrapolated = []
    idx_extrapolated = []
    ssn_slope = []

    # Iterate over the monthly noisy data
    for i in np.arange(window_len,len(ssn_monthly_noisy)):
        # Extract time history
        current_month = ssn_monthly_noisy.index[i]
        backward_dates = ssn_monthly_noisy.index[i-window_len:i]
        xval = np.arange(window_len)

        # Fit a 2nd degree polynomial to the data
        coef = np.polyfit(xval,ssn_monthly_noisy[backward_dates],deg=2)

        # Compute the slope and current value at the specified index
        slope = 2 * coef[0] * xval[idx_to_use] + coef[1]
        current_val = coef[0] * xval[idx_to_use]**2 + coef[1] * xval[idx_to_use] + coef[2]

        # Extrapolate the value into the future
        next_val = current_val + slope * horizon

        ssn_extrapolated.append(next_val)
        idx_extrapolated.append(current_month)
        ssn_slope.append(slope)

    # Create pandas Series for extrapolated values and slopes
    ssn_extrapolated_monthly = pd.Series(ssn_extrapolated,index = idx_extrapolated)
    ssn_slope = pd.Series(ssn_slope,index = idx_extrapolated)

    # Create a date range for the hourly data
    ssn_dates = pd.date_range(ssn_extrapolated_monthly.index[0],
                              ssn_extrapolated_monthly.index[-1] + relativedelta(months=1) - timedelta(hours=1),
                              freq='h')
    ssn_hourly = pd.Series(np.zeros(len(ssn_dates), dtype=float), index=ssn_dates)
    ssn_hourly_change = pd.Series(np.zeros(len(ssn_dates), dtype=float), index=ssn_dates)

    # Assign monthly extrapolated values to the hourly series and fill gaps with NaNs for interpolation
    for i in range(len(ssn_extrapolated_monthly)):
        month_start = ssn_extrapolated_monthly.index[i]
        month_end = ssn_extrapolated_monthly.index[i] + relativedelta(months=1) - timedelta(hours=1)
        binary = (ssn_hourly.index >= month_start) & (ssn_hourly.index < month_end)
        ssn_hourly[month_end] = ssn_extrapolated_monthly[i]
        ssn_hourly[binary] = np.nan
        ssn_hourly_change[month_end] = ssn_slope[i]
        ssn_hourly_change[binary] = np.nan

    # Interpolate the NaN values to fill in the hourly data
    ssn_hourly = ssn_hourly.interpolate()
    ssn_hourly_change = ssn_hourly_change.interpolate()


    return ssn_hourly, ssn_hourly_change


def create_ml_dataset(grid_resolution, start=datetime(2010, 6, 1, 0), end=datetime(2019, 12, 31, 23), map_resolution=256):
    """
    Create the machine learning dataset.

    Parameters
    ----------
    grid_resolution : str
        Grid resolution.
    start : datetime.datetime, optional
        Start date of the dataset, by default 2010-6-1 00:00.
    end : datetime.datetime, optional
        End date, by default 2019-12-31 23:00.
    map_resolution : int, optional
        Map resolution, by default 256.

    Returns
    -------
    pandas.DataFrame, pandas.Series
        Input DataFrame and output Series.
    """

    path = os.path.dirname(os.getcwd())

    # Subtract 30 days from the start date because the data history is needed to compute features.
    start_pred = start
    start = start - relativedelta(months=1)

    print('Loading dataset.')
    # Check if any dataset exists
    files = os.listdir(path + '/data/datasets/ml_data')
    existing_dataset = False
    if len(files) > 0:
        existing_dataset = True
        data = pd.read_csv(path + '/data/datasets/ml_data/{}'.format(files[0]),index_col=0,parse_dates=[0])
        existing_input = data.iloc[:,1:]
        existing_output = data.iloc[:,0]

        # Extract features that do not depend on the grid resolution
        drop_features = []
        for col in existing_input.columns:
            if col[0] == 'D' or col[0] == 'S':
                drop_features.append(col)
        existing_input = existing_input.drop(columns=drop_features)
    else:
        # If features have not been computed yet, do that using OMNI and sunspot data
        print('Reading OMNIweb.')
        sw = pd.read_csv(path + "/data/solar_wind/omni.csv", index_col=0, parse_dates=[0])

        print('Reading sunspot data.')
        sunspots = pd.read_csv(path + "/data/solar_wind/sunspots.csv", index_col=0, parse_dates=[0]).squeeze()

    # load alpha parameter as it is needed to compute the coronal hole area features and to rotate the grid
    alpha = pd.read_csv(path + "/data/datasets/alpha.csv",index_col=0,parse_dates=[0]).squeeze()


    print('Reading coronal hole data.')
    try:
        area = pickle.load(open(path + "/data/ch_area/area_ts_{}.pickle".format(grid_resolution), 'rb'))
    except FileNotFoundError:
        path_area = path + '/data/ch_area'
        if not os.path.exists(path_area):
            os.mkdir(path_area)
        path_area = path + '/data/ch_area/{}'.format(grid_resolution)
        if not os.path.exists(path_area):
            os.mkdir(path_area)

        # Extract grid resolution
        col_names = []
        M, N = grid_resolution.split('x')
        M = int(M)
        N = int(N)

        # Compute columns names
        for i in range(M):
            for j in range(N):
                col_names.append('A[{},{}]'.format(i + 1, j + 1))

        current_month = datetime(start.year, start.month, 1)
        end_month = datetime(end.year, end.month, 1)
        area = pd.DataFrame([], [])
        missing_months = []

        # Check for which months the area has already been extracted from the grid cells
        while current_month <= end_month:
            file_name = path_area + '/' + current_month.strftime('%Y-%m') + '.pickle'
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    a = pickle.load(f)
                    a.columns = col_names
                    area = pd.concat([area, a])
            else:
                missing_months.append(current_month)
            current_month = current_month + relativedelta(months=1)

        # Extract the area for missing months
        if len(missing_months) > 0:
            for month in missing_months:
                a = preprocessing_segmentation_maps.calculate_ch_area([month], map_resolution, alpha, grid_resolution)
                a.columns = col_names
                area = pd.concat([area, a])

        area = area.sort_index()

        # restrict to dataset time span
        binary = (area.index >= start) & (area.index <= end)
        area = area.iloc[binary, :]

        # save coronal hole area DataFrame
        file_name = path + "/data/ch_area/area_ts_{}.pickle".format(grid_resolution)
        with open(file_name, 'wb') as f:
            pickle.dump(area, f)


    print('Creating machine learning dataset.')
    # If a dataset exists, only compute area features and then unit with existing features.
    if existing_dataset:
        X, Y = create_input_output(start_pred, end, area, alpha=alpha, only_area=True)
        X = pd.concat([existing_input,X],axis=1)
        Y = existing_output
        print(X,Y)
        print(X.columns)
    else:
        X, Y = create_input_output(start_pred, end, area, sw=sw, ssn_monthly=sunspots, only_area=False)

    print('Saving dataset.')
    path_dataset = path + '/data/datasets/ml_data'
    if not os.path.exists(path_dataset):
        os.mkdir(path_dataset)

    file_name = grid_resolution + '.pickle'
    path_dataset = path_dataset + '/' + file_name
    with open(path_dataset, 'wb') as f:
        pickle.dump([X, Y], f)


    return X, Y


def assign_cme_hss(enhancement_list, icme_list, add_all_icme=True):
    """
    Label solar wind speed enhancements as CMEs (Coronal Mass Ejections) and HSSs (High-Speed Streams) to based on a
    given list of ICMEs observed at Earth, by default the Richardson and Cane ICME list.

    Parameters
    ----------
    enhancement_list : pandas.DataFrame
        DataFrame containing information about solar wind speed enhancements, including start and end times.
    icme_list : pandas.DataFrame
        DataFrame containing information about ICMEs (Interplanetary Coronal Mass Ejections) observed at Earth.
    add_all_icme : bool, optional
        Default: True. Indicates whether to add ICME plasma times to the CME list, even if the disturbance is not
        classified as a solar wind speed enhancement.

    Returns
    -------
    tuple
        A tuple containing three DataFrames:
        1. Updated enhancement list with CME flag,
        2. DataFrame of identified CMEs,
        3. DataFrame of identified HSSs.
    """

    cme_list = []
    hss_list = []
    enhancement_cme_flag = []

    for j in range(enhancement_list.shape[0]):
        # create binary vector of ICMEs that occur inside or 2 days before an enhancement
        icme_bin = (enhancement_list.loc[j, 'start'] - timedelta(hours=48) <= icme_list.loc[:, 'end']) & \
                   (enhancement_list.loc[j, 'end'] >= icme_list.loc[:, 'start'])
        icme_bin = icme_bin.to_numpy()

        # if there is an ICME occurring, mark that enhancement as CME
        if np.sum(icme_bin) > 0:
            cme_list.append(j)
            enhancement_cme_flag.append(True)

            # then merge enhancement intervals and ICME intervals (possibly multiple) and delete that ICME from the ICME list
            if np.sum(icme_bin) == 1:  # if there is a single ICME inside an enhancement interval:

                # if ICME starts before enhancement, set start of enhancement to the start of the ICME
                if icme_list.loc[icme_bin, 'start'].values < enhancement_list.loc[j, 'start']:
                    start_icme = pd.to_datetime(icme_list.loc[icme_bin, 'start'].values[0])
                    # if ICME intersects with previous enhancement, adjust also end of previous enhancement
                    if j > 0 and enhancement_list.loc[j - 1, 'end'] >= start_icme:
                        enhancement_list.loc[j - 1, 'end'] = start_icme - timedelta(hours=1)
                    enhancement_list.loc[j, 'start'] = start_icme

                # if ICME ends after enhancement, set end of enhancement to the end of the ICME
                if icme_list.loc[icme_bin, 'end'].values > enhancement_list.loc[j, 'end']:
                    end_icme = pd.to_datetime(icme_list.loc[icme_bin, 'end'].values[0])
                    # if ICME intersects with next enhancement, adjust also start of next enhancement
                    if j < (len(enhancement_list) - 1) and enhancement_list.loc[j + 1, 'start'] <= end_icme:
                        enhancement_list.loc[j + 1, 'start'] = end_icme + timedelta(hours=1)
                    enhancement_list.loc[j, 'end'] = end_icme

                # drop processed ICMEs from the ICME list
                icme_list = icme_list.drop(index=np.where(icme_bin == True)[0][0])

            else:  # if there are multiple ICMEs inside an enhancement interval:
                icmes = icme_list.loc[icme_bin, :]

                # if first ICME starts before enhancement, set start of enhancement to the start of the ICME
                if icmes.loc[icmes.index[0], 'start'] < enhancement_list.loc[j, 'start']:
                    start_icme = icmes.loc[icmes.index[0], 'start']
                    # if ICME intersects with previous enhancement, adjust also end of previous enhancement
                    if j > 0 and enhancement_list.loc[j - 1, 'end'] >= start_icme:
                        enhancement_list.loc[j - 1, 'end'] = start_icme - timedelta(hours=1)
                    enhancement_list.loc[j, 'start'] = start_icme

                # if last ICME ends after enhancement, set end of enhancement to the end of the ICME
                if icmes.loc[icmes.index[-1], 'end'] >= enhancement_list.loc[j, 'end']:
                    end_icme = icmes.loc[icmes.index[-1], 'end']
                    # if ICME intersects with next enhancement, adjust also start of next enhancement
                    if j < (len(enhancement_list) - 1) and enhancement_list.loc[j + 1, 'start'] < end_icme:
                        enhancement_list.loc[j + 1, 'start'] = end_icme + timedelta(hours=1)
                    enhancement_list.loc[j, 'end'] = end_icme

                # drop processed ICMEs from the ICME list
                icme_list = icme_list.drop(index=np.where(icme_bin == True)[0])
            icme_list = icme_list.reset_index(drop=True)

        # if there is no ICME occurring, mark that enhancement as HSS
        else:
            hss_list.append(j)
            enhancement_cme_flag.append(False)

    enhancement_cme_flag = pd.DataFrame(np.array(enhancement_cme_flag).reshape((-1, 1)), columns=['cme_flag'])

    # assign marked enhancements as CMEs and HSSs
    cme_list = enhancement_list.loc[cme_list, :].reset_index(drop=True)
    hss_list = enhancement_list.loc[hss_list, :].reset_index(drop=True)

    # add remaining ICMEs from ICME list to the CME list by excluding 1 day prior and 2 days after the indicated ICME times
    if add_all_icme:
        for i in range(icme_list.shape[0]):
            icme_list.iloc[i, 0] = icme_list.iloc[i, 0] - timedelta(hours=24)
            icme_list.iloc[i, 1] = icme_list.iloc[i, 1] + timedelta(hours=48)
        cme_list = pd.concat([cme_list, icme_list], axis=0).sort_values('start', axis=0).reset_index(drop=True)

    # Merge overlapping CMEs from CME list
    i = 0
    while i < cme_list.shape[0] - 1:
        drop = False
        if cme_list.loc[i, 'end'] >= cme_list.loc[i + 1, 'start']:
            drop = True
            cme_list.loc[i, 'end'] = cme_list.loc[i + 1, 'end']
        if cme_list.loc[i, 'end'] >= cme_list.loc[i + 1, 'end']:
            drop = True
            cme_list.loc[i, 'end'] = cme_list.loc[i + 1, 'end']
        if drop:
            cme_list = cme_list.drop(index=i + 1).reset_index(drop=True)
        else:
            i = i + 1

    # compute peaks for events in all lists and add them to the lists

    enhancement_list = pd.concat([enhancement_list, enhancement_cme_flag], axis=1)

    return enhancement_list, cme_list, hss_list

def create_cme_hss_list(sw, icme_list,add_all_icme=True):
    """
    Creates lists of CMEs and HSSs based on solar wind speed time series and ICME list.

    Parameters
    ----------
    sw : pandas.Series
        Solar wind speed time series.
    icme_list : pandas.DataFrame
        DataFrame containing information about ICMEs (Interplanetary Coronal Mass Ejections), usually the
        Richardson and Cane ICME list.
    add_all_icme : bool, optional
        Default: True. Indicates whether to add ICME plasma times to the CME list, even if the disturbance is not
        classified as a solar wind speed enhancement.

    Returns
    -------
    tuple
        A tuple containing three DataFrames:
        1. DataFrame of identified CMEs,
        2. DataFrame of identified HSSs,
        3. Updated enhancement list with CME flag.
    """

    enhancement_list = evaluation.find_peaks_in_time_series(sw)

    # Filter ICME list based on the time interval of solar wind speed time series
    interval_bin = (sw.index[0] <= icme_list.loc[:, 'end']) & (sw.index[-1] >= icme_list.loc[:, 'start'])
    icme_list = icme_list.iloc[interval_bin.to_numpy(), :].reset_index(drop=True)

    # Assign CMEs and HSSs to enhancements
    enhancement_list, cme_list, hss_list = assign_cme_hss(enhancement_list, icme_list,add_all_icme=add_all_icme)

    return cme_list, hss_list, enhancement_list



def read_cme_hss_list(path):
    """
    Reads CME, HSS, and enhancement lists from csv files or generates them if not available.

    Parameters
    ----------
    path : str
        Path to the directory containing the pickle files.

    Returns
    -------
    tuple
        A tuple containing three DataFrames:
        1. DataFrame of CMEs,
        2. DataFrame of HSSs,
        3. Combined DataFrame solar wind speed enhancements.
    """

    try:
        cme_list = pd.read_csv(path + '/data/datasets/enhancements/cme_list.csv', index_col=0, parse_dates=[1, 3, 4, 5])
        hss_list = pd.read_csv(path + '/data/datasets/enhancements/hss_list.csv', index_col=0, parse_dates=[1, 3, 4, 5])
        enhancement_list = pd.read_csv(path + '/data/datasets/enhancements/enhancement_list.csv', index_col=0,
                                       parse_dates=[1, 3, 4, 5])

    except FileNotFoundError:
        # read SW in-situ measurements from OMNI and restrict to SW speed
        omni = pd.read_csv(path + '/data/solar_wind/omni.csv',index_col=0, parse_dates=[0])
        sw_speed = omni.loc[:, 'speed']

        # read the Richardson and Cane ICME list restrict to plasma field start and end times
        rc_list = pd.read_csv(path + '/data/solar_wind/icme.csv',index_col=0,parse_dates=[1,2])
        rc_list_start_end = rc_list.loc[:, ['ICME Plasma/Field Start Y/M/D (UT)', 'ICME Plasma/Field End Y/M/D (UT)']]
        rc_list = pd.DataFrame(rc_list_start_end.values, columns=['start', 'end'])

        # create a list of solar storms from the SW speed time series, and classify as CMEs and HSSs
        cme_list, hss_list, enhancement_list = create_cme_hss_list(sw_speed, rc_list,add_all_icme=True)

        cme_list.to_csv(path + '/data/datasets/enhancements/cme_list.csv', index=True)
        hss_list.to_csv(path + '/data/datasets/enhancements/hss_list.csv',index=True)
        enhancement_list.to_csv(path + '/data/datasets/enhancements/enhancement_list.csv', index=True)

    return cme_list, hss_list, enhancement_list



def read_ml_dataset(path, grid):
    """
    Reads machine learning dataset from csv file or generates it if not available.

    Parameters
    ----------
    path : str
        Path to the directory containing the pickle file.
    grid : str
        Resolution of the grid for the dataset.

    Returns
    -------
    tuple
        A tuple containing features (X) and output (Y) of the dataset.
    """

    try:
        data = pd.read_csv(path + '/data/datasets/ml_data/{}.csv'.format(grid),index_col=0,parse_dates=[0])
        Y = data.iloc[:,0]
        X = data.iloc[:,1:]
    except FileNotFoundError:
        X, Y = create_ml_dataset(grid)
        data = pd.concat([Y, X], axis=1)
        data.to_csv(path + '/data/datasets/ml_data/{}.csv'.format(grid), index=True)

    return X, Y



def extract_baseline_features(X,grid):
    """
    Extract baseline features from the input data, i.e., the coronal hole area of the central grid cell(s) and the
    solar wind speed 27 days ago.

    Parameters
    ----------
    X : DataFrame
        The input data.

    Returns
    -------
    X_base : DataFrame
        DataFrame containing the extracted baseline features.
    """
    # If X is the 2x2 array of the data split, apply the function to each of the entries.
    if np.array_equal(X.index, pd.Index(['no_cme', 'with_cme'])):
        for idx in X.index:
            for col in X.columns:
                X.loc[idx, col] = extract_baseline_features(X.loc[idx, col],grid)
        X_base = X
    else:
        # Get grid resolution and compute location of equator and central meridian.
        grid = grid.split('x')
        M = int(grid[0])
        N = int(grid[1])
        equator = M/2
        meridian = N/2
        # Collect and add coronal hole area features around the intersection of equator and central meridian
        area_features = []
        for col in X.columns:
            if col[0] == 'S' and col[-2] == '4':
                idx_split = col.split(',')
                m = int(idx_split[0].split('[')[1])
                n = int(idx_split[1].split(']')[0])

                if equator <= m <= equator+0.5:
                    if meridian <= n <= meridian + 1:
                        area_features.append(col)

        area = X.loc[:, area_features].sum(axis=1)

        # Add solar wind speed 27 days ago to the coronal hole area four days ago
        sw_27 = X.loc[:, 'speed(-27)']
        X_base = pd.concat([sw_27, area], axis=1)

        # Rename columns
        X_base = pd.DataFrame(X_base.values, index=X_base.index, columns=['speed(-27)', 'A(-4)'])

    return X_base


