import numpy as np
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from hampel import hampel
import pickle
import skimage.measure
import math
from copy import copy,deepcopy


def downsample_maps(month, resolution):
    """
    Downsample segmentation maps to a specified resolution.

    Parameters
    ----------
    month : datetime
        The month for which maps are downsampled.
    resolution : int
        The desired resolution of the downsampled maps.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Downsampled segmentation maps and corresponding dates.
    """
    path = os.path.dirname(os.getcwd())

    if resolution > 1024 or resolution < 1 or np.log2(resolution) % 1 > np.finfo(float).eps:
        raise ValueError('Invalid resolution. Possible resolutions are 1024, 512, 256, ..., 1.')

    path_downsampled = path + '/data/segmentation_maps/raw/downsampled/' + str(resolution)
    if not os.path.exists(path_downsampled):
        os.mkdir(path_downsampled)

    # Path to original maps
    path_original = path + '/data/segmentation_maps/raw/original'
    file_name = path_original + '/' + month.strftime('%Y-%m') + '.pickle'
    with open(file_name, 'rb') as f:
        ch_data = pickle.load(f)
        ch_map, ch_date = ch_data

    # Compute downsampling factor
    pooling_factor = int(1024 / resolution)

    ch_map_new = []
    # Downsample using average pooling
    for i in range(len(ch_map)):
        ch_map_new.append(skimage.measure.block_reduce(ch_map[i], (pooling_factor, pooling_factor), np.mean))

    ch_map_new = np.array(ch_map_new, dtype=float)

    file_name = path_downsampled + '/' + month.strftime('%Y-%m') + '.pickle'
    pickle.dump([ch_map_new, ch_date], open(file_name, 'wb'))

    return ch_map_new, ch_date

def cut_maps(ch_map):
    """
    Cut segmentation maps to remove the outer region.

    Parameters
    ----------
    ch_map : numpy.ndarray
        Segmentation maps to be cut.

    Returns
    -------
    numpy.ndarray
        Cut segmentation maps.
    """
    resolution = ch_map[0].shape[0]
    r = 960  # 960 arcsec radius of sun;
    # Calculate area of each pixel (1024 px have 2.4 arcsec per pixel)
    arcsec_per_pixel = (1024 / resolution) * 2.4

    cutout = math.floor(2 * r / arcsec_per_pixel)  # Diameter in pixels
    buffer = int((resolution - cutout) / 2)

    ch_map_new = []
    for i in range(len(ch_map)):
        ch_map_new.append(ch_map[i][buffer:(ch_map[0].shape[0] - buffer), buffer:(ch_map[0].shape[0] - buffer)])
    ch_map_new = np.array(ch_map_new, dtype=float)

    return ch_map_new


def curate_maps(months,resolution):
    """
    Curate segmentation maps, i.e. downsample, cut off space around solar disk, delete outliers, fill gaps.

    Parameters
    ----------
    months : list of datetime
        List of months for which maps are curated.
    resolution : int
        The desired resolution of the curated maps.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Curated segmentation maps and corresponding dates.
    """
    path = os.path.dirname(os.getcwd())

    # The variable months is expected to be a list of datetime objects of different months

    # Make a list of files that have to be loaded. In order to find outliers and interpolate missing dates at the beginning
    # and the end of months, also the preceding and following months have to be loaded.

    months_extended = deepcopy(months)
    for month in months:
        months_extended.append(month - relativedelta(months=1))
        months_extended.append(month + relativedelta(months=1))
    months_extended = np.array(sorted(list(set(months_extended))))

    months_extended = months_extended[(months_extended >= datetime(2010,5,1)) & (months_extended <= datetime(2019,12,31))]

    ch_map = []
    ch_date = []

    # Load downsampled maps, or compute them if not available yet. Then cut off space around the solar disk.
    path_downsampled = path + '/data/segmentation_maps/raw/downsampled/' + str(resolution)
    for month in months_extended:
        file_name = path_downsampled + '/' + month.strftime('%Y-%m') + '.pickle'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                ch_data = pickle.load(f)
                ch_map.extend(cut_maps(ch_data[0]))
                ch_date.extend(ch_data[1])
        else:
            print('Downsampling {}.'.format(month.strftime('%Y-%m')))
            ch_data = downsample_maps(month,resolution)

            ch_date.extend(ch_data[1])
            ch_map.extend(cut_maps(ch_data[0]))

    ch_map = np.array(ch_map)
    ch_date = np.array(ch_date)

    # sort
    sort_index = np.argsort(ch_date)
    ch_date = ch_date[sort_index]
    ch_map = ch_map[sort_index]

    # delete bad data by outlier detection
    total_area = []
    for i in range(len(ch_map)):
        total_area.append(np.sum(ch_map[i]))
    total_area = pd.Series(total_area)
    outlier_indices = hampel(total_area, window_size=10, n=5)

    binary_mask = np.full(len(ch_map), True)
    binary_mask[outlier_indices] = False
    ch_map = ch_map[binary_mask]
    ch_date = ch_date[binary_mask]

    # find missing dates
    missing_dates = []
    for month in months_extended:
        expected_date = datetime(month.year,month.month,1,0)
        dates_month = ch_date[(ch_date >= expected_date) & (ch_date < (expected_date + relativedelta(months=1)))]
        for i in range(len(dates_month)):
            actual_date = dates_month[i]
            while not actual_date == expected_date:
                missing_dates.append(expected_date)
                expected_date = expected_date + timedelta(hours=1)
            expected_date = expected_date + timedelta(hours=1)
        # account for the case that there are missing dates between the last map and the end of the month
        while expected_date < (datetime(month.year,month.month,1,0) + relativedelta(months=1)):
            missing_dates.append(expected_date)
            expected_date = expected_date + timedelta(hours=1)


    # replace missing dates by nan maps
    ch_date = np.concatenate([ch_date, missing_dates])
    sort_index = np.argsort(ch_date)
    ch_date = ch_date[sort_index]

    nan_array = np.empty((len(missing_dates), ch_map[0].shape[0], ch_map[0].shape[1]))
    nan_array[:] = np.nan

    ch_map = np.concatenate([ch_map, nan_array])
    ch_map = ch_map[sort_index]

    # interpolate maps
    sh = ch_map.shape
    ch_map = ch_map.reshape((len(ch_map), -1))
    ch_map = pd.DataFrame(ch_map)
    ch_map = ch_map.interpolate(limit_direction='both')

    ch_map = ch_map.values.reshape(sh)

    # save curated maps in monthly files
    curated_dates = []
    curated_maps = []
    path_curated = path + '/data/segmentation_maps/curated'
    if not os.path.exists(path_curated):
        os.mkdir(path_curated)

    for month in months:
        start_month = datetime(month.year, month.month, 1, 0)
        end_month = start_month + relativedelta(months=1)
        binary = (ch_date >= start_month) & (ch_date < end_month)
        dates_month = ch_date[binary]
        maps_month = ch_map[binary]
        curated_dates.extend(dates_month)
        curated_maps.extend(maps_month)
        file_name = path_curated + '/' + month.strftime('%Y-%m') + '.pickle'
        pickle.dump([maps_month,dates_month], open(file_name, 'wb'))

    return curated_maps, curated_dates


def extract_areas(ch_map,ch_date, arcsec_per_pixel, angle, latitude_max=90, n_latitude_intervals=4, longitude_max=90,
                 n_longitude_intervals=3, symmetric_to_equator=False):
    """
    Extracts coronal hole area from segmentation maps using a grid.

    Parameters
    ----------
    ch_map : array_like
        Segmentation map data.
    ch_date : array_like
        Dates associated with the map data.
    arcsec_per_pixel : float
        Arcseconds per pixel.
    angle : array_like
        Latitudinal helioshperic angle of Earth (alpha).
    latitude_max : float, optional
        Maximum latitude, by default 90.
    n_latitude_intervals : int, optional
        Number of latitude intervals, by default 4.
    longitude_max : float, optional
        Maximum longitude, by default 90.
    n_longitude_intervals : int, optional
        Number of longitude intervals, by default 3.
    symmetric_to_equator : bool, optional
        Whether to symmetricize to equator, by default False.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the extracted areas.
    """

    # it is assumed that ch_map is centered and quadratic
    # calculate factor for area of each pixel
    dim_pixel = ch_map[0].shape[0]
    middle_point = dim_pixel / 2 - 0.5
    r = 960 # radius of sun in arcsec

    # longitudinal grid lines
    phi_bounds = np.linspace(-longitude_max,longitude_max,n_longitude_intervals+1)
    phi_bounds = np.deg2rad(phi_bounds)

    # latitudinal grid lines
    theta_bounds = np.linspace(90-latitude_max,90+latitude_max,n_latitude_intervals+1)
    theta_bounds = np.deg2rad(theta_bounds)

    # calculate latitudinal and longitudinal bounds for each grid cells
    sector_bounds = np.zeros(((len(theta_bounds)-1)*(len(phi_bounds)-1),4),dtype=float)
    for i in range(len(theta_bounds)-1):
        for j in range(len(phi_bounds)-1):
            sector_bounds[i*(len(phi_bounds)-1)+j,:] = np.array([theta_bounds[i],theta_bounds[i+1],phi_bounds[j],phi_bounds[j+1]])


    area = []
    n_samples = len(ch_date)
    old_angle_hour = 999

    # Calculate for each segmentation map (each hour) and each pixel therein the grid cell it belongs to, taking into
    # account the angle between the equatorial plane of the Sun and the current position of Earth.
    for hour in range(n_samples):
        if hour % 10 == 0:
            print('\rProgression: {:.0%}.'.format(hour/n_samples),end='',flush=True)

        angle_hour = angle.iloc[hour]

        # recompute grid cells only if the angle changes
        if angle_hour != old_angle_hour:

            cell_array = np.zeros((dim_pixel, dim_pixel), dtype=int)
            cells = np.zeros(len(sector_bounds), dtype=object)
            for i in range(len(cells)):
                cells[i] = []

            # iterate over all pixels and compute their heliospheric coordinates to match them to one of the grid cells
            for i in range(dim_pixel):
                for j in range(dim_pixel):
                    # calculate cartesian coordinates of pixels of solar disk in arcsec, assuming the center of the sun to be (0,0,0).
                    # y = width, z = height. x as the coordinate of the direction opposite to the line of sight can be calculated
                    # given the constant radius r = 960 arcsec with r = sqrt(x^2+y^2+z^2).
                    y = (j - middle_point) * arcsec_per_pixel
                    z = (middle_point - i) * arcsec_per_pixel
                    val = r**2 - y**2 - z**2
                    if val >= 0:
                        x = np.sqrt(val)
                    else:
                        x = np.nan

                    # rotate pixels in opposite direction of angle to align
                    # matrix that rotates 3D point around y-axis
                    alpha = np.deg2rad(- angle_hour)
                    R = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                                  [0, 1, 0],
                                  [-np.sin(alpha), 0, np.cos(alpha)]])
                    vec = np.array([x, y, z]).reshape(-1,1)
                    rot = R @ vec
                    rot.reshape(-1)
                    x = rot[0]
                    y = rot[1]
                    z = rot[2]

                    # given the cartesian coordinates of the pixels, the spherical coordinates of each pixel can be calculated.
                    # theta = inclination, phi = azimuth. r = radius is known.
                    # theta = arccos(z/r), phi = atan2(y,x)
                    theta = np.arccos(z/r)
                    phi = np.arctan2(y,x)

                    # given the latitudinal and longitudinal bounds for each cell, group pixels
                    # if they belong to the same cell
                    for k in range(len(sector_bounds)):
                        if sector_bounds[k,0] <= theta <= sector_bounds[k,1]:
                            if sector_bounds[k, 2] <= phi <= sector_bounds[k, 3]:
                                cell_array[i,j] = k
                                cells[k].append([i,j])

            bin = np.zeros(len(cells),dtype=bool)
            # associate cells with same absolute latitude and same longitude
            if symmetric_to_equator:
                for i in range(math.floor(n_latitude_intervals/2)):
                    for j in range(n_longitude_intervals):
                        cells[i*n_longitude_intervals+j].extend(cells[(n_latitude_intervals-1-i)*n_longitude_intervals+j])
                        bin[(n_latitude_intervals - 1 - i) * n_longitude_intervals + j] = True
            cells = cells[np.invert(bin)]

        # for each segmentation map (each hour), sum the pixel values belonging to each grid cell
        area_hour = np.zeros((len(cells),1),dtype=float)
        for i in range(len(cells)):
            if len(cells[i]) > 0:
                index_list = tuple(np.array(cells[i]).T)
                map_local = ch_map[hour]
                area_hour[i] = np.sum(map_local[index_list])

        area.append(area_hour)
        old_angle_hour = angle_hour


    area = np.concatenate(area,axis=1)
    area = pd.DataFrame(area.T,index=ch_date)
    print('\rProgression: {:.0%}.'.format(1), end='\n', flush=True)
    return area



def calculate_ch_area(months,resolution,angle,grid):
    """
    Calculate the area of coronal holes in segmentation maps.

    Parameters
    ----------
    months : list of datetime.datetime
        List of months.
    resolution : int
        Resolution of the segmentation map.
    angle : pandas.Series
        Heliospheric latitudinal angle of Earth (alpha)).
    grid : str
        Grid resolution in format 'MxN'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the calculated coronal hole areas.
    """
    path = os.path.dirname(os.getcwd())

    path_curated = path + '/data/segmentation_maps/curated/'

    M, N = grid.split('x')
    grid = (int(M), int(N))

    ch_map = []
    ch_date = []
    missing_months = []

    # Iterate over all months
    for current_month in months:
        file_name = path_curated + '/' + current_month.strftime('%Y-%m') + '.pickle'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                ch_data = pickle.load(f)

                ch_map.extend(ch_data[0])
                ch_date.extend(ch_data[1])
        else:
            # collect months that need to be curated and then curate all at once
            missing_months.append(current_month)

    # Collect data for missing months
    if len(missing_months) > 0:
        for month in missing_months:
            print('Curating {}.'.format(month.strftime('%Y-%m')))
            ch_data = curate_maps([month], resolution)
            ch_map.extend(ch_data[0])
            ch_date.extend(ch_data[1])


    ch_map = np.array(ch_map)
    ch_date = np.array(ch_date)

    # Sort data by date
    sort_index = np.argsort(ch_date)
    ch_date = ch_date[sort_index]
    ch_map = ch_map[sort_index]


    # calculate arcsec of each pixel (1024 px have 2.4 arcsec per pixel)
    arcsec_per_pixel = (1024 / resolution) * 2.4

    # Set up directory for saving area data
    path_area = path + '/data/ch_area/{}x{}'.format(grid[0],grid[1])
    if not os.path.exists(path_area):
        os.mkdir(path_area)

    # Generate column names for the DataFrame
    col_names = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            col_names.append('A[{},{}]'.format(i + 1, j + 1))

    # Initialize DataFrame for storing area data
    area = pd.DataFrame([],[])
    for current_month in months:
        print('Calculating area for {}.'.format(current_month.strftime('%Y-%m')))
        binary = (ch_date >= datetime(current_month.year,current_month.month,1,0)) & (ch_date < datetime(current_month.year,current_month.month,1,0)+relativedelta(months=1))
        maps_temp = ch_map[binary]
        dates_temp = ch_date[binary]
        binary = (angle.index >= datetime(current_month.year,current_month.month,1,0)) & (angle.index < datetime(current_month.year,current_month.month,1,0)+relativedelta(months=1))
        angle_temp = angle[binary]

        # Extract areas for the current month
        a = extract_areas(maps_temp, dates_temp, arcsec_per_pixel,angle_temp,n_latitude_intervals=grid[0],n_longitude_intervals=grid[1])

        # Concatenate the areas to the main DataFrame
        area = pd.concat([area,a])
        area.columns = col_names
        a.columns = col_names

        # Save monthly extracted areas
        month_str = current_month.strftime('%Y-%m')
        file_name = path_area + '/' + month_str + '.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(a, f)

    return area

