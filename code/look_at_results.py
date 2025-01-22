import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

def plot_ts(pred, obs, peaks, cme, grid, dates=None):
    """
    Plot time series of predicted and observed solar wind speed, including CME intervals and peak associations.

    Parameters
    ----------
    pred : pandas.Series
        Predicted solar wind speed time series.
    obs : pandas.Series
        Observed solar wind speed time series.
    peaks : pandas.DataFrame
        DataFrame containing HSS peak information for observed and predicted time series.
    cme : pandas.DataFrame
        DataFrame containing CME start and end times.
    grid : str
        Identifier for the grid or specific run.
    dates : list of datetime, optional
        List of two datetime objects specifying the zoom window. Default is None, which sets the dates to a specific range.
    """

    if dates is None:
        dates = [datetime(2016, 9, 15, 0), datetime(2017, 3, 27, 0)]
    dir_path = os.path.dirname(os.getcwd())

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(21 * cm, 10 * cm))

    # restrict to specified date range
    zoom_from = dates[0]
    zoom_to = dates[1]
    obs_zoom = obs[zoom_from:zoom_to]
    pred_zoom = pred[zoom_from:zoom_to]
    Y = pd.concat([obs_zoom, pred_zoom], axis=0)

    ax.plot(obs_zoom, 'k', label='Obs.')
    ax.plot(pred_zoom, 'tab:red', label='Pred.')


    # plot CME disturbance intervals
    binary_cme = ((cme.loc[:, 'end'] >= obs_zoom.index[0]) & (cme.loc[:, 'start'] <= obs_zoom.index[-1])).to_numpy()
    cme_temp = cme.iloc[binary_cme, :]
    for j in range(cme_temp.shape[0]):
        ax.axvspan(cme_temp.loc[cme_temp.index[j], 'start'],
                      cme_temp.loc[cme_temp.index[j], 'end'], facecolor='red', alpha=0.3,
                      label='CME dist.')

    # Plot HSS peak data and binary classification (hit/miss/false alarm)
    obs_peaks = peaks.loc['obs']
    pred_peaks = peaks.loc['pred']

    binary = obs_peaks.loc[:, 'hit'].to_numpy()
    if len(binary) > 1:
        ax.plot(obs_peaks.index[binary], np.max(Y) * 1.05 * np.ones(np.sum(binary)), 'P', color='green',
                   label='Hit')
    binary = obs_peaks.loc[:, 'miss'].to_numpy()
    if len(binary) > 1:
        ax.plot(obs_peaks.index[binary], np.max(Y) * 1.05 * np.ones(np.sum(binary)), 'X', color='red',
                   label='Miss')
    binary = pred_peaks.loc[:, 'false_alarm'].to_numpy()
    if len(binary) > 1:
        ax.plot(pred_peaks.index[binary], np.max(Y) * 1.05 * np.ones(np.sum(binary)), 'o', color='orange',
                   label='False Alarm')

    ax.plot(obs_peaks.loc[:, 'peak_value'], 'v', color='tab:blue', label='Obs. HSS')
    ax.plot(pred_peaks.loc[:, 'peak_value'], 'v', color='tab:orange', label='Pred. HSS')

    ax.set_xlim([obs_zoom.index[0], obs_zoom.index[-1]])
    ax.set_ylim([np.min(Y) * 0.97, np.max(Y) * 1.08])
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    # set legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.25),
                   fancybox=True, ncol=4, fontsize=13)

    ax.set_ylabel('Solar wind speed (km/s)', fontsize=13)
    ax.set_xlabel('Date', fontsize=13)

    # save into figures folder
    fig.savefig(dir_path + '/results/figures/ts_zoom_{}.png'.format(grid), bbox_inches='tight', dpi=500)
    return


############################################
# Configure plot here.
############################################

# specify grid-based model
grid = '4x3'

# set time interval that should be plotted
time_interval = [datetime(2016, 9, 15, 0), datetime(2017, 3, 27, 0)]

# plot the predictions with or without the distribution transformation
distribution_transformation = True

############################################
# End of configuration.
############################################

dir_path = os.path.dirname(os.getcwd())

cme_list = pd.read_csv(dir_path + '/data/datasets/enhancements/cme_list.csv',index_col=0,parse_dates=[1, 3, 4, 5])
if not os.path.exists(dir_path + '/results/figures'):
    os.mkdir(dir_path + '/results/figures')

path = dir_path + '/results/model_pred/{}'.format(grid)

# load most recent file
files = os.listdir(path)
files.sort()
data = pickle.load(open(path + '/' + files[-1], "rb"))

if distribution_transformation:
    pred_mode = 'trans_with_cme'
else:
    pred_mode = 'with_cme'

# predictions
pred = data.pred_data.loc[pred_mode,'test']

# observations
obs = data.obs_data.loc['with_cme','test']

if distribution_transformation:
    pred_mode = 'trans'
else:
    pred_mode = 'no_cme'

# HSS peak data
peaks = data.peak_data.loc[pred_mode,:]

plot_ts(pred, obs, peaks, cme_list, grid, dates=time_interval)
plt.show()
