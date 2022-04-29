# -*- coding: utf-8 -*-
"""Decomposition-AA-RNN-Published
"""

import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rcParams

df = pd.read_excel('./dataset/tax-sales-hurricane.csv')
df_dec = df[df.region == 'Orange']
df_dec.set_index('Date', inplace=True)
df_dec = df_dec[['observed']]
df_dec


def anomaly_detection(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh


def AA_decompose(df, period=12, lo_frac=0.6, lo_delta=0.01, thresh=3.5):
    lowess = sm.nonparametric.lowess
    _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)
    observed = np.asanyarray(df).squeeze()
    trend = lowess(observed, [x for x in range(len(observed))],
                   frac=lo_frac,
                   delta=lo_delta * len(observed),
                   return_sorted=False)
    detrended = observed / trend
    period = min(period, len(observed))
    period_median = np.array([pd_nanmean(detrended[i::period])
                              for i in range(period)])
    seasonal = np.tile(period_median, len(observed) //
                       period + 1)[:len(observed)]
    resid_inter = detrended / seasonal
    resid_inter[0] = 1
    resid = resid_inter.copy()
    anomalies = resid_inter.copy()
    b = anomaly_detection(resid, thresh=thresh)
    for j in range(len(b)):
        if b[j] == True:
            resid[j] = 1
        if b[j] == False:
            anomalies[j] = 1
    results = list(map(_pandas_wrapper, [seasonal, trend, resid, observed]))
    fig, axes = plt.subplots(5, 1, sharex=True)

    fig.tight_layout()
    axes[0].plot(observed)
    axes[0].set_ylabel('Observed')
    axes[1].plot(trend)
    axes[1].set_ylabel('Trend')
    axes[2].plot(seasonal)
    axes[2].set_ylabel('Seasonal')
    axes[4].plot(anomalies, color='r')
    axes[4].set_ylabel('Anomalies')
    axes[4].set_xlabel('Time')

    axes[3].plot(resid)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    return trend, seasonal, anomalies, resid 


plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

params = {
    'axes.labelsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
rcParams.update(params)

trend, seasonal, anomalies, resid  = AA_decompose(df_dec, period=12, thresh=2)
