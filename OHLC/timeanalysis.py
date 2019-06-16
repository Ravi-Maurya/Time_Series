import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf


class Timeseries(object):
    """docstring for Timeseries"""

    def __init__(self):
        init_notebook_mode(connected=True)
        super(Timeseries, self).__init__()

    def test_stationarity(data, date=None, var=None, window=7):
        timeseries = data[var]
        datestamp = data[date]

        rolmean = timeseries.rolling(window).mean()
        rolstd = timeseries.rolling(window).std()

        trace_mean = go.Scatter(x=datestamp, y=rolmean,
                                name="Rolling Mean", line=dict(color='#17BECF'), opacity=0.8)
        trace_std = go.Scatter(x=datestamp, y=rolstd,
                               name="Rolling Std", line=dict(color='#7F7F7F'), opacity=0.8)
        trace_close = go.Scatter(x=datestamp, y=timeseries,
                                 name=var, line=dict(color='#FFA500'), opacity=0.8)
        plot = [trace_mean, trace_std, trace_close]
        iplot(plot, filename='Rolling Statistics')
        # Adfuler Test
        print('*' * 30)
        print('H0: Not Stationary\nH1: Stationary')
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        # Kpss Test
        print('*' * 30)
        print('H0: Stationary\nH1: Not Stationary')
        print ('Results of KPSS Test:')
        dftest = kpss(timeseries, regression='c')
        dfoutput = pd.Series(dftest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
        for key, value in dftest[3].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    def plot_ohlc_candlestick(data, date, open, high, low, close):
        trace = go.Candlestick(x=data[date],
                               open=data[open], high=data[high],
                               low=data[low], close=data[close],
                               name="CandleStick")

        plot = [trace]
        iplot(plot, filename='OHLC_CandleStick')

    def plot_ohlc_line(data, date, open, high, low, close):
        trace_open = go.Scatter(x=data[date], y=data[open],
                                name="Open", line=dict(color='#0EB0F2'), opacity=0.8)
        trace_high = go.Scatter(x=data[date], y=data[high],
                                name="High", line=dict(color='#39F20E'), opacity=0.8)
        trace_low = go.Scatter(x=data[date], y=data[low],
                               name="Low", line=dict(color='#F60505'), opacity=0.8)
        trace_close = go.Scatter(x=data[date], y=data[close],
                                 name="Close", line=dict(color='#F26A0E'), opacity=0.8)

        plot = [trace_open, trace_high, trace_low, trace_close]
        iplot(plot, filename='OHLC_Line')

    def plot_acf_pacf(data, nlag=7):
        timeseries = data.dropna()
        lag_acf = acf(timeseries, nlags=nlag)
        lag_pacf = pacf(timeseries, nlags=nlag, method='ols')
        xaxis = np.arange(len(lag_acf))

        trace_acf = go.Scatter(x=xaxis, y=lag_acf,
                               name="ACF", line=dict(color='#0EB0F2'), opacity=0.8)
        trace_pacf = go.Scatter(x=xaxis, y=lag_pacf,
                                name="PACF", line=dict(color='#F26A0E'), opacity=0.8)
        trace_min = go.Scatter(x=xaxis, y=-1.96 / np.sqrt(len(timeseries)) * np.ones(len(lag_acf)),
                               name="Lower Bound", line=dict(color='#808080'), opacity=0.8)
        trace_plus = go.Scatter(x=xaxis, y=1.96 / np.sqrt(len(timeseries)) * np.ones(len(lag_acf)),
                                name="Upper Bound", line=dict(color='#808080'), opacity=0.8)

        plot = [trace_min, trace_plus, trace_acf, trace_pacf]
        iplot(plot, filename='ACF and PACF')
