# Time-Series-Predictions

Time Series is a collection of data points collected at constant time intervals. These are analyzed to determine the long term trend so as to forecast the future or perform some other form of analysis. But what makes a Time Series different from say a regular regression problem? There are 2 things:
<br>
                  <pre><b>1.</b> It is time dependent. So the basic assumption of a linear regression model that the observations are independent doesn’t hold in this case.<br><b>2.</b> Along with an increasing or decreasing trend, most TS have some form of seasonality trends, i.e. variations specific to a particular time frame. For example, if you see the sales of a woolen jacket over time, you will invariably find higher sales in winter seasons.</pre>

A time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. Examples of time series are heights of ocean tides, counts of sunspots, and the daily closing value of the Dow Jones Industrial Average.

Time series are very frequently plotted via line charts. Time series are used in statistics, signal processing, pattern recognition, econometrics, mathematical finance, weather forecasting, earthquake prediction, electroencephalography, control engineering, astronomy, communications engineering, and largely in any domain of applied science and engineering which involves temporal measurements.

Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the use of a model to predict future values based on previously observed values. While regression analysis is often employed in such a way as to test theories that the current values of one or more independent time series affect the current value of another time series, this type of analysis of time series is not called "time series analysis", which focuses on comparing values of a single time series or multiple dependent time series at different points in time. Interrupted time series analysis is the analysis of interventions on a single time series.

Time series data have a natural temporal ordering. This makes time series analysis distinct from cross-sectional studies, in which there is no natural ordering of the observations (e.g. explaining people's wages by reference to their respective education levels, where the individuals' data could be entered in any order). Time series analysis is also distinct from spatial data analysis where the observations typically relate to geographical locations (e.g. accounting for house prices by the location as well as the intrinsic characteristics of the houses). A stochastic model for a time series will generally reflect the fact that observations close together in time will be more closely related than observations further apart. In addition, time series models will often make use of the natural one-way ordering of time so that values for a given period will be expressed as deriving in some way from past values, rather than from future values.

## Data Source
<b>Reliance Industries Limited</b>
> <b>Chairman & MD</b>: Mr. Mukesh Dhirubhai Ambani<br>
> <b>Sector</b>: Energy<br>
> <b>Industry</b>: Oil & Gas Refining & Marketing<br>
> <b>Location</b>: Mumbai, India<br>
> <b>Data API</b>: Yahoo Finance<br>
> <b>Time Span of 5 Years</b>: 1st Jan 2014 - 31 Dec 2018<br>

Reliance Industries Limited (RIL) is an Indian conglomerate holding company headquartered in Mumbai, Maharashtra, India. Reliance owns businesses across India engaged in energy, petrochemicals, textiles, natural resources, retail, and telecommunications. Reliance is one of the most profitable companies in India, the largest publicly traded company in India by market capitalization, and the second largest company in India as measured by revenue after the government-controlled Indian Oil Corporation. On 18 October 2007, Reliance Industries became the first Indian company to breach $100 billion market capitalization. The company is ranked 203rd on the Fortune Global 500 list of the world's biggest corporations as of 2017. It is ranked 8th among the Top 250 Global Energy Companies by Platts as of 2016. Reliance continues to be India’s largest exporter, accounting for 8% of India's total merchandise exports with a value of Rs 147,755 crore and access to markets in 108 countries. Reliance is responsible for almost 5% of the government of India's total revenues from customs and excise duty. It is also the highest income tax payer in the private sector in India.<br>
<b>The Data is varying non linearly.<b><br>
![Close Value](https://github.com/Ravi-Maurya/Time-Series-Predictions/blob/master/Plots/ActualCloseValue.png)

## Libraries

> <b>Numpy</b>: 1.16.0<br>
> <b>Pandas</b>: 0.23.4<br>
> <b>Matplotlib</b>: 3.0.2<br>
> <b>Scikit Learn</b>: 0.20.2<br>
> <b>TensorFlow</b>: 1.12.0<br>

# Time Series Models
## Moving Average
In time series analysis, the moving-average model (MA model), also known as moving-average process, is a common approach for modeling univariate time series. The moving-average model specifies that the output variable depends linearly on the current and various past values of a stochastic (imperfectly predictable) term.<br>
Using This Model was not musch of use as the data was varying day by day inconsistently.<br>
![Moving Average Model](https://github.com/Ravi-Maurya/Time-Series-Predictions/blob/master/Plots/MovingAvCloseValue.png)
<hr>

## LSTM
Long short-term memory (LSTM) units are units of a recurrent neural network (RNN). An RNN composed of LSTM units is often called an LSTM network (or just LSTM). A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.<br>
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and vanishing gradient problems that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, hidden Markov models and other sequence learning methods in numerous applications.<br>
![LSTM Close](https://github.com/Ravi-Maurya/Time-Series-Predictions/blob/master/Plots/LSTM_Deeplearning_CloseValue.png)
<br>
Using this model resulted in very good predictions as it only gave loss of about: <b>0.0009</b>
