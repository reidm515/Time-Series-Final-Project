import base64
import io
import pandas as pd
from pandas.plotting import *
from pandas import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('agg')
  # Step 1: Read the Dataset
  # Set "Datetime" column to be parsed as a Date-Time object, not plain text.
  # Set Datetime column as index column.
  
def arima_forecast(file, file_gaps, column_name):
  import io
  import base64
  df = pd.read_csv(file, sep=',', skiprows=1, encoding='latin1')
  df = df.dropna(axis='columns', how='all')
  df = df.dropna()
  df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
  df = df.set_index('Datetime')

  df_gaps = pd.read_csv(file_gaps, sep=',', encoding='latin1')
  df_gaps['Datetime'] = pd.to_datetime(df_gaps['Datetime'], format='%Y-%m-%d %H:%M:%S')
  df_gaps = df_gaps.set_index('Datetime')
  
  #plt.xlabel('Date')
  #plt.ylabel('Rainfall')
  #plt.plot(df[column_name])
  
  # From the graph above me can see:
  # Seasonality and Error/Irregularities.
  
  # 8754 number of values recorded per year.
  rolling_avg = df[column_name].rolling(window=12).mean()
  rolling_std = df[column_name].rolling(window=12).std()
  
  orig = plt.plot(df[column_name], label = 'Original')
  mean = plt.plot(rolling_avg, label = 'Rolling Mean')
  std = plt.plot(rolling_std, label = 'Rolling STD')
  
  plt.legend(loc='best')
  plot1 = io.BytesIO()  # here we package the plot as a png image using base64 io, so it can be returned to our flask app and html later
  plt.savefig(plot1, format='png',dpi=300, bbox_inches='tight')
  plot1.seek(0)
  plot1 = base64.b64encode(plot1.getvalue()).decode('utf-8')
  plt.close()
  






  def adcf_test(timeseries):
    timeseries.dropna(inplace=True)
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(timeseries)
    print(dftest, dftest[1])

  adcf_test(df[column_name])

  # p-value: 9.238147877165352e-11
  # number of lags: 64
  # number of observations: 78556

  # Our the output indicates that the time series is stationary, as
  # evidenced by the test statistic being more negative than the
  # critical values and the p-value being very small.



  #plt.show()

  #df.dropna(inplace=True)
  from statsmodels.tsa.stattools import adfuller
  #dftest = adfuller(df)
  #print(dftest, dftest[1])
  
  #rainfall_log_scaled = np.log(df[column_name])
  #plt.plot(rainfall_log_scaled)
  
  # Our Test Statistic: -7.365906491729031.
  # MacKinnon's approximate P-Value: 9.238147877165352e-11
  # Number of lags used: 64
  # Number of observations: 78556
  
  # So in summary, the output indicates that the time series represented by df[column_name] is stationary,
  # as evidenced by the test statistic being more negative than the critical values and the p-value
  # being less than 0.05.
  
  from statsmodels.tsa.stattools import acf
  from statsmodels.tsa.stattools import pacf
  
  from statsmodels.graphics.tsaplots import plot_acf
  from statsmodels.graphics.tsaplots import plot_pacf
  
  mx = 40
  #plot_acf(df[column_name])
  #plot_pacf(df[column_name])
  
  # From both of our ACF and PACF plots, we can set our:
  # Q = 2 or 3
  # P = 1
  # D = 1
  
  dataset_rain = (df[column_name][::-1])
  dataset_rain.tail()
  
  from statsmodels.tsa.arima.model import ARIMA
  
  from sklearn.model_selection import train_test_split
  
  dataset_rain.dropna(inplace=True)
  train, test = train_test_split(dataset_rain, test_size = 0.30, shuffle=False)
  
  model_arima = ARIMA(train, order=(3, 1, 1))
  model_arima_fit = model_arima.fit()
  
  
  
  start_date = train.index[0]
  end_date = train.index[-1]
  predictions = model_arima_fit.predict(start=start_date, end=end_date)
  


  train.head()
  
  from sklearn.metrics import mean_squared_error, mean_absolute_error
  mse = mean_squared_error(train, predictions)
  mae = mean_absolute_error(train, predictions)
  
  resultsone = (f'Test MSE: {mse}')
  resultstwo = (f'Test MSE: {mae}')
  
  # order=(1, 1, 1) = MSE 0.9359331576525654
  
  # order=(2, 1, 1) = Test MSE 0.8375151305843395
  
  # order=(3, 1, 1) = Test MSE 0.8333223767959606
  
  # order=(4, 1, 1) = Test MSE 0.8187191942907253
  
  # order=(5, 1, 1) = Test MSE 0.8097824234195028
  
  # order=(6, 1, 1) = Test MSE: 0.8006828966183165
  # order=(6, 1, 1) = Test MSE: 0.6170634190706661
  
  n = int(len(train) * 0.005)
  plt.plot(predictions, color='blue')               #MAIN GRAPH
  plt.plot(df_gaps[column_name])
  plot2 = io.BytesIO()  # here we package the plot as a png image using base64 io, so it can be returned to our flask app and html later
  plt.savefig(plot2, format='png', dpi=300, bbox_inches='tight')
  plot2.seek(0)
  plot2 = base64.b64encode(plot2.getvalue()).decode('utf-8')
  plt.close()
  
  
  # Check for Autocorrelation using LagPlot()
  lag_plot(df[column_name])
  plot3 = io.BytesIO()  # here we package the plot as a png image using base64 io, so it can be returned to our flask app and html later
  plt.savefig(plot3, format='png', dpi=300, bbox_inches='tight')
  plot3.seek(0)
  plot3 = base64.b64encode(plot3.getvalue()).decode('utf-8')
  plt.close()
  
  # Check for Autocorrelation using Pearson Correlation Coefficient:
  
  series = df[column_name]
  values = DataFrame(series.values)
  dataframe = concat([values.shift(1), values], axis=1)
  dataframe.columns = ['t-1', 't+1']
  result = dataframe.corr()
  print(result)
  
  # This confirms that there us a strong correlation between the observation and the lag = 1 value.
  
  # Use Pandas' Autocorrelation plot to get a better observation visually:
  series.dropna(inplace=True)
  autocorrelation_plot(series)
  plt.show()

  return resultsone, resultstwo, plot1, plot2, plot3
