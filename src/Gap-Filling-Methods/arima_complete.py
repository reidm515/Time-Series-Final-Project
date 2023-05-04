import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Read the Dataset
# Set "Datetime" column to be parsed as a Date-Time object, not plain text.
# Set Datetime column as index column.

# df = pd.read_csv('/content/drive/MyDrive/CSV Hourly Datasets/DAD.csv', sep=',', skiprows=1, encoding='latin1',
df = pd.read_csv('../Hourly Datasets (CSV)/DAD.csv', sep=',', skiprows=1, encoding='latin1',
parse_dates=["Datetime"], index_col=["Datetime"])

# Total Rainfall for Daily & Monthly
tr_D = df['avg'].resample('D').sum()
tr_M = df['avg'].resample('MS').sum()

tr_M.plot()

# Analysing monthly Data
decomposition = sm.tsa.seasonal_decompose(tr_M, model="additive")
fig = decomposition.plot()
plt.show()

# Analysing Daily Data
# “residuals” in a time series model are what is left over after fitting/decomposing a model.

decomposition = sm.tsa.seasonal_decompose(tr_D, model="additive")
fig = decomposition.plot()
plt.show()

# From our results, we can now say that this Dataset is Stationary.

# For further validation on the stationarity of the dataset we use the ADFTest (Augmented Dickey-Fuller).
# If our Dataset has a P-Value of greater then 0.05, it is not stationary.

from statsmodels.tsa.stattools import adfuller
adftest_D = adfuller(tr_D)
print('P-Value of ADFuller test is: ', adftest_D[1])

# As our PValue < 0.05, we can now confirm our dataset is stationary.

# Split the dataset into Training, Testing and Validation sets.

train_start_date = '2013-01-01 11:00:00'
train_end_date = '2019-12-31 23:00:00'
val_start_date = '2020-01-01 00:00:00'
val_end_date = '2021-06-30 23:00:00'
test_start_date = '2021-07-01 00:00:00'
test_end_date = '2021-12-31 09:00:00'

train_start = pd.to_datetime(train_start_date)
train_end = pd.to_datetime(train_end_date)
val_start = pd.to_datetime(val_start_date)
val_end = pd.to_datetime(val_end_date)
test_start = pd.to_datetime(test_start_date)
test_end = pd.to_datetime(test_end_date)

train_df = df[(df.index >= train_start) & (df.index <= train_end)]
val_df = df[(df.index >= val_start) & (df.index <= val_end)]
test_df = df[(df.index >= test_start) & (df.index <= test_end)]

# train_df.head(n = 5)
# val_df.head(n = 5)
# test_df.head(n = 5)

# We now need to find the appropriate P,D,Q Hyperparametres for our ARIMA Model.
# Our first method will be a for loop which will iterate through all possible P,Q,D values and return each's mean-squared error value.

# P - AutoRegression - The correlation with its previous values.
# D - Integrated - The order ie. how many times we want to difference our data.
# Q - The order of the Moving Average - depends on the error of the previous lagged values - we remove trend.

import itertools
from sklearn.metrics import mean_squared_error

p = range(0,8)
q = range(0,8)
d = range(0,2)

# Gives all possible combinations of P,Q,D.
pdq_combination = list(itertools.product(p,q,d))
len(pdq_combination)

rmse = []
order1 = []

# for pdq in pdq_combination:
#   try:
#     model = sm.tsa.arima.ARIMA(train_df['avg'], order = pdq).fit()
#     pred = model.predict(start=len(train_df), end=(len(test_df) - 1))
#     error = np.sqrt(mean_squared_error(test_df, pred))
#     order1.append(pdq)
#     rmse.append(error)
#   except:
#     continue

# We now need to find the appropriate P,D,Q Parametres for our ARIMA Model.
model = sm.tsa.arima.ARIMA(train_df['avg'], order=(5,0,4))
result = model.fit

# Our second method will be using an ACF + PACF Plots to analyse the Autocorrelation and Partial Autocorrelation, which will help us identify the best P,Q,D values.
# UNdo COMMENT AFTER
fig, axes = plt.subplots(2,3)
axes[0,0].plot(tr_M)
plot_acf(tr_M, ax = axes[0,1])
plot_acf(tr_M.diff().dropna(), ax = axes[0,2])
axes[1, 0].plot(tr_M)
plot_pacf(tr_M.diff().dropna(), ax = axes[1, 1])
plot_pacf(tr_M.dropna(), ax = axes[1,2])
plt.show()

# The graphs above for Autocorrelation & Partial Autocorrelation show a damped exponential pattern.

train_df_copy = train_df.copy()
# train_df_copy['avg'] = train_df_copy['avg'].dropna(inplace = True)
print(train_df_copy['avg'].isna().sum())

# Our third method will be using the built in function Auto-Arima.

# !pip install pmdarima
# from pmdarima import auto_arima
# auto_arima(train_df_copy['avg'].dropna(), m = 12, start_P = 0, seasonal = True, d = 1, trace = True, error_action = 'ignore', suppress_warnings = True, stepwise = True)

# We have found the best values for (P, D, Q) using ACF & PACF
# P =
# D = 0 [As the time-series is Stationary].
# Q =

# !pip install pandas
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot

autocorrelation_plot(df['avg'].dropna())
pyplot.show()