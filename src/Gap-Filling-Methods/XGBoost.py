import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

color_pal = sns.color_palette()
plt.style.use('ggplot')

file_name = input("Enter file name: ")
df = pd.read_csv("../Hourly Datasets (CSV)/" + file_name, sep=',', skiprows=1, encoding='latin1')
df = df.dropna(axis='columns', how='all') #This drops any fully Null columns in the dataset
df = df.dropna()                      #remove any data entries left with null values
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)  #set datatime as index and converted it to datetime format

train = df.loc[df.index < '01-01-2019'] # This splits data into training/testing
test = df.loc[df.index >= '01-01-2019']

def time_series(df):# This function extracts time series Elements
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = time_series(df)
train = time_series(train)
test = time_series(test)               # Extract time series elements from the df, training and testing
Elements = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year'] # This defines time series elements to use for prediction
Column_name = 'Mainavg'                    # Column we wish to forecast
x_training = train[Elements]             
y_training = train[Column_name]              
x_testing = test[Elements]                
y_testing = test[Column_name]                 

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', # This initialises XGBoost model with parameters
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.1)

reg.fit(x_training, y_training,             # This trains the XGBoost model using the training set
        eval_set=[(x_training, y_training), (x_testing, y_testing)], # This evaluates the model on training and testing sets
        verbose=100)

test['prediction'] = reg.predict(x_testing)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['Mainavg']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Original', 'Model'])
ax.set_title('Original vs Model')
plt.show()