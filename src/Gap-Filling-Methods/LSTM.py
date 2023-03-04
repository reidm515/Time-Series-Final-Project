import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

column_name = input("Enter column to graph: ")
df = pd.read_csv("../Hourly Datasets (CSV)/DAD.csv", sep=',', skiprows=1, encoding='latin1') # This loads in the DAD.csv dataset from hourly datasets
df = df.dropna(axis='columns', how='all') #This drops any fully Null columns in the dataset
df = df.dropna() #remove any data entries left with null values
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
df = df.set_index('Datetime') #set datatime as index and converted it to datetime format

variable_name = df[column_name]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam



def to_X_y(df, window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[i:i+window_size, np.newaxis])
        y.append(df[i+window_size])
    return np.array(X), np.array(y)#this creates two empty lists x and y, extracts window size of 5 from the dataframe. X contains input data for prediction model and Y is expeteced outputs for input

WINDOW_SIZE = 5

X, y = to_X_y(variable_name.values, WINDOW_SIZE)# This converts the variable data column into input and output sequences using the to_X_y() function

X_training, Y_training = X[:60000], y[:60000]
X_Validation, Y_Validation = X[60000:65000], y[60000:65000]
X_testing, y_testing = X[65000:], y[65000:]# This splits the data into training, validation, and testing sets

model = Sequential([
    InputLayer((5, 1)),
    LSTM(64),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')# This creates a sequential model with an LSTM layer, two dense layers and an input shape of 5,1
])

checkpoint = ModelCheckpoint('model/', save_best_only=True)# This checkpoint function will save the best performing model

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1), metrics=[RootMeanSquaredError()])# This compiles the model with mean squared error loss, Adam optimizer, and root mean squared error as the metric

model.fit(X_training, Y_training, validation_data=(X_Validation, Y_Validation), epochs=10, callbacks=[checkpoint])# Here we train the model on the training set and validate on the validation set

from tensorflow.keras.models import load_model# This will load the checkpoint model we got before
model = load_model('model/')

train_predictions = model.predict(X_training).flatten()# This will create predictions using the trained model from earlier

train_results = pd.DataFrame(data={'Training Forecasts':train_predictions, 'Real Data':Y_training})# Here we store the predictions and the actual values side by side in a dataframe

df_filled = df.copy()#Creating a copy of the dataframe to perform gap insertion/filling

gap_percentage = 0.2
gap_indices = np.random.choice(df_filled.index, size=int(len(df_filled) * gap_percentage), replace=False)#where gaps are randomlly selected
df_filled.loc[gap_indices, column_name] = np.nan#this sets the values at those points to null

X_filled, _ = to_X_y(df_filled[column_name].values, WINDOW_SIZE)#input and output sequences

filled_predictions = model.predict(X_filled).flatten()#model predicts x_filled values

filled_predictions = filled_predictions[:len(gap_indices)]#keeps only missing value predictions

df_filled.loc[gap_indices, column_name] = filled_predictions

plt.plot(df[column_name][50:10000])
plt.plot(df_filled[column_name][50:10000])
plt.legend(['Filled', 'Original'])
plt.show()