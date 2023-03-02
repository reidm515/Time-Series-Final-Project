# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


df = pd.read_csv("../Hourly Datasets (CSV)/DAD.csv", sep=',', skiprows=1, encoding='latin1') # This loads in the DAD.csv dataset from hourly datasets
df = df.drop(df.columns[[7, 8]], axis=1) #Here I dropped the NaN Columns at position 7&8 in the DAD.csv dataset
df = df.dropna() #remove any data entries left with null values
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
df = df.set_index('Datetime') #set datatime as index and converted it to datetime format

temp_data = df['Mainavg']

def to_X_y(df, window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[i:i+window_size, np.newaxis])
        y.append(df[i+window_size])
    return np.array(X), np.array(y)#this creates two empty lists x and y, extracts window size of 5 from the dataframe. X contains input data for prediction model and Y is expeteced outputs for input 
WINDOW_SIZE = 5

X, y = to_X_y(temp_data.values, WINDOW_SIZE)# This converts the 'Mainavg' data column into input and output sequences using the to_X_y() function

X_training, Y_training = X[:60000], y[:60000]
X_Validation, Y_Validation = X[60000:65000], y[60000:65000]
X_testing, y_testing = X[65000:], y[65000:] # This splits the data into training, validation, and testing sets

model = Sequential([
    InputLayer((5, 1)),
    LSTM(64),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])  # This creates a sequential model with an LSTM layer, two dense layers and an input shape of 5,1

model.summary() # This shows a summary of the model

checkpoint = ModelCheckpoint('model/', save_best_only=True) # This checkpoint function will save the best performing model

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# This compiles the model with mean squared error loss, Adam optimizer, and root mean squared error as the metric

model.fit(X_training, Y_training, validation_data=(X_Validation, Y_Validation), epochs=10, callbacks=[checkpoint]) # Here we train the model on the training set and validate on the validation set

from tensorflow.keras.models import load_model # This will load the checkpoint model we got before
model = load_model('model/')

train_predictions = model.predict(X_training).flatten() # This will create predictions using the trained model from earlier

train_results = pd.DataFrame(data={'Training Forecasts':train_predictions, 'Real Data':Y_training}) # Here we store the predictions and the actual values side by side in a dataframe


plt.plot(train_results['Training Forecasts'][50:1000])
plt.plot(train_results['Real Data'][50:1000])
plt.show() #Matplotlib to plot the values against eachother.