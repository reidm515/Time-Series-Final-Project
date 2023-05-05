# Import necessary libraries
import pandas as pd
import io
import base64
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
plt.style.use('ggplot')
matplotlib.use('agg')
from GapInsertion import random_gaps, annual_maintenance_gaps, weather_outage_gaps

def lstm_forecast(file_path, column_name):
    df = pd.read_csv(file_path, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('Datetime')


    temp_data = df[column_name]

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

    trained_model = model.fit(X_training, Y_training, validation_data=(X_Validation, Y_Validation), epochs=1, callbacks=[checkpoint]) # Here we train the model on the training set and validate on the validation set

    from tensorflow.keras.models import load_model # This will load the checkpoint model we got before
    model = load_model('model/')

    train_predictions = model.predict(X_training).flatten() # This will create predictions using the trained model from earlier

    train_results = pd.DataFrame(data={'Training Forecasts':train_predictions, 'Real Data':Y_training}) # Here we store the predictions and the actual values side by side in a dataframe
    df_filled = df.copy()
    df_gaps = GapInsertion(df)


    plt.plot(df_gaps[column_name][50:1000])
    plt.plot(df_filled[column_name][50:1000])
    plt.legend(['Filled', 'Original'])
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    plot_lstm = base64.b64encode(img1.getvalue()).decode('utf-8')
    plt.close()



    #plot of rolling average against the real data in the dataset
    rolling_avg = df[column_name][50:1000].rolling(window=12).mean()
    plt.plot(df[column_name][50:1000], color="blue")
    plt.plot(rolling_avg, label='Rolling Mean')
    plt.legend(['Real Data', 'Rolling Mean'])
    img2 = io.BytesIO()#here we package the plot as a png image using base64 io, so it can be returned to our flask app and html later
    plt.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    plot2_lstm = base64.b64encode(img2.getvalue()).decode('utf-8')
    plt.close()






    return trained_model.history, plot_lstm, plot2_lstm#returning the model results in text, and the png images of the plots
