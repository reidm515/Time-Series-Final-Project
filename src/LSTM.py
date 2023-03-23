import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


def lstm_forecast(column_name, csv_path):
    df = pd.read_csv(csv_path, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('Datetime')

    variable_name = df[column_name]

    def to_X_y(df, window_size=5):
        X = []
        y = []
        for i in range(len(df) - window_size):
            X.append(df[i:i+window_size, np.newaxis])
            y.append(df[i+window_size])
        return np.array(X), np.array(y)

    WINDOW_SIZE = 5
    X, y = to_X_y(variable_name.values, WINDOW_SIZE)

    X_training, Y_training = X[:60000], y[:60000]
    X_Validation, Y_Validation = X[60000:65000], y[60000:65000]
    X_testing, y_testing = X[65000:], y[65000:]

    model = Sequential([
        InputLayer((5, 1)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    checkpoint = ModelCheckpoint('model/', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1), metrics=[RootMeanSquaredError()])
    trained_model = model.fit(X_training, Y_training, validation_data=(X_Validation, Y_Validation), epochs=10, callbacks=[checkpoint])

    model = load_model('model/')

    train_predictions = model.predict(X_training).flatten()
    train_results = pd.DataFrame(data={'Training Forecasts':train_predictions, 'Real Data':Y_training})

    df_filled = df.copy()

    gap_percentage = 0.2
    gap_indices = np.random.choice(df_filled.index, size=int(len(df_filled) * gap_percentage), replace=False)
    df_filled.loc[gap_indices, column_name] = np.nan

    X_filled, _ = to_X_y(df_filled[column_name].values, WINDOW_SIZE)
    filled_predictions = model.predict(X_filled).flatten()
    filled_predictions = filled_predictions[:len(gap_indices)]

    df_filled.loc[gap_indices, column_name] = filled_predictions

    plt.plot(df[column_name][50:10000])
    plt.plot(df_filled[column_name][50:10000])
    plt.legend(['Filled', 'Original'])
    plt.show()

    return trained_model.history


column_name = input("Enter column to graph: ")
csv_path = "Hourly Datasets (CSV)/DAD.csv"
trained_model_results = lstm_forecast(column_name, csv_path)
print(trained_model_results)