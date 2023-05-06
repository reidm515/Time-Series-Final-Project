from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import base64
import io

def lstm_forecast(file_path, file_gaps, column_name):
    df = pd.read_csv(file_path, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    df_gaps = pd.read_csv(file_gaps, sep=',', encoding='latin1')
    df_gaps['Datetime'] = pd.to_datetime(df_gaps['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df_gaps = df_gaps.set_index('Datetime')

    temp_data = df[column_name]

    def to_X_y(df, window_size=5):
        X = []
        y = []
        for i in range(len(df) - window_size):
            X.append(df[i:i+window_size, np.newaxis])
            y.append(df[i+window_size])
        return np.array(X), np.array(y)

    WINDOW_SIZE = 5
    X, y = to_X_y(temp_data.values, WINDOW_SIZE)

    X_training, Y_training = X[:60000], y[:60000]
    X_Validation, Y_Validation = X[60000:65000], y[60000:65000]
    X_testing, y_testing = X[65000:], y[65000:]

    model = Sequential([
        InputLayer((5, 1)),
        LSTM(64),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.summary()

    checkpoint = ModelCheckpoint('model/', save_best_only=True)

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model.fit(X_training, Y_training, validation_data=(X_Validation, Y_Validation), epochs=10, callbacks=[checkpoint])

    rmse = history.history['root_mean_squared_error'][-1]
    vrmse = history.history['val_root_mean_squared_error'][-1]


    from tensorflow.keras.models import load_model
    model = load_model('model/')

    train_predictions = model.predict(X_training).flatten()
    train_results = pd.DataFrame(data={'Training Forecasts':train_predictions, 'Real Data':Y_training})

    plt.plot(train_results['Training Forecasts'][50:20000], label='Model')
    plt.plot(train_results['Real Data'][50:20000], label='Real Data')
    plt.legend()

    img1 = io.BytesIO()
    plt.savefig(img1, format='png', dpi=300, bbox_inches='tight')
    img1.seek(0)
    plot_lstm2 = base64.b64encode(img1.getvalue()).decode('utf-8')
    plt.close()

    # Combine the real data and gap-filled data
    combined_data = pd.concat([df[column_name], df_gaps[column_name]])
    combined_data = combined_data.sort_index()

    # Create input sequences for the combined data
    X_combined, _ = to_X_y(combined_data.values, WINDOW_SIZE)

    # Predict the values using the model
    combined_predictions = model.predict(X_combined).flatten()

    # Create a new dataframe containing the predictions and the actual combined data side-by-side
    combined_results = pd.DataFrame(data={'Combined Forecasts': combined_predictions, 'Combined Data': combined_data[WINDOW_SIZE:].values}, index=combined_data.index[WINDOW_SIZE:])

    # Plot the results
    plt.plot(combined_results['Combined Forecasts'], label='Model')
    plt.plot(df_gaps[column_name], label='Data with Gaps')
    plt.legend()

    img3 = io.BytesIO()
    plt.savefig(img3, format='png', dpi=300, bbox_inches='tight')
    img3.seek(0)
    plot_combined = base64.b64encode(img3.getvalue()).decode('utf-8')
    plt.close()

    return rmse,vrmse, plot_lstm2, plot_combined

# You can call the function like this:
# model_summary, plot_lstm, plot_lstm2, plot_combined = lstm_forecast(file_path, file_gaps, column_name)
