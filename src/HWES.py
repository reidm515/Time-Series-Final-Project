import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
plt.style.use('ggplot')
matplotlib.use('agg')




def hwes_forecast(file, file_gaps, column_name):
    df = pd.read_csv(file, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('Datetime')
    df_gaps = pd.read_csv(file_gaps, sep=',', encoding='latin1')
    df_gaps['Datetime'] = pd.to_datetime(df_gaps['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df_gaps = df_gaps.set_index('Datetime')

    data = df[column_name]

    # Create a Holt-Winters Exponential Smoothing model
    model = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add', damped_trend=True)

    # Fit the model to your data
    model_fit = model.fit()

    # Forecast the values for the entire dataset
    forecast = model_fit.fittedvalues

    # Calculate the RMSE and MAPE
    rmse = np.sqrt(mean_squared_error(data, forecast))
    mae = mean_absolute_error(data, forecast)


    plt.plot(forecast[50:500], label='Model')
    plt.plot(df_gaps[column_name][50:500], label='Gap Filled Data')
    plt.legend()

    img1 = io.BytesIO()
    plt.savefig(img1, format='png', dpi=300, bbox_inches='tight')
    img1.seek(0)
    plot_hwes = base64.b64encode(img1.getvalue()).decode('utf-8')
    plt.close()

    plt.plot(forecast[50:50000], label='Model')
    plt.plot(df_gaps[column_name][50:50000], label='Gap Filled Data')
    plt.legend()

    img1 = io.BytesIO()
    plt.savefig(img1, format='png', dpi=300, bbox_inches='tight')
    img1.seek(0)
    plot2_hwes = base64.b64encode(img1.getvalue()).decode('utf-8')
    plt.close()

    # Plot the rolling average against the real data in the dataset
    rolling_avg = data[50:1000].rolling(window=12).mean()
    plt.plot(data[50:1000], color="blue")
    plt.plot(rolling_avg, label='Rolling Mean')
    plt.legend(['Real Data', 'Rolling Mean'])
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', dpi=300, bbox_inches='tight')
    img2.seek(0)
    plot3_hwes = base64.b64encode(img2.getvalue()).decode('utf-8')
    plt.close()

    return plot_hwes, plot2_hwes, plot3_hwes, rmse, mae

