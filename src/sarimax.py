import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from copy import deepcopy
import pandas as pd
import numpy as np
import base64
import io
def sarima_forecast(dataset,df_gaps, column1,column2):
    df = pd.read_csv(dataset, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()


    df_gaps = pd.read_csv(df_gaps, sep=',', encoding='latin1')
    df_gaps['Datetime'] = pd.to_datetime(df_gaps['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df_gaps = df_gaps.set_index('Datetime')

    def keep_two_columns(dataset, column1, column2):
        return dataset[[column1, column2]]


    def split_dataset(dataset):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame")
        dataset = dataset.dropna()

        # Split the dataset into 70% training and 30% temporary (test + validation)
        train_data, temp_data = train_test_split(dataset, test_size=0.3, shuffle=False)

        # Split the temporary dataset into 50% test and 50% validation (15% of the original dataset each)
        test_data, val_data = train_test_split(temp_data, test_size=0.5, shuffle=False)
        print(train_data)

        return train_data, test_data, val_data


    def sarimax_forecast(train_data, test_data, columns):
        model = SARIMAX(
            endog=train_data[columns[0]],
            exog=train_data[columns[1]],
            order=(3, 0, 1),
            trend=(0, 0),
            seasonal_order=(0, 1, 0, 12)
        ).fit()

        forecast = model.forecast(steps=test_data.shape[0], exog=test_data[columns[1]])

        start_date = train_data.index.min()
        end_date = test_data.index.max()

        plt.figure(figsize=(20, 6))

        plt.plot(train_data[columns[0]], label='Historical Precipitation')
        plt.plot(test_data.index, test_data[columns[0]], color="green")
        plt.plot(test_data.index, forecast, label='Future Precipitation - Prediction', color='blue')

        plt.xlim(start_date, end_date)
        plt.legend({'Average': test_data[columns[0]], 'forecast': forecast})
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')

        img.seek(0)
        plot_sarimax = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        return plot_sarimax

    df = keep_two_columns(df, column1, column2)
    train_data, test_data, val_data = split_dataset(df)
    return sarimax_forecast(train_data, test_data, [column1, column2])

