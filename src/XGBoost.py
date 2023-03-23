import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

plt.style.use('ggplot')

def xgboost_forecast(file_name, column_name):
    df = pd.read_csv("Hourly Datasets (CSV)/" + file_name, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    train = df.loc[df.index < '01-01-2019']
    test = df.loc[df.index >= '01-01-2019']

    def time_series(df):
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
    test = time_series(test)
    elements = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']

    x_training = train[elements]
    y_training = train[column_name]
    x_testing = test[elements]
    y_testing = test[column_name]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:linear',
                            max_depth=3,
                            learning_rate=0.1)

    reg.fit(x_training, y_training,
            eval_set=[(x_training, y_training), (x_testing, y_testing)],
            verbose=100)

    test['prediction'] = reg.predict(x_testing)
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    ax = df[[column_name]].plot(figsize=(15, 5))
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Original', 'Model'])
    ax.set_title('Original vs Model')
    plt.show()

    return reg


file_name = input("Enter file name: ")
column_name = 'Mainavg'
trained_model = xgboost_forecast(file_name, column_name)