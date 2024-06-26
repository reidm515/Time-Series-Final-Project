import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import base64
import io
plt.style.use('ggplot')

def xgboost_forecast(file_path, file_gaps, column_name):
    df = pd.read_csv(file_path, sep=',', skiprows=1, encoding='latin1')
    df = df.dropna(axis='columns', how='all')

    #This drops any fully Null columns in the dataset

    df = df.dropna()
    #remove any data entries left with null values

    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    df_gaps = pd.read_csv(file_gaps, sep=',', encoding='latin1')
    df_gaps['Datetime'] = pd.to_datetime(df_gaps['Datetime'], format='%Y-%m-%d %H:%M:%S')
    df_gaps = df_gaps.set_index('Datetime')

    #set datatime as index and converted it to datetime format


    train = df.loc[df.index < '01-01-2019']
    # This splits data into training/testing
    test = df.loc[df.index >= '01-01-2019']


    # This function extracts time series Elements
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



    # Extract time series elements from the df, training and testing
    df = time_series(df)
    train = time_series(train)
    test = time_series(test)
    # This defines time series elements to use for prediction
    elements = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']

    x_training = train[elements]
    y_training = train[column_name]
    x_testing = test[elements]
    y_testing = test[column_name]

    # This initialises XGBoost model with parameters
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:linear',
                            max_depth=3,
                            learning_rate=0.1)


    # This trains the XGBoost model using the training set
    reg.fit(x_training, y_training,
            eval_set=[(x_training, y_training), (x_testing, y_testing)],# This evaluates the model on training and testing sets
            verbose=100)
    # This evaluates the model on training and testing sets

    test['prediction'] = reg.predict(x_testing)
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    ax = df_gaps[[column_name]].plot(figsize=(15, 5))
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Original', 'Model'])
    ax.set_title('Original vs Model')
    ax.set_title('Original vs Model')
    img = io.BytesIO()
    #package plot into png image to be sent to html
    plt.savefig(img, format='png',dpi =300, bbox_inches='tight')
    img.seek(0)
    plot_xgb = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()



    eval_results = reg.evals_result()    # This extracts the evaluation results in text form

    return reg,plot_xgb , eval_results #return back plot and model results


#file_name = input("Enter file name: ")
#column_name = 'Mainavg'
#trained_model = xgboost_forecast(file_name, column_name)
