import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv("../Hourly Datasets (CSV)/")
cleaned_dataset = []

def clean_dataset(data):

    # Remove missing and invalid values from Dataset.

    # Parameters:
    # data: Dataset/ CSV file containing the weather dataset after being converted to Pandas Dataframe.

    # Returns:
    # Dataset with outliers removed using Z-Scores
    
    data.dropna(inplace = True)

def outlier_removal_zScore(data, threshold):
    
    # Function to remove outliers from hourly datasets using Z-Scores

    # Parameters:
    # data: Dataset/ CSV file containing the weather dataset after being converted to Pandas Dataframe.
    # threshold: The maximum Z-score value before a data point is considered an outlier

    # Returns:
    # Dataset with outliers removed using Z-Scores

    if threshold <= 0:
        raise ValueError("The threshold value must be a postive number.")

    mean = np.mean(data)
    std = np.std(data)
    cleaned_dataset = data[(np.abs(data - mean) / std) <= threshold]
    return cleaned_dataset


def outlier_removal_IQR(data, column_name):

    # Function to remove outliers from hourly datasets using Interquartile Range 

    # Parameters:
    # data: Dataset / CSV file containing the weather dataset after being converted to Pandas Dataframe.
    #column: Column in dataset that outliers will be removed from.

    # Returns:
    # Dataset with outliers removed using Interquartile Range
    
    if column_name not in data.columns:
        raise ValueError("The column name is not present in Dataset.")

    Q1 = data.height.quantile(0.25)
    Q3 = data.height.quantile(0.75)
    IQR  = Q3 - Q1

    lower_limit = Q1 - (IQR * 1.5)
    upper_limit = Q1 + (IQR * 1.5)

    cleaned_dataset = data[(data.height < lower_limit) & (data.height < upper_limit)]
    return cleaned_dataset

def standardize_data(data, column):

    # Function to standardize data points inside Meteorological Dataset using . 
    # Use over Normalization when preserving outliers as normalization shrinks the variance/distribution. 

    scaler = StandardScaler()
    scaler.fit_transform(data[column])


def normalize_data_min_max(data, column):

    # Function to normalize data points inside Meteorological Dataset using Mix/Max Normalization. 
    # Use over Standardization when outliers have been removed/not present in dataset.

    scaler = MinMaxScaler()
    scaler.fit_transform(data[column])
    