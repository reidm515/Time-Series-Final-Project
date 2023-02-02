import pandas as pd
import numpy as np


def change_datetime_in_dataset(file_name, year=None, month=None, day=None):


    # Check if the file exists and is a valid CSV file
    if not file_name.endswith('.csv'):
        raise ValueError("File must be a CSV file")
    try:
        with open("../Hourly Datasets (CSV)/" + file_name, 'r') as f:
            pass
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv("../Hourly Datasets (CSV)/" + file_name, sep=',', skiprows=1, encoding='latin1')

    # Convert the Datetime column to datetime type
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # Null conversion for day, month & year
    if year:
        df.loc[df['Datetime'].dt.year == year, 'Datetime'] = np.nan
    if month:
        df.loc[df['Datetime'].dt.month == month, 'Datetime'] = np.nan
    if day:
        df.loc[df['Datetime'].dt.day == day, 'Datetime'] = np.nan

    print(df.head())

    return df


if __name__ == '__main__':
    file_name = input("Enter the name of the csv file: ")
    year = input("Enter the year (optional): ")
    year = int(year) if year.strip() else None
    month = input("Enter the month (optional): ")
    month = int(month) if month.strip() else None
    day = input("Enter the day (optional): ")
    day = int(day) if day.strip() else None
    df = change_datetime_in_dataset(file_name, year, month, day)
