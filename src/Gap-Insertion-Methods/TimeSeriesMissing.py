import pandas as pd
import numpy as np


def insert_gaps_into_dataset(file_name):

    """

    Parameters:
    file_name (str): The name of the CSV file containing the weather dataset.


    Returns:
    pandas.DataFrame: The weather dataset with gaps inserted.
    """

    # Check if the file exists and is a valid CSV file
    if not file_name.endswith('.csv'):
        raise ValueError("File must be a CSV file")
    try:
        with open("../Hourly Datasets (CSV)/" + file_name, 'r') as f:
            pass
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv("../Hourly Datasets (CSV)/" + file_name, sep=',', skiprows = 1, encoding='latin1')
    df['Datetime'] = np.nan #Time-Series Data Removed
    print(df.head())



    # return df


if __name__ == '__main__':
    file_name = input("Enter the name of the csv file: ")
    df = insert_gaps_into_dataset(file_name)
