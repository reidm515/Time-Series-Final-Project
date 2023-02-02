import pandas as pd
import numpy as np


def insert_random_gaps_into_dataset(file_name, num_rows_to_sample):

    """
    This function inserts gaps into a weather dataset by randomly selecting and removing the values for a specified number of rows.
    The function reads the dataset from a CSV file, generates random row indices, and sets the values for the selected rows to NaN.

    Parameters:
    file_name (str): The name of the CSV file containing the weather dataset.
    num_rows_to_sample (int): The number of rows to select and remove the values for.

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
    print(df.head())

    # Generate the random row indices
    random_row_indices = np.random.permutation(num_rows_to_sample)

    # Select the rows based on the random indices and set their values to NaN
    df.iloc[random_row_indices, :] = np.NaN
    return df


if __name__ == '__main__':
    file_name = input("Enter the name of the csv file: ")
    num_rows_to_sample = int(input("Enter the number of rows to sample: "))
    df =  insert_random_gaps_into_dataset(file_name, num_rows_to_sample)
