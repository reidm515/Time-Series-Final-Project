import pandas as pd
import numpy as np

def randomsampling(file_name, num_rows):


    #The function reads the dataset from a CSV file, generates random row indices, and sets the values for the selected rows to NaN.

    #Parameters:
    #file_name (str): The name of the CSV file containing the weather dataset.
    #num_rows (int): The number of rows to select and remove the values for.


    # This Checks if the file is a valid CSV file and actually exists
    if not file_name.endswith('.csv'):
        raise ValueError("The input file needs to be in CSV format.")
    try:
        with open("../Hourly Datasets (CSV)/" + file_name, 'r') as f:
            pass
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv("../Hourly Datasets (CSV)/" + file_name, sep=',', skiprows = 1, encoding='latin1')
    print(df.head())

    # Generate the random row indices
    random_row_indices = np.random.permutation(num_rows)

    # Select the rows based on the random indices and set their values to NaN
    return df.iloc[random_row_indices, :]
    # = np.NaN

    # return df


def spatially_temporally_correlated_gaps(file_name, num_gaps, gap_size):
   #This function inserts spatially and temporally correlated gaps into our dataset by selecting a group of contiguous rows and setting their values to NaN.

    # Check if the file exists and is a valid CSV file
    if not file_name.endswith('.csv'):
        raise ValueError("File must be a CSV file")
    try:
        with open("../Hourly Datasets (CSV)/" + file_name, 'r') as f:
            pass
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    #The function uses pandas read_csv function to read the CSV file into a pandas DataFrame.
    #It skips the first row and sets the delimiter to a comma.
    df = pd.read_csv("../Hourly Datasets (CSV)/" + file_name, sep=',', skiprows=1, encoding='latin1')

    # The function converts the 'Datetime' column to a datetime type using pandas to_datetime function
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # The function generates a list of random starting indices for the gaps using numpy random choice function.
    # It uses the number of gaps and gap size arguments to determine the number of rows in each gap.
    starting_indices = np.random.choice(range(len(df) - gap_size), num_gaps, replace=False)

    # Set the values for each group of contiguous rows to NaN
    # The function sets the values of the selected rows for each gap to NaN using the pandas iloc function.
    for start_idx in starting_indices:
        end_idx = start_idx + gap_size - 1
        df.iloc[start_idx:end_idx + 1, 1:] = np.nan

    return df


def TimeSeriesMissing(file_name):
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

    # Loop through user input for year, month, and day, setting Datetime to NaN for matching rows
    while True:
        year = input("Enter the year (optional), or 'done' if finished: ")
        if year.lower() == 'done':
            break
        year = int(year) if year.strip() else None

        month = input("Enter the month (optional): ")
        month = int(month) if month.strip() else None

        day = input("Enter the day (optional): ")
        day = int(day) if day.strip() else None


        if year:
            df.loc[df['Datetime'].dt.year == year, 'Datetime'] = np.nan
        if month:
            df.loc[df['Datetime'].dt.month == month, 'Datetime'] = np.nan
        if day:
            df.loc[df['Datetime'].dt.day == day, 'Datetime'] = np.nan

    # Return the modified dataframe with NaN values set for matching dates
    return df

if __name__ == '__main__':
    file_name = input("Enter the name of the csv file: ")
    choice = input("Enter 'random-sampling' or 'TimeSeriesMissing' or 'spatially-temporally-correlated' to choose the program to run: ")

    if choice == 'random-sampling':
        num_rows = int(input("Enter the number of rows to sample: "))
        df = randomsampling(file_name, num_rows)
        print(df.head())
    elif choice == 'TimeSeriesMissing':
        df = TimeSeriesMissing(file_name)
        print(df.head())
    elif choice == 'spatially-temporally-correlated':
        num_gaps = int(input("Enter the number of gaps to insert: "))
        gap_size = int(input("Enter the size of each gap: "))
        df = spatially_temporally_correlated_gaps(file_name, num_gaps, gap_size)
        print(df.head())
    else:
        print("Invalid choice. Please enter 'random-sampling' or 'TimeSeriesMissing' or 'spatially-temporally-correlated'")