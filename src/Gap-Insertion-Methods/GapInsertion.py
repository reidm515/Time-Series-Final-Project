import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Example Function Call:
#  df_with_gaps = random_gaps(df_cut, gap_count, gap_size)
def random_gaps(df, gap_count, gap_size):
    """
    Generate random gaps in a DataFrame by setting values to NaN.

    Parameters:
        df (pandas.DataFrame): The DataFrame to insert gaps into.
        gap_count (int): The number of gaps to insert.
        gap_size (int): The size of each gap.

    Returns:
        pandas.DataFrame: The DataFrame with gaps inserted.
    """
    # Generate gap_count random numbers in the range of 0 and len(dataset)
    gap_indices = random.sample(range(len(df)), gap_count)

    for idx in gap_indices:
        df['avg'][idx:idx+gap_size+1] = np.nan

    # Print the list of random indices
    print(f"Inserted {gap_count} of size {gap_size} gaps at: ")
    for idx in gap_indices:
        print(idx)

    # Plot the graph with gaps inserted.
    plt.plot(df['avg'])
    return df


def annual_maintenance_gaps(df, maintenance_duration):
    # Make a copy of the input DataFrame
    df_copy = df.copy()

    # Get the start and end dates for the DataFrame
    start_of_year = df_copy.resample('D').first().iloc[0].name
    end_of_year = df_copy.resample('D').last().iloc[-1].name
    
    # Number of observations per year
    observations_per_year = df_copy.loc[(df_copy.index >= start_of_year) & (df_copy.index <= end_of_year)].shape[0]
    
    # Choose random date for maintenance
    maintenance_interval = random.randint(1, observations_per_year)
    
    # Set the value at the maintenance interval to NaN
    df_copy.iloc[maintenance_interval,:] = np.nan
    
    # Set values at yearly intervals to NaN
    for i in range(maintenance_interval + observations_per_year, len(df_copy), observations_per_year):
        df_copy.iloc[i, :] = np.nan    
    return df_copy

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