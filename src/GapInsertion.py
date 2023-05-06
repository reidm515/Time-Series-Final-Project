import pandas as pd
import numpy as np
import random


def random_gaps(file, gap_count, gap_size):
    # Generate gap_count random numbers in the range of 0 and len(dataset)
    gap_indices = random.sample(range(len(file)), gap_count)

    for idx in gap_indices:
        file[idx:idx + gap_size + 1] = np.nan

    # Print the list of random indices
    #print(f"Inserted {gap_count} of size {gap_size} gaps at: ")
    #for i in gap_indices:
        #print(i)

    return file

def annual_maintenance_gaps(df, maintenance_duration):

    # Make a copy of the input DataFrame
    df_copy = df.reindex(index=df.index[::-1])




    # Get the start and end dates for the DataFrame
    start_of_year = df.resample('D').first().iloc[0].name
    end_of_year = df.resample('D').last().iloc[-1].name

    # Number of observations per year
    observations_per_year = df.loc[(df.index >= start_of_year) & (df.index <= end_of_year)].shape[0]

    # Choose random date for maintenance
    maintenance_interval = random.randint(1, observations_per_year)

    # Set the value at the maintenance interval to NaN
    df.iloc[maintenance_interval, :] = np.nan

    # Set values at yearly intervals to NaN
    for i in range(maintenance_interval + observations_per_year, len(df), observations_per_year):
        df.iloc[i:(i + maintenance_duration), :] = np.nan
    return df

def weather_outage_gaps(file, num_outage_days, outage_duration, col):
    df_copy = file.copy()

    # Get the first and last years in the dataset
    first_year = pd.to_datetime(df_copy.index.min()).year
    last_year = pd.to_datetime(df_copy.index.max()).year

    # iterate through each year and insert gaps.
    while (int(first_year) != (int(last_year))):
        gap_period_start = pd.Timestamp(year=first_year, month=7, day=1, hour=0)
        gap_period_end = pd.Timestamp(year=first_year, month=9, day=1, hour=23)

        # Random dates between gap_period_start to gap_period_end
        gap_dates = pd.date_range(start=gap_period_start, end=gap_period_end, freq='H')
        gap_dates = random.sample(list(gap_dates), k=num_outage_days)

        # Add object to each Timestamp object in the list
        gap_end = [x + pd.Timedelta(hours=outage_duration) for x in gap_dates]

        # Set the values in the selected rows to NaN
        for gap_start, gap_stop in zip(gap_dates, gap_end):
            mask = (df_copy.index >= gap_start) & (df_copy.index <= gap_stop)
            df_copy.loc[mask, col] = np.nan

        # print(gap_dates)
        # Increment year counter
        first_year += 1
        df_copy = df_copy.reindex(index=df_copy.index[::-1])
    # Return the modified copy of the DataFrame
    return df_copy
