import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the columns of the dataframe
columns = ['Datetime', 'HC Air temperature [°C] mean', 'HC Air temperature [°C] max', 'HC Air temperature [°C] min',
           'Dew Point [°C] mean', 'Dew Point [°C] min', 'HC Relative humidity [%] mean', 'HC Relative humidity [%] max',
           'HC Relative humidity [%] min', 'Precipitation [mm] sum', 'Leaf Wetness [min] time', 'Battery voltage [mV] last']

# Create the dataframe
df = pd.read_csv('A1D.csv',encoding = 'unicode_escape', names=columns, skiprows=1, parse_dates=[0])


print(df)