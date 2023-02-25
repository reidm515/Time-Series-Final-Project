import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cleaned_dataset = []

# Function to remove outliers using Z-Scores
def remove_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    cleaned_dataset = data[np.abs(data - mean) / std) <= threshold]
    return cleaned_dataset
