import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from copy import deepcopy
import pandas as pd
import numpy as np

# Update params
def step(params): 
    # Copy of the params dictionary...avoid modifying original
    updated_params = deepcopy(params)
    
    # Select a random param
    selected_variable = np.random.choice(['p', 'd', 'q', 'P', 'D', 'Q', 'a0', 'a1'], size=1)[0]
    
    # Get current value selected
    current_value = updated_params[selected_variable]

    # Update current value 
    updated_value = current_value + np.random.choice([-1, 1], size=1)[0]

    # Put limits on selected variable
    if(selected_variable in ['p', 'q']):
        low, high = 0, 6
    else:
        low, high = 0, 1
    
    updated_value = min([max([low, updated_value]), high])

    # Update selected param
    updated_params[selected_variable] = updated_value
    return updated_params

# Num validation steps as 25%
validation_steps = int(len(test_df) * 0.25)

# Define the training set length as the remaining data length after removing validation steps
training_set_length = data.shape[0] - validation_steps

# Score each param set
def score(params, data):    
    # List to store error scores
    error_scores = []

    # Loop through step values + compute error score for each
    for validation_steps in [1,2,3,4,5,6,7,8,9,10]:
        # Set training set length based on current validation step value
        training_set_length = data.shape[0] - validation_steps

        # Fit model using the given params
        model = SARIMAX(
            endog= df['MeanfirstDiff'].head(training_set_length),
            exog= df['MinfirstDiff'].head(training_set_length), 
            order = (params['p'], params['d'], params['q']), # p, d, q order
            trend = [params['a0'], params['a1']], # A(t) = a0 + a1*t + a2*t^2
            seasonal_order = (params['P'], params['D'], params['Q'], 12) # P, D, Q seasonal order
        ).fit()

        # forecast for validation step using the model + last exogenous value
        forecast = model.forecast(steps=validation_steps, exog=data['MinfirstDiff'].tail(validation_steps))
        actuals = data['MeanfirstDiff'].tail(validation_steps)
        
        # Compute mean absolute error against actual values
        error = (forecast.tail(1) - actuals.tail(1)).abs().mean()
        error_scores.append(error)

    # return the average error score across all validation steps
    return np.mean(error_scores)

# Define the initial hyperparameter set
x0 = {'p': 1, 'd': 1, 'q': 0, 'P': 0, 'D': 0, 'Q': 0, 'a0': 1, 'a1': 1}
