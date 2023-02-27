# Ordinary Kriging - Spatial Interpolation

import numpy as np
import pandas as pd
from math import *

# These values are just from tutoral likely to need to be changed.
nugget = 2.5
sill = 7.5
rang = 10

# Calculates distances between each X and Y coordinate and appends them to NumpyArray / Distance Matrix.
def distance_matrix(X, Y):
    n = len(X)
    m = len(Y)
    distance =  [np.sqrt((X[i] - Y[j])**2 + (X[i+1] - Y[j+1])**2) for i in range(n) for j in range(m)]
    distances = np.array(distance).reshape(n, m)
    return distances

def distance_to_unknown(X1, Y1, X2, Y2):
    distances = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
    return distances


def semivariance (nug, sil, ran, h):
    sv = nug + sill * (3 / 2 * h / ran - 0.5 * (h / ran) ** 3)
    if sv.shape[0] > 1:
        col1 = np.ones(sv.shape(0))
        sv = np.insert(sv, sv.shape[1], col1, axis=1)
        row1 = np.ones(sv.shape[1])
        sv = np.insert(sv, sv.shape[0], row1, axis=0)
        sv[sv.shape[0] - 1][sv.shape[1] - 1] = 0
    else:
        col1 = np.ones(sv.shape(0))
        sv = np.insert(sv, sv.shape(1), col1, axis=1)
        sv.transpose()
    return sv


def ordinary_kriging(dataX, dataY, unknownX, unknownY, Variable):
    Var1 = np.reshape(Variable, (Variable.shape[0] ,1))
    Var1 = Var1.transpose()
    matdist_N = distance_matrix(dataX, dataY)
    matdist_U = distance_matrix(dataX, dataY)
    N_SV = semivariance(nugget, sill, rang, matdist_N)
    U_SV = semivariance(nugget, sill, rang, matdist_U)

    # Get Inverse of Known SemiVariance
    inv_N_SV = np.linalg.inv(N_SV)
    weights = np.matmul(inv_N_SV, U_SV)
    weights = np.delete(weights, weights.shape[1] - 1, 0)

    # Get dot product of variable and weights
    estimate = np.dot(Var1, weights)
    return estimate[0]

def interpolation(X, Y, Variable, Res):
    Resolution = Res
    X_mesh = np.linspace(np.amin(X) - 1, np.amax(X) + 1, Resolution)
    Y_mesh = np.linspace(np.amin(Y) - 1, np.amax(Y) + 1, Resolution)
    XX, YY = np.meshgrid(X_mesh, Y_mesh)
    
    EstimateX = []
    EstimateY = []
    EstimateZ = []

    for x in np.nditer(XX):
        EstimateX.append(x)
    for y in np.nditer(YY):
        EstimateY.append(y)

    grid1 = pd.DataFrame(data={'X':EstimateX, 'Y':EstimateY})
    for index, rows in grid1.iterrows():
        estimated = ordinary_kriging(X, Y, rows['X'], rows['Y'], Variable)
        EstimateZ.append(estimated)
    Grid = pd.DataFrame(data = {'X':EstimateX, 'Y':EstimateY, 'Z':EstimateZ})
    return Grid

print("Please enter the name of your Dataset: ")
file_name = input()
data = pd.read_csv("../Hourly Datasets (CSV)/" + file_name, sep=',', skiprows = 1, encoding='latin1')

X = data['X'].to_numpy()
Y = data['Y'].to_numpy()
Var = data['Var'].to_numpy()

test = interpolation(X, Y, Var)
print(test)