import numpy as np
import importlib
import Runs.Torun

V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)


init_x1 = 1
init_q1 = 0

def fv(X):
    return np.array([0.,0.,1.5*(X[1]/40.)**2,0.,0.,0.])

init_x2 = 0
init_q2 = 0
