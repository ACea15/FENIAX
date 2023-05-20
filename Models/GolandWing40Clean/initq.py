import numpy as np
import importlib
import Runs.Torun

V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)


init_x1 = 0
init_q1 = 1
q01 = np.zeros(V.NumModes - V.NumModes_res)
q01[0] = 0.2
q01[1] = 0.5

init_x2 = 0
init_q2 = 0
