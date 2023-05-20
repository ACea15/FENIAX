import numpy as np
import os
model=os.getcwd().split('/')[-1]
import importlib
V=importlib.import_module("Runs"+'.'+model+'.'+'V')

Aname = 'A'
Amatrix = os.getcwd()+'/AERO'+'/AICs.npy'
u_inf = 8
rho_inf = 1.22
c =1
NumAeroStates=5


if (__name__ == '__main__'):
    
    import IntrinsicSolver.Tools.write_config_file
    IntrinsicSolver.Tools.write_config_file.write_aero(Aname,V,locals())
    print('Force Interpolation Saved')

