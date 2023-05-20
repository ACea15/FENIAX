import numpy as np
import os
model=os.getcwd().split('/')[-1]
import importlib
V=importlib.import_module("Runs"+'.'+model+'.'+'V')

Aname = 'A'
Amatrix = os.getcwd()+'/AERO'+'/AICs00_4r12.npy'
LocPoles = os.getcwd()+'/AERO'+'/Poles00_4r12.npy'
u_inf = 216. #840.#836.#100
rho_inf = 0.0023771#(1.02/14.5939)*0.3048**3
q_inf=0.5*rho_inf*u_inf**2
c = 6
NumPoles = 4


if (__name__ == '__main__'):
    
    import intrinsic.Tools.write_config_file
    intrinsic.Tools.write_config_file.write_aero(Aname,V,locals())
    print('Force Interpolation Saved')

