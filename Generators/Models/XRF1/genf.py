import pdb
import subprocess
import numpy as np
import Generators.modelGen as mgen
import Utils.FEM_MatrixBuilder as femmb
import Utils.common as common
import Utils.GridOrder as gridorder
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import importlib
# import Runs.Torun
# Runs.Torun.torun = 'HaleX1x'
# Runs.Torun.variables = 'V'
# run_V = 0
# run_nastran = 1
# run_model=1
# run_fems=1
# run_aerodynamics=1
# run_AICs=1
# if run_V:
#     V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
# else:
#     class variables:
#         pass
#     V=variables()
#     V.NumModes=5
################################################################################
# AERODYNAMICS
################################################################################

mesh3=BDF(debug=True,log=None)
m3 = mgen.Model('beam_inpf',mesh3)
am3 = mgen.Aerodynamic_model('aero_inp_elevator_trim',m3)
am3.surfaces()
m3.write_model(model='/aerodynamics_engine.bdf')

machs = [0.]
reduced_freqs = np.linspace(1e-9,1,101)
write_file = '/MKAERO/MKaero'
for mi in machs:
    model = BDF(debug=True,log=None)
    mgen.Aerodynamic_model.mkaero(model,mi,reduced_freqs,m3.path+write_file+common.remove_dot(mi)+'.bdf')
################################################################################
