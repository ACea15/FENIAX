# [[file:modelgeneration.org::parameters_dlm0][parameters_dlm0]]
import json
import feniax.unastran.aero as aero
from pyNastran.bdf.bdf import BDF
import numpy as np
import feniax.unastran.op4handler as op4handler
import feniax.aeromodal.panels as panels
import feniax.plotools.grid
import copy
PRINT_CAEROS = True
######## Set discretisation MODEL ##############

nchord_wing = 7
nchord_htp = 7
label_dlm = f"d1c{nchord_wing}"
dlm_aeros = dict(RWing1=dict(nspan=2, nchord=nchord_wing),
             RWing2=dict(nspan=3, nchord=nchord_wing),
             RWing3=dict(nspan=9, nchord=nchord_wing),
             RWing4=dict(nspan=6, nchord=nchord_wing),
             RWing5=dict(nspan=4, nchord=nchord_wing),
             RHTP=dict(nspan=6, nchord=nchord_htp)
           )

dlm_aeros["LWing1"] = copy.copy(dlm_aeros["RWing1"])
dlm_aeros["LWing2"] = copy.copy(dlm_aeros["RWing2"])
dlm_aeros["LWing3"] = copy.copy(dlm_aeros["RWing3"])
dlm_aeros["LWing4"] = copy.copy(dlm_aeros["RWing4"])
dlm_aeros["LWing5"] = copy.copy(dlm_aeros["RWing5"])
dlm_aeros["LHTP"] = copy.copy(dlm_aeros["RHTP"])

# CAEROS IDs in the original model (right side only)
aeros2ids = dict(RWing1=3504001,
                 RWing2=3500001,
                 RWing3=3501001,
                 RWing4=3502001,
                 RWing5=3503001,
                 RHTP=3600001)

with open(f"./NASTRAN/DLMs/input_{label_dlm}.json", "w") as fp:
    json.dump(dlm_aeros, fp)  # encode dict into JSON
# parameters_dlm0 ends here

# [[file:modelgeneration.org::DLMbuild][DLMbuild]]
# Read old model with right side of CAEROS
bdfaero = BDF()#debug=False)
bdfaero.read_bdf("./NASTRAN/BUGaero1.bdf", validate=False, punch=False)

if PRINT_CAEROS:
    for ki, vi in bdfaero.caeros.items():
        print(f"*{ki}*-p1: {vi.p1}")
        print(f"*{ki}*-p4: {vi.p4}")
        print(f"*{ki}*-x12: {vi.x12}")
        print(f"*{ki}*-x43: {vi.x43}")

# copy info from old model
for ki, i in aeros2ids.items():
    dlm_aeros[ki]['p1'] = bdfaero.caeros[i].p1
    dlm_aeros[ki]['p4'] = bdfaero.caeros[i].p4
    dlm_aeros[ki]['x12'] = bdfaero.caeros[i].x12
    dlm_aeros[ki]['x43'] = bdfaero.caeros[i].x43
    ki_l=('L'+ki[1:])
    # symmetry to left side
    dlm_aeros[ki_l]['p1'] = bdfaero.caeros[i].p1*np.array([1.,-1.,1.])
    dlm_aeros[ki_l]['p4'] = bdfaero.caeros[i].p4*np.array([1.,-1.,1.])
    dlm_aeros[ki_l]['x12'] = bdfaero.caeros[i].x12
    dlm_aeros[ki_l]['x43'] = bdfaero.caeros[i].x43

dlm_aeros['RWing1']['set1x'] = [1004, 2001] 
dlm_aeros['RWing2']['set1x'] = [2003, 2005, 2008, 2010] 
dlm_aeros['RWing3']['set1x'] = list(range(2012, 2030, 2))
dlm_aeros['RWing4']['set1x'] = list(range(2030, 2044, 2))
dlm_aeros['RWing5']['set1x'] = list(range(2044,2053, 2))
dlm_aeros['RHTP']['set1x'] = list(range(4000, 4014))
#####
dlm_aeros['LWing1']['set1x'] = [1004, 10002001] 
dlm_aeros['LWing2']['set1x'] = [10002003, 10002005, 10002008, 10002010] 
dlm_aeros['LWing3']['set1x'] = list(range(10002012, 10002030, 2))
dlm_aeros['LWing4']['set1x'] = list(range(10002030, 10002044, 2))
dlm_aeros['LWing5']['set1x'] = list(range(10002044,10002053, 2))
dlm_aeros['LHTP']['set1x'] = [4000]+list(range(10004001, 10004014))

dlm = aero.GenDLMPanels.from_dict(dlm_aeros) # pass your dictionary with DLM model
dlm.build_model()
dlm.model.write_bdf(f"./NASTRAN/DLMs/{label_dlm}.bdf") # write the bdf file
dlm.save_yaml(f"./NASTRAN/DLMs/model_{label_dlm}.yaml") # write the bdf file
# DLMbuild ends here

# [[file:modelgeneration.org::DLMGrid][DLMGrid]]
dlmgrid = aero.GenDLMGrid(dlm.model)
dlmgrid.plot_pyvista(f"./paraview/dlm{label_dlm}")
collocationpoints = dlmgrid.get_collocation()
np.save(f"./AERO/Collocation_{label_dlm}.npy", collocationpoints)
#bdfdef.vtkRef("./NASTRAN/Paraview/BUG_103cao.bdf")  # write full FE paraview
# DLMGrid ends here
