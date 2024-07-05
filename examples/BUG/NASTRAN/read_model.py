from pyNastran.bdf.bdf import BDF
import pandas as pd
from fem4inas.unastran.asetbuilder import BuildAsetModel
from fem4inas.unastran.aero import GenDLMPanels
import fem4inas.aeromodal.panels as panels
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef

bdf = BDF()#debug=False)
bdf.read_bdf("./BUG_103cao.bdf", validate=False)

bdfaero = BDF()#debug=False)
bdfaero.read_bdf("./BUGaero1.bdf", validate=False, punch=False)

# bdf_conm2 = BDF()
# conm2_ids = list(range(314, 345)) + [376, 377, 378]
# for cmi in conm2_ids:
#     conm2 = bdf.masses[cmi]
#     bdf_conm2.add_conm2(conm2.eid, conm2.nid, conm2.mass, conm2.cid, conm2.X, conm2.I)

# bdf_conm2.write_bdf("./Parts/MTOW_FUEL_RWBOXmod.bdf")


######## BUILD STRUCTURAL MODEL ##############


components_ids = dict()
components_ids['FusWing'] = [2000]
components_ids['FusBack'] = [1006, 1007, 1008, 1009]
components_ids['FusFront'] = [1004, 1003, 1002, 1001, 1000]
components_ids['RWing'] = list(range(2001, 2053))
components_ids['LWing'] = list(range(10002001, 10002053))
components_ids['FusTail'] = [1010]
components_ids['VTP'] = list(range(3000, 3010))
components_ids['HTP'] = [4000]
components_ids['VTPTail'] = [3010]
components_ids['RHTP'] = list(range(4001, 4014))
components_ids['LHTP'] = list(range(10004001, 10004014))

# model = BuildAsetModel(components_ids, clamped_node=1005)
# model.write_asets("./Config/asets_clamped.bdf")


components_ids = dict()
#components_ids['FusWing'] = [2000]
components_ids['FusBack'] = [1006, 1007, 1008, 1009]
components_ids['FusFront'] = [1004, 1003, 1002, 1001, 1000]
components_ids['RWing'] = [2001, 2003, 2005, 2008, 2010] + list(range(2012, 2053, 2))
components_ids['LWing'] = ([10002001, 10002003, 10002005, 10002008, 10002010] +
                           list(range(10002012, 10002053, 2)))
components_ids['FusTail'] = [1010]
components_ids['VTP'] = list(range(3000+1, 3010-1))
components_ids['HTP'] = [4000]
components_ids['VTPTail'] = [3010]
components_ids['RHTP'] = list(range(4001, 4014))
components_ids['LHTP'] = list(range(10004001, 10004014))

# model_red = BuildAsetModel(components_ids, clamped_node=1005)
# model_red.write_asets("./Config/asets_clamped_reduced.bdf")

# string = ""
# for i, ai in enumerate(bdfaero.asets[0].ids):#model_red.asets_ids):

#     if i%8 == 0 and i!= 0:
#         string += "\n"
#     string += str(ai)+","




######## BUILD AERO MODEL ##############

aeros = dict(RWing1=dict(nspan=2, nchord=8),
             RWing2=dict(nspan=3, nchord=8),
             RWing3=dict(nspan=9, nchord=8),
             RWing4=dict(nspan=6, nchord=8),
             RWing5=dict(nspan=4, nchord=8),
             RHTP=dict(nspan=6, nchord=8))

aeros2ids = dict(RWing1=3504001,
                 RWing2=3500001,
                 RWing3=3501001,
                 RWing4=3502001,
                 RWing5=3503001,
                 RHTP=3600001)

PRINT_CAEROS = False
if PRINT_CAEROS:
    for ki, vi in bdfaero.caeros.items():
        print(f"*{ki}*-p1: {vi.p1}")
        print(f"*{ki}*-p4: {vi.p4}")
        print(f"*{ki}*-x12: {vi.x12}")
        print(f"*{ki}*-x43: {vi.x43}")

for ki, i in aeros2ids.items():
    aeros[ki]['p1'] = bdfaero.caeros[i].p1
    aeros[ki]['p4'] = bdfaero.caeros[i].p4
    aeros[ki]['x12'] = bdfaero.caeros[i].x12
    aeros[ki]['x43'] = bdfaero.caeros[i].x43

print(sorted(bdfaero.asets[0].ids))
print(bdfaero.Nodes([2001, 2003, 2005, 2008, 2010] + list(range(2012, 2053, 2)))) # wing nodes

aeros['RWing1']['set1x'] = [1004, 2001] 
aeros['RWing2']['set1x'] = [2003, 2005, 2008, 2010] 
aeros['RWing3']['set1x'] = list(range(2012, 2030, 2))
aeros['RWing4']['set1x'] = list(range(2030, 2044, 2))
aeros['RWing5']['set1x'] = list(range(2044,2053, 2))
aeros['RHTP']['set1x'] = list(range(4000, 4014))

dlm = GenDLMPanels.from_dict(aeros) # pass your dictionary with DLM model
dlm.build_model()
dlm.model.write_bdf("./dlm_model.bdf") # write the bdf file

grid = panels.caero2grid(dlm.components, dlm.caero1) # build grid from dlm model
panels.build_gridmesh(grid, 'dlm_mesh')  #  write paraview mesh
bdfdef.vtkRef("./BUG_103cao.bdf")  # write full FE paraview
