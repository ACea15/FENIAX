from pyNastran.bdf.bdf import BDF
import pandas as pd

bdf = BDF()#debug=False)
bdf.read_bdf("./BUG_103cao.bdf", validate=False)

# bdf_conm2 = BDF()
# conm2_ids = list(range(314, 345)) + [376, 377, 378]
# for cmi in conm2_ids:
#     conm2 = bdf.masses[cmi]
#     bdf_conm2.add_conm2(conm2.eid, conm2.nid, conm2.mass, conm2.cid, conm2.X, conm2.I)

# bdf_conm2.write_bdf("./Parts/MTOW_FUEL_RWBOXmod.bdf")

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

component_names = list(components_ids.keys())
asets_ids = []
for vi in components_ids.values():
    asets_ids += vi

asets_ids_ordered = sorted(asets_ids)
asets_ids_fe = {ai: asets_ids_ordered.index(ai) for ai in asets_ids}


grid = dict()
clamped_node = 1005

grid['x'] = []
grid['y'] = []
grid['z'] = []
grid['fe_order'] = list()
grid['components'] = list()
if clamped_node is not None:
    x1, x2, x3 = bdf.Node(clamped_node).get_position()
    grid['x'].append(x1)
    grid['y'].append(x2)
    grid['z'].append(x3)
    grid['fe_order'].append(-1)    
    grid['components'].append(component_names[0])


for ai in asets_ids:
    x1, x2, x3 = bdf.Node(ai).get_position()
    grid['x'].append(x1)
    grid['y'].append(x2)
    grid['z'].append(x3)
grid['fe_order'] += list(asets_ids_fe.values())
grid['components'] += [k for k, v in components_ids.items() for _ in v]

df = pd.DataFrame(grid)


bdf_asets = BDF()
bdf_asets.add_aset1(asets_ids_ordered, '123456')
bdf_asets.write_bdf("./Config/asets_clamped.bdf")
