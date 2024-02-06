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

class BuildAsetModel:

    def __init__(self, components_ids, clamped_node=None):

        self.components_ids = components_ids
        self.clamped_node = clamped_node
        self.component_names = list(components_ids.keys())
        self.asets_ids = None
        self.asets_ids_ordered = None
        self.asets_ids_fe = None
        self.grid = None

        self._get_asets()
        self._get_grid()
    def _get_asets(self):

        self.asets_ids = []
        for vi in self.components_ids.values():
            self.asets_ids += vi
        self.asets_ids_ordered = sorted(self.asets_ids)
        self.asets_ids_fe = {ai: self.asets_ids_ordered.index(ai)
                             for ai in self.asets_ids}
        if self.clamped_node is not None:
            assert self.clamped_node not in self.asets_ids, "Clamped node in ASETids"
            
    def _get_grid(self):

        self.grid = dict()
        self.grid['x'] = []
        self.grid['y'] = []
        self.grid['z'] = []
        self.grid['fe_order'] = list()
        self.grid['components'] = list()
        if self.clamped_node is not None:
            x1, x2, x3 = bdf.Node(clamped_node).get_position()
            self.grid['x'].append(x1)
            self.grid['y'].append(x2)
            self.grid['z'].append(x3)
            self.grid['fe_order'].append(-1)    
            self.grid['components'].append(self.component_names[0])

        for ai in self.asets_ids:
            x1, x2, x3 = bdf.Node(ai).get_position()
            self.grid['x'].append(x1)
            self.grid['y'].append(x2)
            self.grid['z'].append(x3)
        self.grid['fe_order'] += list(self.asets_ids_fe.values())
        self.grid['components'] += [k for k, v in self.components_ids.items() for _ in v]
        self.df = pd.DataFrame(grid)
        
    def write_asets(self, out_file, dof='123456'):

        bdf_asets = BDF()
        bdf_asets.add_aset1(self.asets_ids_ordered, dof)
        bdf_asets.write_bdf(out_file)
    
    def write_grid(self, out_file, dof='123456'):

        bdf_asets = BDF()
        bdf_asets.add_aset1(self.asets_ids_ordered, dof)
        bdf_asets.write_bdf(out_file)


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

model = BuildAsetModel(components_ids, clamped_node=1005)
model.write_asets("./Config/asets_clamped.bdf")



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

model_red = BuildAsetModel(components_ids, clamped_node=1005)
model_red.write_asets("./Config/asets_clamped_reduced.bdf")


string = ""
for i, ai in enumerate(model_red.asets_ids):

    if i%8 == 0 and i!= 0:
        string += "\n"
    string += str(ai)+","

# grid = dict()
# clamped_node = 1005

# grid['x'] = []
# grid['y'] = []
# grid['z'] = []
# grid['fe_order'] = list()
# grid['components'] = list()
# if clamped_node is not None:
#     x1, x2, x3 = bdf.Node(clamped_node).get_position()
#     grid['x'].append(x1)
#     grid['y'].append(x2)
#     grid['z'].append(x3)
#     grid['fe_order'].append(-1)    
#     grid['components'].append(component_names[0])


# for ai in asets_ids:
#     x1, x2, x3 = bdf.Node(ai).get_position()
#     grid['x'].append(x1)
#     grid['y'].append(x2)
#     grid['z'].append(x3)
# grid['fe_order'] += list(asets_ids_fe.values())
# grid['components'] += [k for k, v in components_ids.items() for _ in v]

# df = pd.DataFrame(grid)


# bdf_asets = BDF()
# bdf_asets.add_aset1(asets_ids_ordered[::2], '123456')
# bdf_asets.write_bdf("./Config/asets_clamped.bdf")
