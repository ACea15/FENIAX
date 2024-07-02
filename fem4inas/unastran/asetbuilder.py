from pyNastran.bdf.bdf import BDF
import pandas as pd

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
            x1, x2, x3 = bdf.Node(self.clamped_node).get_position()
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
        self.df = pd.DataFrame(self.grid)
        
    def write_asets(self, out_file, dof='123456'):

        bdf_asets = BDF()
        bdf_asets.add_aset1(self.asets_ids_ordered, dof)
        bdf_asets.write_bdf(out_file)
    
    def write_grid(self, out_file, dof='123456'):

        bdf_asets = BDF()
        bdf_asets.add_aset1(self.asets_ids_ordered, dof)
        bdf_asets.write_bdf(out_file)
