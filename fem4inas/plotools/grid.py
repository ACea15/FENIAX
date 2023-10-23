from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from pyNastran.bdf.bdf import BDF

@dataclass
class AeroGrid:
    caeros: list
    labels: dict
    npanels: dict
    npoints: dict
    panel_ids: dict
    points: dict
    cells: dict
    def get(self, name, label):
        caero = self.labels[label]
        attr = getattr(self, name)
        return attr[caero]
    
    @classmethod
    def build_grid(cls, model: BDF) -> AeroGrid:
        npanels = dict()
        npoints = dict()
        panel_ids = dict()
        points = dict()
        cells = dict()
        caeros = list()
        labels = dict()
        for i, ci in enumerate(model.caeros.keys()):
            caeros.append(ci)
            labels[model.caeros[ci].comment.strip()] = ci
            _npanels, _npoints = model.caeros[ci].get_npanel_points_elements()
            npanels[ci] = _npanels
            npoints[ci] = _npoints
            panel_ids[ci] = model.caeros[ci]._init_ids()
            _points, _cells = model.caeros[ci].panel_points_elements()
            points[ci] = _points
            cells[ci] = _cells
        return cls(caeros, labels, npanels, npoints, panel_ids, points, cells)

@dataclass
class Component:
    link_m0m1: dict
    link_m0mx: dict
    data_m0: dict
    data_m1: dict
    data_m1_tensor: jnp.array
    pa_m1: jnp.array = None

    def __init__(self):
        self.link_mx = list(self.link_m0mx.values())
    
class Model:

    def __init__(self, **kwargs):

        ...

    def link_models(self):
        ...

    def link_solution(self):
        ...

    def map_solution(self):
        ...
        
class ASETModel(Model):

    def __init__(self, aerogrid: AeroGrid,
                 model0_ids,
                 modelx_data,
                 model, **kwargs):
        self.aerogrid = aerogrid
        self.model0_ids = model0_ids
        self.model = model
        self.components = {ki: None for ki in self.aerogrid.labels.keys()}
        self.component_names = list(self.components.keys())
        
    def link_models(self):

        for i, ki in enumerate(self.component_names):
            m0_ids = self.model0_ids[i]
            link_m0m1, data_m0, data_m1 = self.point2aset(self.model,
                                                          self.aerogrid.get('points', ki),
                                                          m0_ids)
            link_m0mx = self.link_solution(m0_ids, data_m0)
            data_m1_tensor = self.build_m1_tensor(data_m1, link_m0m1)
            self.components[ki] = Component(link_m0m1, link_m0mx,
                                            data_m0, data_m1, data_m1_tensor)
    def link_solution(self, asets, data_m0):
        link_m0mx = self.aset2id_intrinsic(self.modex_data, asets, data_m0)
        return link_m0mx

    def set_solution(self, ra, Rab):
        self.ra = ra
        self.Rab = Rab
        for i, ki in enumerate(self.component_names):
            beam_ids = self.components[ki].link_mx
            ra_ci = ra[:, beam_ids]
            Rab_ci = Rab[:, :, beam_ids]
            self.components[ki].ra = ra_ci
            self.components[ki].Rab = Rab_ci

    def build_m1_tensor(self, data_m1, link_m0m1):

        pa_points = list(link_m0m1.values())
        narms = len(pa_points[0])
        nasets = len(pa_points)
        pa_tensor = jnp.zeros(3, narms, nasets)
        for i, arm_i in enumerate(pa_points):
            assert len(arm_i) == narms, f"unequal arms length for arm {i}"
            pa_tensor = pa_tensor.at[:,:, i].set(data_m1[arm_i])
        return pa_tensor

    def map_(self):
        
        for i, ki in enumerate(self.component_names):
            self.components[ki].pa_m1 = transform_grid_rigid(self.components[ki].ra,
                                                             self.components[ki].Rab,
                                                             self.components[ki].data_m1_tensor)

    @staticmethod
    def point2aset(model, points, asets) -> tuple[np.array]:
        """
        Links arbitrary points to the asets (beam) nodes
        """
        aset_map = {ai: [] for ai in asets}
        asets_vect = np.array([model.nodes[i].get_position() for i in asets])
        pa = list()
        for i, pi in enumerate(points):
            cos_a = list()
            for j, vj in enumerate(asets_vect):
                if j < len(asets_vect)-1:
                    axis = asets_vect[j+1] - vj
                else:
                    axis = asets_vect[j-1] - vj
                pa_i = pi - vj
                pa_i /= np.linalg.norm(pa_i)
                axis /= np.linalg.norm(axis) 
                cos_a.append(abs(pa_i.dot(axis)))
            index = cos_a.index(min(cos_a))
            aset_map[asets[index]].append(i)
            pa.append(pi - asets_vect[index])
        return aset_map, asets_vect, np.array(pa)

    @staticmethod
    def aset2id_intrinsic(X, asets, asets_vect, tolerance=1e-6):
        """
        Maps a Nastran aset id to the corresponding intrinsic Grid id
        """
        intrinsic_map = dict()
        for i, vi in enumerate(asets_vect):
            norm_ = np.linalg.norm(vi.reshape((1,3)) - X, axis=1)
            min_ = np.min(norm_)
            if min_ < tolerance:
                index = np.where(norm_ == min_)[0][0]
                intrinsic_map[asets[i]] = index
            else:
                intrinsic_map[asets[i]] = None
        return intrinsic_map


def map_beam2grid(intrinsic_map, pa_ci, ra, Rab):

    beam_ids = list(intrinsic_map.values())
    ra_ci = ra[:, beam_ids]
    Rab_ci = Rab[:, :, beam_ids]
    pa_deform = transform_grid_rigid(ra_ci, Rab_ci, pa_ci)
    return pa_deform

def tensor2array(pa_tensor, aset_map):
    pa_points = list(aset_map.values())
    narms = len(pa_points[0])
    nasets = len(pa_points)
    pa_array = np.zeros(3, npoints)
    for i, arm_i in enumerate(pa_points):
        pa_array[arm_i] = pa_tensor[:,:, i]
    return pa_array

def transform_grid_rigid(ra, Rab, pa):
    f = jax.vmap(lambda ra_i, Rab_i, pa_i: ra_i + Rab_i @ pa_i,
                 in_axes=(1, 2, 2), out_axes=2)
    pa_deform = f(ra, Rab, pa)
    return pa_deform
