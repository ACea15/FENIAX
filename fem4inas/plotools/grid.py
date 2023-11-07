from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from pyNastran.bdf.bdf import BDF
import pathlib

def transform_grid_rigid(ra, Rab, R0ab, pa):
    f = jax.vmap(lambda ra_i, Rab_i, R0ab_i, pa_i: ra_i.reshape((3, 1)) + Rab_i @ (R0ab_i.T @ pa_i),
                 in_axes=(1, 2, 2, 2), out_axes=2)
    pa_deform = f(ra, Rab, R0ab, pa)
    return pa_deform

def transform_arm_rigid(Rab, R0ab, pa):
    f = jax.vmap(lambda Rab_i, R0ab_i, pa_i: Rab_i @ (R0ab_i.T @ pa_i),
                 in_axes=(2, 2, 2), out_axes=2)
    pa_deform = f(Rab, R0ab, pa)
    return pa_deform

@dataclass
class AeroGrid:
    """
    Data container for a Component-based definition of aerodynamic panels
    in a NASTRAN style fashion
    """
    
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
    def build_DLMgrid(cls, model: BDF):
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
            _npoints, _npanels = model.caeros[ci].get_npanel_points_elements()
            npanels[ci] = _npanels
            npoints[ci] = _npoints
            panel_ids[ci] = model.caeros[ci]._init_ids()
            _points, _cells = model.caeros[ci].panel_points_elements()
            points[ci] = _points
            cells[ci] = _cells
        return cls(caeros, labels, npanels, npoints, panel_ids, points, cells)

    @classmethod
    def build_DLMcollocation(cls, model: BDF,
                             collocation_chordwise=0.5):
        
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
            box_ids = model.caeros[ci]._init_ids()
            nchord = model.caeros[ci].nchord
            nspan = model.caeros[ci].nspan
            panel_ids[ci] = np.zeros((nchord-1, nspan-1))
            _points = []
            _cells = []
            _npanels = 0
            _npoints = 0
            for jx in range(nspan):
                for ix in range(nchord):
                    box_id = box_ids[ix, jx]
                    point = model.caeros[ci]._get_box_x_chord_center(box_id,
                                                                     collocation_chordwise)
                    _points.append(point)
                    _npoints += 1
                    if (ix < nchord -1) and (jx < nspan -1):
                        cell = [ix + jx * nchord, ix + (jx + 1) * nchord,
                                (ix + 1) + (jx + 1) * nchord, (ix + 1) + jx * nchord] #clockwise
                        _cells.append(cell)
                        panel_ids[ci][ix, jx] = ix + jx * nchord
                        _npanels += 1
            npanels[ci] = _npanels
            npoints[ci] = _npoints
            points[ci] = np.array(_points)
            cells[ci] = np.array(_cells)
        return cls(caeros, labels, npanels, npoints, panel_ids, points, cells)

@dataclass
class Component:
    link_m0m1: dict
    link_m0mx: dict
    data_m0: dict
    data_m1: dict
    pa_m1: jnp.array
    pa_m1_tensor: jnp.array
    ra: jnp.array = None
    Rab: jnp.array = None
    data_mx: jnp.array = None
    data_mx_tensor: jnp.array = None
    pa_mx: jnp.array = None
    pa_mx_tensor: jnp.array = None

    def __post_init__(self):
        self.len_pa = len(self.pa_m1)
        self.link_m0 = list(self.link_m0mx.keys())
        self.link_mx = list(self.link_m0mx.values())
        self.link_m1 = list(self.link_m0m1.values())

class Model:

    def __init__(self, **kwargs):

        ...

    def link_models(self):
        ...

    def link_solution(self):
        ...

    def map_mxm1(self):
        ...
        
class ASETModel(Model):
    """
    This class maps data between a set of ASET or beam nodes, model0,
    onto a surface, model1; model1 can be used to interpolate to any data
    set, modelx.
    """
    
    def __init__(self, aerogrid: AeroGrid,
                 model0_ids,
                 modelx_data,
                 model,
                 tol_identification=1e-6,
                 **kwargs):
        self.aerogrid = aerogrid
        self.model0_ids = model0_ids
        self.modelx_data = modelx_data
        self.model = model
        self.tol_identification = tol_identification
        self.components = {ki: None for ki in self.aerogrid.labels.keys()}
        self.component_names = list(self.components.keys())
        self.link_models()
        self.merge_components()

    def link_models(self):
        
        for i, ki in enumerate(self.component_names):
            print(f"setting component {ki}")
            m0_ids = self.model0_ids[i]
            data_m1 = self.aerogrid.get('points', ki)
            link_m0m1, data_m0, pa_m1 = self.point2aset(self.model,
                                                        data_m1,
                                                        m0_ids)
            link_m0mx = self.link_solution(m0_ids, data_m0)
            pa_m1_tensor = self.build_m1_tensor(pa_m1, link_m0m1)
            self.components[ki] = Component(link_m0m1, link_m0mx,
                                            data_m0, data_m1, pa_m1, pa_m1_tensor)

    def link_solution(self, asets, data_m0):
        link_m0mx = self.aset2id_intrinsic(self.modelx_data, asets, data_m0,
                                           self.tol_identification)
        return link_m0mx

    def set_solution(self, ra, Rab, R0ab):
        """
        """
        self.ra = ra
        self.Rab = Rab
        for i, ki in enumerate(self.component_names):
            beam_ids = self.components[ki].link_mx
            ra_ci = ra[:, beam_ids]
            Rab_ci = Rab[:, :, beam_ids]
            R0ab_ci = R0ab[:, :, beam_ids]
            self.components[ki].ra = ra_ci
            self.components[ki].Rab = Rab_ci
            self.components[ki].R0ab = R0ab_ci
            self.map_mxm1(ki)
        self.merge_data("data_mx")
        self.merge_data("pa_mx")

    def mesh_plot(self, folder_path:str , data_name: str):
        """
        """
        import pyvista
        path = pathlib.Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        for i, ki in enumerate(self.component_names):
            _cells = self.aerogrid.get('cells', ki)
            cells = np.hstack([4*np.ones(len(_cells),dtype=int).reshape(
                len(_cells),1), _cells])
            data_ci = getattr(self.components[ki], data_name)
            mesh = pyvista.PolyData(data_ci, cells)
            mesh.save(path / f"{ki}.ply",
                      binary=False)

    def build_m1_tensor(self, pa_m1, link_m0m1):

        pa_points = list(link_m0m1.values())
        narms = len(pa_points[0])
        nasets = len(pa_points)
        pa_tensor = jnp.zeros((3, narms, nasets))
        for i, arm_i in enumerate(pa_points):
            # print(arm_i)
            if len(arm_i) == 0:
                pa_tensor = pa_tensor.at[:,:, i].set(pa_m1[pa_points[i-1]].T)
            else:
                assert len(arm_i) == narms, f"unequal arms length for arm {i}"
                pa_tensor = pa_tensor.at[:,:, i].set(pa_m1[arm_i].T)
        return pa_tensor

    def tensor2array(self, ki, pa_tensor):
        pa_points = self.components[ki].link_m1
        pa_array = np.zeros((3, self.components[ki].len_pa))
        #import pdb; pdb.set_trace()
        for i, arm_i in enumerate(pa_points):
            pa_array[:, arm_i] = pa_tensor[:,:, i]
        return pa_array.T

    def map_mxm1(self, ki):

        self.components[ki].data_mx_tensor = transform_grid_rigid(self.components[ki].ra,
                                                                  self.components[ki].Rab,
                                                                  self.components[ki].R0ab,
                                                                  self.components[ki].pa_m1_tensor)
        self.components[ki].data_mx = self.tensor2array(ki,
                                                        self.components[ki].data_mx_tensor)
        self.components[ki].pa_mx_tensor = transform_arm_rigid(self.components[ki].Rab,
                                                               self.components[ki].R0ab,
                                                               self.components[ki].pa_m1_tensor)
        self.components[ki].pa_mx = self.tensor2array(ki,
                                                      self.components[ki].pa_mx_tensor)

    def merge_components(self):

        self.index_merged = {}
        for i, ki in enumerate(self.component_names):
            index_pa_merged = np.array([])
            if i == 0:
                self.pa_merged = self.components[ki].pa_m1
                self.pa = self.components[ki].pa_m1
                self.datam1_merged = self.components[ki].data_m1
                self.datam1 = self.components[ki].data_m1

            else:
                index = []
                for i, datam1_i in enumerate(self.components[ki].data_m1):
                    ind = np.where(np.linalg.norm(datam1_i - self.datam1_merged, axis=1) <
                                   self.tol_identification)[0]
                    if len(ind) > 0:
                        #import pdb; pdb.set_trace()
                        #print(ind)
                        index_pa_merged = np.hstack([index_pa_merged, ind])
                        index.append(i)
                index = np.array(index)
                self.index_merged[ki] = [len(self.datam1_merged), index, index_pa_merged]
                self.datam1_merged = np.vstack([self.datam1_merged,
                                                self.components[ki].data_m1])
                self.pa_merged = np.vstack([self.pa_merged, self.components[ki].pa_m1])
                self.pa = np.vstack([self.pa_merged, self.components[ki].pa_m1])
                self.datam1 = np.vstack([self.datam1, self.components[ki].data_m1])
                if len(index) > 0:
                    self.pa_merged = np.delete(self.pa_merged,
                                          self.index_merged[ki][0] + index, axis=0)
                    self.datam1_merged = np.delete(self.datam1_merged,
                                                   self.index_merged[ki][0] + index, axis=0)
    def merge_data(self, data_name:str):

        for i, ki in enumerate(self.component_names):
            if i == 0:
                data = getattr(self.components[ki], data_name)
                data_merged = getattr(self.components[ki], data_name)

            else:
                # index = []
                # for i, data_i in enumerate(getattr(self.components[ki], data_name)):
                #     ind = np.where(np.linalg.norm(data_i - self.datam1_merged, axis=1) <
                #                    self.tol_identification)[0]
                #     if len(ind) > 0:
                #         #import pdb; pdb.set_trace()
                #         #print(ind)
                #         index_pa_merged = np.hstack([index_pa_merged, ind])
                #         index.append(i)
                # index = np.array(index)
                # self.index_merged[ki] = [len(self.datam1_merged), index, index_pa_merged]
                data_ki = getattr(self.components[ki], data_name)
                #import pdb; pdb.set_trace()
                data = np.vstack([data,
                                  data_ki])
                data_merged = np.vstack([data_merged,
                                         data_ki])
                index = self.index_merged[ki][1]
                if len(index) > 0:
                    data_merged = np.delete(data_merged,
                                            self.index_merged[ki][0] + index,
                                            axis=0)
        setattr(self, f"{data_name}", data)
        setattr(self, f"{data_name}_merged", data_merged)
    
    @staticmethod
    def point2aset2(model, points, asets) -> tuple[dict, np.ndarray, np.ndarray]:
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
    def point2aset(model, points, asets):
  
        aset_map = {ai: [] for ai in asets}
        asets_vect = np.array([model.nodes[i].get_position() for i in asets])
        pa = list()
        for i, pi in enumerate(points):
            cos_a = list()
            for j, vj in enumerate(asets_vect):
                normal2arm = np.linalg.norm([0, pi[1]-vj[1], pi[2]-vj[2]])
                cos_a.append(normal2arm)
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

