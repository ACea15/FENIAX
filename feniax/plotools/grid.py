from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from pyNastran.bdf.bdf import BDF
import pathlib
import feniax.plotools.interpolation as interpolation
import feniax.preprocessor.solution as solution
import feniax.plotools.nastranvtk.bdfdef as bdfdef
import feniax.plotools.grid as grid
import feniax.unastran.bdfextractor as bdfextractor


def transform_rigid(ra, Rab, R0ab, pa):
    return ra + Rab @ (R0ab.T @ pa)


def transform_grid_rigid(ra, Rab, R0ab, pa):
    f = jax.vmap(
        lambda ra_i, Rab_i, R0ab_i, pa_i: ra_i.reshape((3, 1))
        + Rab_i @ (R0ab_i.T @ pa_i),
        in_axes=(1, 2, 2, 2),
        out_axes=2,
    )
    pa_deform = f(ra, Rab, R0ab, pa)
    return pa_deform


def transform_arm_rigid(Rab, R0ab, pa):
    f = jax.vmap(
        lambda Rab_i, R0ab_i, pa_i: Rab_i @ (R0ab_i.T @ pa_i),
        in_axes=(2, 2, 2),
        out_axes=2,
    )
    pa_deform = f(Rab, R0ab, pa)
    return pa_deform


@dataclass
class AeroGrid:
    """
    Data container for a Component-based definition of aerodynamic panels
    in a NASTRAN style fashion
    """

    caeros: list = None
    labels: dict = None
    npanels: dict = None
    npoints: dict = None
    panel_ids: dict = None
    points: dict = None
    cells: dict = None

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


    @staticmethod
    def _get_box_x_chord_center(caero, box_id: int, x_chord: float) -> np.ndarray:
        """
        The the location of the x_chord of the box along the centerline.
        """
        
        ispan, ichord = caero.get_box_index(box_id)

        le_vector = caero.p4 - caero.p1
        delta_xyz = le_vector * ((ispan + 0.5)/caero.nspan)
        yz = delta_xyz[1:3] + caero.p1[1:3]
        chord = ((ispan + 0.5)/caero.nspan) * (caero.x43 - caero.x12) + caero.x12
        x = (ichord + x_chord)/caero.nchord * chord + caero.p1[0] + delta_xyz[0]
        return np.array([x, yz[0], yz[1]])
    
    
    @classmethod
    def build_DLMcollocation(cls, model: BDF, collocation_chordwise=0.5):
        npanels = dict()
        npoints = dict()
        panel_ids = dict()
        points = dict()
        cells = dict()
        caeros = list()
        labels = dict()
        normals = dict()
        for i, ci in enumerate(model.caeros.keys()):
            caeros.append(ci)
            labels[model.caeros[ci].comment.strip()] = ci
            p1234 = model.caeros[ci].get_points()
            normals[ci] = np.cross(p1234[3] - p1234[0], p1234[1] - p1234[0])
            #box_ids = model.caeros[ci]._init_ids()
            box_ids = model.caeros[ci].box_ids
            nchord = model.caeros[ci].nchord
            nspan = model.caeros[ci].nspan
            panel_ids[ci] = np.zeros((nchord - 1, nspan - 1))
            _points = []
            _cells = []
            _npanels = 0
            _npoints = 0
            for jx in range(nspan):
                for ix in range(nchord):
                    box_id = box_ids[jx, ix]
                    point = AeroGrid._get_box_x_chord_center(model.caeros[ci],
                        box_id, collocation_chordwise
                    )
                    # point = model.caeros[ci]._get_box_x_chord_center(
                    #     box_id, collocation_chordwise
                    # )

                    _points.append(point)
                    _npoints += 1
                    if (ix < nchord - 1) and (jx < nspan - 1):
                        cell = [
                            ix + jx * nchord,
                            ix + (jx + 1) * nchord,
                            (ix + 1) + (jx + 1) * nchord,
                            (ix + 1) + jx * nchord,
                        ]  # clockwise
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
    def __init__(self, **kwargs): ...

    def link_models(self): ...

    def link_solution(self): ...

    def map_mxm1(self): ...


class RBE3Model:
    def __init__(
        self,
        bdf_model: BDF,
        model2_coord: list | jnp.ndarray,
        tol_identification=1e-6,
        rbe3s_full=True,
        **kwargs,
    ):
        self.bdf_model = bdf_model
        self.model2_coord = model2_coord
        self.tol_identification = tol_identification
        self.rbe3s_full = rbe3s_full
        self._link_models()
        self._link_solution()

    def _link_models(self):
        self.rbe3 = bdfextractor.build_RBE3(self.bdf_model)
        self.model0_nodes = self.rbe3.dependent_nodes
        self.model0_coord = self.rbe3.dnodes_coord
        self.model1_nodes = self.rbe3.independent_nodes
        self.model1_coord = self.rbe3.inodes_coord
        self.model1_map = self.rbe3.inodes_map
        self.link_m0m1 = self.rbe3.dinpendent_link
        self.model01_coord = self.rbe3.arm_coord

    def _link_solution(self):
        self.link_m0m2 = ASETModel.aset2id_intrinsic(
            self.model2_coord,
            self.model0_nodes,
            self.model0_coord,
            self.tol_identification,
        )
        self.link_m0m2_valid = {
            k: vi for k, vi in self.link_m0m2.items() if vi is not None
        }
        self.model1_coord_valid = []
        self.repeated = []
        for niref in self.link_m0m2_valid.keys():
            # nmrom_node = self.link_m0m2[niref]
            i = self.model0_nodes.index(niref)
            for j, nj in enumerate(self.link_m0m1[i]):
                node_index = self.model1_map[nj]
                # model1_coord = self.model1_coord[node_index]
                coord = self.model1_coord[node_index]
                if len(self.model1_coord_valid) > 0:
                    check_repeated = (
                        np.linalg.norm(
                            [coord - ci for ci in self.model1_coord_valid], axis=1
                        )
                        < self.tol_identification * 1e-3
                    ).any()
                else:
                    check_repeated = False
                if not check_repeated:
                    self.model1_coord_valid.append(coord)
                else:
                    self.repeated.append((i, j))
        self.model1_coord_valid = np.array(self.model1_coord_valid)

    def map_m1mx(self, ra, Rab, R0ab, arm):
        model1x_coord_ij = transform_rigid(ra, Rab, R0ab, arm)
        return model1x_coord_ij

    def set_solution(self, ra, Rab, R0ab):
        """ """
        model1x_coord = []
        if self.rbe3s_full:
            for i, niref in enumerate(self.model0_nodes):
                nmrom_node = self.link_m0m2[niref]
                for j, nj in enumerate(self.link_m0m1[i]):
                    node_index = self.model1_map[nj]
                    # model1_coord = self.model1_coord[node_index]
                    arm = self.model01_coord[node_index]
                    model1x_coord_ij = self.map_m1mx(
                        ra[:, nmrom_node],
                        Rab[:, :, nmrom_node],
                        R0ab[:, :, nmrom_node],
                        arm,
                    )
                    model1x_coord.append(model1x_coord_ij)
        else:
            for niref, nmrom_node in self.link_m0m2_valid.items():
                # nmrom_node = self.link_m0m2[niref]
                i = self.model0_nodes.index(niref)
                for j, nj in enumerate(self.link_m0m1[i]):
                    if (i, j) not in self.repeated:
                        node_index = self.model1_map[nj]
                        # model1_coord = self.model1_coord[node_index]
                        arm = self.model01_coord[node_index]
                        model1x_coord_ij = self.map_m1mx(
                            ra[:, nmrom_node],
                            Rab[:, :, nmrom_node],
                            R0ab[:, :, nmrom_node],
                            arm,
                        )
                        model1x_coord.append(model1x_coord_ij)

        self.model1x_coord = np.array(model1x_coord)


class ASETModel(Model):
    """
    This class maps data between a set of ASET or beam nodes, model0,
    onto a surface, model1; model1 can be used to interpolate to any data
    set, modelx.
    """

    def __init__(
        self,
        aerogrid: AeroGrid,
        model0_ids: list,
        modelx_data: list | jnp.ndarray,
        bdf_model: BDF,
        tol_identification=1e-6,
        **kwargs,
    ):
        self.aerogrid = aerogrid
        self.model0_ids = model0_ids
        self.modelx_data = modelx_data
        self.bdf_model = bdf_model
        self.tol_identification = tol_identification
        self.components = {ki: None for ki in self.aerogrid.labels.keys()}
        self.component_names = list(self.components.keys())
        self.link_models()
        self.merge_components()

    def link_models(self):
        """
        links m0 to m1 models (aset nodes to panel grid)
        """
        for i, ki in enumerate(self.component_names):
            print(f"setting component {ki}")
            m0_ids = self.model0_ids[i]
            data_m1 = self.aerogrid.get("points", ki)
            link_m0m1, data_m0, pa_m1 = self.point2aset(self.bdf_model, data_m1, m0_ids)
            link_m0mx = self.link_solution(m0_ids, data_m0)
            pa_m1_tensor = self.build_m1_tensor(pa_m1, link_m0m1)
            self.components[ki] = Component(
                link_m0m1, link_m0mx, data_m0, data_m1, pa_m1, pa_m1_tensor
            )

    def link_solution(self, asets, data_m0):
        link_m0mx = self.aset2id_intrinsic(
            self.modelx_data, asets, data_m0, self.tol_identification
        )
        return link_m0mx

    def set_solution(self, ra, Rab, R0ab):
        """ """
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

    # def mesh_plot(self, folder_path:str , data_name: str):
    #     """
    #     """
    #     import pyvista
    #     path = pathlib.Path(folder_path)
    #     path.mkdir(parents=True, exist_ok=True)
    #     for i, ki in enumerate(self.component_names):
    #         _cells = self.aerogrid.get('cells', ki)
    #         cells = np.hstack([4*np.ones(len(_cells),dtype=int).reshape(
    #             len(_cells),1), _cells])
    #         data_ci = getattr(self.components[ki], data_name)
    #         mesh = pyvista.PolyData(data_ci, cells)
    #         mesh.save(path / f"{ki}.ply",
    #                   binary=False)

    def mesh_plot(self, folder_path: str, data_name: str):
        """ """
        import pyvista

        path = pathlib.Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        data, cells = self.get_meshdata(data_name)
        for i, ki in enumerate(self.component_names):
            # _cells = self.aerogrid.get('cells', ki)
            # cells = np.hstack([4*np.ones(len(_cells),dtype=int).reshape(
            #     len(_cells),1), _cells])
            # data_ci = getattr(self.components[ki], data_name)
            # import pdb; pdb.set_trace();
            mesh = pyvista.PolyData(data[i], cells[i])
            mesh.save(path / f"{ki}.ply", binary=False)

    def get_meshdata(self, data_name: str):
        """ """

        Cells = list()
        Data_ci = list()
        for i, ki in enumerate(self.component_names):
            _cells = self.aerogrid.get("cells", ki)
            cells = np.hstack(
                [4 * np.ones(len(_cells), dtype=int).reshape(len(_cells), 1), _cells]
            )
            data_ci = getattr(self.components[ki], data_name)
            Cells.append(cells)
            Data_ci.append(data_ci)
        return Data_ci, Cells

    def build_m1_tensor(self, pa_m1, link_m0m1):
        pa_points = list(link_m0m1.values())
        narms = len(pa_points[0])
        nasets = len(pa_points)
        pa_tensor = jnp.zeros((3, narms, nasets))
        for i, arm_i in enumerate(pa_points):
            # print(arm_i)
            if len(arm_i) == 0:
                pa_tensor = pa_tensor.at[:, :, i].set(pa_m1[pa_points[i - 1]].T)
            else:
                assert len(arm_i) == narms, f"unequal arms length for arm {i}"
                pa_tensor = pa_tensor.at[:, :, i].set(pa_m1[arm_i].T)
        return pa_tensor

    def tensor2array(self, ki, pa_tensor):
        pa_points = self.components[ki].link_m1
        pa_array = np.zeros((3, self.components[ki].len_pa))
        # import pdb; pdb.set_trace()
        for i, arm_i in enumerate(pa_points):
            pa_array[:, arm_i] = pa_tensor[:, :, i]
        return pa_array.T

    def map_mxm1(self, ki):
        self.components[ki].data_mx_tensor = transform_grid_rigid(
            self.components[ki].ra,
            self.components[ki].Rab,
            self.components[ki].R0ab,
            self.components[ki].pa_m1_tensor,
        )
        self.components[ki].data_mx = self.tensor2array(
            ki, self.components[ki].data_mx_tensor
        )
        self.components[ki].pa_mx_tensor = transform_arm_rigid(
            self.components[ki].Rab,
            self.components[ki].R0ab,
            self.components[ki].pa_m1_tensor,
        )
        self.components[ki].pa_mx = self.tensor2array(
            ki, self.components[ki].pa_mx_tensor
        )

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
                    ind = np.where(
                        np.linalg.norm(datam1_i - self.datam1_merged, axis=1)
                        < self.tol_identification
                    )[0]
                    if len(ind) > 0:
                        # import pdb; pdb.set_trace()
                        # print(ind)
                        index_pa_merged = np.hstack([index_pa_merged, ind])
                        index.append(i)
                index = np.array(index)
                self.index_merged[ki] = [
                    len(self.datam1_merged),
                    index,
                    index_pa_merged,
                ]
                self.datam1_merged = np.vstack(
                    [self.datam1_merged, self.components[ki].data_m1]
                )
                self.pa_merged = np.vstack([self.pa_merged, self.components[ki].pa_m1])
                self.pa = np.vstack([self.pa_merged, self.components[ki].pa_m1])
                self.datam1 = np.vstack([self.datam1, self.components[ki].data_m1])
                if len(index) > 0:
                    self.pa_merged = np.delete(
                        self.pa_merged, self.index_merged[ki][0] + index, axis=0
                    )
                    self.datam1_merged = np.delete(
                        self.datam1_merged, self.index_merged[ki][0] + index, axis=0
                    )

    def merge_data(self, data_name: str):
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
                # import pdb; pdb.set_trace()
                data = np.vstack([data, data_ki])
                data_merged = np.vstack([data_merged, data_ki])
                index = self.index_merged[ki][1]
                if len(index) > 0:
                    data_merged = np.delete(
                        data_merged, self.index_merged[ki][0] + index, axis=0
                    )
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
                if j < len(asets_vect) - 1:
                    axis = asets_vect[j + 1] - vj
                else:
                    axis = asets_vect[j - 1] - vj
                pa_i = pi - vj
                pa_i /= np.linalg.norm(pa_i)
                axis /= np.linalg.norm(axis)
                cos_a.append(abs(pa_i.dot(axis)))
            index = cos_a.index(min(cos_a))
            aset_map[asets[index]].append(i)
            pa.append(pi - asets_vect[index])
        return aset_map, asets_vect, np.array(pa)

    @staticmethod
    def point2aset(model: BDF, points, asets):
        aset_map = {ai: [] for ai in asets}
        asets_vect = np.array([model.nodes[i].get_position() for i in asets])
        pa = list()
        for i, pi in enumerate(points):
            cos_a = list()
            for j, vj in enumerate(asets_vect):
                normal2arm = np.linalg.norm([0, pi[1] - vj[1], pi[2] - vj[2]])
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
            norm_ = np.linalg.norm(vi.reshape((1, 3)) - X, axis=1)
            min_ = np.min(norm_)
            if min_ < tolerance:
                index = np.where(norm_ == min_)[0][0]
                intrinsic_map[asets[i]] = index
            else:
                intrinsic_map[asets[i]] = None
        return intrinsic_map


class Interpol: ...


import feniax.unastran.aero as nasaero
# import feniax.plotools.grid

# dlm_panels= nasaero.GenDLMPanels.from_file("./dlm_model.yaml")
# aerogrid = feniax.plotools.grid.AeroGrid.build_DLMgrid(dlm_panels.model)
# panelmodel = feniax.plotools.grid.ASETModel(aerogrid, dlm_panels.set1x, X, bdf_model)


class PanelsBDFInterpol(Interpol):
    def __init__(self, file_dlm, file_bdf, Xref, folder_sol=None):
        self.sol = None
        self.sys_name = None
        self.Xref = Xref
        self.inputmodel_ref = None
        self.targetmodel_ref = None
        self.inputmodel_current = None
        self.targetmodel_current = None
        self.inputmodel_disp = None
        self.targetmodel_disp = None
        if folder_sol is not None:
            self.solfolder = folder_sol
        self._read_bdf(file_bdf)
        self._read_dlm(file_dlm)
        self._set_inputmodel_ref()

    @property
    def solfolder(self):
        return self._solfolder

    @solfolder.setter
    def solfolder(self, value):
        self._solfolder = value
        self.sol = self._read_sol(self._solfolder)
        self.set_sysname()

    def _read_sol(self, folder):
        """
        Input and read an intrinsic solution
        """
        sol = solution.IntrinsicReader(folder)
        return sol.data

    def _read_bdf(self, file_bdf):
        self.bdf = bdfdef.DefBdf(file_bdf)
        self._set_targetmodel_ref()

    def _read_dlm(self, file_dlm):
        self.dlm_panels = nasaero.GenDLMPanels.from_file(file_dlm)
        self.dlm_panels.build_model()
        # import pdb;pdb.set_trace()
        self.aerogrid = grid.AeroGrid.build_DLMgrid(self.dlm_panels.model)

    def _set_inputmodel_ref(self):
        self.asetmodel = grid.ASETModel(
            self.aerogrid, self.dlm_panels.set1x, self.Xref, self.bdf.mbdf
        )
        self.inputmodel_ref = self.asetmodel.datam1_merged

    def set_sysname(self, sys_name=None):
        if sys_name is None:
            if self.sys_name is None:
                systems = [
                    si for si in dir(self.sol) if si[0] != "_" and "system" in si
                ]
                self.sys_name = systems[0]
            else:
                self.sys_name = sys_name

    def set_inputmodel_current(self, ti=0):
        sys = getattr(self.sol, f"{self.sys_name}")
        ra = sys.ra[ti]
        Rab = sys.Cab[ti]
        self.asetmodel.set_solution(ra, Rab, self.sol.modes.C0ab)
        self.inputmodel_current = self.asetmodel.data_mx_merged
        self._set_targetmodel_current()

    def _set_targetmodel_ref(self):
        self.targetmodel_ref = self.bdf.get_nodes(sort=True)

    def _set_targetmodel_current(self):
        disp, coord = interpolation.compute(
            self.inputmodel_ref, self.inputmodel_current, self.targetmodel_ref
        )
        self.targetmodel_current = coord
        self.targetmodel_disp = disp
        self.bdf.update_bdf(coord, self.bdf.sorted_nodeids)

    def vtk(self, folder_path, file_name, plot_timesteps=[0]):
        """Plot vtk of the target model at the given time steps."""
        folder_path = pathlib.Path(folder_path)
        for ti in plot_timesteps:
            self.set_inputmodel_current(ti)
            file_path = folder_path / f"{file_name}_{ti}"
            self.bdf.plot_vtk(file_path)
