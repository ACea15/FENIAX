from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import numpy as np
import copy
import pathlib
import fem4inas.plotools.nastranvtk.bdf2vtk as bdf2vtk

def vtkRef(bdf_file, size_card=8, write_path=None):

    bdfile = pathlib.Path(bdf_file)
    mbdf = BDF()
    mbdf.read_bdf(bdf_file)
    nids, ntransform, icd = mbdf.get_displacement_index()
    if write_path is None:
        write_path = bdfile.parent / bdfile.name.split('.')[0]
        write_path.mkdir(parents=True, exist_ok=True)
    write_vtk = f"{write_path}/Ref.vtk"
    mbdf.write_bdf(f"{write_path}/Ref.bdf", size=size_card)
    bdf2vtk.run(f"{write_path}/Ref.bdf", None, write_vtk, False, fileformat="ascii")

def vtkModes_fromop2(bdf_file, op2_file, scale = 100., modes2plot=None, size_card=8, write_path=None):

    bdfile = pathlib.Path(bdf_file)
    mbdf = BDF()
    mop2 = OP2()
    mbdf.read_bdf(bdf_file)
    nids, ntransform, icd = mbdf.get_displacement_index()
    mop2.read_op2(op2_file)
    mop2.transform_displacements_to_global(icd, mbdf.coords)
    eigv = mop2.eigenvectors[1].data
    if modes2plot is None:
        modes2plot = range(len(eigv))
    nodes_sorted = sorted(list(mbdf.node_ids))
    # run the reference
    if write_path is None:
        write_path = bdfile.parent / bdfile.name.split('.')[0]
        write_path.mkdir(parents=True, exist_ok=True)
    write_vtk = f"{write_path}/Ref.vtk"
    mbdf.write_bdf(f"{write_path}/Ref.bdf", size=size_card)
    bdf2vtk.run(f"{write_path}/Ref.bdf", None, write_vtk, False, fileformat="ascii")
    # run the modes
    for mode_i in modes2plot:
        # mbdfi = copy.deepcopy(mbdf)
        mbdfi = BDF()
        mbdfi.read_bdf(bdf_file)
        nodes_sorted = sorted(list(mbdf.node_ids))
        for i, ni in enumerate(nodes_sorted):
            try:
                # r = mbdfi.Node(ni).get_position()
                # cdi = mbdfi.Node(ni).Cd()
                # if cdi != 0:
                #     #rl = mbdfi.Node(ni).get_position_wrt(mbdfi, cdi)
                #     #mbdfi.Node(ni).set_position(mbdfi, rl + scale * eigv[mode_i, i, :3], cid=cdi)
                #     r_new = r #mbdfi.Node(ni).get_position()
                # else:
                #     r_new = r + scale * eigv[mode_i, i, :3]
                # cpi = mbdfi.Node(ni).Cp()
                # # import pdb; pdb.set_trace()
                # if cpi != 0:
                #     mbdfi.Node(ni).set_position(mbdfi, r_new)
                #     r_new2 = mbdfi.Node(ni).get_position_wrt(mbdf, cpi)
                #     mbdfi.Node(ni).set_position(mbdfi, r_new2, cpi)
                # else:
                #     mbdfi.Node(ni).set_position(mbdfi, r_new)
                r = mbdfi.Node(ni).get_position()
                u = scale * eigv[mode_i, i, :3]
                mbdfi.Node(ni).set_position(mbdfi, r + u)
                
            except AttributeError:
                print(f"Node {ni} was not read")
        write_pathi = f"{write_path}/M{mode_i}.bdf"
        write_vtki = f"{write_path}/M{mode_i}.vtk"
        mbdfi.write_bdf(write_pathi, size=size_card)
        bdf2vtk.run(write_pathi, None, write_vtki, False, fileformat="ascii")

def vtkSol_op2Modes(bdf_file, op2_file, q, label="Sol",
                    size_card=8, write_path=None, write_ref=True):

    num_modes = len(q)
    bdfile = pathlib.Path(bdf_file)
    mbdf = BDF()
    mop2 = OP2()
    mbdf.read_bdf(bdf_file)
    nids, ntransform, icd = mbdf.get_displacement_index()
    mop2.read_op2(op2_file)
    mop2.transform_displacements_to_global(icd, mbdf.coords)
    eigv = mop2.eigenvectors[1].data
    nodes_sorted = sorted(list(mbdf.node_ids))
    # run the reference
    if write_path is None:
        write_path = bdfile.parent / bdfile.name.split('.')[0]
        write_path.mkdir(parents=True, exist_ok=True)
    else:
        write_path = pathlib.Path(write_path)
        write_path.mkdir(parents=True, exist_ok=True)
    if write_ref:
        write_vtk = f"{write_path}/Ref.vtk"
        mbdf.write_bdf(f"{write_path}/Ref.bdf", size=size_card)
        bdf2vtk.run(f"{write_path}/Ref.bdf", None, write_vtk, False, fileformat="ascii")

    # mbdfi = copy.deepcopy(mbdf)
    mbdfi = BDF()
    mbdfi.read_bdf(bdf_file)
    nodes_sorted = sorted(list(mbdf.node_ids))
    # run the modes
    for mode_i in range(num_modes):
        for i, ni in enumerate(nodes_sorted):
            try:
                # r = mbdfi.Node(ni).get_position()
                # cdi = mbdfi.Node(ni).Cd()
                # if cdi != 0:
                #     #rl = mbdfi.Node(ni).get_position_wrt(mbdfi, cdi)
                #     #mbdfi.Node(ni).set_position(mbdfi, rl + scale * eigv[mode_i, i, :3], cid=cdi)
                #     r_new = r #mbdfi.Node(ni).get_position()
                # else:
                #     r_new = r + scale * eigv[mode_i, i, :3]
                # cpi = mbdfi.Node(ni).Cp()
                # # import pdb; pdb.set_trace()
                # if cpi != 0:
                #     mbdfi.Node(ni).set_position(mbdfi, r_new)
                #     r_new2 = mbdfi.Node(ni).get_position_wrt(mbdf, cpi)
                #     mbdfi.Node(ni).set_position(mbdfi, r_new2, cpi)
                # else:
                #     mbdfi.Node(ni).set_position(mbdfi, r_new)
                r = mbdfi.Node(ni).get_position()
                u = q[mode_i] * eigv[mode_i, i, :3]
                mbdfi.Node(ni).set_position(mbdfi, r + u)
                
            except AttributeError:
                print(f"Node {ni} was not read")
    write_pathi = f"{write_path}/{label}.bdf"
    write_vtki = f"{write_path}/{label}.vtk"
    mbdfi.write_bdf(write_pathi, size=size_card)
    bdf2vtk.run(write_pathi, None, write_vtki, False, fileformat="ascii")

def vtkSol_fromop2(bdf_file, op2_file, scale = 1., loads2plot=None, size_card=8, write_path=None, plot_ref=False):

    "TODO: make into a class together with previous function"
    bdfile = pathlib.Path(bdf_file)
    mbdf = BDF()
    mop2 = OP2()
    mbdf.read_bdf(bdf_file)
    nids, ntransform, icd = mbdf.get_displacement_index()
    mop2.read_op2(op2_file)
    mop2.transform_displacements_to_global(icd, mbdf.coords)
    eigv = mop2.displacements[1].data
    if loads2plot is None:
        loads2plot = range(len(eigv))
    nodes_sorted = sorted(list(mbdf.node_ids))
    # run the reference
    if write_path is None:
        write_path = bdfile.parent / bdfile.name.split('.')[0]
        write_path.mkdir(parents=True, exist_ok=True)
    else:
        write_path = pathlib.Path(write_path)
        write_path.mkdir(parents=True, exist_ok=True)
    if plot_ref:
        write_vtk = f"{write_path}/Ref.vtk"
        mbdf.write_bdf(f"{write_path}/Ref.bdf", size=size_card)
        bdf2vtk.run(f"{write_path}/Ref.bdf", None, write_vtk, False, fileformat="ascii")
    # run the modes
    for mode_i in loads2plot:
        # mbdfi = copy.deepcopy(mbdf)
        mbdfi = BDF()
        mbdfi.read_bdf(bdf_file)
        nodes_sorted = sorted(list(mbdf.node_ids))
        for i, ni in enumerate(nodes_sorted):
            try:
                # r = mbdfi.Node(ni).get_position()
                # cdi = mbdfi.Node(ni).Cd()
                # if cdi != 0:
                #     #rl = mbdfi.Node(ni).get_position_wrt(mbdfi, cdi)
                #     #mbdfi.Node(ni).set_position(mbdfi, rl + scale * eigv[mode_i, i, :3], cid=cdi)
                #     r_new = r #mbdfi.Node(ni).get_position()
                # else:
                #     r_new = r + scale * eigv[mode_i, i, :3]
                # cpi = mbdfi.Node(ni).Cp()
                # # import pdb; pdb.set_trace()
                # if cpi != 0:
                #     mbdfi.Node(ni).set_position(mbdfi, r_new)
                #     r_new2 = mbdfi.Node(ni).get_position_wrt(mbdf, cpi)
                #     mbdfi.Node(ni).set_position(mbdfi, r_new2, cpi)
                # else:
                #     mbdfi.Node(ni).set_position(mbdfi, r_new)
                r = mbdfi.Node(ni).get_position()
                u = scale * eigv[mode_i, i, :3]
                mbdfi.Node(ni).set_position(mbdfi, r + u)
                
            except AttributeError:
                print(f"Node {ni} was not read")
        write_pathi = f"{write_path}/L{mode_i}.bdf"
        write_vtki = f"{write_path}/L{mode_i}.vtk"
        mbdfi.write_bdf(write_pathi, size=size_card)
        bdf2vtk.run(write_pathi, None, write_vtki, False, fileformat="ascii")


class DefBdf:

    def __init__(self, bdf_file):
        self.mbdf = BDF()
        self.bdf_file = bdf_file

    @property
    def bdf_file(self):
        return self._bdf_file
    
    @bdf_file.setter
    def bdf_file(self, value):
        self._bdf_file = value
        self._read_bdf()
        
    def _read_bdf(self):
        self.mbdf.read_bdf(self.bdf_file)
        self.sorted_nodeids = sorted(self.mbdf.node_ids)

    def get_nodes(self, sort=True):

        if sort:
            nodes = [ni.get_position() for ni in
                     self.mbdf.Nodes(self.sorted_nodeids)]
        else:
            nodes = [ni.get_position() for ni in
                     self.mbdf.Nodes(self.mbdf.node_ids)]
        return np.array(nodes)

    def update_bdf(self, nposition, nid):
        for i, ni in enumerate(nid):
            self.mbdf.Node(ni).set_position(self.mbdf, nposition[i])

    def save_vtkpath(self, file_path):
        
        self.path_bdf4vtk = pathlib.Path(file_path)
        path_folder = self.path_bdf4vtk.parent
        path_folder.mkdir(parents=True, exist_ok=True)

    def plot_vtk(self, label="", size_cards=8):

        path = self.path_bdf4vtk.parent / self.path_bdf4vtk.stem
        path_bdf = f"{path}_{label}.bdf"
        path_vtk = f"{path}_{label}.vtk"
        self.mbdf.write_bdf(path_bdf, size=size_cards)
        bdf2vtk.run(path_bdf, None, path_vtk, False, fileformat="ascii")
        
if (__name__ == "__main__"):
    vtk_fromop2("/media/acea/work/projects/FEM4INAS/examples/SailPlane/NASTRAN/SOL103/run_cao.bdf",
                "/media/acea/work/projects/FEM4INAS/examples/SailPlane/NASTRAN/SOL103/run_cao.op2")
