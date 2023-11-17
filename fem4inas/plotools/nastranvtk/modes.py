from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import numpy as np
import copy
import pathlib
import fem4inas.plotools.nastranvtk.bdf2vtk as bdf2vtk

def vtk_fromop2(bdf_file, op2_file, scale = 100., modes2plot=None):

    bdfile = pathlib.Path(bdf_file)
    mbdf = BDF()
    mop2 = OP2()
    mbdf.read_bdf(bdf_file)
    mop2.read_op2(op2_file)
    eigv = mop2.eigenvectors[1].data
    if modes2plot is None:
        modes2plot = range(len(eigv))
    nodes_sorted = sorted(list(mbdf.node_ids))
    # run the reference
    write_path = bdfile #f"{bdfile.parent / bdfile.name.split('.')[0]}REF.bdf"
    write_vtk = f"{bdfile.parent / bdfile.name.split('.')[0]}Ref.vtk"
    bdf2vtk.run(write_path, None, write_vtk, False, fileformat="ascii")
    # run the modes
    for mode_i in modes2plot:
        mbdfi = copy.deepcopy(mbdf)
        for i, ni in enumerate(nodes_sorted):
            r = mbdfi.Node(ni).get_position()
            mbdfi.Node(ni).set_position(mbdfi, r + scale * eigv[mode_i, i, :3])
        write_path = f"{bdfile.parent / bdfile.name.split('.')[0]}M{mode_i}.bdf"
        write_vtk = f"{bdfile.parent / bdfile.name.split('.')[0]}M{mode_i}.vtk"
        mbdfi.write_bdf(write_path)
        bdf2vtk.run(write_path, None, write_vtk, False, fileformat="ascii")

class ParseBDF:

    def __init__(self, bdf_file):
        self.mbdf = BDF()
        self.bdf_file = bdf_file

    @property
    def bdf_file(self):
        return self._bdf_file
    
    @bdf_file.setter
    def bdf_file(self, value):
        self._bdf_file = value
        self.read_bdf()
        
    def read_bdf(self):
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
            
    def plot_vtk(self, file_path):
        path = pathlib.Path(file_path)
        path_folder = path.parent
        path_folder.mkdir(parents=True, exist_ok=True)
        path_vtk = path.with_suffix(".vtk")
        self.mbdf.write_bdf(path)
        bdf2vtk.run(str(path), None, str(path_vtk), False, fileformat="ascii")
        
if (__name__ == "__main__"):
    vtk_fromop2("/media/acea/work/projects/FEM4INAS/examples/SailPlane/NASTRAN/SOL103/run_cao.bdf",
                "/media/acea/work/projects/FEM4INAS/examples/SailPlane/NASTRAN/SOL103/run_cao.op2")
