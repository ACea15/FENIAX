from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
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
    for mode_i in modes2plot:
        mbdfi = copy.deepcopy(mbdf)
        for i, ni in enumerate(nodes_sorted):
            r = mbdfi.Node(ni).get_position()
            mbdfi.Node(ni).set_position(mbdfi, r + scale * eigv[mode_i, i, :3])
        write_path = f"{bdfile.parent / bdfile.name.split('.')[0]}M{mode_i}.bdf"
        write_vtk = f"{bdfile.parent / bdfile.name.split('.')[0]}M{mode_i}.vtk"
        mbdfi.write_bdf(write_path)
        bdf2vtk.run(write_path, None, write_vtk, False, fileformat="ascii")

vtk_fromop2("/media/acea/work/projects/FEM4INAS/examples/SailPlane/NASTRAN/SOL103/run_cao.bdf",
            "/media/acea/work/projects/FEM4INAS/examples/SailPlane/NASTRAN/SOL103/run_cao.op2")
