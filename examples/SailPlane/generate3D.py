import fem4inas.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import pandas as pd
import importlib
import numpy as np
import jax.numpy as jnp
from fem4inas.preprocessor import solution
import fem4inas.plotools.interpolation as interpolation
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef
import fem4inas.unastran.op2reader as op2reader
importlib.reload(bdfdef)
importlib.reload(grid)

df_grid = pd.read_csv('./FEM/structuralGrid', comment="#", sep=" ",
                    names=['x1', 'x2', 'x3', 'fe_order', 'component'])
bdf_model = BDF(debug=True)
bdf_model.read_bdf("./NASTRAN/SailPlane_MakeMatc.bdf", punch=False)
#bdf_model.read_bdf("./NASTRAN/static400/run_all.bdf", punch=False)

X = df_grid[['x1','x2','x3']].to_numpy()
gridmodel = grid.RBE3Model(bdf_model, X, 1e-3)

soli = solution.IntrinsicReader("./results_static")
R0ab = soli.data.modes.C0ab

bdf_def = bdfdef.DefBdf("./NASTRAN/SailPlane_MakeMatc.bdf")
# bdf_def = bdfdef.DefBdf("./NASTRAN/static400/run_all.bdf")

bdf_def.plot_vtk("./generateParaview_static/sp_ref.bdf")

nodesX = bdf_def.get_nodes()
time_plot = 1
time = list(range(6))
uintrinsic = np.zeros((6, len(nodesX), 3))
rintrinsic = np.zeros((6, len(nodesX), 3))

for i, ti in enumerate(time):
    ra = soli.data.staticsystem_s1.ra[ti]
    Rab = soli.data.staticsystem_s1.Cab[ti]
    gridmodel.set_solution(ra, Rab, R0ab)
    disp, coord = interpolation.compute(gridmodel.model1_coord,
                                        gridmodel.model1x_coord,
                                        nodesX)
    uintrinsic[i] = disp
    rintrinsic[i] = coord
    if time_plot is not None:
        if i % time_plot == 0:
            tstep = int(i / time_plot)
            bdf_def.update_bdf(coord, bdf_def.sorted_nodeids)
            bdf_def.plot_vtk(f"./generateParaview_static/sp_L{i}.bdf")


op2model = op2reader.NastranReader("./NASTRAN/static400/run_all.op2",
                                   "./NASTRAN/static400/run_all.bdf",
                                   static=True)
op2model.readModel()
tnastran, unastran = op2model.displacements()


bdf_def2 = bdfdef.DefBdf("./NASTRAN/static400/run_all.bdf")
nodesX2 = bdf_def2.get_nodes()

for i, ti in enumerate(time):
    if i % time_plot == 0:
        tstep = int(i / time_plot)
        coord = nodesX2 + unastran[i,:,:3]
        bdf_def2.update_bdf(coord, bdf_def.sorted_nodeids)
        bdf_def2.plot_vtk(f"./generateParaview_static/sp_op2L{i}.bdf")


err = jnp.array([jnp.linalg.norm(unastran[i,:,:3] - uintrinsic[i]) / 3313 for i in range(6)])
jnp.save("./sp_err.npy",err)


# op2_model = OP2()
# op2_model.set_additional_matrices_to_read({b'OPHP':False, b'OUG1':False, b'OUGV1': False})

# op2_model.read_op2("/Users/ac5015/pCloud Drive/Imperial/Computations/FEM4INAS/Models/SailPlane/SP400/SailPlane_MakeMatc.op2", skip_undefined_matrices=True, combine=False)
# #op2_model.read_op2("./NASTRAN/wing400d.op2", skip_undefined_matrices=True)

# op2model = op2reader.NastranReader("/Users/ac5015/pCloud Drive/Imperial/Computations/FEM4INAS/Models/SailPlane/SP400/SailPlane_MakeMatc.op2",
#                                    "/Users/ac5015/pCloud Drive/Imperial/Computations/FEM4INAS/Models/SailPlane/SP400/SailPlane_MakeMatc.bdf",
#                                    static=True)
# op2model.readModel()
# tnastran, unastran = op2model.displacements()
