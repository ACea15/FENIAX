import fem4inas.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
import pandas as pd
import importlib
import numpy as np
from fem4inas.preprocessor import solution
import fem4inas.unastran.op2reader as op2reader
import fem4inas.plotools.interpolation as interpolation
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef
importlib.reload(bdfdef)
importlib.reload(grid)

df_grid = pd.read_csv('./FEM/structuralGrid', comment="#", sep=" ",
                    names=['x1', 'x2', 'x3', 'fe_order', 'component'])
bdf_model = BDF(debug=True)
bdf_model.read_bdf("./NASTRAN/wing400d.bdf", punch=False)
X = df_grid[['x1','x2','x3']].to_numpy()
gridmodel = grid.RBE3Model(bdf_model, X, 1e-3)


soli = solution.IntrinsicReader("./results_dynamics")
R0ab = soli.data.modes.C0ab

bdf_def = bdfdef.DefBdf("./NASTRAN/wing400d.bdf")
bdf_def.plot_vtk("./generateParaview_dynamics/spref.bdf")

nodesX = bdf_def.get_nodes()
time_plot = 1
time = np.linspace(0, 15000, 10001, dtype=int)
uintrinsic = np.zeros((10001, len(nodesX), 3))
rintrinsic = np.zeros((10001, len(nodesX), 3))

for i, ti in enumerate(time):
    ra = soli.data.dynamicsystem_s1.ra[ti]
    Rab = soli.data.dynamicsystem_s1.Cab[ti]
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
            bdf_def.plot_vtk(f"./generateParaview_dynamics/sp_{tstep}.bdf")

op2model = op2reader.NastranReader("./NASTRAN/wing400d_full.op2",
                                   "./NASTRAN/wing400d_full.bdf")
op2model.readModel()
tnastran, unastran = op2model.displacements()
