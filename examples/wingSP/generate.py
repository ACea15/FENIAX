import feniax.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
import pandas as pd
import importlib
import numpy as np
from feniax.preprocessor import solution
import feniax.unastran.op2reader as op2reader
import feniax.plotools.interpolation as interpolation
import feniax.plotools.nastranvtk.bdfdef as bdfdef
importlib.reload(bdfdef)

importlib.reload(grid)
df_grid = pd.read_csv('./FEM/structuralGrid', comment="#", sep=" ",
                    names=['x1', 'x2', 'x3', 'fe_order', 'component'])
bdf_model = BDF(debug=True)
bdf_model.read_bdf("./NASTRAN/wing400d.bdf", punch=False)
X = df_grid[['x1','x2','x3']].to_numpy()
gridmodel = grid.RBE3Model(bdf_model, X, 1e-3)

op2model = op2reader.NastranReader("./NASTRAN/wing400d.op2")
op2model.readModel()
t, u = op2model.displacements()

soli = solution.IntrinsicReader("./results_dynamics")

R0ab = soli.data.modes.C0ab
ra = soli.data.dynamicsystem_s1.ra[0]
Rab = soli.data.dynamicsystem_s1.Cab[0]

bdf_def = bdfdef.DefBdf("./NASTRAN/wing400d.bdf")
bdf_def.plot_vtk("./generate_dynamics/spref.bdf")

nodesX = bdf_def.get_nodes()
PLOT = False
if PLOT:
    time = np.linspace(0,15000,1000,dtype=int)
    for i, ti in enumerate(time):
        ra = soli.data.dynamicsystem_s1.ra[ti]
        Rab = soli.data.dynamicsystem_s1.Cab[ti]
        gridmodel.set_solution(ra, Rab, R0ab)
        disp, coord = interpolation.compute(gridmodel.model1_coord,
                                            gridmodel.model1x_coord,
                                            nodesX)
        bdf_def.update_bdf(coord, bdf_def.sorted_nodeids)
        bdf_def.plot_vtk(f"./generate_dynamics/sp_{i}.bdf")

time2 = np.linspace(0,15000,10001,dtype=int)
nodesX = bdf_def.get_nodes()
uintrinsic = np.zeros((10001, len(nodesX), 3))
rintrinsic = np.zeros((10001, len(nodesX), 3))
for i, ti in enumerate(time2):

    ra = soli.data.dynamicsystem_s1.ra[ti]
    Rab = soli.data.dynamicsystem_s1.Cab[ti]
    gridmodel.set_solution(ra, Rab, R0ab)
    disp, coord = interpolation.compute(gridmodel.model1_coord,
                                        gridmodel.model1x_coord,
                                        nodesX)
    uintrinsic[i] = disp
    rintrinsic[i] = coord
