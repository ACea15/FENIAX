from pyNastran.op2.op2 import OP2
import feniax.plotools.nastranvtk.bdfdef as bdfdef
import feniax.plotools.interpolation as interpolation
from feniax.preprocessor import solution
import feniax.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
import pandas as pd
import feniax.plotools.reconstruction as rec
import feniax.preprocessor.configuration as configuration
import jax.numpy as jnp
import feniax.plotools.nastranvtk as nastranvtk

Nastran = False

if Nastran:
    op2 = OP2()
    op2.read_op2("./NASTRAN/runs/XRF1-144trim.op2")

    bdfdef.vtkSol_fromop2("./NASTRAN/XRF1-144trim.bdf", "./NASTRAN/runs/XRF1-144trim.op2", size_card=16)


# op2 = OP2()
# op2.read_op2("./NASTRAN/103/XRF1-103cfo.op2")



results = "results4g"
config = configuration.Config.from_file(f"./{results}/config.yaml")
sol = solution.IntrinsicReader(f"./{results}")

r, u = rec.rbf_based("./NASTRAN/XRF1-144trim.bdf",
                     config.fem.X,
                     jnp.array([0,1,2,3]), #sol.data.staticsystem_s1.t,
                     sol.data.staticsystem_s1.ra,
                     sol.data.staticsystem_s1.Cab,
                     sol.data.modes.C0ab,
                     vtkpath="./results4gVTK/conf",
                     plot_timeinterval=1,
                     plot_ref=False,
                     tolerance=1e-2,
                     size_cards=16,
                     rbe3s_full=False)


results = "results4gl"
configl = configuration.Config.from_file(f"./{results}/config.yaml")
soll = solution.IntrinsicReader(f"./{results}")
load = 0
q2 = soll.data.staticsystem_s1.q[load, 0:-1]
q0i = - q2[2:]/ soll.data.modes.omega[2:]
q0 = jnp.hstack([q2[:2], q0i])  
nastranvtk.vtkSol_op2Modes("./NASTRAN/XRF1-144trim.bdf",
                           "./NASTRAN/runs/XRF1-144trim.op2",
                           q0,
                           label="Sol",
                           size_card=16,
                           write_path="./results4glVTK/",
                           write_ref=True)




OLD_rec = False
if OLD_rec:
    df_grid = pd.read_csv('./FEM/structuralGridc.txt', comment="#", sep=" ",
                        names=['x1', 'x2', 'x3', 'fe_order', 'component'])
    bdf_model = BDF(debug=True)
    bdf_model.read_bdf("./NASTRAN/XRF1-144trim.bdf", punch=False)
    X = df_grid[['x1','x2','x3']].to_numpy()
    gridmodel = grid.RBE3Model(bdf_model, X, 1e-3)

    soli = solution.IntrinsicReader("./results4g")

    R0ab = soli.data.modes.C0ab
    ra = soli.data.staticsystem_s1.ra[0]
    Rab = soli.data.staticsystem_s1.Cab[0]

    time = [0,1,2,3,4]#np.linspace(0,15000,1000,dtype=int)

    #bdf_def = bdfdef.DefBdf("./NASTRAN/XRF1-144trim.bdf")
    bdf_def = bdfdef.DefBdf("/media/acea/work/projects/FENIAX/examples/XRF1/NASTRAN/XRF1-146run.bdf")
    #bdf_def.plot_vtk("./generate_trim/ref.bdf")

    nodesX = bdf_def.get_nodes()

    for i, ti in enumerate(time):
        ra = soli.data.staticsystem_s1.ra[ti]
        Rab = soli.data.staticsystem_s1.Cab[ti]
        gridmodel.set_solution(ra,Rab,R0ab)
        disp, coord = interpolation.compute(gridmodel.model1_coord,
                                            gridmodel.model1x_coord,
                                            nodesX)
        bdf_def.update_bdf(coord, bdf_def.sorted_nodeids)
        bdf_def.plot_vtk(f"./generate_trim/xrf1_{i}.bdf")
