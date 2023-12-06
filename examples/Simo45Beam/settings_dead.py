import pathlib
import pdb
import sys
import jax.numpy as jnp
import datetime

import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import fem4inas.plotools.upyvista as upyvista

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'Beam1':None}
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 90
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./resultsDead_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_00101'
inp.systems.sett.s1.xloads.dead_forces = True
inp.systems.sett.s1.xloads.dead_points = [[15, 2]]
inp.systems.sett.s1.xloads.x = list(range(11))
inp.systems.sett.s1.xloads.dead_interpolation = [jnp.arange(0,3300,300)]
inp.systems.sett.s1.t = list(range(1,11))
config =  configuration.Config(inp)
sol = fem4inas.fem4inas_main.main(input_obj=config)


import importlib
importlib.reload(upyvista)
istruct = upyvista.IntrinsicStruct(config.fem)
istruct.add_solution(sol.staticsystem_s1.ra)
pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
pl.show_grid()
pl.view_xy()
for k, v in istruct.mappoints.items():
    pl = upyvista.render_wireframe(points=v, lines=istruct.lines, pl=pl)
#pl.save_graphic("try1.pdf")
#pl.save("try1.vtk")
pl.show()




mesh = upyvista.render_mesh(points=config.fem.X, lines=istruct.lines)
mesh.save("try.vtk")
for k, v in istruct.mappoints.items():
    mesh = upyvista.render_mesh(points=v, lines=istruct.lines)
    mesh.save(f"try{k}.vtk")
