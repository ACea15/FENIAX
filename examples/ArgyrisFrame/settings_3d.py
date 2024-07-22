import pathlib
import jax.numpy as jnp
import pdb
import sys
import datetime
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import fem4inas.plotools.upyvista as upyvista

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[1], []]
#inp.fem.connectivity = [[]]
#inp.fem.grid = "structuralGrid2"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 120
inp.fem.fe_order_start = 1
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results3D_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
#inp.ex.Cab_xtol = 1e-4
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_001001'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[20, 1],
                                              [20, 4]]
inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inp.systems.sett.s1.xloads.follower_interpolation = [
    [0., -50., -75,    -100, -150, -200,  -250,    -300,  -400,    -500,  -550,    -600],
    [0., -5e3, -6.5e3, -8e3, -9e3, -10e3, -12.5e3, -15e3, -17.5e3, -20e3, -22.5e3, -25e3]]

inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

config =  configuration.Config(inp)

sol = fem4inas.fem4inas_main.main(input_obj=config)

import importlib
importlib.reload(upyvista)
istruct = upyvista.IntrinsicStruct(config.fem)
istruct.add_solution(sol.staticsystem_s1.ra)
pl = upyvista.render_wireframe(points=istruct.mappoints[1], lines=istruct.lines)
for k, v in istruct.mappoints.items():
    if k != 1:
        pl = upyvista.render_wireframe(points=v, lines=istruct.lines, pl=pl)
        
