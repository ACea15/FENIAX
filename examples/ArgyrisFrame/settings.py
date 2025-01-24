import pathlib
import jax.numpy as jnp
import pdb
import sys
import datetime
import feniax.plotools.upyvista as upyvista
import feniax.plotools.utils as putils
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main

inp = Inputs()
inp.engine = "intrinsicmodal"
# inp.fem.connectivity = [[1], []]
inp.fem.connectivity = [[]]
inp.fem.grid = "structuralGrid2"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 120
inp.fem.fe_order_start = 1
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
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
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[20, 1]]
inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                     -50.,
                                                     -100.,
                                                     -300.,
                                                     -430.,
                                                     -700.,
                                                      -1040.,
                                                      -1200.,
                                                      -1350.,
                                                      -1500.,
                                                      -1700.,
                                                      -1900
                                                      ]
                                                     ]

inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


config =  configuration.Config(inp)

path2config = pathlib.Path("./config.yaml")
#config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config)

sol = feniax.feniax_main.main(input_obj=config)

PLOT = False
if PLOT:
    import importlib
    importlib.reload(upyvista)
    istruct = putils.IntrinsicStruct(config.fem)
    cstruct = putils.IntrinsicStructComponent(config.fem)
    istruct.add_solution(sol.staticsystem_s1.ra)
    cstruct.add_solution(sol.staticsystem_s1.ra)
    pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
    pl.show_grid()
    pl.view_xy()
    for k, v in istruct.map_ra.items():
        pl = upyvista.render_wireframe(points=v, lines=istruct.lines, pl=pl)
    pl.show()
