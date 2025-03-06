import pathlib
import pdb
import sys
import jax.numpy as jnp
import datetime
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.upyvista as upyvista

# ka = "Ka.npy"
# lk = list(pathlib.Path('./FEM/').glob("**/*Ka.npy*"))

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'Beam1':None}
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 90
inp.fem.eig_type = "inputs"
#inp.fem.fe_order_start = 1
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
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
inp.systems.sett.s1.xloads.follower_points = [[15, 2]]
inp.systems.sett.s1.xloads.x = list(range(11))
inp.systems.sett.s1.xloads.follower_interpolation = [[float(li) for li in np.arange(0.,3300.,300)]]
inp.systems.sett.s1.t = list(range(1,11))
config =  configuration.Config(inp)

# for k, v in config._data_dict['fem'].items():
#     print(f"{k}:{type(v[0])}")


path2config = pathlib.Path("./config.yaml")


configuration.dump_to_yaml(path2config, config, with_comments=True)


sol = feniax.feniax_main.main(input_obj=config)


import importlib
importlib.reload(upyvista)
import feniax.plotools.utils as putils
istruct = putils.IntrinsicStruct(config.fem)
istruct.add_solution(sol.staticsystem_s1.ra)
pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
pl.show_grid()
#pl.view_xy()
for k, v in istruct.map_ra.items():
    pl = upyvista.render_wireframe(points=v, lines=istruct.lines, pl=pl)
pl.show()
