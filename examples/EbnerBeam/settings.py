import pathlib
import datetime

import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'Beam1': ['Beam2'],
                        'Beam2': None}
#inp.fem.connectivity = [[1], []]

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 60
inp.fem.eig_type = "inputs"
inp.fem.eig_cutoff = 1e-4
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
#                                             tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_00101'
inp.systems.sett.s1.xloads.dead_forces = True
# inp.systems.sett.s1.xloads.gravity_forces = True
inp.systems.sett.s1.xloads.dead_points = [[9, 2], [18, 2]]
inp.systems.sett.s1.xloads.x = [0, 1, 2]
inp.systems.sett.s1.xloads.dead_interpolation = [[0, 0.85 / 2, 0.85],
                                                 [0, 1.35 / 2, 1.35]
                                                 ]
inp.systems.sett.s1.t = [1, 2] # jnp.linspace(0,2,21)

config = configuration.Config(inp)

# config.systems.mapper['s1'].xloads.build_gravity(config.fem.Ma, config.fem.Mfe_order)

# config.systems.mapper['s1'].xloads.build_point_dead(config.fem.num_nodes)
# config.systems.mapper['s1'].xloads.force_dead

sol = feniax.feniax_main.main(input_obj=config)

