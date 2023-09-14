import pathlib
import pdb
import sys

import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[]]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
#inp.fem.fe_order_start = 1
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.ex.Cab_xtol = 1e-4
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm=jnp.linalg.norm,
                                           kappa=0.01)
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
inp.systems.sett.s1.label = 'dq_0'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[25, 1]]
inp.systems.sett.s1.xloads.follower_interpolation = [[[1., -3.7e3],
                                                      [2, -12.1e3],
                                                      [3, -17.5e3],
                                                      [4, -39.3e3],
                                                      [5, -61.0e3],
                                                      [6, -94.5e3],
                                                      [7, -120e3]
                                                      ]
                                                     ]
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6, 7]
config =  configuration.Config(inp)

# for k, v in config._data_dict['fem'].items():
#     print(f"{k}:{type(v[0])}")


path2config = pathlib.Path("./config.yaml")
#config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)


# def y(*args):
#     print(args)

