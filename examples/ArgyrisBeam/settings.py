import pathlib
import pdb
import sys
import numpy as np
import datetime
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[]]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 150
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
# inp.systems.sett.s1.label = 'dq_001001'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[25, 1]]
inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6, 7]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                     -3.7e3,
                                                     -12.1e3,
                                                     -17.5e3,
                                                     -39.3e3,
                                                     -61.0e3,
                                                     -94.5e3,
                                                     -120e3]
                                                     ]
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6, 7]
#config =  configuration.Config(inp)

# for k, v in config._data_dict['fem'].items():
#     print(f"{k}:{type(v[0])}")


# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)

# import pickle
# with open('../../tests/intrinsic/structural_static/data/.pickle', 'wb') as fp:
#     pickle.dump(sol, fp)
