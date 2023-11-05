import pathlib
import jax.numpy as jnp
import pdb
import sys
import datetime
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path=None
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.tn = 15001

#inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
#inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_101001'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
config =  configuration.Config(inp)
sol = fem4inas.fem4inas_main.main(input_obj=config)
