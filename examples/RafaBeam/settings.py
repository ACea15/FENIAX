import pathlib
import pdb
import sys
import numpy as np
import datetime
import time 
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 100
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 2.5
inp.systems.sett.s1.tn = 2501
inp.systems.sett.s1.solver_library = "runge_kutta" #"diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.init_states = dict(q1=["axial_parabolic",
                                           ([0., 3., 3., 0., 0., 0], 20.)
                                           ])
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_101'
config =  configuration.Config(inp)
time1 = time.time()
sol = feniax.feniax_main.main(input_obj=config)
time2 = time.time()
