import pathlib
import pdb
import sys
import numpy as np
import datetime
import time 
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import scipy.linalg

Ka = jnp.load('./FEMshell50/Ka.npy')
Ma = jnp.load('./FEMshell50/Ma.npy')
w, v = scipy.linalg.eigh(Ka, Ma)

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'0': None}
inp.fem.folder = pathlib.Path('./FEMshell50')
inp.fem.num_modes = 300
inp.fem.eig_type = "scipy"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_try")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.dt = 1e-3
inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
inp.systems.sett.s1.xloads.dead_forces = True
inp.systems.sett.s1.xloads.dead_points = [[49, 0],
                                          [49, 5]]
inp.systems.sett.s1.xloads.x = [0., 2.5, 2.5+1e-6, 10.5]
inp.systems.sett.s1.xloads.dead_interpolation = [[8., 8., 0., 0.],
                                                 [-80., -80., 0., 0.]
                                                 ]

config =  configuration.Config(inp)

time1 = time.time()
sol = fem4inas.fem4inas_main.main(input_obj=config)
time2 = time.time()
print(time2-time1)
