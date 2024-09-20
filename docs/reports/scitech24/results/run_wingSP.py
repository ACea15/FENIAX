name="wingSP"
import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader

wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
inp.fem.num_modes = 50
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"{name}")
#inp.driver.sol_path=None
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.tn = 15001
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
config_wsp =  configuration.Config(inp)
sol_wsp = feniax.feniax_main.main(input_obj=config_wsp)
