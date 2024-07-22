name="ArgyrisBeam"
import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import fem4inas.plotools.uplotly as uplotly
import fem4inas.plotools.utils as putils
import fem4inas.preprocessor.solution as solution
import fem4inas.unastran.op2reader as op2reader

argyrisbeam_folder = fem4inas.PATH / "../examples/ArgyrisBeam"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[]]
inp.fem.folder = pathlib.Path(f'{argyrisbeam_folder}/FEM/')
inp.fem.num_modes = 150
inp.fem.eig_type = "inputs"
#inp.fem.fe_order_start = 1
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./{name}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
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
config_argy =  configuration.Config(inp)
sol_argy = fem4inas.fem4inas_main.main(input_obj=config_argy)
