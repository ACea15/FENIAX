name="Simo45Dead"
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

simo45beam_folder = feniax.PATH / "../examples/Simo45Beam"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'Beam1':None}
inp.fem.folder = pathlib.Path(f'{simo45beam_folder}/FEM/')
inp.fem.num_modes = 90
inp.fem.eig_type = "inputs"
#inp.fem.fe_order_start = 1
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
inp.systems.sett.s1.xloads.dead_forces = True
inp.systems.sett.s1.xloads.dead_points = [[15, 2]]
inp.systems.sett.s1.xloads.x = list(range(11))
inp.systems.sett.s1.xloads.dead_interpolation = [[float(li) for li in np.arange(0.,3300.,300)]]
inp.systems.sett.s1.t = list(range(1,11))
config_simo45d =  configuration.Config(inp)
sol_simodead = feniax.feniax_main.main(input_obj=config_simo45d)
