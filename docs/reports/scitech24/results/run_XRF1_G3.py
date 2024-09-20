name="Gust3"
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

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.tn = 3001
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust_settings.shift = 0.
inp.systems.sett.s1.aero.gust_settings.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust_settings.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225  
inp.systems.sett.s1.aero.gust_settings.intensity = 14.0732311562*0.01 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust_settings.length = 67.
inp.systems.sett.s1.aero.gust_settings.step = 0.5
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
config_gust3 =  configuration.Config(inp)
sol_gust3 = feniax.feniax_main.main(input_obj=config_gust3)
