import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import pyNastran.op4.op4 as op4


inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load("./FEM/Dreal50.npy")
inp.fem.eigenvecs = jnp.load("./FEM/Vreal50.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./resultsGust_g1i1_m50diffrax")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 7.5
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#"rk4")
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"./NASTRAN/AERO/AICs081_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"./NASTRAN/AERO/AICsQhj081_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"./NASTRAN/AERO/Poles081_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 1.
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = jnp.load("./NASTRAN/AERO/Dihedral.npy")
inp.systems.sett.s1.aero.gust.collocation_points = "./NASTRAN/AERO/Control_nodes.npy"

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)
