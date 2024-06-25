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
                        [26], [], [28], [], [30], [], []
                        ]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results_fg2b")
inp.simulation.typeof = "serial"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-7,
                                           atol=1e-7,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
inp.systems.sett.s1.xloads.gravity = 9.807*2
inp.systems.sett.s1.t = [0.5, 1.]
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"./NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = jnp.load(f"./NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy")
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]

inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5")#"rk4")
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.xloads.modalaero_forces = True
inp.systems.sett.s2.xloads.gravity_forces = True
#inp.systems.sett.s2.xloads.gravity = 9.807*2
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.
inp.systems.sett.s2.dt = 5e-3
# inp.systems.sett.s2.aero.c_ref = 7.271
# inp.systems.sett.s2.aero.u_inf = 200.
# inp.systems.sett.s2.aero.rho_inf = 1.225
# inp.systems.sett.s2.aero.A = f"./NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.poles = f"./NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.D = f"./NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.gust_profile = "mc"
inp.systems.sett.s2.aero.gust.intensity = 14.0732311562*1 #11.304727674272842/10000
inp.systems.sett.s2.aero.gust.length = 67.
inp.systems.sett.s2.aero.gust.step = 1.
inp.systems.sett.s2.aero.gust.shift = 0.
inp.systems.sett.s2.aero.gust.panels_dihedral = jnp.load("./NASTRAN/AERO/Dihedral.npy")
inp.systems.sett.s2.aero.gust.collocation_points = "./NASTRAN/AERO/Control_nodes.npy"


# inp.systems.sett.s2.solution = "dynamic"
# inp.systems.sett.s2.target = "level"
# inp.systems.sett.s2.bc1 = 'free'
# inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
# inp.systems.sett.s2.solver_function = "ode"
# inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5")#"rk4")
# inp.systems.sett.s2.q0treatment = 1
# inp.systems.sett.s2.xloads.modalaero_forces = True
# inp.systems.sett.s2.xloads.gravity_forces = True
# inp.systems.sett.s2.t1 = 1.
# inp.systems.sett.s2.dt = 5e-3
# inp.systems.sett.s2.aero.c_ref = 7.271
# inp.systems.sett.s2.aero.u_inf = 200.
# inp.systems.sett.s2.aero.rho_inf = 1.225
# inp.systems.sett.s2.aero.A = f"./NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
# inp.systems.sett.s2.aero.poles = f"./NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
# inp.systems.sett.s2.aero.D = f"./NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
# inp.systems.sett.s2.aero.gust_profile = "mc"
# inp.systems.sett.s2.aero.gust.intensity = 14.0732311562*1 #11.304727674272842/10000
# inp.systems.sett.s2.aero.gust.length = 67.
# inp.systems.sett.s2.aero.gust.step = 1.
# inp.systems.sett.s2.aero.gust.shift = 0.
# inp.systems.sett.s2.aero.gust.panels_dihedral = jnp.load("./NASTRAN/AERO/Dihedral.npy")
# inp.systems.sett.s2.aero.gust.collocation_points = "./NASTRAN/AERO/Control_nodes.npy"



config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)

