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
inp.fem.eigenvals = jnp.load("./FEM/Dreal100.npy")
inp.fem.eigenvecs = jnp.load("./FEM/Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results4gl")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.nonlinear = -1
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
inp.systems.sett.s1.xloads.gravity = 9.807 * 4
#inp.systems.sett.s1.xloads.gravity = 0.5
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = f"./NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = jnp.load(f"./NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy")
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]

config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)



fus0_vect = [sol.staticsystem_s1.ra[0,:,1] -
            sol.staticsystem_s1.ra[0,:,0] for ti in range(len(inp.systems.sett.s1.t))]
fus_vect = [sol.staticsystem_s1.ra[ti,:,1] -
            sol.staticsystem_s1.ra[ti,:,0] for ti in range(len(inp.systems.sett.s1.t))]
fus0_norm = [jnp.linalg.norm(fus0_vect[ti]) for ti in range(len(inp.systems.sett.s1.t))]
fus_norm = [jnp.linalg.norm(fus_vect[ti]) for ti in range(len(inp.systems.sett.s1.t))]
aoa = [180/jnp.pi*jnp.arccos(fus_vect[ti].dot(fus0_vect[ti]) / (fus0_norm[ti] * fus_norm[ti]))
                         for ti in range(len(inp.systems.sett.s1.t))]

load=-1
q2 = sol.staticsystem_s1.q[load, 0:-1]
q0i = - q2[2:]/ sol.modes.omega[2:]
q0 = jnp.hstack([q2[:2], q0i])  
X0 = jnp.tensordot(sol.modes.phi1, q0, axes=(0, 0))
print(X0[-2,0])
print(sol.staticsystem_s1.q[load,-1])
