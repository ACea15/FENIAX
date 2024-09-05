import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main

inp = Inputs()
inp.engine = "intrinsicmodal"
# WARNING: eigs need to be input as they are implicit in the aero matrices
inp.fem.eig_type = "input_memory"
# inp.fem.eigenvals = jnp.load("./FEM/Dreal50.npy")
# inp.fem.eigenvecs = jnp.load("./FEM/Vreal50.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.eig_names = ["Dreal50.npy", "Vreal50.npy"]
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
inp.driver.ad_on = True
inp.driver.sol_path = pathlib.Path(
    "./resultsGust_g1i2_m50AD")
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamicAD"
inp.system.t1 = 7.5
inp.system.tn = 2001
inp.system.solver_library = "runge_kutta" #"diffrax"  
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4") #Dopri5
inp.system.xloads.modalaero_forces = True
inp.system.q0treatment = 2
inp.system.aero.c_ref = 7.271
inp.system.aero.u_inf = 200.
inp.system.aero.rho_inf = 1.225
inp.system.aero.A = f"./NASTRAN/AERO/AICs081_8r{inp.fem.num_modes}.npy"
inp.system.aero.D = f"./NASTRAN/AERO/AICsQhj081_8r{inp.fem.num_modes}.npy"
inp.system.aero.poles = f"./NASTRAN/AERO/Poles081_8r{inp.fem.num_modes}.npy"
inp.system.aero.gust_profile = "mc"
inp.system.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.system.aero.gust.length = 67.
inp.system.aero.gust.step = 1.
inp.system.aero.gust.shift = 0.
inp.system.aero.gust.panels_dihedral = jnp.load("./NASTRAN/AERO/Dihedral.npy")
inp.system.aero.gust.collocation_points = "./NASTRAN/AERO/Control_nodes.npy"
inp.system.ad = dict(inputs=dict(length = 67., intensity = 14.0732311562*2, u_inf=200., rho_inf = 1.225),
                     input_type="gust1",
                     grad_type="jacfwd",
                     objective_fun="max",
                     objective_var="X2",
                     objective_args=dict(nodes=(5,), components=(2,3), axis=0)
                     )

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = feniax.feniax_main.main(input_obj=config)
# sol.dynamicsystem_sys1.jac
