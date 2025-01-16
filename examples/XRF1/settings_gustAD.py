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
inp.fem.eig_names = ["Dreal70.npy", "Vreal70.npy"]
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.driver.ad_on = True
inp.driver.sol_path = pathlib.Path(
    "./resultsGust_g1i2_m70AD")
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.t1 = 5
inp.system.tn = 1001
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
node = 5
inp.system.ad = dict(inputs=dict(length = 67., intensity = 14.0732311562*2, u_inf=200., rho_inf = 1.225),
                     input_type="gust1",
                     grad_type="jacrev",
                     objective_fun="max",
                     objective_var="X2",
                     objective_args=dict(nodes=(node,), components=(0,1,2,3,4,5))
                     )

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = feniax.feniax_main.main(input_obj=config)
# sol.dynamicsystem_s1.jac

ad_check = False
if ad_check:
    #################################################################################
    epsilon = 1e-4
    inp.system.ad = dict(inputs=dict(length = 67. + epsilon, intensity = 14.0732311562*2, u_inf=200., rho_inf = 1.225),
                         input_type="gust1",
                         grad_type="jacrev",
                         objective_fun="max",
                         objective_var="X2",
                         objective_args=dict(nodes=(5,), components=(0,1,2,3,4,5))
                         )
    config2 =  configuration.Config(inp)
    sol2 = feniax.feniax_main.main(input_obj=config2)
    print((sol2.dynamicsystem_s1.f_ad - sol.dynamicsystem_s1.f_ad) / epsilon)
    print(sol.dynamicsystem_s1.jac['length'])

    #################################################################################

    #################################################################################
    epsilon = 1e-2
    inp.system.ad = dict(inputs=dict(length = 67., intensity = 14.0732311562*2 + epsilon, u_inf=200., rho_inf = 1.225),
                         input_type="gust1",
                         grad_type="jacrev",
                         objective_fun="max",
                         objective_var="X2",
                         objective_args=dict(nodes=(5,), components=(0,1,2,3,4,5))
                         )
    config2 =  configuration.Config(inp)
    sol2 = feniax.feniax_main.main(input_obj=config2)
    print((sol2.dynamicsystem_s1.f_ad - sol.dynamicsystem_s1.f_ad) / epsilon)
    print(sol.dynamicsystem_s1.jac['intensity'])


    #################################################################################

    #################################################################################
    epsilon = 1e-2
    inp.system.ad = dict(inputs=dict(length = 67., intensity = 14.0732311562*2, u_inf=200., rho_inf = 1.225 + epsilon),
                         input_type="gust1",
                         grad_type="jacrev",
                         objective_fun="max",
                         objective_var="X2",
                         objective_args=dict(nodes=(5,), components=(0,1,2,3,4,5))
                         )
    config2 =  configuration.Config(inp)
    sol2 = feniax.feniax_main.main(input_obj=config2)
    print((sol2.dynamicsystem_s1.f_ad - sol.dynamicsystem_s1.f_ad) / epsilon)
    print(sol.dynamicsystem_s1.jac['rho_inf'])

    #################################################################################

import feniax.intrinsic.objectives as objectives
inp.fem.eigenvals = jnp.load("./FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load("./FEM/Vreal70.npy").T
inp.system.ad = None
inp.driver.ad_on = False
inp.driver.sol_path = pathlib.Path(
    "./resultsGustAD3")

node = 5
config3 =  configuration.Config(inp)
sol3 = feniax.feniax_main.main(input_obj=config3)
fobj3 = objectives.X2_MAX(sol3.dynamicsystem_s1.X2,
                          nodes=jnp.array([node]),
                          components=jnp.array([0,1,2,3,4,5]),
                          t=jnp.arange(config3.system.tn),
                          axis=0)
#################################################################################
inp.driver.sol_path = pathlib.Path(
    "./resultsGustAD4")
epsilon = 1e-4
inp.system.aero.gust.intensity = 14.0732311562*2 + epsilon
config4 =  configuration.Config(inp)
sol4 = feniax.feniax_main.main(input_obj=config4)
fobj4 = objectives.X2_MAX(sol4.dynamicsystem_s1.X2,
                          nodes=jnp.array([node]),
                          components=jnp.array([0,1,2,3,4,5]),
                          t=jnp.arange(config4.system.tn),
                          axis=0)

print((fobj4 - fobj3) / epsilon)
print(sol.dynamicsystem_s1.jac['intensity'])

#################################################################################
epsilon = 1e-4
inp.system.aero.gust.intensity = 14.0732311562*2
inp.system.aero.gust.length = 67. + epsilon

inp.driver.sol_path = pathlib.Path(
    "./resultsGustAD5")

config5 =  configuration.Config(inp)
sol5 = feniax.feniax_main.main(input_obj=config5)
fobj5 = objectives.X2_MAX(sol5.dynamicsystem_s1.X2,
                          nodes=jnp.array([node]),
                          components=jnp.array([0,1,2,3,4,5]),
                          t=jnp.arange(config5.system.tn),
                          axis=0)

print((fobj5 - fobj3) / epsilon)
print(sol.dynamicsystem_s1.jac['length'])
#################################################################################
epsilon = 1e-4
inp.system.aero.gust.intensity = 14.0732311562*2
inp.system.aero.gust.length = 67.
inp.system.aero.rho_inf = 1.225 + epsilon

inp.driver.sol_path = pathlib.Path(
    "./resultsGustAD6")

config6 =  configuration.Config(inp)
sol6 = feniax.feniax_main.main(input_obj=config6)
fobj6 = objectives.X2_MAX(sol6.dynamicsystem_s1.X2,
                          nodes=jnp.array([node]),
                          components=jnp.array([0,1,2,3,4,5]),
                          t=jnp.arange(config6.system.tn),
                          axis=0)

print((fobj6 - fobj3) / epsilon)
print(sol.dynamicsystem_s1.jac['rho_inf'])
