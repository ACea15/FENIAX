import pathlib
import pdb
import sys
import numpy as np
import datetime
import time 
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import scipy.linalg


inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'0': None}
inp.fem.grid = ""
inp.fem.num_modes = 3
inp.fem.Ka = Ka2
inp.fem.Ma = Ma2
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.array([0, 0, 0])
inp.fem.eigenvecs = v2
inp.driver.typeof = "intrinsicmultibody"
inp.driver.sol_path= pathlib.Path(
    f"./results_try")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 20.
inp.systems.sett.s1.tn = 2000
inp.systems.sett.s1.solver_library = "diffrax" #"runge_kutta" #"diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")# "rk4")
inp.systems.sett.s1.xloads.gravity_forces = True
inp.multibody.fems_input.b1.connectivity = {'0': None}       
inp.multibody.fems_input.b1.grid = ""                        
inp.multibody.fems_input.b1.num_modes = 3                    
inp.multibody.fems_input.b1.Ka = Ka2                         
inp.multibody.fems_input.b1.Ma = Ma2                         
inp.multibody.fems_input.b1.eig_type = "input_memory"        
inp.multibody.fems_input.b1.eigenvals = jnp.array([0, 0, 0]) 
inp.multibody.fems_input.b1.eigenvecs = v2                   
inp.multibody.systems_input.b1.xloads.gravity_forces = True
inp.multibody.constraints_input.c12.type_name = "spherical"
inp.multibody.constraints_input.c12.node = 0
inp.multibody.constraints_input.c12.body = "b2"
inp.multibody.constraints_input.c12.node_father = -1
inp.multibody.constraints_input.c12.body_father = "b0"

config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
