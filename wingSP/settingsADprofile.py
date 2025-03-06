import pathlib
import jax.numpy as jnp
import pdb
import sys
import datetime
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax
jax.config.update("jax_enable_x64", True)

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.ad_on = True
inp.driver.sol_path= pathlib.Path(
    "./resultsAD")

#inp.driver.sol_path=None
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.t1 = 15.
inp.system.tn = 10001

inp.system.solver_library = "diffrax"
# inp.system.solver_library = "runge_kutta"
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="Dopri5") #Dopri5
inp.system.xloads.follower_forces = True
inp.system.xloads.follower_points = [[23, 0],
                                     [23, 2]]
inp.system.xloads.x = [0, 4, 4+1e-6, 20]
inp.system.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                            [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                            ]
inp.system.ad = dict(inputs=dict(alpha=1.),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="max",
                     objective_var="X2",
                     objective_args=dict(nodes=(1,), components=(2,))
                     )
config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
