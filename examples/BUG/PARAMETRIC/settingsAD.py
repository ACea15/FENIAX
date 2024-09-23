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
inp.fem.eig_type = "inputs"
inp.fem.connectivity = dict(FusBack=['FusTail',
                                     'VTP'],
                            FusFront=None,
                            RWing=None,
                            LWing=None,
                            FusTail=None,
                            VTP=['HTP', 'VTPTail'],
                            HTP=['RHTP', 'LHTP'],
                            VTPTail=None,
                            RHTP=None,
                            LHTP=None,
                            )
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
inp.driver.ad_on = True
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "staticAD"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                  atol=1e-6,
                                  max_steps=50,
                                  norm="linalg_norm",
                                  kappa=0.01)
inp.system.xloads.follower_forces = True
inp.system.xloads.follower_points = [[35, 2], [61, 2]]

inp.system.xloads.x = [0, 1, 2, 3, 4, 5, 6]
load=[0.,4e3,7e3,1.1e4,1.4e4,1.7e4,2e4]
#load=[0.,5e4,1e5,1.3e5,1.5e5,1.8e5,2e5]
inp.system.xloads.follower_interpolation = [load,load]

inp.system.t = [1, 2, 3, 4, 5, 6]
inp.driver.sol_path = pathlib.Path("./results_staticADfem")

inp.system.ad = dict(inputs=dict(Ka=jnp.load("./FEM/Ka.npy"),
                                 Ma=jnp.load("./FEM/Ma.npy"),
                                 eigenvals=jnp.load("./FEM/eigenvals.npy"),
                                 eigenvecs=jnp.load("./FEM/eigenvecs.npy")
                                 ),
                     input_type="fem",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(35,), components=(2,))
                     )
config =  configuration.Config(inp)

sol = feniax.feniax_main.main(input_obj=config)