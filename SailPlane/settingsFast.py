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
inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                           'LWingInner'],
                            FuselageBack=['BottomTail',
                                          'Fin'],
                            RWingInner=['RWingOuter'],
                            RWingOuter=None,
                            LWingInner=['LWingOuter'],
                            LWingOuter=None,
                            BottomTail=['LHorizontalStabilizer',
                                        'RHorizontalStabilizer'],
                            RHorizontalStabilizer=None,
                            LHorizontalStabilizer=None,
                            Fin=None
                            )

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
inp.driver.sol_path = pathlib.Path(
    "./results_staticFast")
inp.driver.fast_on = True
inp.simulation.typeof = "single"
inp.system.operationalmode = "fast"
inp.system.solution = "static"
#inp.system.name = "s1"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.system.xloads.follower_forces = True
inp.system.xloads.follower_points = [[25, 2], [48, 2]]

inp.system.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.system.xloads.follower_interpolation = [[0.,
                                             2e5,
                                             2.5e5,
                                             3.e5,
                                             4.e5,
                                             4.8e5,
                                             5.3e5],
                                            [0.,
                                             2e5,
                                             2.5e5,
                                             3.e5,
                                             4.e5,
                                             4.8e5,
                                             5.3e5]
                                            ]
inp.system.t = [1, 2, 3, 4, 5, 6]
config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
