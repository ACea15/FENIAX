import pathlib
import pdb
import sys
import datetime
#import jax.numpy as jnp
#import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain

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

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    f"./results_staticShard")
inp.simulation.typeof = "single"
inp.system.name = "sh1"
inp.system.solution = "static"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                  atol=1e-6,
                                  max_steps=50,
                                  norm="linalg_norm",
                                  kappa=0.01)
inp.system.xloads.follower_forces = True

inp.system.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.system.t = [1, 2, 3, 4, 5, 6]
inputforces = dict(follower_points=[[[25, 2], [48, 2]],
                                    [[20, 2], [43, 2]],
                                    [[15, 2], [38, 2]],
                                    [[10, 2], [33, 2]],
                                    [[25, 5], [48, 5]],
                                    [[20, 5], [43, 5]],
                                    [[15, 5], [38, 5]],
                                    [[10, 5], [33, 5]],
                                    ],
                   follower_interpolation= [[[0.,2e5,2.5e5,3.e5,4.e5,4.8e5,5.3e5],
                                            [0.,2e5,2.5e5,3.e5,4.e5,4.8e5,5.3e5]
                                            ]
                                            ]*8
                   )
inp.system.shard = dict(input_type="pointforces",
                        inputs=inputforces)
#config =  configuration.Config(inp)

# path2config = pathlib.Path("./config.yaml")
#config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)
