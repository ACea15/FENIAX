import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import numpy as np

fem_folder='./FEM/'
sol_path='./results_struct'
#load=[0.,2e4,2.5e4,3e4,4e4,4.8e4,5.3e4]#*0.5
#load=[0.,5e5,1e6,1.3e6,1.5e6,1.8e6,2e6]#*0.5
load=[0.,5e4,1e5,1.3e5,1.5e5,1.8e5,2e5]

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

inp.fem.folder = pathlib.Path(fem_folder)
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(sol_path)
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                           tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_001001'
inp.systems.sett.s1.xloads.follower_forces = True
#inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]
inp.systems.sett.s1.xloads.follower_points = [[35, 2], [61, 2]]

inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
force=load
inp.systems.sett.s1.xloads.follower_interpolation = [force,force]
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

#config =  configuration.Config(inp)

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)
