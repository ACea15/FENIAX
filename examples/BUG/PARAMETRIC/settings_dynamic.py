import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

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
inp.fem.num_modes = 15
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results_dynamic")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.t1 = 0.5
inp.systems.sett.s1.tn = 11
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                           tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_001001'
inp.systems.sett.s1.xloads.follower_forces = True
#inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]
inp.systems.sett.s1.xloads.follower_points = [[35, 2], [61, 2]]

inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
force=[0.05 * -5e4, 1 * -5e4, 0., 0.]
inp.systems.sett.s1.xloads.follower_interpolation = [force,force]

#config =  configuration.Config(inp)

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)
