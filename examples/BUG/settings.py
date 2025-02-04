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

inp.fem.connectivity = dict(# FusWing=['RWing',
                            #          'LWing'],
                            FusBack=['FusTail',
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
inp.fem.eig_names = ["eigenvals_50.npy", "eigenvecs_50.npy"]
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results_static")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.x = [0.,1.]
inp.systems.sett.s1.t = [0., 0.25, 0.5, 0.75, 1]
inp.systems.sett.s1.aero.c_ref = 3.
inp.systems.sett.s1.aero.u_inf = 160.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.poles = "./AERO/PolesDd1c7F1Scao-50.npy"
inp.systems.sett.s1.aero.A = f"./AERO/ADd1c7F1Scao-50.npy"
#inp.systems.sett.s1.aero.C = f"./AERO/QhxDd1c7F1Scao-50.npy"
inp.systems.sett.s1.aero.Q0_rigid = f"./AERO/QhxDd1c7F1Scao-50.npy"
inp.systems.sett.s1.aero.qalpha = jnp.array(([0.,  0., 0, 0, 0, 0],
                                             [0.,  4 * jnp.pi / 180, 0, 0, 0, 0]))

config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = feniax.feniax_main.main(input_obj=config)
