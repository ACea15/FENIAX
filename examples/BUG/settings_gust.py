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
    "./results1gust")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 7.5
inp.systems.sett.s1.tn = 4001
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 3.
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 1.
inp.systems.sett.s1.aero.poles = "./AERO/PolesDd1c7F1Scao-50p5.npy"
inp.systems.sett.s1.aero.A = f"./AERO/ADd1c7F1Scao-50p5.npy"
inp.systems.sett.s1.aero.D = f"./AERO/DDd1c7F1Scao-50p5.npy"
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*0.1
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.1
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = jnp.load("./AERO/Dihedral_d1c7.npy")
inp.systems.sett.s1.aero.gust.collocation_points = "./AERO/Collocation_d1c7.npy"

config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = feniax.feniax_main.main(input_obj=config)
