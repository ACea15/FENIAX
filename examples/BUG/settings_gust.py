import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import numpy as np
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import pyNastran.op4.op4 as op4

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
inp.driver.sol_path = pathlib.Path("./results_gust")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 5.
inp.systems.sett.s1.tn = 5001
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 2.
inp.systems.sett.s1.aero.u_inf = 150.
inp.systems.sett.s1.aero.rho_inf = 1.225
###################################
inp.systems.sett.s1.aero.A = f"./NASTRAN/data_out/Qhh0_8-{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"./NASTRAN/data_out/Qhj0_8-{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"./NASTRAN/data_out/Poles0_8-{inp.fem.num_modes}.npy"

Qhh = op4.read_op4(f"./NASTRAN/data_out/Qhh50-50.op4")
Qhalpha = op4.read_op4(f"./NASTRAN/data_out/Qhj0_8-50.op4")
Qhx = np.array(Qhalpha['Q_HJ'].data)[1][:,1:]
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.aero.Qk_struct = [[0.], np.array(Qhh['Q_HH'].data)[1][0].real]
inp.systems.sett.s1.aero.Q0_rigid = Qhx
inp.systems.sett.s1.aero.qalpha = jnp.array([1,0,0,0,0]) * jnp.pi / 180
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.5
inp.systems.sett.s1.aero.c_ref = 2.
inp.systems.sett.s1.xloads.modalaero_forces = True
config =  configuration.Config(inp)
sol = fem4inas.fem4inas_main.main(input_obj=config)