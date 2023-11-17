import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import pyNastran.op4.op4 as op4

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
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    f"./resultsGust")
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
inp.systems.sett.s1.aero.A = f"./NASTRAN/data_out/Qhh0_8-{inp.fem.num_modes}r5.npy"
inp.systems.sett.s1.aero.D = f"./NASTRAN/data_out/Qhj0_8-{inp.fem.num_modes}r5.npy"
inp.systems.sett.s1.aero.poles = f"./NASTRAN/data_out/Poles0_8-{inp.fem.num_modes}r5.npy"
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust_settings.intensity = 1.#11.304727674272842/10000
inp.systems.sett.s1.aero.gust_settings.length = 18.
inp.systems.sett.s1.aero.gust_settings.step = 0.5
inp.systems.sett.s1.aero.gust_settings.shift = 0.
inp.systems.sett.s1.aero.gust_settings.panels_dihedral = jnp.load("./NASTRAN/data_out/dihedral.npy")[:,0]
inp.systems.sett.s1.aero.gust_settings.collocation_points = "./NASTRAN/data_out/collocation_points.npy"

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)


# import jax
# from functools import partial
# @partial(jax.jit, static_argnames=['npoles'])
# def foo(x, npoles, modes):
#     nmodes =len(modes)
#     y = jnp.zeros(npoles * nmodes)
#     z  = jnp.ones(modes.shape)
#     for i in range(npoles):
#         y = y.at[i*nmodes:(i+1)*nmodes].set(i)
#     return y, z

# b, b2 = foo(1,5,jnp.zeros((10,3)))


# #@partial(jax.jit, static_argnames=['npoles'])
# @jax.jit
# def foo2(x):
#     #y = jnp.zeros(npol)
#     z  = jnp.hstack(x)
#     return z

# b3 = foo2(jnp.zeros((10,3)))
