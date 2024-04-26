import pathlib
import pdb
import sys
import numpy as np
import datetime
import time 
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import scipy.linalg

Ka = jnp.load('./FEM2nodes/Ka.npy')
Ma = jnp.load('./FEM2nodes/Ma.npy')
w, v = scipy.linalg.eigh(Ka, Ma)

Ka2 = jnp.insert(Ka, 6, 0., axis=0)
Ka2 = jnp.insert(Ka2, 7, 0., axis=0)
Ka2 = jnp.insert(Ka2, 8, 0., axis=0)
Ka2 = jnp.insert(Ka2, 6, 0., axis=1)
Ka2 = jnp.insert(Ka2, 7, 0., axis=1)
Ka2 = jnp.insert(Ka2, 8, 0., axis=1)

Ma2 = jnp.insert(Ma, 6, 0., axis=0)
Ma2 = jnp.insert(Ma2, 7, 0., axis=0)
Ma2 = jnp.insert(Ma2, 8, 0., axis=0)
Ma2 = jnp.insert(Ma2, 6, 0., axis=1)
Ma2 = jnp.insert(Ma2, 7, 0., axis=1)
Ma2 = jnp.insert(Ma2, 8, 0., axis=1)

v2 = jnp.insert(v, 6, 0., axis=0)
v2 = jnp.insert(v2, 7, 0., axis=0)
v2 = jnp.insert(v2, 8, 0., axis=0)


inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'0': None}
inp.fem.folder = pathlib.Path('./FEM2nodes')
inp.fem.num_modes = 3
inp.fem.Ka = Ka2
inp.fem.Ma = Ma2
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.array([0, 0, 0])
inp.fem.eigenvecs = v2
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_try")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 5.
inp.systems.sett.s1.tn = 1000
inp.systems.sett.s1.solver_library = "diffrax" #"runge_kutta" #"diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")# "rk4")
# inp.systems.sett.s1.init_states = dict(q1=["axial_parabolic",
#                                            ([0., 3., 3., 0., 0., 0], 20.)
#                                            ])
inp.systems.sett.s1.xloads.gravity_forces = True
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_101'
config =  configuration.Config(inp)

#time1 = time.time()
sol = fem4inas.fem4inas_main.main(input_obj=config)
#time2 = time.time()

import fem4inas.plotools.uplotly as uplotly
import fem4inas.plotools.utils as putils

x, y = putils.pickIntrinsic2D(sol.dynamicsystem_s1.t,
                              sol.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=-1, dim=0))
fig=None
fig = uplotly.lines2d(x,y, fig,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))
fig.show()

# from functools import partial
# import jax
# @partial(jax.jit, static_argnums=1)
# def g(x, *n):
#   nn = n[0]  
#   for i in range(nn):

#     x = x ** 2

#   return x
