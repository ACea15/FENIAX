import pathlib
import pdb
import sys
import numpy as np
import datetime
import time 
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
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
#inp.fem.folder = pathlib.Path('./FEM2nodes')
inp.fem.num_modes = 3
inp.fem.Ka = Ka
inp.fem.Ma = Ma
inp.fem.eigenvecs = v
inp.fem.grid = "structuralGridConstrained"
#inp.fem.grid = "structuralGrid"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.hstack([jnp.array([0, 0, 0]), w[3:]])
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results2")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 20.
inp.systems.sett.s1.tn = 2000
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
sol = feniax.feniax_main.main(input_obj=config)
#time2 = time.time()

import feniax.unastran.op2reader as op2reader
import importlib
importlib.reload(op2reader)

op2 = op2reader.NastranReader("./NASTRAN/2node/n400.op2",
                        "./NASTRAN/2node/n400.bdf")
op2.readModel()
tnastran, unastran = op2.displacements()
tnastran, rnastran = op2.position()

import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils

x0, y0 = putils.pickIntrinsic2D(sol.dynamicsystem_s1.t,
                              sol.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=-1, dim=0))
x2, y2 = putils.pickIntrinsic2D(sol.dynamicsystem_s1.t,
                              sol.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=-1, dim=2))

fig=None
fig = uplotly.lines2d(x0, y0, fig,
                      dict(name="NMROM-x",
                           line=dict(color="navy")
                           ))
fig = uplotly.lines2d(x2, y2, fig,
                      dict(name="NMROM-y",
                           line=dict(color="black")
                           ))
fig = uplotly.lines2d(tnastran[0], unastran[0,:,1,0]+1, fig,
                      dict(name="NASTRAN-x",
                           line=dict(color="red", dash="dot")
                           ))

fig = uplotly.lines2d(tnastran[0], unastran[0,:,1,2], fig,
                      dict(name="NASTRAN-y",
                           line=dict(color="green", dash="dot")
                           ))
fig.update_layout(      xaxis_title='time',
                        yaxis_title='tip-position')

fig.show()


import pandas as pd
df = pd.read_csv("./FEM2nodes/structuralGrid", comment="#", sep=" ",
                              names=['x1', 'x2', 'x3', 'fe_order', 'component', 'try1'])

# from functools import partial
# import jax
# @partial(jax.jit, static_argnums=1)
# def g(x, *n):
#   nn = n[0]  
#   for i in range(nn):

#     x = x ** 2

#   return x

