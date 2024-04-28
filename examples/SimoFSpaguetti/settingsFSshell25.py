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

Ka = jnp.load('./FEMshell25/Ka.npy')
Ma = jnp.load('./FEMshell25/Ma.npy')
w, v = scipy.linalg.eigh(Ka, Ma)

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'0': None}
inp.fem.folder = pathlib.Path('./FEMshell25')
inp.fem.num_modes = 150
inp.fem.eig_type = "scipy"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_try")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.dt = 1e-4
inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
inp.systems.sett.s1.xloads.dead_forces = True
inp.systems.sett.s1.xloads.dead_points = [[24, 0],
                                          [24, 5]]
inp.systems.sett.s1.xloads.x = [0., 2.5, 2.5+1e-6, 10.5]
inp.systems.sett.s1.xloads.dead_interpolation = [[8., 8., 0., 0.],
                                                 [-80., -80., 0., 0.]
                                                 ]

config =  configuration.Config(inp)

#time1 = time.time()
sol = fem4inas.fem4inas_main.main(input_obj=config)
#time2 = time.time()


import fem4inas.intrinsic.functions as functions
import importlib
importlib.reload(functions)

R = functions.center_mass(config.fem.Ma, sol.dynamicsystem_s1.ra)


def FFB25_cgx(t):

  k0=2./2.5
  k1=-2./2.5
  k2 = 0.

  s0 = 0.
  v00=0.

  v10=v00+k0*(2.5)**2/2
  v20=v10 + 2*(5.-2.5)+k1*(5.-2.5)**2/2

  s10 = s0+v00*2.5+k0*2.5**3/6
  s20 = s10+v10*(5.-2.5)+(5.-2.5)**2+k1*(5-2.5)**3/6


  if t<2.5:
    x1 = s0+v00*t+k0*t**3/6
    return x1

  elif t<5.:

    x2 = s10+v10*(t-2.5)+(t-2.5)**2+k1*(t-2.5)**3/6
    return x2

  else:
    x3 = s20+v20*(t-5.)+k2*(t-5.)**3/6
    return x3

def FFB25_cgx2d(t):

  a0 = 8./10.  
  s0 = 0.
  v00=0.
  v10=v00+a0*(2.5)
  s10 = s0+v00*2.5+a0*2.5**2/2

  if t<2.5:
    x1 = s0+v00*t+a0*t**2/2
    return x1

  else:

    x2 = s10+v10*(t-2.5)
    return x2


def FFB25_cg2d(t):
    return np.array([FFB25_cgx2d(t)+3,-4.,0.])

Rth = np.array([FFB25_cg2d(ti) for ti in config.systems.mapper['s1'].t])
print(jnp.linalg.norm(R-Rth)/len(R))


# import fem4inas.plotools.uplotly as uplotly
# import fem4inas.plotools.utils as putils

# x, y = putils.pickIntrinsic2D(sol.dynamicsystem_s1.t,
#                               sol.dynamicsystem_s1.ra,
#                               fixaxis2=dict(node=-1, dim=0))
# fig=None
# fig = uplotly.lines2d(x,y, fig,
#                       dict(name="NMROM",
#                            line=dict(color="navy")
#                            ))
# fig.show()

# from functools import partial
# import jax
# @partial(jax.jit, static_argnums=1)
# def g(x, *n):
#   nn = n[0]  
#   for i in range(nn):

#     x = x ** 2

#   return x
