from  feniax.systems.system import System
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.dq_static as dq_static
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.intrinsic.postprocess as postprocess
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
import feniax.intrinsic.initcond as initcond
import feniax.intrinsic.args as libargs
import feniax.intrinsic.modes as modes
import feniax.intrinsic.couplings as couplings
import feniax.intrinsic.dq_common as common
import feniax.intrinsic.xloads as xloads
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
from jax.config import config; config.update("jax_enable_x64", True)


tarray = jnp.array([1,2,3,5])
tarray_len = len(tarray)
t = 4
out = jnp.argwhere(jax.lax.select(t < tarray,
                                  jnp.ones(tarray_len),
                                  jnp.zeros(tarray_len)),
                   size=1
                   )
tarray_new = tarray[:out[0][0]]
tarray_new

@partial(jax.jit, static_argnames=['t','tarray_len'])
def t1(tarray, t, tarray_len):
    out = jnp.argwhere(jax.lax.select(t < tarray,
                                      jnp.ones(tarray_len),
                                      jnp.zeros(tarray_len)),
                       size=1
                       )
    tarray_new =jax.lax.dynamic_slice(tarray,(0,),(out[0][0],))
    return tarray_new

#t1(tarray, t, tarray_len)


def fscan(x, args):
    x1, x2 = args

    def f2scan(carryin, xin):
        yout = xin **2 / carryin + 3* x1 + 2 * x2
        carryout = (xin+1)
        return carryout, yout

    last_carry, y = jax.lax.scan(f2scan, 1, x)
    return last_carry, y

carry0, y = fscan(jnp.array([1,2,3,4,7]), (100,1000))



@partial(jax.jit, static_argnames=['config'])
def f1(x, config):
    return jnp.linalg.norm(x * config.const.e1)

#f1(5,config)


@partial(jax.jit)
def f2(x):
    y = 2 * x
    return f1(y, config)

#f2(4)
############################################################

############################################################
@partial(jax.jit, static_argnames=[])
def linear_interpolation(t, x, data_tensor):
    # jax.debug.breakpoint()
    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(jax.lax.select(x >= t,
                                               jnp.ones(len_x),
                                               jnp.zeros(len_x)),
                                size=1)[0][0]#jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(jax.lax.select(x == t,
                                             jnp.ones(len_x),
                                             jnp.zeros(len_x)),
                          dtype=int)
    xindex_lower = xindex_upper - 1 + index_equal #jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (t - x_lower) / (x_upper - x_lower))
    weight_lower = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (x_upper - t) / (x_upper - x_lower))

    f_upper = data_tensor[xindex_upper]
    f_lower = data_tensor[xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower  * f_lower
    return f_interpol

@jax.jit
def eta_pointfollower(t, phi1, x, force_follower) -> jnp.array:
    # jax.debug.breakpoint()

    f = linear_interpolation(t, x, force_follower)
    eta = jnp.tensordot(phi1, f, axes=([1, 2],
                                       [0, 1]))
    return eta

def modestemp(X,
              Ka,
              Ma,
              eigenvals,
              eigenvecs,
              config):
    """Structural static with follower point forces."""

    # X is the transponse of config.fem.X
    #eigenvals, eigenvecs = self._compute_eigs()
    modal_analysis = modes.shapes(X.T,
                                  Ka,
                                  Ma,
                                  eigenvals,
                                  eigenvecs,
                                  config
                                  )
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = modes.scale(*modal_analysis)

    return (phi1, psi1, phi2,                  
            phi1l, phi1ml, psi1l, phi2l, psi2l,
            omega, X_xdelta, C0ab, C06ab)

#@partial(jax.jit)
def sys_10g11(q,
              *args):
    """Structural static with follower point forces."""
    
    (gamma2, omega, phi1l, x_forceinterpol,
     y_forceinterpol, t) = args[0]
    # print(y_forceinterpol)
    F = omega * q - common.contraction_gamma2(gamma2, q)
    F += xloads.eta_pointfollower(t,
                           phi1l,
                           x_forceinterpol,
                           y_forceinterpol)
    #jax.debug.breakpoint()
    # jax.debug.print("system t: {t}",t=t)
    return F

def q_10g11(q0, args):
    
    solver = optx.Newton(rtol=1e-6,
                         atol=1e-6,
                         #kappa=kappa,
                         norm=jnp.linalg.norm)
    # sol = optx.root_find(sys_10g11, solver, q0, args=args, max_steps=300
    #                      )
    sol = optx.root_find(dq_static.dq_10g11, solver, q0,
                         args=args, max_steps=300
                         )
    
    #sol = solver(F, q0, args, jac)
    return sol.value

@partial(jax.jit, static_argnames=[])
def iter_10g11(tarray, q0, args):
    # jax.debug.print("tarray: {t}",t=tarray)
    @partial(jax.jit)
    def sol_10g11(q0, t):
        
        args_in = args + (t,)
        q = q_10g11(q0, args_in)
        return q, q

    q = jax.lax.scan(sol_10g11, q0, tarray)
    #q, q2 = sol_10g11(q0, 1)
    return q

def rec_10g11(q, tn, X, phi2l, psi2l, fem, X_xdelta, C0ab):

    #tn = len(q)
    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X2 = postprocess.compute_internalforces(phi2l,
                                            q)
    X3 = postprocess.compute_strains(psi2l,
                                     q)
    Cab, ra = postprocess.integrate_strains_t(ra0,
                                              Cab0,
                                              X3,
                                              fem,
                                              X_xdelta,
                                              C0ab
                                              )
    
    return X2, X3, ra

def OBJ1_10g11(ra, node):

    return ra[-1,2,node]


@partial(jax.jit, static_argnames=['config'])
def main_10g11(t,
               t_array,
               q0,
               Ka,
               Ma,
               # eigenvals,
               # eigenvecs,
               node,
               config):

    t_array = jnp.hstack([t_array, t])
    tn = len(t_array)
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = modestemp(X,
                                               Ka,
                                               Ma,
                                               reduced_eigenvals,
                                               reduced_eigenvecs,
                                               config)

    # gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )
    config.system.xloads.build_point_follower(
                config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = config.system.xloads.force_follower
    args = (gamma2, omega, phi1l, x_forceinterpol,
            y_forceinterpol)
    # args = args_10g11(gamma2, omega, phi1l, t, config)    
    qcarry, q = iter_10g11(t_array, q0, args)
    # f = linear_interpolation(t, x_forceinterpol, y_forceinterpol)
    # F = sys_10g11(q0,
    #                args)
    X2, X3, ra = rec_10g11(q, tn, X, phi2l,
                           psi2l, config.fem, X_xdelta, C0ab)
    return OBJ1_10g11(ra, node)
    #return q 
    

# class IntrinsicADSystem(System, cls_name="intrinsicAD"):

#     def __init__(self,
#                  name: str,
#                  settings: intrinsicmodal.Dsystem,
#                  fem: intrinsicmodal.Dfem,
#                  sol: solution.IntrinsicSolution,
#                  config):

#         self.name = name
#         self.settings = settings
#         self.fem = fem
#         self.sol = sol
#         self.config = config
#         #self._set_xloading()
#         #self._set_generator()
#         #self._set_solver()

import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import pathlib

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

inp.fem.folder = pathlib.Path('/home/ac5015/programs/FENIAX.examples/SailPlane/FEM/')
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    '/home/ac5015/programs/FENIAX.examples/SailPlane/resultsAD')
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "static"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                  atol=1e-6,
                                  max_steps=50,
                                  norm="linalg_norm",
                                  kappa=0.01)
inp.system.xloads.follower_forces = True
inp.system.xloads.follower_points = [[25, 2], [48, 2]]

inp.system.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.system.xloads.follower_interpolation = [[0.,
                                             2e5,
                                             2.5e5,
                                             3.e5,
                                             4.e5,
                                             4.8e5,
                                             5.3e5],
                                            [0.,
                                             2e5,
                                             2.5e5,
                                             3.e5,
                                             4.e5,
                                             4.8e5,
                                             5.3e5]
                                            ]
inp.system.t = [1, 2, 3, 4, 5, 6]

#config =  configuration.Config(inp)

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#main_10g11
F0 = main_10g11(1.49,
               t_array=jnp.array([1]),#jnp.array(config.system.t[:-1]),
               q0=jnp.zeros(config.fem.num_modes),
               Ka=config.fem.Ka,
               Ma=config.fem.Ma,
               node=25,
               config=config)
fprime = jax.value_and_grad(main_10g11)
F, Fp  =fprime(1.5,
               t_array=jnp.array([1]), #jnp.array(config.system.t[:-1]),
               q0=jnp.zeros(config.fem.num_modes),
               Ka=config.fem.Ka,
               Ma=config.fem.Ma,
               node=25,
               config=config)


@partial(jax.jit, static_argnames=[])
def interpolation(t, x, data_tensor):
    # print(data_tensor)
    return linear_interpolation(t, x, data_tensor)
