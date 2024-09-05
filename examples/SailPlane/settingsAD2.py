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
import feniax.intrinsic.objectives as objectives
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
from jax.config import config; config.update("jax_enable_x64", True)

import equinox

class RootFindArgs(equinox.Module):

    max_steps: int = 256

class NewtonArgs(equinox.Module):

    atol: float = 1e-7
    rtol: float = 1e-7


@partial(jax.jit, static_argnames=['args'])
def f1(x, args):

    print(x)
    print(args.atol)
    return (x, args.atol, args.rtol)


    
def _compute_modes(X,
                  Ka,
                  Ma,
                  eigenvals,
                  eigenvecs,
                  config):

    modal_analysis = modes.shapes(X.T,
                                  Ka,
                                  Ma,
                                  eigenvals,
                                  eigenvecs,
                                  config
                                  )

    return modes.scale(*modal_analysis)

@partial(jax.jit, static_argnames=['dq', 'config'])
def _static_optx(dq, q0, dq_args, config):
    """
    see
    https://docs.kidger.site/optimistix/api/root_find/#optimistix.Newton
    """

    solver_args = config.solver_settings
    solver = optx.Newton(solver_args['rtol'], solver_args['atol'])
    # sol = optx.root_find(sys_10g11, solver, q0, args=args, max_steps=300
    #                      )
    sol = optx.root_find(dq,
                         solver,
                         q0,
                         args=dq_args,
                         max_steps=solver_args['max_steps']#root_args.max_steps
                         )

    return sol.value

@partial(jax.jit, static_argnames=['dq','config'])
def _solve(dq, t_loads, q0, dq_args, config):

    def _iter(qim1, t):
        args = dq_args + (t,)
        qi = _static_optx(dq, qim1, args, config)
        return qi, qi

    qcarry, qs = jax.lax.scan(_iter,
                              q0,
                              t_loads)
    return qs

def recover_fields(q, tn, X, q1_index, q2_index,
                   phi1l, phi2l, psi2l, X_xdelta, C0ab, fem):

    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X1 = postprocess.compute_velocities(phi1l,
                                        q1)
    
    X2 = postprocess.compute_internalforces(phi2l,
                                            q2)
    X3 = postprocess.compute_strains(psi2l,
                                     q2)
    Cab, ra = postprocess.integrate_strains_t(ra0,
                                              Cab0,
                                              X3,
                                              fem,
                                              X_xdelta,
                                              C0ab
                                              )

    return X1, X2, X3, ra, Cab


def recover_staticfields(q, tn, X, q2_index, phi2l, psi2l, X_xdelta, C0ab, fem):

    q2 = q[:, q2_index]
    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X2 = postprocess.compute_internalforces(phi2l,
                                            q2)
    X3 = postprocess.compute_strains(psi2l,
                                     q2)
    Cab, ra = postprocess.integrate_strains_t(ra0,
                                              Cab0,
                                              X3,
                                              fem,
                                              X_xdelta,
                                              C0ab
                                              )

    return X2, X3, ra, Cab

@partial(jax.jit, static_argnames=['config', 'f_obj'])
def main_10g11(t,
               t_array,
               q0,
               Ka,
               Ma,
               config,
               f_obj,
               obj_args=None
               ):

    if obj_args is None:
        obj_args = dict()

    t_loads = jnp.hstack([t_array, t])
    tn = len(t_loads)
    config.system.build_states(config.fem.num_modes)
    q2_index = config.system.states['q2']
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = _compute_modes(X,
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
    dq_args = (gamma2, omega, phi1l, x_forceinterpol,
               y_forceinterpol)
    q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system)
    X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
                                           phi2l, psi2l, X_xdelta, C0ab, config.fem)
    objective = f_obj(X2=X2, X3=X3, ra=ra, Cab=Cab, **obj_args)
    return objective


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
inp.system.xloads.follower_points = [[25, 2],
                                     [48, 2]]
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
config =  configuration.Config(inp)

fprime = jax.value_and_grad(main_10g11)
solver_args=dict(rtol=1e-6,
                 atol=1e-6)
root_args=dict(max_steps=300)

if solver_args is None:
    solver_args = NewtonArgs()
else:
    solver_args = NewtonArgs(**solver_args)
if root_args is None:
    root_args = RootFindArgs()
else:
    root_args = RootFindArgs(root_args)

F, Fp  =fprime(1.5,
               t_array=jnp.array([1]), #jnp.array(config.system.t[:-1]),
               q0=jnp.zeros(config.fem.num_modes),
               Ka=config.fem.Ka,
               Ma=config.fem.Ma,
               config=config,
               f_obj=objectives.OBJ_ra,
               obj_args=dict(node=25,
                             component=2))

