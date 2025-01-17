import jax.numpy as jnp
import jax
from functools import partial
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.dq_static as dq_static
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.couplings as couplings
import feniax.systems.sollibs.diffrax as libdiffrax


newton = partial(jax.jit, static_argnames=["F", "sett"])(libdiffrax.newton)
_solve = partial(jax.jit, static_argnames=["eqsolver", "dq", "sett"])(isys._staticSolve)

def _get_inputs(config, **kwargs):

    kwargs_list = list(kwargs.keys())
    if "Ka" in kwargs_list:
        Ka = kwargs.get("Ka")
    else:
        Ka = config.fem.Ka
    if "Ma" in kwargs_list:
        Ma = kwargs.get("Ma")
    else:
        Ma = config.fem.Ma
    if "eigenvals" in kwargs_list:
        eigenvals = kwargs.get("eigenvals")
    else:
        eigenvals = config.fem.eigenvals
    if "eigenvecs" in kwargs_list:
        eigenvecs = kwargs.get("eigenvecs")
    else:
        eigenvecs = config.fem.eigenvecs
    if "t_loads" in kwargs_list:
        t_loads = kwargs.get("t_loads")
    else:
        t_loads = config.system.t

    return Ka, Ma, eigenvals, eigenvecs, t_loads

def _build_intrinsic(config, **kwargs):

    Ka, Ma, eigenvals, eigenvecs, t_loads = _get_inputs(config, **kwargs)
    X = config.fem.X
    (
        phi1,
        psi1,
        phi2,
        phi1l,
        phi1ml,
        psi1l,
        phi2l,
        psi2l,
        omega,
        X_xdelta,
        C0ab,
        C06ab
    ) = adcommon._compute_modes(X, Ka, Ma, eigenvals, eigenvecs, config)
    gamma2 = couplings.f_gamma2(phi1ml, phi2l, psi2l, X_xdelta)
    output_dict = dict(phi1 = phi1,
                       psi1 = psi1,
                       phi2 = phi2,
                       phi1l = phi1l,
                       phi1ml = phi1ml,
                       psi1l = psi1l,
                       phi2l = phi2l,
                       psi2l = psi2l,
                       omega = omega,
                       X_xdelta = X_xdelta,
                       C0ab = C0ab,
                       C06ab = C06ab,
                       gamma2 = gamma2,
                       t_loads = t_loads
                       )
    return output_dict

def _build_solution(q, output_dict, config):

    X = config.fem.X
    t_loads = output_dict['t_loads']                
    tn = len(t_loads)
    q2_index = config.system.states["q2"]                              
    q2 = q[:, q2_index]
    X2, X3, ra, Cab = isys.recover_staticfields(
        q2,
        tn,
        X,
        output_dict['phi2l'],
        output_dict['psi2l'],
        output_dict['X_xdelta'],
        output_dict['C0ab'],
        config
    )
    X1 = jnp.zeros_like(X2)
    output_dict['q'] = q
    output_dict['X1'] = X1
    output_dict['X2'] = X2
    output_dict['X3'] = X3    
    output_dict['ra'] = ra
    output_dict['Cab'] = Cab

def _get_aeromatrices(config):

    A0 = config.system.aero.A[0]
    C0 = config.system.aero.Q0_rigid
    A0hat = config.system.aero.q_inf * A0
    C0hat = config.system.aero.q_inf * C0
    return A0hat, C0hat
    
# @partial(jax.jit, static_argnames=["config"])
def main_10g11(
    q0,
    config,
    *args,
    **kwargs,
):
    """Structural static with follower point forces."""

    output = _build_intrinsic(config, **kwargs)
                              
    t_loads = output['t_loads']                
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    eta0 = jnp.zeros(config.fem.num_modes)
    config.system.xloads.build_point_follower(config.fem.num_nodes, output['C06ab'])
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = config.system.xloads.force_follower
    dq_args = (eta0, output['gamma2'], output['omega'], output['phi1l'],
               x_forceinterpol, y_forceinterpol)

    q = _solve(
        newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings
    )
    _build_solution(q, output, config)                      
    return output

#@partial(jax.jit, static_argnames=["config"])
def main_10g15(
    q0,
    config,
    *args,
    **kwargs,
):
    """Manoeuvre under qalpha."""

    output = _build_intrinsic(config, **kwargs)
                              
    t_loads = output['t_loads']                
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    eta0 = jnp.zeros(config.fem.num_modes)
    A0hat, C0hat = _get_aeromatrices(config)
    dq_args = (eta0, output['gamma2'], output['omega'], output['phi1l'],
               config.system.xloads.x, config.system.aero.qalpha,
               A0hat, C0hat)

    q = _solve(
        newton, dq_static.dq_10g15, t_loads, q0, dq_args, config.system.solver_settings
    )
    _build_solution(q, output, config)                              
    return output
