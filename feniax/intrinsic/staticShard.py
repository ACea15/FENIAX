import jax.numpy as jnp
import jax
from functools import partial
import feniax.intrinsic.xloads as xloads
import feniax.intrinsic.postprocess as postprocess
import feniax.intrinsic.dq_common as common
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.dq_static as dq_static
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.couplings as couplings
import feniax.systems.sollibs.diffrax as libdiffrax


newton = partial(jax.jit, static_argnames=["F", "sett"])(libdiffrax.newton)
_solve = partial(jax.jit, static_argnames=["eqsolver", "dq", "sett"])(isys._staticSolve)


@partial(jax.jit, static_argnames=["config"])
def main_10g11_1(
    inputs,  # 
    q0,
    config,
    args,
    **kwargs,
):

    t_loads = config.system.t
    tn = len(t_loads)
    q2_index = config.system.states["q2"]
    X = config.fem.X
    phi2l, psi2l, X_xdelta, C0ab, _dqargs = args
    
    # @jax.jit
    def _main_10g11_1(inp):
        
        dq_args = _dqargs + (inp,)
        q = _solve(
            newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings
        )
        q2 = q[:, q2_index]
        #jax.debug.breakpoint()
        X2, X3, ra, Cab = isys.recover_staticfields(
            q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config
        )
        return dict(q=q, X2=X2, X3=X3, ra=ra, Cab=Cab)

    main_vmap = jax.vmap(_main_10g11_1)
    results = main_vmap(inputs)
    return results

def dq_10g15(q, *args):
    """Manoeuvre under qalpha."""

    (eta_0, gamma2, omega, x, qalpha, A0hat, C0hat, t) = args[0]
    q0 = -q / omega
    F = omega * q - common.contraction_gamma2(gamma2, q)
    F += xloads.eta_steadyaero(q0, A0hat)
    F += xloads.eta_manoeuvre(t, x, qalpha, C0hat)
    F += eta_0
    return F

@partial(jax.jit, static_argnames=["config"])
def main_10g15_1(
    inputs,  # 
    q0,
    config,
    args,
    **kwargs,
):

    t_loads = config.system.t
    tn = len(t_loads)
    q2_index = config.system.states["q2"]
    X = config.fem.X
    phi2l, psi2l, X_xdelta, C0ab, _dqargs = args
    
    # @jax.jit
    def _main_10g11_1(inp):
        
        dq_args = _dqargs + (inp,)
        q = _solve(
            newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings
        )
        q2 = q[:, q2_index]
        #jax.debug.breakpoint()
        X2, X3, ra, Cab = isys.recover_staticfields(
            q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config
        )
        return dict(q=q, X2=X2, X3=X3, ra=ra, Cab=Cab)

    main_vmap = jax.vmap(_main_10g11_1)
    results = main_vmap(inputs)
    return results
