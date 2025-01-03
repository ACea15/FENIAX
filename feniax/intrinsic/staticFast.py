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


@partial(jax.jit, static_argnames=["config"])
def main_10g11(
    q0,
    config,
    *args,
    **kwargs,
):
    """
    Static computation with follower forces
    """
    
    Ka = kwargs["Ka"] #config.fem.Ka
    Ma = kwargs["Ma"] #config.fem.Ma
    reduced_eigenvals = kwargs["eigenvals"] 
    reduced_eigenvecs = kwargs["eigenvecs"] 
    t_loads = kwargs["t_loads"]
    tn = len(t_loads)
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = config.system.states["q2"]
    # solver_args = config.system.solver_settings
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
    ) = adcommon._compute_modes(X, Ka, Ma, reduced_eigenvals, reduced_eigenvecs, config)

    gamma2 = couplings.f_gamma2(phi1ml, phi2l, psi2l, X_xdelta)
    eta0 = jnp.zeros(config.fem.num_modes)
    config.system.xloads.build_point_follower(config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = config.system.xloads.force_follower
    dq_args = (eta0, gamma2, omega, phi1l, x_forceinterpol, y_forceinterpol)

    q = _solve(
        newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings
    )
    q2 = q[:, q2_index]
    X2, X3, ra, Cab = isys.recover_staticfields(
        q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config
    )
    X1 = jnp.zeros_like(X2)
    return (phi1,psi1,
            phi2,
            phi1l,
            phi1ml,
            psi1l,
            phi2l,
            psi2l,
            omega,
            X_xdelta,
            C0ab,
            C06ab,
            gamma2,
            q,
            X1, X2, X3, ra, Cab
            )


