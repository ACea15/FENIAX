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

    kwargs_list = list(kwargs.keys())
    #print(config.fem.eigenvals)
    # Ka = jax.lax.select("Ka" in kwargs_list, kwargs.get("Ka"), config.fem.Ka)
    # Ma = jax.lax.select("Ma" in kwargs_list, kwargs.get("Ma"), config.fem.Ma)
    # # eigenvals = config.fem.eigenvals #jax.lax.cond("eigenvals" in kwargs_list,
    # #                          #lambda kwargs: kwargs.get("eigenvals"),
    # #                          #lambda kwargs: config.fem.eigenvals,
    # #                          #kwargs)
    # # eigenvecs = config.fem.eigenvecs #jax.lax.select("eigenvecs" in kwargs_list,
    # #                            #kwargs.get("eigenvecs"), config.fem.eigenvecs)
    # # t_loads = kwargs.get("t_loads") #config.system.t #jax.lax.select("t_loads" in kwargs_list,
    # #                           # kwargs.get("t_loads"), config.system.t)
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
    ) = adcommon._compute_modes(X, Ka, Ma, eigenvals, eigenvecs, config)

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
    return dict(phi1 = phi1,
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
            q = q,
            X1 = X1,
            X2 = X2,
            X3 = X3,
            ra = ra,
            Cab = Cab
            )


