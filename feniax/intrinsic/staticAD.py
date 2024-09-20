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


@partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_30g11_1(
    inputs,  # alpha
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    t = inputs["t"]
    Ka = config.fem.Ka
    Ma = config.fem.Ma

    t_loads = jnp.hstack([config.system.t, t])
    tn = len(t_loads)
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = config.system.states["q2"]
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[: config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, : config.fem.num_modes]
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
        C06ab,
    ) = adcommon._compute_modes(X, Ka, Ma, reduced_eigenvals, reduced_eigenvecs, config)

    # gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(phi1ml, phi2l, psi2l, X_xdelta)
    eta0 = jnp.zeros(config.fem.num_modes)
    config.system.xloads.build_point_follower(config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = config.system.xloads.force_follower
    dq_args = (eta0, gamma2, omega, phi1l, x_forceinterpol, y_forceinterpol)

    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q = _solve(
        newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings
    )
    q2 = q[:, q2_index]
    # X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    X2, X3, ra, Cab = isys.recover_staticfields(
        q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config
    )
    X1 = jnp.zeros_like(X2)
    return adcommon._objective_output(
        q=q,
        X1=X1,
        X2=X2,
        X3=X3,
        ra=ra,
        Cab=Cab,
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t),
    )


@partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_30g11_3(
    inputs,  # alpha
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    Ka = inputs["Ka"]
    Ma = inputs["Ma"]
    eigenvals = inputs[
        "eigenvals"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs[
        "eigenvecs"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[1])

    t_loads = config.system.t
    tn = len(t_loads)
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = config.system.states["q2"]
    # eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    # eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[: config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, : config.fem.num_modes]
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
        C06ab,
    ) = adcommon._compute_modes(X, Ka, Ma, reduced_eigenvals, reduced_eigenvecs, config)

    # gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(phi1ml, phi2l, psi2l, X_xdelta)
    eta0 = jnp.zeros(config.fem.num_modes)
    config.system.xloads.build_point_follower(config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = config.system.xloads.force_follower
    dq_args = (eta0, gamma2, omega, phi1l, x_forceinterpol, y_forceinterpol)

    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q = _solve(
        newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings
    )
    q2 = q[:, q2_index]
    # X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    X2, X3, ra, Cab = isys.recover_staticfields(
        q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config
    )
    X1 = jnp.zeros_like(X2)
    return adcommon._objective_output(
        q=q,
        X1=X1,
        X2=X2,
        X3=X3,
        ra=ra,
        Cab=Cab,
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t),
        axis=obj_args.axis,
    )
