import jax.numpy as jnp
import jax
from functools import partial
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.dq_static as dq_static
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.couplings as couplings
import feniax.systems.sollibs.diffrax as libdiffrax
import feniax.intrinsic.staticFast as staticFast

newton = partial(jax.jit, static_argnames=["F", "sett"])(libdiffrax.newton)
_solve = partial(jax.jit, static_argnames=["eqsolver", "dq", "sett"])(isys._staticSolve)


@partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_10g11_1(
    inputs,  # alpha
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    """
    Static follower force, pseudo time input
    """
    
    t = inputs["t"]
    Ka = config.fem.Ka
    Ma = config.fem.Ma

    t_loads = jnp.hstack([config.system.t, t])
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[: config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, : config.fem.num_modes]

    sol_dict= staticFast.main_10g11(q0,
                               config,
                               #Ka=Ka,
                               #Ma=Ma,
                               #eigenvals=reduced_eigenvals,
                               #eigenvecs=reduced_eigenvecs,
                               t_loads=t_loads)

    phi1 = sol_dict.get("phi1"),
    psi1 = sol_dict.get("psi1"),
    phi2 = sol_dict.get("phi2"),
    phi1l = sol_dict.get("phi1l"),
    phi1ml = sol_dict.get("phi1ml"),
    psi1l = sol_dict.get("psi1l"),
    phi2l = sol_dict.get("phi2l"),
    psi2l = sol_dict.get("psi2l"),
    omega = sol_dict.get("omega"),
    X_xdelta = sol_dict.get("X_xdelta"),
    C0ab = sol_dict.get("C0ab"),
    C06ab = sol_dict.get("C06a")
    gamma2 = sol_dict.get("gamma2")
    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")
    
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
def main_10g11_3(
    inputs,  # alpha
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    """
    Static follower force, Ka, Ma, phi1
    """
    
    Ka = inputs["Ka"]
    Ma = inputs["Ma"]
    eigenvals = inputs[
        "eigenvals"
    ]
    eigenvecs = inputs[
        "eigenvecs"
    ]

    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    reduced_eigenvals = eigenvals[: config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, : config.fem.num_modes]
    # solver_args = config.system.solver_settings
    (phi1,
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
     gamma2,
     q,
     X1, X2, X3, ra, Cab
     ) = staticFast.main_10g11(q0,
                               config,
                               Ka=Ka,
                               Ma=Ma,
                               eigenvals=reduced_eigenvals,
                               eigenvecs=reduced_eigenvecs)
    
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
