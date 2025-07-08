import jax.numpy as jnp
import jax
from functools import partial
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.gust as igust
import feniax.intrinsic.couplings as couplings
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.dynamicFast as dynamicFast

# @partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_20g11_1(
    inputs,  # alpha
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    alpha = inputs["alpha"]

    sol_dict= dynamicFast.main_20g11(q0,
                                     config,
                                     alpha=alpha)

    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")
    
    return adcommon._objective_output(
        sol_dict,
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t),
        axis=obj_args.axis,
    )


# @partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_20g11_3(
        inputs,  # Ka, Ma
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    # alpha = inputs["alpha"]
    Ka = inputs["Ka"]
    Ma = inputs["Ma"]
    eigenvals = inputs[
        "eigenvals"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs[
        "eigenvecs"
    ]  


    sol_dict= dynamicFast.main_20g11(q0,
                                      config,
                                      Ka=Ka,
                                      Ma=Ma,
                                      eigenvals=eigenvals,
                                      eigenvecs=eigenvecs
                                      )
    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")
    
    return adcommon._objective_output(
        sol_dict,
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t),
        axis=obj_args.axis,
    )

# @partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_20g21_2(
    inputs,  # gust_intensity, gust_length, u_inf, rho_inf,
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    gust_intensity = inputs["intensity"]
    gust_length = inputs["length"]
    u_inf = inputs["u_inf"]
    rho_inf = inputs["rho_inf"]

    sol_dict= dynamicFast.main_20g21(q0,
                                     config,
                                     gust_intensity=gust_intensity,
                                     gust_length=gust_length,
                                     u_inf=u_inf,
                                     rho_inf=rho_inf
                                     )
    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")

    return adcommon._objective_output(
        sol_dict,        
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t)
    )


# @partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_20g21_3(
    inputs,  
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    Ka = inputs["Ka"]
    Ma = inputs["Ka"]
    eigenvals = inputs[
        "eigenvals"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs[
        "eigenvecs"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[1])


    sol_dict= dynamicFast.main_20g21(q0,
                                     config,
                                     Ka=Ka,
                                     Ma=Ma,
                                     eigenvals=eigenvals,
                                     eigenvecs=eigenvecs
                                     )
    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")

    return adcommon._objective_output(
        sol_dict,        
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t),
        axis=obj_args.axis,
    )

# @partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_20g546_2(
    inputs,  # gust_intensity, gust_length, u_inf, rho_inf,
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    gust_intensity = inputs["intensity"]
    gust_length = inputs["length"]
    u_inf = inputs["u_inf"]
    rho_inf = inputs["rho_inf"]

    sol_dict= dynamicFast.main_20g546(q0,
                                     config,
                                     gust_intensity=gust_intensity,
                                     gust_length=gust_length,
                                     u_inf=u_inf,
                                     rho_inf=rho_inf
                                     )
    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")

    return adcommon._objective_output(
        sol_dict,        
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t)
    )


# @partial(jax.jit, static_argnames=["config", "f_obj", "obj_args"])
def main_20g546_3(
    inputs,  
    q0,
    config,
    f_obj,
    obj_args,
    *args,
    **kwargs,
):
    Ka = inputs["Ka"]
    Ma = inputs["Ka"]
    eigenvals = inputs[
        "eigenvals"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs[
        "eigenvecs"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[1])


    sol_dict= dynamicFast.main_20g546(q0,
                                     config,
                                     Ka=Ka,
                                     Ma=Ma,
                                     eigenvals=eigenvals,
                                     eigenvecs=eigenvecs
                                     )
    q = sol_dict.get("q")
    X1 = sol_dict.get("X1")
    X2 = sol_dict.get("X2")
    X3 = sol_dict.get("X3")
    ra = sol_dict.get("ra")
    Cab = sol_dict.get("Cab")

    return adcommon._objective_output(
        sol_dict,        
        f_obj=f_obj,
        nodes=jnp.array(obj_args.nodes),
        components=jnp.array(obj_args.components),
        t=jnp.array(obj_args.t),
        axis=obj_args.axis,
    )
