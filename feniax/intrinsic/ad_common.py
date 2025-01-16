import feniax.intrinsic.objectives as objectives
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
import feniax.intrinsic.modes as modes
import feniax.intrinsic.couplings as couplings
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.postprocess as postprocess
import equinox


def _compute_modes(X, Ka, Ma, eigenvals, eigenvecs, config):
    modal_analysis = modes.shapes(X.T, Ka, Ma, eigenvals, eigenvecs, config)

    return modes.scale(*modal_analysis)


@partial(jax.jit, static_argnames=["f_obj"])
def _objective_output(q, X1, X2, X3, ra, Cab, f_obj, *args, **kwargs):
    obj = f_obj(X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab, **kwargs)
    #objective = jnp.hstack(jnp.hstack(obj))
    # objective = f_obj(X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab, axis=axis, **kwargs)
    # lax cond or select not working here as both branches are evaluated.
    # made sure obj is an array in f_obj
    # objective = jax.lax.select(len(obj.shape) > 0, jnp.hstack(obj), obj)
    # objective = jax.lax.cond(len(obj.shape) > 0,
    #                          lambda x : jnp.hstack(x),
    #                          lambda x : x, obj)
    #f_out = (objective, q, X1, X2, X3, ra, Cab)
    f_out = (obj, q, X1, X2, X3, ra, Cab)
    return (obj, f_out)
