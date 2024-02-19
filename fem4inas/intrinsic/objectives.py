import jax.numpy as jnp

def OBJ_ra(ra, node, component, *args, **kwargs):

    return ra[-1, component, node]

def OBJ_X2(X2, node, component, *args, **kwargs):

    return jnp.max(X2[:, component, node])
