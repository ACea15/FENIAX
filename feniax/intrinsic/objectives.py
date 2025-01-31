import jax.numpy as jnp
import jax
from functools import partial

def factory(obj: str):
    return __NAMES__[obj]


__NAMES__ = dict()


def name(f, *args, **kwargs):
    __NAMES__[f.__name__] = f
    return f


def make_toarray(x):
    """
    Necessary for x to be an array as hstack will be applied to it
    """
    out = jax.lax.select(len(x.shape) > 0, x, jnp.array([x]))
    return out


@name
def X1_VAR(X1, nodes, components, t, *args, **kwargs):
    return X1[jnp.ix_(t, components, nodes)]


@name
@partial(jax.jit, static_argnames=["axis"])
def X1_MAX(X1, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.max(X1[jnp.ix_(t, components, nodes)], axis=axis)


@name
@partial(jax.jit, static_argnames=["axis"])
def X1_MIN(X1, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.min(X1[jnp.ix_(t, components, nodes)], axis=axis)


@name
def X2_VAR(X2, nodes, components, t, *args, **kwargs):
    return X2[jnp.ix_(t, components, nodes)]


@name
def X2_MAX(X2, nodes, components, t, axis=0, *args, **kwargs):

    return jnp.max(jnp.abs(X2[jnp.ix_(t, components, nodes)]), axis=axis)


@name
def X2_MIN(X2, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.min(X2[jnp.ix_(t, components, nodes)], axis=axis)


@name
def ra_VAR(ra, nodes, components, t, *args, **kwargs):
    return ra[jnp.ix_(t, components, nodes)]


@name
@partial(jax.jit, static_argnames=["axis"])
def ra_MAX(ra, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.max(ra[jnp.ix_(t, components, nodes)], axis=axis)


@name
@partial(jax.jit, static_argnames=["axis"])
def ra_MIN(ra, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.min(ra[jnp.ix_(t, components, nodes)], axis=axis)


@name
@partial(jax.jit, static_argnames=["axis"])
def ra_PMAX(ra, nodes, components, t, axis=None, *args, **kwargs):
    if axis is None:
        axis = 0
    axis += 1
    ra_max = jnp.max(ra[jnp.ix_(t, components, nodes)], axis=axis)
    return jnp.lax.pmax(ra, axis_name="i")


@name
@partial(jax.jit, static_argnames=["axis"])
def ra_PMIN(ra, nodes, components, t, axis=None, *args, **kwargs):
    if axis is None:
        axis = 0
    axis += 1
    ra_min = jnp.min(ra[jnp.ix_(t, components, nodes)], axis=axis)
    return jnp.lax.pmin(ra, axis_name="i")
