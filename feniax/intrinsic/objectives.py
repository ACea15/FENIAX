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
# @partial(jax.jit, static_argnames=["axis"])
def X2_PMAX(X2, nodes, components, t, axis=None, *args, **kwargs):
    if axis is None:
        axis = 0
    axis += 1
    # x2max = jnp.max(jnp.abs(X2[jnp.ix_(jnp.array([0]), t, components, nodes)]), axis=axis)
    # return x2max
    X2filter = X2[jnp.ix_(jnp.array([0]), t, components, nodes)]
    X2_max = jnp.max(X2filter, axis=axis)
    return jax.lax.pmean(X2_max, axis_name="x")


@name
# @partial(jax.jit, static_argnames=["axis"])
def X2_PMIN(X2, nodes, components, t, axis=None, *args, **kwargs):
    if axis is None:
        axis = 0
    axis += 1
    X2_min = jnp.min(X2[jnp.ix_(t, components, nodes)], axis=axis)
    return jax.lax.pmin(X2_min, axis_name="x")


@name
def ra_VAR(ra, nodes, components, t, *args, **kwargs):
    return ra[jnp.ix_(t, components, nodes)]


@name
# @partial(jax.jit, static_argnames=["axis"])
def ra_MAX(ra, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.max(ra[jnp.ix_(t, components, nodes)], axis=axis)


@name
# @partial(jax.jit, static_argnames=["axis"])
def ra_MIN(ra, nodes, components, t, axis=None, *args, **kwargs):
    return jnp.min(ra[jnp.ix_(t, components, nodes)], axis=axis)


@name
# @partial(jax.jit, static_argnames=["axis"])
def ra_PMAX(ra, nodes, components, t, axis=None, *args, **kwargs):
    if axis is None:
        axis = 0
    axis += 1
    ra_max = jnp.max(ra[jnp.ix_(t, components, nodes)], axis=axis)
    return jax.lax.pmax(ra_max, axis_name="x")


@name
# @partial(jax.jit, static_argnames=["axis"])
def ra_PMIN(ra, nodes, components, t, axis=None, *args, **kwargs):
    if axis is None:
        axis = 0
    axis += 1
    ra_min = jnp.min(ra[jnp.ix_(t, components, nodes)], axis=axis)
    return jax.lax.pmin(ra_min, axis_name="x")
