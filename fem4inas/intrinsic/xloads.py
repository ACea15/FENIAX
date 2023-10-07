import jax.numpy as jnp
import jax
from functools import partial

# @partial(jax.jit, static_argnames=["t", "x"])
# def linear_interpolation(t, x, force_tensor):

#     xindex_upper = jnp.where(x >= t)[0][0]
#     xindex_lower = jnp.where(x <= t)[0][-1]
#     x_upper = x[xindex_upper]
#     x_lower = x[xindex_lower]
#     weight_upper = jax.lax.select(xindex_upper == xindex_lower,
#                                   0.5,
#                                   (t - x_lower) / (x_upper - x_lower))
#     weight_lower = jax.lax.select(xindex_upper == xindex_lower,
#                                   0.5,
#                                   (x_upper - t) / (x_upper - x_lower))

#     f_upper = force_tensor[xindex_upper]
#     f_lower = force_tensor[xindex_lower]
#     f_interpol = weight_upper * f_upper + weight_lower  * f_lower
#     return f_interpol

def linear_interpolation(t, x, force_tensor):

    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(jax.lax.select(x >= t,
                                               jnp.ones(len_x),
                                               jnp.zeros(len_x)),
                                size=1)[0][0]#jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(jax.lax.select(x == t,
                                             jnp.ones(len_x),
                                             jnp.zeros(len_x)),
                          dtype=int)
    xindex_lower = xindex_upper - 1 + index_equal #jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (t - x_lower) / (x_upper - x_lower))
    weight_lower = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (x_upper - t) / (x_upper - x_lower))

    f_upper = force_tensor[xindex_upper]
    f_lower = force_tensor[xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower  * f_lower
    return f_interpol

def eta_001001(t, phi1, x, force_follower):

    f =  linear_interpolation(t, x, force_follower)
    eta = jnp.tensordot(phi1, f, axes=([1, 2],
                                       [0, 1]))
    return eta

def eta_001011(t, phi1, x, force_dead, Rab):

    f1 = jax.vmap(lambda R, x: R @ x , in_axes=1, out_axes=2)
    f =  linear_interpolation(t, x, force_dead)
    f_fd = f1(Rab, f)
    eta = jnp.tensordot(phi1, f_fd, axes=([1, 2],
                                       [0, 1]))
    return eta


def project_phi1(force, phi1):

    eta = jnp.tensordot(phi1, force, axes=([1, 2],
                                           [0, 1]))
    return eta
