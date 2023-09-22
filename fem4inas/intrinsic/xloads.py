import jax.numpy as jnp
import jax

def linear_interpolation(t, x, force_tensor):

    xindex_upper = jnp.where(x >= t)[0][0]
    xindex_lower = jnp.where(x <= t)[0][-1]
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


def project_phi1(force, phi1):

    eta = jnp.tensordot(phi1, force, axes=([1, 2],
                                           [0, 1]))
    return eta
