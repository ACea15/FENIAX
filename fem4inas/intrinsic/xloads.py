import jax.numpy as jnp
from jax import jit

def eta_0(q, t, phi1, follower_force):

    f =  follower_force #force_follower(t)
    eta = jnp.tensordot(phi1, f, axes=([1, 2],
                                       [0, 1]))
    return eta

def project_phi1(force, phi1):

    eta = jnp.tensordot(phi1, force, axes=([1, 2],
                                           [0, 1]))
    return eta
