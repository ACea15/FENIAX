import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, lax
from fem4inas.preprocessor import configuration
from functools import partial

###############################
# Exponential map integration #
###############################

@jit
def tilde(vector3: jnp.ndarray):
    """Compute matrix that yields the cross product.

    Parameters
    ----------
    vector3 : jnp.ndarray
      Vector of length 3

    """
    tilde = jnp.array([[0.,    -vector3[2], vector3[1]],
                       [vector3[2],  0.,       -vector3[0]],
                       [-vector3[1], vector3[0], 0.]])
    return tilde

@jit
def H0(Itheta: float, Ipsi: jnp.ndarray):

    I3 = jnp.eye(3)
    if Itheta == 0.0:
        return I3
    else:
        return (I3 + jnp.sin(Itheta) / Itheta * tilde(Ipsi)
                + (1 - jnp.cos(Itheta)) / Itheta**2 * jnp.matmul(tilde(Ipsi),
                                                                 tilde(Ipsi)))

@jit
def H1(Itheta: float, Ipsi: jnp.ndarray, ds: float):

    I3 = jnp.eye(3)
    if Itheta == 0.0:
        return I3*ds
    else:
        return ds *(I3 + (1 - jnp.cos(Itheta)) / Itheta**2 * tilde(Ipsi)
                    + (Itheta - jnp.sin(Itheta)) / (Itheta**3) * jnp.matmul(tilde(Ipsi),
                                                                            tilde(Ipsi)))

@jit
def L1(x1: jnp.ndarray):

    L1 = jnp.zeros((6, 6))
    v = x1[0:3]
    w = x1[3:6]

    L1 = L1.at[0:3, 0:3].set(tilde(w))
    L1 = L1.at[3:6, 3:6].set(tilde(w))
    L1 = L1.at[3:6, 0:3].set(tilde(v))
    return L1

def tilde_np(vector3: np.ndarray):
    """Compute matrix that yields the cross product.

    Parameters
    ----------
    vector3 : jnp.ndarray
      Vector of length 3

    """
    tilde = np.array([[0.,    -vector3[2], vector3[1]],
                      [vector3[2],  0.,       -vector3[0]],
                      [-vector3[1], vector3[0], 0.]])
    return tilde

def L1_np(x1: np.ndarray):

    L1 = np.zeros((6, 6))
    v = x1[0:3]
    w = x1[3:6]

    L1[0:3, 0:3] = (tilde_np(w))
    L1[3:6, 3:6] = (tilde_np(w))
    L1[3:6, 0:3] = (tilde_np(v))
    return L1

@jit
def L2(x2: jnp.ndarray):

    L2 = jnp.zeros((6, 6))
    f = x2[0:3]
    m = x2[3:6]

    L2 = L2.at[0:3, 3:6].set(tilde(f))
    L2 = L2.at[3:6, 3:6].set(tilde(m))
    L2 = L2.at[3:6, 0:3].set(tilde(f))
    return L2

@partial(jit, static_argnames=['config'])
def compute_C0ab(X_diff: jnp.ndarray, X_xdelta: jnp.ndarray,
                 config: configuration.Config) -> jnp.ndarray:

    x = X_diff / X_xdelta
    x = x.at[:, 0].set(jnp.array([1, 0, 0])) # WARNING: this says the first node FoR at time 0
    # aligns with the global reference frame.
    # TODO: check this works when x_local = [0,0,1]
    cond = jnp.linalg.norm(x - jnp.array([[0,0,1]]).T) > config.ex.Cab_xtol # if not true,
    # local-x is almost parallel to global-z, z direction parallel to global y
    y = lax.select(cond,
                   jnp.cross(jnp.array([0, 0, 1]), x, axisb=0, axisc=0),
                   jnp.cross(jnp.array([0, 1, 0]), x, axisb=0, axisc=0))
    y /= jnp.linalg.norm(y, axis=0)
    z = jnp.cross(x, y, axisa=0, axisb=0, axisc=0)
    C0ab = jnp.stack([x, y, z], axis=1)
    return C0ab

