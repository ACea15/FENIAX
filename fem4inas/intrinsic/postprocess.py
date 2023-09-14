import jax.numpy as jnp
import jax
from functools import partial
from fem4inas.intrinsic.functions import H0, H1

def compute_velocities(phi1l: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:

    X1 = jnp.tensordot(phi1l, q1, axes=(0,0))  # 6xNnxNt
    return X1

def compute_internalforces(phi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X2 = jnp.tensordot(phi2l, q2, axes=(0,0))  # 6xNnxNt
    return X2

def compute_strains(cphi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X3 = jnp.tensordot(cphi2l, q2, axes=(0,0))  # 6xNnxNt
    return X3

def velocity_Rab():
    ...

def strains_Rab():
    ...
    
def velocity_ra():
    ...

def strains_ra():
    ...

def integrate_X3(carry, x):

    kappa = x[0, :3]
    strain = x[0, 3:]
    ds = x[1]
    Cab0 = carry[:,:3]
    ra0 = carry[:, 3:]
    Ipsi = kappa * ds
    Itheta = jnp.linalg.norm(Ipsi)
    Cab = Cab0 @ H0(Itheta,Ipsi)
    ra = ra0 + Cab0 @ (H1(Itheta, Ipsi, ds) @ strain)
    y = jnp.hstack([Cab, ra.reshape((3,1))])
    return y, y

def integrate_strains(X3t, sol, fem):

    integrate_X3
    ds = sol.modes.X_xdelta
    C0ab = sol.modes.C0ab  # 3x3xNn
    # TODO: make as fori loop
    for i, ci in fem.component_names:
        ds_i = ds[fem.component_nodes[ci]]
        X3t_i = X3t[fem.component_nodes[ci]].T
        xs = jnp.concatenate([])
        jax.lax.scan(integrate_X3, init, xs)
