import jax.numpy as jnp


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
