import jax.numpy as jnp
import jax

@jax.jit
def contraction_gamma1(gamma1: jnp.ndarray,
                       q1:jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('ijk,jk->i', gamma1,
                     jnp.tensordot(q1, q1, axes=0))
    return res

@jax.jit
def contraction_gamma2(gamma2: jnp.ndarray,
                       q2:jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('ijk,jk->i', gamma2,
                     jnp.tensordot(q2, q2, axes=0))
    return res

@jax.jit
def contraction_gamma3(gamma2: jnp.ndarray,
                       q1: jnp.ndarray,
                       q2: jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('jik,jk->i', gamma2,
                     jnp.tensordot(q1, q2, axes=0))
    return res

@jax.jit
def f_12(omega, gamma1, gamma2, q1, q2):

    F1 = (omega * q2 - contraction_gamma1(gamma1, q1)
          - contraction_gamma2(gamma2, q2))
    F2 = -omega * q1 + contraction_gamma3(gamma2, q1, q2)

    return F1, F2
