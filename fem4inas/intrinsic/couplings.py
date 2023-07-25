import jax
import jax.numpy as jnp
from jax import jit
import fem4inas.intrinsic.functions as functions

# TODO: add quadratic approx.

def f_Gamma1(phi1: jnp.array, psi1: jnp.array):

    f1 = jax.vmap(lambda u, v: jnp.tensordot(functions.L1(u), v, axes=(1, 1)),
              in_axes=(1,2), out_axes=2)  # iterate nodes
    f2 = jax.vmap(f1, in_axes=(0, None), out_axes=0)  # modes in 1st tensor
    L1 = f2(phi1, psi1) # Nmx6xNmxNm
    gamma1 = jnp.einsum('isn,jskn->ijk', phi1, L1)
    return gamma1

def f_Gamma2(phi1m: jnp.array,
             phi2: jnp.array,
             psi2: jnp.array,
             delta_s: jnp.array):

    f1 = jax.vmap(lambda u, v: jnp.tensordot(functions.L2(u), v, axes=(1, 1)),
              in_axes=(1,2), out_axes=2)  # iterate nodes
    f2 = jax.vmap(f1, in_axes=(0, None), out_axes=0)  # modes in 1st tensor
    L2 = f2(phi2, psi2) # Nmx6xNmxNm
    gamma2 = jnp.einsum('isn,jskn,n->ijk', phi1m, L2, delta_s)
    return gamma2

@jit    
def g1f_Gamma1(phi1: jnp.array, psi1: jnp.array):
    ...

    
