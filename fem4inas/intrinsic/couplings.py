import jax
import jax.numpy as jnp
from jax import jit

#from . import functions
def f_Gamma1(phi1: jnp.array, psi1: jnp.array):

    jnp.sum(functions.L1())

def f_Gamma2(phi1m: jnp.array,
             phi2: jnp.array,
             psi2: jnp.array,
             delta_s: jnp.array):
    ...

@jit    
def g1f_Gamma1(phi1: jnp.array, psi1: jnp.array):
    ...

    
