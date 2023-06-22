from jax import jit
import jax.numpy as jnp
import jax.scipy as jscipy
from fem4inas.preprocessor.containers.intrinsicmodal import Dfem
from functools import partial

#TODO: implement from jnp.eigh and compare with jscipy.eigh
#https://math.stackexchange.com/questions/4518062/rewrite-generalized-eigenvalue-problem-as-standard-eigenvalue-problem

@partial(jit, static_argnames=['num_modes'])
def compute_eigs(Ka: jnp.Array, Ma: jnp.Array,
                 num_modes: int) -> (jnp.ndarray, jnp.ndarray):
 
    eigenvals, eigenvecs = jscipy.linalg.eigh(Ka, Ma)
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs

@partial(jit, static_argnames=['fem'])
def shapes(X: jnp.ndarray, Ka: jnp.ndarray, Ma: jnp.ndarray,
           eigenval: jnp.ndarray, eigenvec: jnp.ndarray, fem: Dfem):

    precision = Djax_np.precision
    num_modes = fem.num_modes
    eigenvals, eigenvecs = compute_eigs(Ka, Ma, num_modes)
    _phi1 = jnp.matmul(fem.FEorder, eigenvecs)
    _phi1 = jnp.insert(_phi1, fem.clamped_indices, jnp.zeros(num_modes), axis=0)
    phi = jnp.reshape(_phi1, (int(x.shape[0] / 6), 6, num))
    # Nodal force in global frame [Compute overall inbalance of the force]
    #=====================================================================
    nodal_force = jnp.dot(Ma, eigenvec, precision=precision) * eigenval

