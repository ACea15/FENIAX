from jax import jit
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from fem4inas.preprocessor.containers.intrinsicmodal import Dfem
from functools import partial

#TODO: implement from jnp.eigh and compare with jscipy.eigh
#https://math.stackexchange.com/questions/4518062/rewrite-generalized-eigenvalue-problem-as-standard-eigenvalue-problem

@partial(jit, static_argnames=['num_modes'])
def compute_eigs(Ka: jnp.ndarray, Ma: jnp.ndarray,
                 num_modes: int) -> (jnp.ndarray, jnp.ndarray):
 
    eigenvals, eigenvecs = jscipy.linalg.eigh(Ka, Ma)
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs

@partial(jit, static_argnames=['config'])
def shapes(X: jnp.ndarray, Ka: jnp.ndarray, Ma: jnp.ndarray,
           eigenval: jnp.ndarray, eigenvec: jnp.ndarray, config: Dfem):

    precision = config.jax_np.precision
    num_modes = config.fem.num_modes  # Nm
    num_nodes = config.fem.num_nodes  # Nn
    delta_X = jnp.linalg.norm(jnp.matmul(config.fem.Mdiff, X), axis=0)
    eigenvals, eigenvecs = compute_eigs(Ka, Ma, num_modes)
    # reorder to the grid coordinate in X
    _phi1 = jnp.matmul(config.fem.Mfe_order, eigenvecs)
    # add 0s for clamped nodes and DoF
    _phi1 = add_clampedDoF(_phi1, config.fem.clamped_dof, num_modes)
    phi1 = reshape_modes(_phi1, num_modes) # Becomes  (Nm, 6, Nn)
    # Define mode components in-between nodes
    phi1m = jnp.tensordot(phi1, config.fem.Maverage_nodes,
                          axes=(2, 0), precision=precision)
    # Define mode components in the initial local-frame
    phi1l = coordinate_transform(config.fem.T0ba, phi1)
    phi1ml = coordinate_transform(config.fem.T0ba, phi1m)
    _psi1 = jnp.matmul(Ma, eigenvec, precision=precision)
    _psi1 = add_clampedDoF(_psi1, config.fem.clamped_dof, num_modes)
    psi1 = reshape_modes(_psi1, num_modes)
    # Nodal forces in global frame (equal to Ka*eigenvec)
    nodal_force = _psi1 * eigenval  # broadcasting (6Nn x Nm * Nm)
    _phi2 = reshape_modes(nodal_force, num_modes)
    # Sum all forces in the load-path from the present node to the free-ends
    # Each column in config.fem.Mload_paths represents the nodes to sum through
    phi2 = jnp.tensordot(_phi2, config.fem.Mload_paths,
                         axes=(2, 0), precision=precision)
    phi2l = coordinate_transform(config.fem.Mglobal2local, phi2)
    psi2l = (jnp.tensordot(phi1l, config.fem.Mdiff, axes=(2, 0),
                           precision=precision) / delta_X
             + jnp.tensordot(phi1ml, config.const.EMAT, axes=(2, 0)))

    return phi1, phi1l, psi1, phi2l, psi2l
    
    
@jit
def coordinate_transform(u1, v1):

    f = jax.vmap(lambda u, v: jnp.matmul(v, u.T),
                 in_axes=(0,1), out_axes=1)
    fuv = f(u1, v1)
    return fuv

@partial(jit, static_argnames=['num_modes'])
def reshape_modes(_phi, num_modes):

    phi = jnp.reshape(_phi, (num_modes, 6, int(_phi.shape[0] / 6)),
                      order='C')
    return phi

@partial(jit, static_argnames=['num_modes', 'clamped_dof'])
def add_clampedDoF(_phi, num_modes, clamped_dof):

    phi = jnp.insert(_phi, clamped_dof,
                     jnp.zeros(num_modes), axis=0)
    
    return phi
