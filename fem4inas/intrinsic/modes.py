from jax import jit
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from fem4inas.preprocessor.containers.intrinsicmodal import Dfem
from fem4inas.intrinsic.functions import compute_C0ab
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
           config: Dfem):

    precision = config.jax_np.precision
    num_modes = config.fem.num_modes  # Nm
    num_nodes = config.fem.num_nodes  # Nn
    X_diff = jnp.matmul(X, config.fem.Mdiff)
    X_xdelta = jnp.linalg.norm(X_diff, axis=0)
    X_xdelta = X_xdelta.at[0].set(1.)
    C0ab = compute_C0ab(X_diff, X_xdelta, config) # shape=(3x3xNn)
    eigenvals, eigenvecs = compute_eigs(Ka, Ma, num_modes)
    # reorder to the grid coordinate in X and add 0s of clamped DoF
    _phi1 = jnp.matmul(config.fem.Mfe_order, eigenvecs)
    phi1 = reshape_modes(_phi1, num_modes, num_nodes) # Becomes  (Nm, 6, Nn)
    # Define mode components in-between nodes
    phi1m = jnp.tensordot(phi1, config.fem.Mavg,
                          axes=(2, 0), precision=precision)
    # Define mode components in the initial local-frame
    phi1l = coordinate_transform(phi1, C0ab) # effectively doing C0ba*phi1
    phi1ml = coordinate_transform(phi1m, C0ab)
    _psi1 = jnp.matmul(Ma, eigenvec, precision=precision)
    _psi1 = jnp.matmul(config.fem.Mfe_order, _psi1)
    psi1 = reshape_modes(_psi1, num_modes, num_nodes)
    # Nodal forces in global frame (equal to Ka*eigenvec)
    nodal_force = _psi1 * eigenval  # broadcasting (6Nn x Nm)
    _phi2 = reshape_modes(nodal_force, num_modes, num_nodes)
    # Sum all forces in the load-path from the present node to the free-ends
    # Each column in config.fem.Mload_paths represents the nodes to sum through
    phi2 = jnp.tensordot(_phi2, config.fem.Mload_paths,
                         axes=(2, 0), precision=precision)
    phi2 += jnp.tensordot(_phi2, config.fem.Mload_paths,
                          axes=(2, 0), precision=precision)
    phi2l = coordinate_transform(config.fem.Mglobal2local, phi2)
    psi2l = (jnp.tensordot(phi1l, config.fem.Mdiff, axes=(2, 0),
                           precision=precision) / X_xdelta
             + jnp.tensordot(phi1ml, config.const.EMAT, axes=(2, 0)))

    return phi1, phi1l, psi1, phi2l, psi2l, C0ab


def coordinates_difftensor(X: jnp.ndarray, Mavg: jnp.ndarray) -> jnp.ndarray:
    """Computes coordinates


    The tensor representes the following: Coordinates, middle point of each element,
    minus the position of each node in the structure

    Parameters
    ----------
    X : jnp.ndarray
        Grid coordinates
    Mavg : jnp.ndarray
        Matrix to calculate the averege point between nodes

    Returns
    -------
    jnp.ndarray: (3xNnxNn) tensor


    """
    
    num_nodes = X.shape[0]
    Xavg = jnp.matmul(X, Mavg)
    ones = jnp.ones(num_nodes)
    return jnp.tensordot(Xavg, ones) - jnp.tensordot(ones, X).T

@jit
def coordinate_transform(u1, v1):

    f = jax.vmap(lambda u, v: jnp.matmul(u, v),
                 in_axes=(2,2), out_axes=2)
    fuv = f(u1, v1)
    return fuv

@partial(jit, static_argnames=['num_modes', 'num_nodes'])
def reshape_modes(_phi, num_modes, num_nodes):

    phi = jnp.reshape(_phi, (num_modes, 6, num_nodes),
                      order='C')
    return phi

@partial(jit, static_argnames=['num_modes', 'clamped_dof'])
def add_clampedDoF(_phi, num_modes: int, clamped_dof):

    phi = jnp.insert(_phi, clamped_dof,
                     jnp.zeros(num_modes), axis=0)
    
    return phi
