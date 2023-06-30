from jax import jit
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from fem4inas.intrinsic.functions import tilde
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
           eigenval: jnp.ndarray, eigenvec: jnp.ndarray, config: Dfem):

    precision = config.jax_np.precision
    num_modes = config.fem.num_modes  # Nm
    num_nodes = config.fem.num_nodes  # Nn
    X_diff = jnp.matmul(X, config.fem.Mdiff)
    X_xdelta = jnp.linalg.norm(X_diff, axis=0)
    X_xdelta = X_xdelta.at[0].set(1.)
    C0ab = compute_C0ab(X_diff, X_xdelta, config)  # shape=(3x3xNn)
    eigenvals, eigenvecs = compute_eigs(Ka, Ma, num_modes)
    # reorder to the grid coordinate in X
    _phi1 = jnp.matmul(config.fem.Mfe_order, eigenvecs)
    # add 0s for clamped nodes and DoF
    _phi1 = add_clampedDoF(_phi1, config.fem.clamped_dof, num_modes)
    phi1 = reshape_modes(_phi1, num_modes, num_nodes)  # Becomes  (Nm, 6, Nn)
    # Define mode components in-between nodes
    phi1m = jnp.tensordot(phi1, config.fem.Mavg,
                          axes=(2, 0), precision=precision)
    # Define mode components in the initial local-frame
    phi1l = coordinate_transform(phi1, C0ab)  # effectively doing C0ba*phi1
    phi1ml = coordinate_transform(phi1m, C0ab)
    _psi1 = jnp.matmul(Ma, eigenvec, precision=precision)
    _psi1 = add_clampedDoF(_psi1, config.fem.clamped_dof, num_modes)
    psi1 = reshape_modes(_psi1, num_modes, num_nodes)
    # Nodal forces in global frame (equal to Ka*eigenvec)
    nodal_force = _psi1 * eigenval  # broadcasting (6Nn x Nm * Nm)
    _phi2 = reshape_modes(nodal_force, num_modes, num_nodes)
    X3 = coordinates_difftensor(X, config.fem.Mavg) # (3xNnxNn)
    X3tilde = axis_tilde(X3)  # (6x6xNnxNn)
    _moments_force = moment_force(_phi2, X3tilde)  # (Nmx6xNnxNn)
    moments_force = contraction(_moments_force,
                                config.fem.Mload_paths,
                                in_axes=(2, 1), out_axes=2)  # (Nmx6xNn)
    # Sum all forces in the load-path from the present node to the free-ends
    # Each column in config.fem.Mload_paths represents the nodes to sum through
    phi2 = jnp.tensordot(_phi2, config.fem.Mload_paths,
                         axes=(2, 0), precision=precision)
    phi2 += moments_force
    phi2l = coordinate_transform(config.fem.Mglobal2local, phi2)
    psi2l = (jnp.tensordot(phi1l, config.fem.Mdiff, axes=(2, 0),
                           precision=precision) / X_xdelta
             + jnp.tensordot(phi1ml, config.const.EMAT, axes=(2, 0)))

    return phi1, phi1l, psi1, phi2l, psi2l, C0ab

@jit
def tilde0010(vector: jnp.ndarray) -> jnp.ndarray:
    """Tilde matrix for cross product (moments due to forces)

    Parameters
    ----------
    vector : jnp.ndarray
        A 3-element array

    Returns
    -------
    jnp.ndarray
        6x6 matrix with (3:6 x 0:3) tilde operator

    """
    
    vector_tilde = jnp.vstack([jnp.zeros((3,6)),
                               jnp.hstack([tilde(vector), jnp.zeros((3,3))])
                               ])
    return vector_tilde

@jit
def axis_tilde(tensor: jnp.ndarray) -> jnp.ndarray:
    """Apply tilde0010 to a tensor

    The input tesor is iterated through axis 2 first, and axis 1
    subsequently; tilde0010 is applied to axis 0.

    Parameters
    ----------
    tensor : jnp.ndarray
        3xN1xN2 tensor

    Returns
    -------
    jnp.ndarray
        6x6xN1xN2 tensor

    """

    f1 = jax.vmap(tilde0010, in_axes=1, out_axes=2)
    f2 = jax.vmap(f1, in_axes=2, out_axes=3)
    f = f2(tensor)

    return f

@jit
def contraction(moments: jnp.ndarray, loadpaths: jnp.ndarray) -> jnp.ndarray:
    """Sums the moments from the nodal forces along the corresponding load path  

    Parameters
    ----------
    moments : jnp.ndarray
        num_modes x 6 x num_nodes(index) x num_nodes(moment at the
        previous index due to forces at this node)
    loadpaths : jnp.ndarray
        num_node x num_node such that [ni, nj] is 1 or 0 depending on
        whether ni is a node in the loadpath of nj respectively

    Returns
    -------
    jnp.ndarray
        num_modes x 6 x num_nodes(index) as the sum of moments
        due to forces at each node

    """

    f = jax.vmap(lambda u, v: jnp.dot(u, v),
                 in_axes=(2, 1), out_axes=2)
    fuv = f(moments, loadpaths)
    return fuv

@jit
def moment_force(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:

    f1 = jax.vmap(lambda u, v: jnp.matmul(u, v), in_axes=(2,2), out_axes=2)
    f2 = jax.vmap(f1, in_axes=(None, 3), out_axes=3)
    fuv = f2(u, v)

    return fuv

@jit
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
    jnp.ndarray: (3xNnxNn) tensor, Xm*1 -(x*1)'


    """
    
    num_nodes = X.shape[0]
    Xavg = jnp.matmul(X, Mavg)
    ones = jnp.ones(num_nodes)
    return jnp.tensordot(Xavg, ones) - jnp.tensordot(ones, X).T

@jit
def coordinate_transform(u1, v1) -> jnp.ndarray:

    f = jax.vmap(lambda u, v: jnp.matmul(u, v),
                 in_axes=(2,2), out_axes=2)
    fuv = f(u1, v1)
    return fuv

@partial(jit, static_argnames=['num_modes', 'num_node'])
def reshape_modes(_phi, num_modes, num_nodes:int) -> jnp.ndarray:

    phi = jnp.reshape(_phi, (num_modes, 6, num_nodes),
                      order='C')
    return phi

@partial(jit, static_argnames=['num_modes', 'clamped_dof'])
def add_clampedDoF(_phi, num_modes, clamped_dof) -> jnp.ndarray:

    phi = jnp.insert(_phi, clamped_dof,
                     jnp.zeros(num_modes), axis=0)
    
    return phi
