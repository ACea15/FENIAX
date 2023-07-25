from jax import jit
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from fem4inas.preprocessor.containers.intrinsicmodal import Dfem
from fem4inas.intrinsic.functions import compute_C0ab
from functools import partial

# TODO: implement from jnp.eigh and compare with jscipy.eigh
# https://math.stackexchange.com/questions/4518062/rewrite-generalized-eigenvalue-problem-as-standard-eigenvalue-problem


@jit
def generalized_eigh(A, B):
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    A_redo = L_inv.dot(A).dot(L_inv.T)
    return jnp.linalg.eigh(A_redo)


@partial(jit, static_argnames=["num_modes"])
def compute_eigs(
    Ka: jnp.ndarray, Ma: jnp.ndarray, num_modes: int
) -> (jnp.ndarray, jnp.ndarray):
    # eigenvals, eigenvecs = jscipy.linalg.eigh(Ka, Ma)
    eigenvals, eigenvecs = generalized_eigh(Ka, Ma)
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs


# @partial(jit, static_argnames=['config'])
def shapes(X: jnp.ndarray, Ka: jnp.ndarray, Ma: jnp.ndarray, config: Dfem):
    precision = config.jax_np.precision
    num_modes = config.fem.num_modes  # Nm
    num_nodes = config.fem.num_nodes  # Nn
    X_diff = jnp.matmul(X.T, config.fem.Mdiff)
    X_xdelta = jnp.linalg.norm(X_diff, axis=0)
    X_xdelta = X_xdelta.at[0].set(1.0)  #  so that there is no devision
    # by 0 below
    C0ab = compute_C0ab(X_diff, X_xdelta, config)  # shape=(3x3xNn)
    C06ab = make_C6(C0ab)  # shape=(6x6xNn)
    eigenvals, eigenvecs = compute_eigs(Ka, Ma, num_modes)
    # reorder to the grid coordinate in X and add 0s of clamped DoF
    _phi1 = jnp.matmul(config.fem.Mfe_order, eigenvecs)
    phi1 = reshape_modes(_phi1, num_modes, num_nodes)  # Becomes  (Nm, 6, Nn)
    # Define mode components in-between nodes
    phi1m = jnp.tensordot(phi1, config.fem.Mavg, axes=(2, 0), precision=precision)
    # Define mode components in the initial local-frame
    phi1l = coordinate_transform(phi1, C06ab)  # effectively doing C0ba*phi1
    phi1ml = coordinate_transform(phi1m, C06ab)
    _psi1 = jnp.matmul(Ma, eigenvecs, precision=precision)
    _psi1 = jnp.matmul(config.fem.Mfe_order, _psi1)
    psi1 = reshape_modes(_psi1, num_modes, num_nodes)
    # Nodal forces in global frame (equal to Ka*eigenvec)
    nodal_force = _psi1 * eigenvals  # broadcasting (6Nn x Nm)
    _phi2 = reshape_modes(nodal_force, num_modes, num_nodes)
    # Sum all forces in the load-path from the present node to the free-ends
    # Each column in config.fem.Mload_paths represents the nodes to sum through
    phi2 = jnp.tensordot(
        _phi2, config.fem.Mload_paths, axes=(2, 0), precision=precision
    )
    phi2 += jnp.tensordot(
        _phi2, config.fem.Mload_paths, axes=(2, 0), precision=precision
    )
    phi2l = coordinate_transform(phi2, C06ab)
    ematt_phi1 = ephi(config.const.EMAT, phi1ml)
    psi2l = jnp.tensordot(
        phi1l, config.fem.Mdiff, axes=(2, 0), precision=precision
    ) / X_xdelta + ematt_phi1

    return (phi1, psi1, phi2,
            phi1l, phi1ml, phi2l, psi2l,
            X_xdelta, C0ab)


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
    f = jax.vmap(lambda u, v: jnp.matmul(u, v), in_axes=(2, 2), out_axes=2)
    fuv = f(u1, v1)
    return fuv

@jit
def ephi(emat, phi):
    f = jax.vmap(lambda u, v: jnp.matmul(u, v), in_axes=(2, None), out_axes=2)
    fuv = f(phi, emat)
    return fuv

@jit
def make_C6(v1):
    f = jax.vmap(
        lambda v: jnp.vstack(
            [jnp.hstack([v, jnp.zeros((3, 3))]), jnp.hstack([jnp.zeros((3, 3)), v])]
        ),
        in_axes=2,
        out_axes=2,
    )
    fv = f(v1)
    return fv


@partial(jit, static_argnames=["num_modes", "num_nodes"])
def reshape_modes(_phi, num_modes, num_nodes):
    phi = jnp.reshape(_phi, (num_nodes, 6, num_modes), order="C")
    return phi.T


@partial(jit, static_argnames=["num_modes", "clamped_dof"])
def add_clampedDoF(_phi, num_modes: int, clamped_dof):
    phi = jnp.insert(_phi, clamped_dof, jnp.zeros(num_modes), axis=0)

    return phi
