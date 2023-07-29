from jax import jit
import numpy as np
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
###########################################

def _T(x):
    return jnp.swapaxes(x, -1, -2)


def _H(x):
    return jnp.conj(_T(x))


def symmetrize(x):
    return (x + _H(x)) / 2


def standardize_angle(w, b):
    if jnp.isrealobj(w):
        return w * jnp.sign(w[0, :])
    else:
        # scipy does this: makes imag(b[0] @ w) = 1
        assert not jnp.isrealobj(b)
        bw = b[0] @ w
        factor = bw / jnp.abs(bw)
        w = w / factor[None, :]
        sign = jnp.sign(w.real[0])
        w = w * sign
        return w


@jax.custom_jvp  # jax.scipy.linalg.eigh doesn't support general problem i.e. b not None
def eigh(a, b):
    """
    Compute the solution to the symmetrized generalized eigenvalue problem.

    a_s @ w = b_s @ w @ np.diag(v)

    where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized versions of the
    inputs and H is the Hermitian (conjugate transpose) operator.

    For self-adjoint inputs the solution should be consistent with `scipy.linalg.eigh`
    i.e.

    v, w = eigh(a, b)
    v_sp, w_sp = scipy.linalg.eigh(a, b)
    np.testing.assert_allclose(v, v_sp)
    np.testing.assert_allclose(w, standardize_angle(w_sp))

    Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which will be
    slow because there is no GPU implementation of `eig` and it's just a generally
    inefficient way of doing it. Future implementations should wrap cuda primitives.
    This implementation is provided primarily as a means to test `eigh_jvp_rule`.

    Args:
        a: [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
        b: [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

    Returns:
        v: eigenvalues of the generalized problem in ascending order.
        w: eigenvectors of the generalized problem, normalized such that
            w.H @ b @ w = I.
    """
    a = symmetrize(a)
    b = symmetrize(b)
    b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)
    v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)
    v = v.real
    # with loops.Scope() as s:
    #     for _ in s.cond_range(jnp.isrealobj)
    if jnp.isrealobj(a) and jnp.isrealobj(b):
        w = w.real
    # reorder as ascending in w
    order = jnp.argsort(v)
    v = v.take(order, axis=0)
    w = w.take(order, axis=1)
    # renormalize so v.H @ b @ H == 1
    norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)
    norm = jnp.sqrt(norm2)
    w = w / norm
    w = standardize_angle(w, b)
    return v, w


@eigh.defjvp
def eigh_jvp_rule(primals, tangents):
    """
    Derivation based on Boedekker et al.

    https://arxiv.org/pdf/1701.00392.pdf

    Note diagonal entries of Winv dW/dt != 0 as they claim.
    """
    a, b = primals
    da, db = tangents
    if not all(jnp.isrealobj(x) for x in (a, b, da, db)):
        raise NotImplementedError("jvp only implemented for real inputs.")
    da = symmetrize(da)
    db = symmetrize(db)

    v, w = eigh(a, b)

    # compute only the diagonal entries
    dv = jax.vmap(
        lambda vi, wi: -wi.conj() @ db @ wi * vi + wi.conj() @ da @ wi, in_axes=(0, 1),
    )(v, w)

    dv = dv.real

    E = v[jnp.newaxis, :] - v[:, jnp.newaxis]

    # diagonal entries: compute as column then put into diagonals
    diags = jnp.diag(-0.5 * jax.vmap(lambda wi: wi.conj() @ db @ wi, in_axes=1)(w))
    # off-diagonals: there will be NANs on the diagonal, but these aren't used
    off_diags = jnp.reciprocal(E) * (_H(w) @ (da @ w - db @ w * v[jnp.newaxis, :]))

    dw = w @ jnp.where(jnp.eye(a.shape[0], dtype=np.bool), diags, off_diags)

    return (v, w), (dv, dw)
####################################################################

@partial(jit, static_argnames=["num_modes"])
def compute_eigs(
    Ka: jnp.ndarray, Ma: jnp.ndarray, num_modes: int
) -> (jnp.ndarray, jnp.ndarray):
    # eigenvals, eigenvecs = jscipy.linalg.eigh(Ka, Ma)
    # eigenvals, eigenvecs = generalized_eigh(Ka, Ma)
    eigenvals, eigenvecs = eigh(Ka, Ma)
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
