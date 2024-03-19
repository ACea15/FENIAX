from jax import jit
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import pathlib
#from fem4inas.preprocessor.containers.intrinsicmodal import Dfem
from fem4inas.preprocessor import configuration
from fem4inas.intrinsic.functions import (compute_C0ab, tilde,
                                          coordinate_transform)
from functools import partial
import fem4inas.intrinsic.couplings as couplings
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
        Ka: jnp.ndarray,
        Ma: jnp.ndarray,
        num_modes: int,
        *args, **kwargs) -> (jnp.ndarray, jnp.ndarray):
    
    eigenvals, eigenvecs = generalized_eigh(Ka, Ma)
    #eigenvals, eigenvecs = eigh(Ka, Ma)
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs

def compute_eigs_scipy(
        Ka: jnp.ndarray,
        Ma: jnp.ndarray,
        num_modes: int,
        *args, **kwargs) -> (jnp.ndarray, jnp.ndarray):
    eigenvals, eigenvecs = scipy.linalg.eigh(Ka, Ma)
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs

def compute_eigs_load(num_modes: int,
                      path: pathlib.Path,
                      eig_names: list[str],
                      *args, **kwargs)-> (jnp.ndarray, jnp.ndarray):
    #eigenvals = jnp.load("/home/ac5015/programs/FEM4INAS/examples/SailPlane/FEM/w.npy")
    #eigenvecs = jnp.load("/home/ac5015/programs/FEM4INAS/examples/SailPlane/FEM/v.npy")
    #eigenvals = jnp.load("/home/ac5015/programs/FEM4INAS/examples/ArgyrisFrame/FEM/w.npy")
    #eigenvecs = jnp.load("/home/ac5015/programs/FEM4INAS/examples/ArgyrisFrame/FEM/v.npy")
    # eigenvals = jnp.load("/home/ac5015/programs/FEM4INAS/examples/ArgyrisBeam/FEM/w.npy")
    # eigenvecs = jnp.load("/home/ac5015/programs/FEM4INAS/examples/ArgyrisBeam/FEM/v.npy")
    if path is not None:
        eigenvals = jnp.load(path / eig_names[0])
        eigenvecs = jnp.load(path / eig_names[1])
    else:
        eigenvals = jnp.load(eig_names[0])
        eigenvecs = jnp.load(eig_names[1])        
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs

def compute_eigs_pass(num_modes, eigenvals, eigenvecs,
        *args, **kwargs):
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs
    
@partial(jit, static_argnames=['config'])
def shapes(X: jnp.ndarray,
           Ka: jnp.ndarray,
           Ma: jnp.ndarray,
           eigenvals: jnp.ndarray,
           eigenvecs: jnp.ndarray,
           config: configuration.Config):
    precision = config.jax_np.precision
    num_modes = config.fem.num_modes  # Nm
    num_nodes = config.fem.num_nodes  # Nn
    X_diff = jnp.matmul(X, config.fem.Mdiff, precision=precision)
    X_xdelta = jnp.linalg.norm(X_diff, axis=0)
    X_xdelta = X_xdelta.at[0].set(1.0)  #  so that there is no division by 0
    # by 0 below
    C0ab = compute_C0ab(X_diff, X_xdelta, config)  # shape=(3x3xNn)
    C06ab = make_C6(C0ab)  # shape=(6x6xNn)
    #eigenvals, eigenvecs = compute_eigs(Ka, Ma, num_modes)
    #eigenvals, eigenvecs = compute_eigs_scpy(Ka, Ma, num_modes)
    #eigenvals, eigenvecs = compute_eigs_load(num_modes)    
    omega = jnp.sqrt(eigenvals)
    # reorder to the grid coordinate in X and add 0s of clamped DoF
    _phi1 = jnp.matmul(config.fem.Mfe_order, eigenvecs, precision=precision)
    phi1 = reshape_modes(_phi1, num_modes, num_nodes)  # Becomes  (Nm, 6, Nn)
    # Define mode components in-between nodes
    phi1m = jnp.tensordot(phi1, config.fem.Mavg, axes=(2, 0), precision=precision)
    # Define mode components in the initial local-frame
    phi1l = coordinate_transform(phi1, C06ab, precision)  # effectively doing C0ba*phi1
    phi1ml = coordinate_transform(phi1m, C06ab, precision)
    _psi1 = jnp.matmul(Ma, eigenvecs, precision=precision)
    _psi1 = jnp.matmul(config.fem.Mfe_order, _psi1, precision=precision)
    psi1 = reshape_modes(_psi1, num_modes, num_nodes)
    psi1l = coordinate_transform(psi1, C06ab, precision=precision)
    # Nodal forces in global frame (equal to Ka*eigenvec)
    nodal_force = _psi1 * -eigenvals  # broadcasting (6Nn x Nm)
    # _nodal_force = jnp.matmul(Ka, -eigenvecs, precision=precision)
    # nodal_force = jnp.matmul(config.fem.Mfe_order, _nodal_force, precision=precision)
    _phi2 = reshape_modes(nodal_force, num_modes, num_nodes)  #(Nmx6xNn)
    #  Note: _phi2 are forces at the Nodes due to deformed shape, phi2 are internal forces
    #  as the sum of _phi2 along load-paths
    X3 = coordinates_difftensor(X, config.fem.Xm, precision)  # (3xNnxNn)
    X3tilde = -axis_tilde(X3)  # (6x6xNnxNn)
    _moments_force = moment_force(_phi2, X3tilde, precision)  # (Nmx6xNnxNn)
    moments_force = contraction(_moments_force,
                                config.fem.Mload_paths,
                                precision)  # (Nmx6xNn)
    # Sum all forces in the load-path from the present node to the free-ends
    # Each column in config.fem.Mload_paths represents the nodes to sum through
    phi2 = jnp.tensordot(
        _phi2, config.fem.Mload_paths, axes=(2, 0), precision=precision
    )
    phi2 += moments_force
    phi2l = coordinate_transform(phi2, C06ab, precision=precision)
    ematt_phi1 = ephi(config.const.EMAT, phi1ml, precision)
    phi1_diff = jnp.tensordot(
        phi1, config.fem.Mdiff, axes=(2, 0), precision=precision
    )
    phi1l_diff= coordinate_transform(phi1_diff, C06ab, precision=precision)
    psi2l = -phi1l_diff / X_xdelta + ematt_phi1

    return (phi1, psi1, phi2,
            phi1l, phi1ml, psi1l, phi2l, psi2l,
            omega, X_xdelta, C0ab, C06ab)

def scale(phi1: jnp.ndarray,
          psi1: jnp.ndarray,
          phi2: jnp.ndarray,
          phi1l: jnp.ndarray,
          phi1ml: jnp.ndarray,
          psi1l: jnp.ndarray,
          phi2l: jnp.ndarray,
          psi2l: jnp.ndarray,
          omega: jnp.ndarray,
          X_xdelta: jnp.ndarray,
          C0ab: jnp.ndarray,
          C06ab: jnp.ndarray,          
          *args, **kwargs):
    """Sacales the intrinsic modes

    The porpuse is that the integrals alpha1 and alpha2 are the
    identity

    Parameters
    ----------
    phi1 : jnp.ndarray
    psi1 : jnp.ndarray
    phi2 : jnp.ndarray
    phi1l : jnp.ndarray
    phi1ml : jnp.ndarray
    psi1l : jnp.ndarray
    phi2l : jnp.ndarray
    psi2l : jnp.ndarray
    omega : jnp.ndarray
    X_xdelta : jnp.ndarray
    C0ab : jnp.ndarray
    C06ab : jnp.ndarray
    *args :
    **kwargs :


    """

    alpha1 = couplings.f_alpha1(phi1, psi1)
    alpha2 = couplings.f_alpha2(phi2l, psi2l, X_xdelta)
    num_modes = len(alpha1)
    # Broadcasting in division
    phi1 /= jnp.sqrt(alpha1.diagonal()).reshape(num_modes, 1, 1)
    psi1 /= jnp.sqrt(alpha1.diagonal()).reshape(num_modes, 1, 1)
    phi1l /= jnp.sqrt(alpha1.diagonal()).reshape(num_modes, 1, 1)
    phi1ml /= jnp.sqrt(alpha1.diagonal()).reshape(num_modes, 1, 1)
    psi1l /= jnp.sqrt(alpha1.diagonal()).reshape(num_modes, 1, 1)
    phi2 /= jnp.sqrt(alpha2.diagonal()).reshape(num_modes, 1, 1)
    phi2l /= jnp.sqrt(alpha2.diagonal()).reshape(num_modes, 1, 1)
    psi2l /= jnp.sqrt(alpha2.diagonal()).reshape(num_modes, 1, 1)

    return (phi1, psi1, phi2,
            phi1l, phi1ml, psi1l, phi2l, psi2l,
            omega, X_xdelta, C0ab, C06ab)


def check_alphas(phi1, psi1,
                 phi2l, psi2l,
                 X_xdelta,
                 tolerance,
                 *args, **kwargs):

    alpha1 = couplings.f_alpha1(phi1, psi1)
    alpha2 = couplings.f_alpha2(phi2l, psi2l, X_xdelta)
    num_modes = len(alpha1)
    assert jnp.allclose(alpha1, jnp.eye(num_modes),
                        **tolerance), \
        f"Alpha1 not equal to Identity: Alpha1: {alpha1}"
    assert jnp.allclose(alpha2, jnp.eye(num_modes),
                        **tolerance), \
        f"Alpha2 not equal to Identity: Alpha2: {alpha2}"
    return alpha1, alpha2

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

@partial(jit, static_argnames=["precision"])
def contraction(moments: jnp.ndarray, loadpaths: jnp.ndarray,
                precision) -> jnp.ndarray:
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

    f = jax.vmap(lambda u, v: jnp.tensordot(u, v, axes=(2, 0), precision=precision),
                 in_axes=(2, 1), out_axes=2)
    fuv = f(moments, loadpaths)
    return fuv

@partial(jit, static_argnames=["precision"])
def moment_force(force: jnp.ndarray, X3t: jnp.ndarray, precision) -> jnp.ndarray:
    """Yields moments associated to each node due to the forces

    Parameters
    ----------
    force : jnp.ndarray
        Force tensor (Nmx6xNn) for which we want to obtain the
        resultant moments
    X3t : jnp.ndarray
        Tilde positions tensor (6x6xNnxNn)

    Returns
    -------
    jnp.ndarray: (Nmx6xNnxNn)
    
    """

    f1 = jax.vmap(lambda u, v: jnp.tensordot(u, v, axes=(1,1), precision=precision),
                  in_axes=(None,2), out_axes=2) # tensordot along coordinate axis (len=6)
    f2 = jax.vmap(f1, in_axes=(2, 3), out_axes=3)
    fuv = f2(force, X3t)

    return fuv

@partial(jit, static_argnames=["precision"])
def coordinates_difftensor(X: jnp.ndarray, Xm: jnp.ndarray, precision) -> jnp.ndarray:
    """Computes coordinates

    The tensor represents the following: Coordinates, middle point of each element,
    minus the position of each node in the structure

    Parameters
    ----------
    X : jnp.ndarray
        Grid coordinates
    Mavg : jnp.ndarray
        Matrix to calculate the averege point between nodes
    num_nodes : int
        Number of nodes  

    Returns
    -------
    X3 : jnp.ndarray: (3xNnxNn)
        Tensor, Xm*1 -(X*1)' : [Coordinates, Middle point of segment, Node]


    """

    #Xm = jnp.matmul(X, Mavg, precision=precision)
    num_nodes = X.shape[1]
    ones = jnp.ones(num_nodes)
    Xm3 = jnp.tensordot(Xm, ones, axes=0, precision=precision)  # copy Xm along a 3rd dimension
    Xn3 = jnp.transpose(jnp.tensordot(X, ones, axes=0, precision=precision),
                        axes=[0, 2, 1]) # copy X along the 2nd dimension
    X3 = (Xm3 - Xn3)
    return X3

@partial(jit, static_argnames=["precision"])
def ephi(emat, phi, precision):
    f = jax.vmap(lambda u, v: jnp.matmul(u, v,
                                         precision=precision),
                 in_axes=(2, None), out_axes=2)
    fuv = f(phi, emat)
    return fuv

@jit
def make_C6(v1) -> jnp.ndarray:
    """Given a 3x3xNn tensor, make the diagonal 6x6xNn

    It iterates over a third dimension in the input tensor

    Parameters
    ----------
    v1 : jnp.ndarray
        A tensor of the form (3x3xNn)

    """
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
def reshape_modes(_phi: jnp.ndarray,
                  num_modes: int,
                  num_nodes: int):
    """Reshapes vectors in the input matrix to form a 3rd-order tensor 

    Each vector is made into a 6xNn matrix

    Parameters
    ----------
    _phi : jnp.ndarray
        Matrix as in the output of eigenvector analysis (6NnxNm)
    num_modes : int
        Number of modes
    num_nodes : int
        Number of nodes


    """
    
    phi = jnp.reshape(_phi, (num_nodes, 6, num_modes), order="C")
    return phi.T
