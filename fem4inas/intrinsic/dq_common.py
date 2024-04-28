import jax.numpy as jnp
import jax
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.intrinsic.functions as functions
from functools import partial

@jax.jit
def contraction_gamma1(gamma1: jnp.ndarray,
                       q1: jnp.ndarray) -> jnp.ndarray:
    """Contraction of Gamma1 with velocity modal coordinate.

    Parameters
    ----------
    gamma1 : jnp.ndarray
        3rd order tensor (NmxNmxNm) with velocity modal couplings
    q1 : jnp.ndarray
        velocity modal coordinate

    Returns
    -------
    jnp.ndarray
        Gamma1xq1xq1 (Nmx1)
    """
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

@jax.jit
def f_12l(omega, q1, q2):

    F1 = omega * q2
    F2 = -omega * q1

    return F1, F2


@jax.jit
def f_12aero(omega, gamma1, gamma2, q1, q2, A2hat):

    _F1 = (omega * q2 - contraction_gamma1(gamma1, q1)
          - contraction_gamma2(gamma2, q2))
    F1 = A2hat @ _F1
    F2 = -omega * q1 + contraction_gamma3(gamma2, q1, q2)

    return F1, F2

@jax.jit
def f_quaternion(phi1l, q1, qr):
    
    qr0 = qr[0]
    qrx = qr[1:]
    X1_0 = jnp.tensordot(phi1l[:,:,0], q1, axes=(0,0)) 
    X1_0w = X1_0[3:6]
    Fr0 = -0.5 * jnp.dot(X1_0w, qrx)
    Frx = 0.5 * (qr0 * X1_0w - functions.tilde(X1_0w) @ qrx)
    Fr = jnp.hstack([Fr0, Frx])
    return Fr

#@partial(jax.jit, static_argnums=5)
def computeRab_node0(psi2l, q2, qr, X_xdelta, C0ab, *args):

    (component_names,
     num_nodes,
     component_nodes,
     component_father) = args

    qr0 = qr[0]
    qrx = qr[1:]
    Rab0 = (2 * jnp.tensordot(qrx, qrx, axes=0) +
            (qr0 ** 2 - jnp.dot(qrx, qrx)) * jnp.eye(3) +
            2 * qr0 * functions.tilde(qrx))

    X3t = postprocess.compute_strains_t(
        psi2l, q2)
    Rab = postprocess.integrate_strainsCab(
        Rab0, X3t,
        X_xdelta, C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father)

    return Rab
