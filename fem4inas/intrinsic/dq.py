import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
import fem4inas.intrinsic.xloads as xloads
import fem4inas.intrinsic.postprocess as postprocess

@jit
def contraction_gamma1(gamma1: jnp.ndarray,
                       q1:jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('ijk,jk->i', gamma1,
                     jnp.tensordot(q1, q1, axes=0))
    return res

@jit
def contraction_gamma2(gamma2: jnp.ndarray,
                       q2:jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('ijk,jk->i', gamma2,
                     jnp.tensordot(q2, q2, axes=0))
    return res

@jit
def contraction_gamma3(gamma2: jnp.ndarray,
                       q1: jnp.ndarray,
                       q2: jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('jik,jk->i', gamma2,
                     jnp.tensordot(q1, q2, axes=0))
    return res

@jit
def f_12(omega, gamma1, gamma2, q1, q2):

    F1 = (omega * q2 - contraction_gamma1(gamma1, q1)
          - contraction_gamma2(gamma2, q2))
    F2 = -omega * q1 + contraction_gamma3(gamma2, q1, q2)

    return F1, F2

def dq_000001(q, *args):

    (omega, phi1, x,
     force_follower, t) = args[0]
    F = omega * q
    F += xloads.eta_000001(t,
                           phi1,
                           x,
                           force_follower)
    return F

def dq_001001(q, *args):

    (gamma2, omega, phi1, x,
     force_follower, t) = args[0]
    F = omega * q - contraction_gamma2(gamma2, q)
    F += xloads.eta_001001(t,
                           phi1,
                           x,
                           force_follower)
    return F

#@jit
#@partial(jit, static_argnames=["args"])
def dq_00101(q, *args):

    (gamma2, omega, phi1l, psi2l,
     x, force_dead,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father, t) = args[0]
    #@jit
    def _dq_00101(q2):
        X3t = postprocess.compute_strains_t(psi2l, q2)
        Rab = postprocess.integrate_strainsCab(
            jnp.eye(3), X3t,
            X_xdelta, C0ab,
            component_names,
            num_nodes,
            component_nodes,
            component_father)
        F = omega * q2 - contraction_gamma2(gamma2, q2)
        F += xloads.eta_00101(t, phi1l, x, force_dead, Rab)
        return F
    F = _dq_00101(q)
    return F

@jit
def dq_0011(q, *args):

    (gamma2, omega,
     u_inf, rho_inf,
     qx, A0, B0) = args[0]
    q0 = -q / omega
    F = omega * q - contraction_gamma2(gamma2, q)
    F += xloads.eta_0011(q0, qx,
                         u_inf, rho_inf,
                         A0, B0)
    return F

@jit
def dq_101(t, q, *args):

    gamma1, gamma2, omega, states = args[0]
    q1 = q[states['q1']]
    q2 = q[states['q2']]
    F1, F2 = f_12(omega, gamma1, gamma2, q1, q2)
    F = jnp.hstack([F1, F2])
    return F

@jit
def dq_101001(t, q, *args):

    (gamma1, gamma2, omega, phi1, x,
     force_follower, states) = args[0]

    q1 = q[states['q1']]
    q2 = q[states['q2']]
    eta = xloads.eta_101001(t,
                            phi1,
                            x,
                            force_follower)
    F1, F2 = f_12(omega, gamma1, gamma2, q1, q2)
    F1 += eta
    F = jnp.hstack([F1, F2])
    return F

@jit
def dq_100001(t, q, *args):

    (omega, phi1, x,
     force_follower, states) = args[0]    
    q1 = q[states['q1']]
    q2 = q[states['q2']]
    eta = xloads.eta_100001(t,
                            phi1,
                            x,
                            force_follower)
    F1 = omega * q2
    F2 = -omega * q1
    F1 += eta
    F = jnp.hstack([F1, F2])
    return F

def dq_10101(t, q, *args):

    (gamma1, gamma2, omega, phi1l, psi2l,
     x, force_dead,
     states,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father) = args[0]
    q1i = q[states['q1']]
    q2i = q[states['q2']]
    #@jit
    def _dq_10101(t, q1, q2):
        X3t = postprocess.compute_strains_t(
            psi2l, q2)
        Rab = postprocess.integrate_strainsCab(
            jnp.eye(3), X3t,
            X_xdelta, C0ab,
            component_names,
            num_nodes,
            component_nodes,
            component_father)
        eta = xloads.eta_10101(t,
                               phi1l,
                               x,
                               force_dead,
                               Rab)
        F1, F2 = f_12(omega, gamma1, gamma2, q1, q2)
        F1 += eta
        F = jnp.hstack([F1, F2])
        return F
    F = _dq_10101(t, q1i, q2i)
    return F

if (__name__ == "__main__"):

    @jit
    def q_static_ein(q2, omega, gamma2, eta=0.):

        res = omega * q2
        res += jnp.einsum('ijk,jk->i',gamma2, jnp.tensordot(q2, q2, axes=0))
        #res += eta(q2)

        return res

    @jit
    def q_static_td(q2, omega, gamma2, eta=0.):

        res = omega * q2
        res += jnp.tensordot(gamma2, jnp.tensordot(q2, q2, axes=0), axes=([1,2],
                                                                          [0,1]))
        #res += eta(q2)

        return res

    def q_static_np(q2, omega, gamma2, eta=0.):

        res = omega * q2
        res += np.tensordot(gamma2, np.tensordot(q2, q2, axes=0), axes=([1,2],
                                                                        [0,1]))
        #res += eta(q2)

        return res

    def q_static_for(q2, omega, gamma2, eta=0.):

        nm = len(q2)
        res = np.zeros(nm)
        for i in range(nm):
            res[i] = omega[i] * q2[i]
            for j in range(nm):
                for k in range(nm):
                    res[i] += gamma2[i][j][k] * q2[j] * q2[k]
                    #res += eta(q2)

        return res
    
    import time

    NUM_MODES = 70
    NUM_ITER = 1000
    st1 = time.time()
    res = np.zeros(NUM_MODES)
    for i in range(NUM_ITER):
        omega = np.arange(i,i+NUM_MODES, dtype=float)
        q2 = np.arange(i,i+NUM_MODES, dtype=float) / NUM_MODES**4
        gamma2 = np.arange(NUM_MODES**3, dtype=float).reshape((NUM_MODES,
                                                               NUM_MODES, NUM_MODES ))
        res += q_static_np(q2, omega, gamma2)
    time1 = time.time() - st1

    st1 = time.time()
    res2 = jnp.zeros(NUM_MODES)
    for i in range(NUM_ITER):
        omega = jnp.arange(i, i+NUM_MODES, dtype=float)
        q2 = jnp.arange(i, i+NUM_MODES, dtype=float) / NUM_MODES**4
        gamma2 = jnp.arange(NUM_MODES**3, dtype=float).reshape((NUM_MODES,
                                                               NUM_MODES, NUM_MODES ))
        res2 += q_static_ein(q2, omega, gamma2)
    time2 = time.time() - st1

    st1 = time.time()
    res3 = jnp.zeros(NUM_MODES)
    for i in range(NUM_ITER):
        omega = jnp.arange(i, i+NUM_MODES, dtype=float)
        q2 = jnp.arange(i, i+NUM_MODES, dtype=float) / NUM_MODES**4
        gamma2 = jnp.arange(NUM_MODES**3, dtype=float).reshape((NUM_MODES,
                                                               NUM_MODES, NUM_MODES ))
        res3 += q_static_td(q2, omega, gamma2)
    time3 = time.time() - st1

    st1 = time.time()
    res4 = np.zeros(NUM_MODES)
    for i in range(NUM_ITER):
        omega = np.arange(i,i+NUM_MODES, dtype=float)
        q2 = np.arange(i,i+NUM_MODES, dtype=float) / NUM_MODES**4
        gamma2 = np.arange(NUM_MODES**3, dtype=float).reshape((NUM_MODES,
                                                               NUM_MODES, NUM_MODES ))
        res4 += q_static_for(q2, omega, gamma2)
    time4 = time.time() - st1

def y(x, *args):
    print(args)
