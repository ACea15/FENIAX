import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
import fem4inas.intrinsic.xloads as xloads

def contraction_gamma1(gamma1: jnp.ndarray,
                       q1:jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('ijk,jk->i', gamma1,
                     jnp.tensordot(q1, q1, axes=0))
    return res

def contraction_gamma2(gamma2: jnp.ndarray,
                       q2:jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('ijk,jk->i', gamma2,
                     jnp.tensordot(q2, q2, axes=0))
    return res

def contraction_gamma3(gamma2: jnp.ndarray,
                       q1: jnp.ndarray,
                       q2: jnp.ndarray) -> jnp.ndarray:

    res = jnp.einsum('jik,jk->i', gamma2,
                     jnp.tensordot(q1, q2, axes=0))
    return res

def f_12(omega, gamma1, gamma2, q1, q2):

    F1 = (omega * q2 + contraction_gamma1(gamma1, q1)
          - contraction_gamma2(gamma2, q2))
    F2 = -omega * q1 + contraction_gamma3(gamma2, q1, q2)

    return F1, F2

#@partial(jit, static_argnums=(2, 3))
#@jit
#@partial(jit, static_argnames=['sol','system'])
def _dq_001001(t, q, sol, system):

    #t, sol, system,  *xargs = args[0]
    gamma2 = sol.data.couplings.gamma2
    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    F = (omega * q
         - contraction_gamma2(gamma2, q)
         + xloads.eta_001001(t, phi1,
                             system.xloads.x,
                             system.xloads.force_follower))
    return F

def dq_001001(q, *args):

    t, sol, system,  *xargs = args[0]
    F = _dq_001001(t, q, sol, system)
    return F

def dq_000001(q, *args):
    
    t, sol, system,  *xargs = args[0]
    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    F = (omega * q
         + xloads.eta_001001(t, phi1,
                             system.xloads.x,
                             system.xloads.force_follower))

    return F

def dq_101001(t, q, *args):
    """Solver for structural dynamics with follower forces."""

    sol, system,  *xargs = args[0]
    gamma1 = sol.couplings.gamma1
    gamma2 = sol.couplings.gamma2
    omega = sol.fem.omega
    phi1 = sol.data.modes.phi1l
    q1 = q[system.states['q1']]
    q2 = q[system.states['q2']]
    eta = xloads.eta_001001(t,
                            phi1,
                            system.xloads.x,
                            system.xloads.force_follower)
    F1, F2 = f_12(omega, gamma1, gamma2, q1, q2)
    F1 += eta
    F = jnp.hstack([F1, F2])
    return F

def dq_100001(t, q, *args):
    """Solver for structural dynamics with follower forces."""

    sol, system,  *xargs = args[0]
    omega = sol.fem.omega
    phi1 = sol.data.modes.phi1l
    q1 = q[system.states['q1']]
    q2 = q[system.states['q2']]
    eta = xloads.eta_001001(t,
                            phi1,
                            system.xloads.x,
                            system.xloads.force_follower)
    F1 = omega * q2
    F2 = -omega * q1
    F1 += eta
    F = jnp.hstack([F1, F2])
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
