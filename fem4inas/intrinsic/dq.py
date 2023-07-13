import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.config import config; config.update("jax_enable_x64", True)

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


if __name__ == "__main__":
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
