import jax.numpy as jnp
import math

# epsilon for testing whether a number is close to zero
_EPS = jnp.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> jnp.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> jnp.allclose(M, jnp.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> jnp.allclose(M, jnp.diag([1, -1, -1, 1]))
    True

    """
    q = jnp.array(quaternion, copy=True)
    n = jnp.dot(q, q)
    if n < _EPS:
        return jnp.identity(4)
    q *= math.sqrt(2.0 / n)
    q = jnp.outer(q, q)
    return jnp.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(jnp.identity(4), True)
    >>> jnp.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(jnp.diag([1, -1, -1, 1]))
    >>> jnp.allclose(q, [0, 1, 0, 0]) or jnp.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> jnp.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> jnp.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> jnp.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True

    """
    M = jnp.array(matrix, copy=False)[:4, :4]
    if isprecise:
        q = jnp.empty((4, ))
        t = jnp.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = jnp.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = jnp.linalg.eigh(K)
        q = V[[3, 0, 1, 2], jnp.argmax(w)]
    if q[0] < 0.0:
        jnp.negative(q, q)
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> jnp.allclose(q, [28, -44, -14, 48])
    True

    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return jnp.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=jnp.float64)


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
    True

    """
    q = jnp.array(quaternion, copy=True)
    jnp.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> jnp.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True

    """
    q = jnp.array(quaternion, copy=True)
    jnp.negative(q[1:], q[1:])
    return q / jnp.dot(q, q)

def quaternion_rotation(v,q):
    #import time
    #start_time = time.time()
    #vr=(q[0]**2-jnp.linalg.norm(q[1:])**2)*v + 2*(jnp.dot(q[1:],v))*q[1:] + 2*q[0]*jnp.cross(q[1:],v)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    vr = quaternion_multiply(q,quaternion_multiply(jnp.asarray([0,v[0],v[1],v[2]]),quaternion_conjugate(q)))
    return vr
