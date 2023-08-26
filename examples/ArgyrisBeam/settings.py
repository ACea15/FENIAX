import pathlib
import pdb
import sys

import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[]]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 10
inp.fem.fe_order_start = 1
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.ex.Cab_xtol = 1e-4
inp.systems.s1.solution = "static"
inp.systems.s1.solver_library = "diffrax"
inp.systems.s1.solver_function = "newton_raphson"
inp.systems.s1.xloads.follower_forces = True
inp.systems.s1.xloads.follower_points = [[0, -1, 1]]
inp.systems.s1.xloads.follower_interpolation = [[[1., 3.7e3],
                                                 [2, 12.1e3],
                                                 [3, 17.5e3],
                                                 [4, 39.3e3],
                                                 [5, 61.0e3],
                                                 [6, 94.5e3],
                                                 [7, 120e3]
                                                 ]
                                                ]
config =  configuration.Config(inp)

# for k, v in config._data_dict['fem'].items():
#     print(f"{k}:{type(v[0])}")


path2config = pathlib.Path("./config.yaml")
#config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config)

sol = fem4inas.fem4inas_main.main(input_obj=config)


import numpy as np
import scipy
import jax.numpy as jnp
import jax
from jax.config import config; config.update("jax_enable_x64", True)

Ka = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25/FEM/Kaa.npy")
Ma = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25/FEM/Maa.npy")
w, v = scipy.linalg.eigh(Ka, Ma, driver='gvx')

def generalized_eigh(A, B):
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    A_redo = L_inv.dot(A).dot(L_inv.T)
    return jnp.linalg.eigh(A_redo)

def generalized_eigh2(A, B):
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    A_redo = L_inv.dot(A).dot(L_inv.T)
    w, v = jnp.linalg.eigh(A_redo)
    return jax.experimental.sparse.linalg.lobpcg_standard(A_redo, jnp.eye(len(A_redo))[:, :20])

w2, v2 = generalized_eigh(Ka, Ma)
w3 = generalized_eigh2(Ka, Ma)


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


#@jax.custom_jvp  # jax.scipy.linalg.eigh doesn't support general problem i.e. b not None
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

w5, v5 = eigh(Ka, Ma)

                                                        
