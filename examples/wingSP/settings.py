import pathlib
import jax.numpy as jnp
import pdb
import sys
import datetime
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 53
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.tn = 15001

inp.systems.sett.s1.solver_library = "diffrax"
# inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode2"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")
#inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
inp.systems.sett.s1.label = 'dq_101001'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]

#inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]


config =  configuration.Config(inp)

path2config = pathlib.Path("./config.yaml")
#config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config)

sol = fem4inas.fem4inas_main.main(input_obj=config)

# import jax
# import fem4inas.intrinsic.functions as functions
# f1 = jax.vmap(lambda u, v: jnp.tensordot(functions.L2(u), v, axes=(1, 1)),
#               in_axes=(1,2), out_axes=2)  # iterate nodes
# f2 = jax.vmap(f1, in_axes=(0, None), out_axes=0)  # modes in 1st tensor
# L2 = f2(sol.modes.phi2l, sol.modes.psi2l) # Nmx6xNmxNm
# gamma2 = jnp.einsum('isn,jskn,n->ijk', sol.modes.phi1ml, L2, sol.modes.X_xdelta)



# config2 = configuration.Config.from_file(path2config)

# yaml = YAML()
# yaml_dict = yaml.load(pth1)
# for k, v in config2.fem.__dict__.items():
#     if v != getattr(config.fem, k):
#         print(k)

# import pathlib
# import re
# p1 = list(pathlib.Path("./FEM/").glob("*Ka*"))[0]
# for pi in p1:
#     print(list(re.match("*Ka*", pi)))

# import numpy as np
# import scipy
# import jax.numpy as jnp
# import jax

# def generalized_eigh(A, B):
#     L = jnp.linalg.cholesky(B)
#     L_inv = jnp.linalg.inv(L)
#     A_redo = L_inv.dot(A).dot(L_inv.T)
#     return jnp.linalg.eigh(A_redo)

# Ka = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisFrame_20/FEM/Kaa.npy")
# Ma = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisFrame_20/FEM/Maa.npy")
# w, v = scipy.linalg.eigh(Ka, Ma)

# w2, v2 = scipy.linalg.eigh(config.fem.Ka, config.fem.Ma)

# # Ka3 = np.load("/media/acea/work/projects/FEM4INAS/examples/ArgyrisFrame/FEM/Ka.npy")
# # Ma3 = np.load("/media/acea/work/projects/FEM4INAS/examples/ArgyrisFrame/FEM/Ma.npy")
# # w3, v3 = scipy.linalg.eigh(Ka3, Ma3)

# w4, v4 = generalized_eigh(config.fem.Ka, config.fem.Ma)



# def _T(x):
#     return jnp.swapaxes(x, -1, -2)


# def _H(x):
#     return jnp.conj(_T(x))


# def symmetrize(x):
#     return (x + _H(x)) / 2


# def standardize_angle(w, b):
#     if jnp.isrealobj(w):
#         return w * jnp.sign(w[0, :])
#     else:
#         # scipy does this: makes imag(b[0] @ w) = 1
#         assert not jnp.isrealobj(b)
#         bw = b[0] @ w
#         factor = bw / jnp.abs(bw)
#         w = w / factor[None, :]
#         sign = jnp.sign(w.real[0])
#         w = w * sign
#         return w


# @jax.custom_jvp  # jax.scipy.linalg.eigh doesn't support general problem i.e. b not None
# def eigh(a, b):
#     """
#     Compute the solution to the symmetrized generalized eigenvalue problem.

#     a_s @ w = b_s @ w @ np.diag(v)

#     where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized versions of the
#     inputs and H is the Hermitian (conjugate transpose) operator.

#     For self-adjoint inputs the solution should be consistent with `scipy.linalg.eigh`
#     i.e.

#     v, w = eigh(a, b)
#     v_sp, w_sp = scipy.linalg.eigh(a, b)
#     np.testing.assert_allclose(v, v_sp)
#     np.testing.assert_allclose(w, standardize_angle(w_sp))

#     Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which will be
#     slow because there is no GPU implementation of `eig` and it's just a generally
#     inefficient way of doing it. Future implementations should wrap cuda primitives.
#     This implementation is provided primarily as a means to test `eigh_jvp_rule`.

#     Args:
#         a: [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
#         b: [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

#     Returns:
#         v: eigenvalues of the generalized problem in ascending order.
#         w: eigenvectors of the generalized problem, normalized such that
#             w.H @ b @ w = I.
#     """
#     a = symmetrize(a)
#     b = symmetrize(b)
#     b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)
#     v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)
#     v = v.real
#     # with loops.Scope() as s:
#     #     for _ in s.cond_range(jnp.isrealobj)
#     if jnp.isrealobj(a) and jnp.isrealobj(b):
#         w = w.real
#     # reorder as ascending in w
#     order = jnp.argsort(v)
#     v = v.take(order, axis=0)
#     w = w.take(order, axis=1)
#     # renormalize so v.H @ b @ H == 1
#     norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)
#     norm = jnp.sqrt(norm2)
#     w = w / norm
#     w = standardize_angle(w, b)
#     return v, w

# w5, v5 = eigh(config.fem.Ka, config.fem.Ma)


# a = jnp.arange(4*6*7).reshape((4,6,7))
