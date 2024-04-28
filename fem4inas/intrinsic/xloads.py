import jax.numpy as jnp
import jax
from functools import partial

def redirect_to(another_function):
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            """Call the replacement function
            (another_function) instead of
            the original function"""
            return another_function(*args, **kwargs)

        return wrapper

    return decorator


# @partial(jax.jit, static_argnames=["t", "x"])
# def linear_interpolation(t, x, force_tensor):

#     xindex_upper = jnp.where(x >= t)[0][0]
#     xindex_lower = jnp.where(x <= t)[0][-1]
#     x_upper = x[xindex_upper]
#     x_lower = x[xindex_lower]
#     weight_upper = jax.lax.select(xindex_upper == xindex_lower,
#                                   0.5,
#                                   (t - x_lower) / (x_upper - x_lower))
#     weight_lower = jax.lax.select(xindex_upper == xindex_lower,
#                                   0.5,
#                                   (x_upper - t) / (x_upper - x_lower))

#     f_upper = force_tensor[xindex_upper]
#     f_lower = force_tensor[xindex_lower]
#     f_interpol = weight_upper * f_upper + weight_lower  * f_lower
#     return f_interpol

@jax.jit
def linear_interpolation(t, x, data_tensor):

    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(jax.lax.select(x >= t,
                                               jnp.ones(len_x),
                                               jnp.zeros(len_x)),
                                size=1)[0][0]#jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(jax.lax.select(x == t,
                                             jnp.ones(len_x),
                                             jnp.zeros(len_x)),
                          dtype=int)
    xindex_lower = xindex_upper - 1 + index_equal #jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (t - x_lower) / (x_upper - x_lower))
    weight_lower = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (x_upper - t) / (x_upper - x_lower))

    f_upper = data_tensor[xindex_upper]
    f_lower = data_tensor[xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower  * f_lower
    return f_interpol

@jax.jit
def linear_interpolation2(t, x, data_tensor):

    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(jax.lax.select(x >= t,
                                               jnp.ones(len_x),
                                               jnp.zeros(len_x)),
                                size=1)[0][0] #jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(jax.lax.select(x == t,
                                         jnp.ones(len_x),
                                         jnp.zeros(len_x)),
                          dtype=int)
    xindex_lower = xindex_upper - 1 + index_equal #jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (t - x_lower) / (x_upper - x_lower))
    weight_lower = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (x_upper - t) / (x_upper - x_lower))

    f_upper = data_tensor[:, :, xindex_upper]
    f_lower = data_tensor[:, :, xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower  * f_lower
    return f_interpol

@jax.jit
def linear_interpolation3(t, x, data_tensor):

    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(jax.lax.select(x >= t,
                                               jnp.ones(len_x),
                                               jnp.zeros(len_x)),
                                size=1)[0][0] #jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(jax.lax.select(x == t,
                                         jnp.ones(len_x),
                                         jnp.zeros(len_x)),
                          dtype=int)
    xindex_lower = xindex_upper - 1 + index_equal #jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (t - x_lower) / (x_upper - x_lower))
    weight_lower = jax.lax.select(xindex_upper == xindex_lower,
                                  0.5,
                                  (x_upper - t) / (x_upper - x_lower))

    f_upper = data_tensor[:, xindex_upper]
    f_lower = data_tensor[:, xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower  * f_lower
    return f_interpol

@jax.jit
def eta_pointfollower(t, phi1, x, force_follower):

    f = linear_interpolation(t, x, force_follower)
    eta = jnp.tensordot(phi1, f, axes=([1, 2],
                                       [0, 1]))
    return eta


@jax.jit
def eta_pointdead(t, phi1, x, force_dead, Rab):

    f1 = jax.vmap(lambda R, x: jnp.vstack(
            [jnp.hstack([R.T, jnp.zeros((3, 3))]),
             jnp.hstack([jnp.zeros((3, 3)), R.T])]) @ x,
                  in_axes=(2, 1), out_axes=1)
    f = linear_interpolation(t, x, force_dead)
    f_fd = f1(Rab, f)
    eta = jnp.tensordot(phi1, f_fd, axes=([1, 2],
                                          [0, 1]))
    return eta

@jax.jit
def eta_pointdead_const(phi1, f, Rab):

    f1 = jax.vmap(lambda R, x: jnp.vstack(
            [jnp.hstack([R.T, jnp.zeros((3, 3))]),
             jnp.hstack([jnp.zeros((3, 3)), R.T])]) @ x,
                  in_axes=(2, 1), out_axes=1)
    # f = linear_interpolation(t, x, force_dead)
    f_fd = f1(Rab, f)
    eta = jnp.tensordot(phi1, f_fd, axes=([1, 2],
                                          [0, 1]))
    return eta

#@partial(jax.jit, static_argnames=["num_modes", "num_poles"])
def eta_rogerstruct(q0, q1, ql,
                    A0hat, A1hat,
                    num_modes, num_poles):

    eta0 = A0hat @ q0 + A1hat @ q1
    num_modes = len(q0)
    num_poles = int(len(ql) / num_modes)
    #lags = jnp.tensordot(A2hat, ql_tensor, axis=(1,0))
    lags_sum = ql[:num_modes]
    for pi in range(1, num_poles):
        lags_sum += ql[pi*num_modes: (pi + 1)*num_modes]
    eta = eta0 + lags_sum
    #jax.debug.breakpoint()
    return eta

@jax.jit
def eta_steadyaero(q0: jnp.ndarray,
                   A0hat: jnp.ndarray):

    eta = A0hat @ q0
    return eta

@jax.jit
def eta_manoeuvre(qalpha: jnp.ndarray,
                  C0hat: jnp.ndarray):

    eta = C0hat @ qalpha
    return eta

@jax.jit
def eta_controls(qx, B0hat: jnp.ndarray):

    eta = B0hat @ qx
    return eta

@jax.jit
def eta_rogergust(t, xgust, F1gust):

    eta = linear_interpolation3(t, xgust, F1gust)
    return eta

#@partial(jax.jit, static_argnames=["num_modes", "num_poles"])
@jax.jit
def lags_rogerstructure(A3hat, q1, ql, u_inf, c_ref, poles,
                        num_modes, num_poles):
    num_modes = len(q1)
    num_poles = int(len(ql) / num_modes)
    ql_dot = jnp.zeros(num_modes * num_poles)
    for pi in range(num_poles):
        ql_dot = ql_dot.at[pi * num_modes: (pi+1) * num_modes].set(A3hat[pi] @ q1 -
                                                                   2 * u_inf /c_ref * poles[pi] *
                                                                   ql[pi * num_modes: (pi+1) * num_modes])
    return ql_dot

# @jax.jit
# def lags_rogerstructure1(A3hat, q1, ql, u_inf, c_ref, poles,
#                         num_modes, num_poles):
#     num_modes = len(q1)
#     num_poles = int(len(ql) / num_modes)
#     ql_dot = jnp.zeros(num_modes * num_poles)
#     for pi in range(num_poles):
#         ql_dot = ql_dot.at[pi * num_modes: (pi+1) * num_modes].set(A3hat[pi] @ q1)
#     return ql_dot

# @jax.jit
# def lags_rogerstructure2(A3hat, q1, ql, u_inf, c_ref, poles,
#                         num_modes, num_poles):
#     num_modes = len(q1)
#     num_poles = int(len(ql) / num_modes)
#     ql_dot = jnp.zeros(num_modes * num_poles)
#     for pi in range(num_poles):
#         ql_dot = ql_dot.at[pi * num_modes: (pi+1) * num_modes].set(-2 * u_inf /c_ref * poles[pi] *
#                                                                    ql[pi * num_modes: (pi+1) * num_modes])
#     return ql_dot

def lags_rogergust(t, xgust, Flgust):

    Flgust_tensor = linear_interpolation2(t, xgust, Flgust)  # NpxNm
    Flgust = jnp.hstack(Flgust_tensor)
    return Flgust

# def eta_rogergust(t, xgust, _wgust, _wgust_dot, _wgust_ddot,
#                   D0hat, D1hat, D2hat):

#     wgust = linear_interpolation(t, xgust, _wgust)
#     wgust_dot = linear_interpolation(t, xgust, _wgust_dot)
#     wgust_ddot = linear_interpolation(t, xgust, _wgust_ddot)
#     eta = D0hat @ wgust + D1hat @ wgust_dot + D2hat @ wgust_ddot
#     return eta

########################
def eta_000001(t, phi1, x, force_follower):

    f =  linear_interpolation(t, x, force_follower)
    eta = jnp.tensordot(phi1, f, axes=([1, 2],
                                       [0, 1]))
    return eta

@redirect_to(eta_000001)
def eta_001001(*args, **kwargs):
    pass

@jax.jit
def eta_00101(t, phi1, x, force_dead, Rab):

    f1 = jax.vmap(lambda R, x: jnp.vstack(
            [jnp.hstack([R.T, jnp.zeros((3, 3))]),
             jnp.hstack([jnp.zeros((3, 3)), R.T])]) @ x,
                  in_axes=(2, 1), out_axes=1)
    f = linear_interpolation(t, x, force_dead)
    f_fd = f1(Rab, f)
    eta = jnp.tensordot(phi1, f_fd, axes=([1, 2],
                                          [0, 1]))
    return eta

def eta_0011(q0: jnp.ndarray,
             qalpha: jnp.ndarray,
             u_inf: float,
             rho_inf: float,
             A0: jnp.ndarray,
             C0: jnp.ndarray):

    eta = 0.5 * rho_inf * u_inf ** 2 * (
        A0 @ q0 + C0 @ qalpha)
    return eta

@redirect_to(eta_000001)
def eta_101001(*args, **kwargs):
    pass

@redirect_to(eta_000001)
def eta_100001(*args, **kwargs):
    pass

@redirect_to(eta_001001)
def eta_10101(*args, **kwargs):
    pass

def project_phi1(force, phi1):

    eta = jnp.tensordot(phi1, force, axes=([1, 2],
                                           [0, 1]))
    return eta

def eta_gust():
    ...
