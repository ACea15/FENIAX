import jax.numpy as jnp
import jax
from functools import partial
import feniax.intrinsic.functions as functions
import itertools
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal

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
    xindex_upper = jnp.argwhere(
        jax.lax.select(x >= t, jnp.ones(len_x), jnp.zeros(len_x)), size=1
    )[0][0]  # jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(
        jax.lax.select(x == t, jnp.ones(len_x), jnp.zeros(len_x)), dtype=int
    )
    xindex_lower = xindex_upper - 1 + index_equal  # jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(
        xindex_upper == xindex_lower, 0.5, (t - x_lower) / (x_upper - x_lower)
    )
    weight_lower = jax.lax.select(
        xindex_upper == xindex_lower, 0.5, (x_upper - t) / (x_upper - x_lower)
    )

    f_upper = data_tensor[xindex_upper]
    f_lower = data_tensor[xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower * f_lower
    return f_interpol


@jax.jit
def linear_interpolation2(t, x, data_tensor):
    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(
        jax.lax.select(x >= t, jnp.ones(len_x), jnp.zeros(len_x)), size=1
    )[0][0]  # jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(
        jax.lax.select(x == t, jnp.ones(len_x), jnp.zeros(len_x)), dtype=int
    )
    xindex_lower = xindex_upper - 1 + index_equal  # jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(
        xindex_upper == xindex_lower, 0.5, (t - x_lower) / (x_upper - x_lower)
    )
    weight_lower = jax.lax.select(
        xindex_upper == xindex_lower, 0.5, (x_upper - t) / (x_upper - x_lower)
    )

    f_upper = data_tensor[:, :, xindex_upper]
    f_lower = data_tensor[:, :, xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower * f_lower
    return f_interpol


@jax.jit
def linear_interpolation3(t, x, data_tensor):
    len_x = x.shape[0]
    xindex_upper = jnp.argwhere(
        jax.lax.select(x >= t, jnp.ones(len_x), jnp.zeros(len_x)), size=1
    )[0][0]  # jnp.where(x >= t)[0][0]
    index_equal = jnp.sum(
        jax.lax.select(x == t, jnp.ones(len_x), jnp.zeros(len_x)), dtype=int
    )
    xindex_lower = xindex_upper - 1 + index_equal  # jnp.where(x <= t)[0][-1]
    x_upper = x[xindex_upper]
    x_lower = x[xindex_lower]
    weight_upper = jax.lax.select(
        xindex_upper == xindex_lower, 0.5, (t - x_lower) / (x_upper - x_lower)
    )
    weight_lower = jax.lax.select(
        xindex_upper == xindex_lower, 0.5, (x_upper - t) / (x_upper - x_lower)
    )

    f_upper = data_tensor[:, xindex_upper]
    f_lower = data_tensor[:, xindex_lower]
    f_interpol = weight_upper * f_upper + weight_lower * f_lower
    return f_interpol


@jax.jit
def eta_pointfollower(t, phi1, x, force_follower):
    f = linear_interpolation(t, x, force_follower)
    eta = jnp.tensordot(phi1, f, axes=([1, 2], [0, 1]))
    return eta


@jax.jit
def eta_pointdead(t, phi1, x, force_dead, Rab):
    f1 = jax.vmap(
        lambda R, x: jnp.vstack(
            [jnp.hstack([R.T, jnp.zeros((3, 3))]), jnp.hstack([jnp.zeros((3, 3)), R.T])]
        )
        @ x,
        in_axes=(2, 1),
        out_axes=1,
    )
    f = linear_interpolation(t, x, force_dead)
    f_fd = f1(Rab, f)
    eta = jnp.tensordot(phi1, f_fd, axes=([1, 2], [0, 1]))
    return eta


@jax.jit
def eta_pointdead_const(phi1, f, Rab):
    f1 = jax.vmap(
        lambda R, x: jnp.vstack(
            [jnp.hstack([R.T, jnp.zeros((3, 3))]), jnp.hstack([jnp.zeros((3, 3)), R.T])]
        )
        @ x,
        in_axes=(2, 1),
        out_axes=1,
    )
    # f = linear_interpolation(t, x, force_dead)
    f_fd = f1(Rab, f)
    eta = jnp.tensordot(phi1, f_fd, axes=([1, 2], [0, 1]))
    return eta


# @partial(jax.jit, static_argnames=["num_modes", "num_poles"])
def eta_rogerstruct(q0, q1, ql, A0hat, A1hat, num_modes, num_poles):
    eta0 = A0hat @ q0 + A1hat @ q1
    num_modes = len(q0)
    num_poles = int(len(ql) / num_modes)
    # lags = jnp.tensordot(A2hat, ql_tensor, axis=(1,0))
    lags_sum = ql[:num_modes]
    for pi in range(1, num_poles):
        lags_sum += ql[pi * num_modes : (pi + 1) * num_modes]
    eta = eta0 + lags_sum
    # jax.debug.breakpoint()
    return eta


@jax.jit
def eta_steadyaero(q0: jnp.ndarray, A0hat: jnp.ndarray):
    eta = A0hat @ q0
    return eta


@jax.jit
def eta_manoeuvre(t, x, qalpha: jnp.ndarray, C0hat: jnp.ndarray):
    
    qalpha_ti = linear_interpolation(t, x, qalpha)
    eta = C0hat @ qalpha_ti
    return eta


@jax.jit
def eta_controls(
    qx, B0hat: jnp.ndarray, elevator_index: jnp.ndarray, elevator_link: jnp.ndarray
):
    eta = jnp.tensordot(B0hat[:, elevator_index] * qx, elevator_link, axes=(1, 0))
    return eta


@jax.jit
def eta_rogergust(t, xgust, F1gust):
    eta = linear_interpolation3(t, xgust, F1gust)
    return eta


# @partial(jax.jit, static_argnames=["num_modes", "num_poles"])
@jax.jit
def lags_rogerstructure(A3hat, q1, ql, u_inf, c_ref, poles, num_modes, num_poles):
    num_modes = len(q1)
    num_poles = int(len(ql) / num_modes)
    ql_dot = jnp.zeros(num_modes * num_poles)
    for pi in range(num_poles):
        ql_dot = ql_dot.at[pi * num_modes : (pi + 1) * num_modes].set(
            A3hat[pi] @ q1
            - 2 * u_inf / c_ref * poles[pi] * ql[pi * num_modes : (pi + 1) * num_modes]
        )
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

def build_point_follower(x, follower_points, follower_interpolation,  num_nodes, C06ab):
    num_interpol_points = len(x)
    forces = jnp.zeros((num_interpol_points, 6, num_nodes))
    num_forces = len(follower_interpolation)
    for li in range(num_interpol_points):
        for fi in range(num_forces):
            fnode = follower_points[fi][0]
            dim = follower_points[fi][1]
            forces = forces.at[li, dim, fnode].set(
                follower_interpolation[fi][li]
            )  # Nx_6_Nn
    force_follower = functions.coordinate_transform(forces, C06ab)

    return force_follower

def shard_point_follower(x, follower_points, follower_interpolation,  num_nodes, C06ab):

    def _mapfollower(points, interpolation):

        force_follower = build_point_follower(x,
                                              points,
                                              interpolation,
                                              num_nodes,
                                              C06ab)
        return force_follower

    vmapfollower = jax.vmap(_mapfollower, in_axes=(0, 0))
    shardforce_follower = vmapfollower(follower_points, follower_interpolation)

    return shardforce_follower #Ns_Nx_6_Nn


def build_point_dead(x, dead_points, dead_interpolation, num_nodes):
    
    num_interpol_points = len(x)
    force_dead = jnp.zeros((num_interpol_points, 6, num_nodes))
    num_forces = len(dead_interpolation)
    for li in range(num_interpol_points):
        for fi in range(num_forces):
            fnode = dead_points[fi][0]
            dim = dead_points[fi][1]
            force_dead = force_dead.at[li, dim, fnode].set(dead_interpolation[fi][li])

    return force_dead

def shard_point_dead(x, dead_points, dead_interpolation,  num_nodes):

    def _mapdead(points, interpolation):

        force_dead = build_point_dead(x,
                                      points,
                                      interpolation,
                                      num_nodes
                                      )
        return force_dead

    vmapdead = jax.vmap(_mapdead, in_axes=(0, 0))
    shardforce_dead = vmapdead(dead_points, dead_interpolation)

    return shardforce_dead #Ns_Nx_6_Nn

def build_gravity(x, gravity, gravity_vect, Ma, Mfe_order):
    num_nodes = Mfe_order.shape[1] // 6
    num_nodes_out = Mfe_order.shape[0] // 6
    if x is not None and len(x) > 1:
        len_x = len(x)
    else:
        len_x = 2
    # force_gravity = jnp.zeros((2, 6, num_nodes))
    gravity = gravity * gravity_vect
    gravity_field = jnp.hstack([jnp.hstack([gravity, 0.0, 0.0, 0.0])] * num_nodes)
    _force_gravity = jnp.matmul(Mfe_order, Ma @ gravity_field)
    gravity_interpol = jnp.vstack([xi * _force_gravity for xi in jnp.linspace(0, 1, len_x)]).T
    force_gravity = functions.reshape_field(
        gravity_interpol, len_x, num_nodes_out
    )  # Becomes  (len_x, 6, Nn)
    # num_forces = len(dead_interpolation)
    # for li in range(num_interpol_points):
    #     for fi in range(num_forces):
    #         fnode = dead_points[fi][0]
    #         dim = dead_points[fi][1]
    #         force_dead = force_dead.at[li, dim, fnode].set(
    #             dead_interpolation[fi][li])
    return force_gravity

def shard_gravity(x, gravity, gravity_vect, Ma, Mfe_order):

    def _mapgravity(points_gravity, points_gravity_vect):

        force_gravity = build_gravity(x,
                                      points_gravity,
                                      points_gravity_vect,
                                      Ma,
                                      Mfe_order)
        return force_gravity

    vmapgravity = jax.vmap(_mapgravity, in_axes=(0, 0))
    shardforce_gravity = vmapgravity(gravity, gravity_vect)

    return shardforce_gravity #Ns_Nx_6_Nn


def shard_gust1(inputs: intrinsicmodal.DShard_gust1) -> jnp.ndarray:

    prod_list = []
    for k, v in inputs.__dict__.items():
        if v is not None:
            prod_list.append(v)
    prod = list(itertools.product(*prod_list))
    return jnp.array(prod)
            
def shard_steadyalpha(inputs: intrinsicmodal.DShard_steadyalpha,
                      default: intrinsicmodal.Daero) -> jnp.ndarray:

    prod_list = []
    default_dict = default.__dict__
    for k, v in inputs.__dict__.items():
        if v is not None:
            prod_list.append(v)
        elif k == "aeromatrix":
            prod_list.append([0])
        else:
            d_k = default_dict[k]
            prod_list.append([d_k])
    prod = list(itertools.product(*prod_list))
    return jnp.array(prod)
    
    

if __name__ == "__main__":

    # NumShards_NumForces_2(node x component)
    dead_points = jnp.array([[[9, 2], [18, 2]],
                             [[9, 1], [18, 1]]
                   ])
    # NumShards_NumForces_NumInterpolationPoints
    dead_interpolation = jnp.array([[[0, 1, 2],
                                     [0, 1, 2]
                                     ],
                                    [[0,1,2],
                                     [0,2,5]]
                                    ])
    x = jnp.array([0, 1, 2])
    num_nodes = 20
    #Ns_Nx_6_Nn
    shard_dead = shard_point_dead(x, dead_points, dead_interpolation,  num_nodes)
    dead0 = build_point_dead(x, dead_points[0], dead_interpolation[0], num_nodes)
    print((dead0 == shard_dead[0]).all())
    dead1 = build_point_dead(x, dead_points[1], dead_interpolation[1], num_nodes)
    print((dead1 == shard_dead[1]).all())
    
    g1 = intrinsicmodal.DShard_gust1(length=[10,20],intensity=[3,4,5], u_inf=[100,150], rho_inf=[0.2,0.3,0.6])
    prod = shard_gust1(g1)
    print(prod)

    steadyalpha = intrinsicmodal.DShard_steadyalpha(u_inf=[100,150], rho_inf=[0.2,0.3,0.6])
    aero = intrinsicmodal.Daero(u_inf=100, rho_inf=0.4)
    prod = shard_steadyalpha(steadyalpha, aero)
    print(prod)

    print("-----------")
    steadyalpha = intrinsicmodal.DShard_steadyalpha(u_inf=[100,150])
    aero = intrinsicmodal.Daero(u_inf=100, rho_inf=0.4)
    prod = shard_steadyalpha(steadyalpha, aero)
    print(prod)
    
