import jax.numpy as jnp

def build_point_follower(xloads, num_nodes):

    num_loadings = len(xloads.follower_interpolation[0])
    num_forces = len(xloads.follower_interpolation)
    force = jnp.zeros((num_loadings, 6, num_nodes))
    for li in range(num_loadings):
        for fi in range(num_forces):
            fnode= xloads.follower_points[fi][0]
            dim = xloads.follower_points[fi][1]
            force = force.at[li, dim, fnode].set(xloads.follower_interpolation[fi][li][1])

    return force
