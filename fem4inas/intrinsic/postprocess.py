import jax.numpy as jnp
import jax
from functools import partial
from fem4inas.intrinsic.functions import H0, H1

def compute_velocities(phi1l: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:

    X1 = jnp.tensordot(phi1l, q1, axes=(0,0))  # 6xNnxNt
    return X1

def compute_internalforces(phi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X2 = jnp.tensordot(phi2l, q2, axes=(0,0))  # 6xNnxNt
    return X2

def compute_strains(cphi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X3 = jnp.tensordot(cphi2l, q2, axes=(0,0))  # 6xNnxNt
    return X3

def velocity_Rab():
    ...

def strains_Rab():
    ...
    
def velocity_ra():
    ...

def strains_ra():
    ...

def integrate_X3(carry, x):

    Cab0_x = x[:,:3]
    kappa = x[:, 3]
    strain = x[:, 4]
    ds = x[0, 5]
    Cab_carry = carry[:, :3]
    Cab0_carry = carry[:, 3:6]
    ra0 = carry[:, 6]
    Ipsi = kappa * ds
    Itheta = jnp.linalg.norm(Ipsi)
    Cab = Cab_carry @ H0(Itheta,Ipsi)
    ra = ra0 + Cab_carry @ (H1(Itheta, Ipsi, ds) @ strain)
    y = jnp.hstack([Cab, ra.reshape((3,1))])
    carry = jnp.hstack([Cab, Cab0, ra.reshape((3,1))])
    return carry, y

def integrate_strains(X3t, sol, fem):

    integrate_X3
    ds = sol.modes.X_xdelta
    C0ab = sol.modes.C0ab  # 3x3xNn
    # TODO: make as fori loop
    Cab = jnp.zeros((3, 3, fem.num_nodes))
    ra = jnp.zeros((3, fem.num_nodes))
    for i, ci in fem.component_names:
        init = 4
        comp_father = fem.component_father
        comp_nodes = fem.component_nodes[ci]
        node_father = fem.component_nodes[comp_father][-1]
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(3, ds_i.shape[0]).T.reshape((comp_nodes, 3, 1))
        strains_i = X3t[:3, comp_nodes].T.reshape((comp_nodes, 3, 1))
        kappas_i = X3t[3:, comp_nodes].T.reshape((comp_nodes, 3, 1))
        C0ab_i = C0ab[:, :, comp_nodes].transponse((1,2,0))
        xs = jnp.concatenate([C0ab_i, strains_i, kappast_i,  ds_i], axis=2)
        last_carry, C_ra = jax.lax.scan(integrate_X3, init, xs)
        
