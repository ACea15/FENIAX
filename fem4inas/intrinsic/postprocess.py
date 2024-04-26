import jax.numpy as jnp
import jax
from functools import partial
from fem4inas.intrinsic.functions import H0, H1, H0_t, H1_t

def compute_velocities(phi1l: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:

    X1 = jnp.tensordot(phi1l, q1, axes=(0, 1))  # 6xNnxNt
    return X1.transpose((2,0,1))

def compute_internalforces(phi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X2 = jnp.tensordot(phi2l, q2, axes=(0, 1))  # 6xNnxNt
    return X2.transpose((2,0,1))

def compute_strains(cphi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X3 = jnp.tensordot(cphi2l, q2, axes=(0, 1))  # 6xNnxNt
    return X3.transpose((2,0,1))

def compute_velocities_t(phi1l: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:

    X1 = jnp.tensordot(phi1l, q1, axes=(0, 0))  # 6xNnxNt
    return X1

def compute_internalforces_t(phi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X2 = jnp.tensordot(phi2l, q2, axes=(0, 0))  # 6xNnxNt
    return X2

def compute_strains_t(psi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:

    X3 = jnp.tensordot(psi2l, q2, axes=(0, 0))  # 6xNnxNt
    return X3


def velocity_Rab():
    ...

def strains_Rab():
    ...
    
def velocity_ra():
    ...

def strains_ra():
    ...

def integrate_node0(X1, dt, ra_n0, Rab_n0):

    v_average = (X1[: -1, :3] + X1[1:, :3]) / 2
    theta_average = (X1[:-1,3:6] + X1[1:,3:6]) / 2 * dt
    theta_norm = jnp.linalg.norm(theta_average, axis=1)
    theta = jnp.hstack([theta_average,
                        theta_norm.reshape((theta_norm.shape[0], 1)),
                        v_average])
    init = jnp.vstack([Rab_n0, ra_n0])
    def integrate(carry, x):

        thetai_average = x[:3]
        thetai_norm = x[3]
        vi_average = x[4:]
        Rab0 = carry[:3]
        ra0 = carry[3]
        Rab1 = Rab0 @ H0(thetai_norm, thetai_average)
        ra1 = ra0 + Rab0 @ H1(thetai_norm, thetai_average, dt) @ vi_average
        out = jnp.vstack([Rab1, ra1])
        return out, out

    last_carry, y = jax.lax.scan(integrate, init, theta)
    Rab = jnp.vstack([Rab_n0.reshape((1,3,3)), y[:, :3]])
    ra = jnp.vstack([ra_n0, y[:,3]])
    return Rab, ra

def integrate_X3Cab(carry, x):

    Cab0_x = x[:, :3]
    # strain = x[:, 3]
    kappa = x[:, 3]
    ds = x[0, 4]
    Cab_carry = carry[:, :3]
    Cab0_carry = carry[:, 3:6]
    #ra0 = carry[:, 6]
    Ipsi = kappa * ds
    Itheta = jnp.linalg.norm(Ipsi)
    Cab = Cab_carry @ Cab0_carry.T @ Cab0_x  @ H0(Itheta, Ipsi)
    # ra = ra0 + Cab_carry @ Cab0_carry.T @ Cab0_x @ (
    #     H1(Itheta, Ipsi, ds) @ (strain + jnp.array([1, 0, 0])))
    # y = jnp.hstack([Cab, ra.reshape((3, 1))])
    carry = jnp.hstack([Cab, Cab0_x])
    return carry, Cab

def integrate_strainsCab(Cab_0n, X3t,
                         X_xdelta, C0ab,
                         component_names,
                         num_nodes,
                         component_nodes,
                         component_father):

    ds = X_xdelta
    C0ab = C0ab  # 3x3xNn
    # TODO: make as fori loop
    Cab = jnp.zeros((3, 3, num_nodes))
    #ra = jnp.zeros((3, num_nodes))

    comp_nodes = jnp.array(component_nodes[component_names[0]])[1:]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.hstack([Cab_0n,
                       Cab0_init])
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i,
                            (3, ds_i.shape[0])).T.reshape((
                                numcomp_nodes, 3, 1))
    #strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    #import pdb; pdb.set_trace()
    C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
    xs = jnp.concatenate([C0ab_i, kappas_i,  ds_i], axis=2)
    last_carry, Cra = jax.lax.scan(integrate_X3Cab, init, xs)
    #ra = ra.at[:, 0].set(ra_0n)
    Cab = Cab.at[:, :, 0].set(Cab_0n)
    #ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
    Cab = Cab.at[:, :, comp_nodes].set(Cra.transpose((1, 2, 0)))

    for ci in component_names[1:]:

        comp_father = component_father[ci]
        comp_nodes = jnp.array(component_nodes[ci])
        numcomp_nodes = len(comp_nodes)
        if comp_father is None:
            node_father = 0
        else:
            node_father = component_nodes[comp_father][-1]
        Cab_init = Cab[:, :, node_father]
        Cab0_init = C0ab[:, :, node_father]
        #ra_init = ra[:, node_father]
        init = jnp.hstack([Cab_init, Cab0_init])
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i,
                                (3, ds_i.shape[0])).T.reshape((numcomp_nodes, 3, 1))
        #strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        xs = jnp.concatenate([C0ab_i, kappas_i,  ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3Cab, init, xs)
        # ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
        Cab = Cab.at[:, :, comp_nodes].set(Cra.transpose((1, 2, 0)))
        
    return Cab
    
def integrate_X3(carry, x):

    Cab0_x = x[:, :3]
    strain = x[:, 3]
    kappa = x[:, 4]
    ds = x[0, 5]
    Cab_carry = carry[:, :3]
    Cab0_carry = carry[:, 3:6]
    ra0 = carry[:, 6]
    Ipsi = kappa * ds
    Itheta = jnp.linalg.norm(Ipsi)
    Cab = Cab_carry @ Cab0_carry.T @ Cab0_x  @ H0(Itheta, Ipsi)
    ra = ra0 + Cab_carry @ Cab0_carry.T @ Cab0_x @ (
        H1(Itheta, Ipsi, ds) @ (strain + jnp.array([1, 0, 0])))
    y = jnp.hstack([Cab, ra.reshape((3, 1))])
    carry = jnp.hstack([Cab, Cab0_x, ra.reshape((3, 1))])
    return carry, y

def integrate_strains(ra_0n, Cab_0n, X3t, sol, fem):

    ds = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab  # 3x3xNn
    # TODO: make as fori loop
    Cab = jnp.zeros((3, 3, fem.num_nodes))
    ra = jnp.zeros((3, fem.num_nodes))

    comp_nodes = jnp.array(fem.component_nodes[fem.component_names[0]])[1:]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.hstack([Cab_0n,
                       Cab0_init,
                       ra_0n.reshape((3, 1))])
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i, (3, ds_i.shape[0])).T.reshape((numcomp_nodes, 3, 1))
    strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    #import pdb; pdb.set_trace()
    C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
    xs = jnp.concatenate([C0ab_i, strains_i, kappas_i,  ds_i], axis=2)
    last_carry, Cra = jax.lax.scan(integrate_X3, init, xs)
    ra = ra.at[:, 0].set(ra_0n)
    Cab = Cab.at[:, :, 0].set(Cab_0n)
    ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
    Cab = Cab.at[:, :, comp_nodes].set(Cra[:, :, :3].transpose((1, 2, 0)))

    for ci in fem.component_names[1:]:

        comp_father = fem.component_father[ci]
        comp_nodes = jnp.array(fem.component_nodes[ci])
        numcomp_nodes = len(comp_nodes)
        if comp_father is None:
            node_father = 0
        else:
            node_father = fem.component_nodes[comp_father][-1]
        Cab_init = Cab[:, :, node_father]
        Cab0_init = C0ab[:, :, node_father]
        ra_init = ra[:, node_father]
        init = jnp.hstack([Cab_init, Cab0_init,
                           ra_init.reshape((3,1))])
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i,
                                (3, ds_i.shape[0])).T.reshape((numcomp_nodes, 3, 1))
        strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        xs = jnp.concatenate([C0ab_i, strains_i, kappas_i,  ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3, init, xs)
        ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
        Cab = Cab.at[:, :, comp_nodes].set(Cra[:, :, :3].transpose((1, 2, 0)))
        
    return Cab, ra

def integrate_X3_t(carry, x):

    f_Cab = jax.vmap(lambda u1, u2, u3, u4: u1 @ u2.T @ u3 @ u4,
                     in_axes=2, out_axes=2)
    f_ra = jax.vmap(lambda ra0, Cab_carry, Cab0_carry, Cab0_x, H1, strain1:
                    ra0 + Cab_carry @ Cab0_carry.T @ Cab0_x @ H1 @ strain1,
                     in_axes=1, out_axes=1)
    Cab0_x = x[:, :3]  # 3x3xNt
    strain = x[:, 3]  # 3xNt
    kappa = x[:, 4]  # 3xNt
    ds = x[0, 5, 0]  # 1
    Cab_carry = carry[:3]  # 3x3xNt
    Cab0_carry = carry[3:6]  # 3x3xNt
    ra0 = carry[6]  # 3xNt
    Ipsi = kappa * ds  # 3xNt
    Itheta = jnp.linalg.norm(Ipsi, axis=0) # Nt
    Cab = f_Cab(Cab_carry, Cab0_carry, Cab0_x, H0_t(Itheta, Ipsi))  # 3x3xNt
    ra = f_ra(ra0, Cab_carry.transpose((0,2,1)), Cab0_carry.transpose((0,2,1)), Cab0_x.transpose((0,2,1)),
              H1_t(Itheta, Ipsi, ds).transpose((0,2,1)),
              (strain + jnp.array([1, 0, 0]).reshape(3, 1)))  # 3xNt
    #y = jnp.hstack([Cab, ra.reshape((3, 1))])
    ra_reshaped = ra.reshape((1,) + ra.shape)
    y = jnp.concatenate([Cab, ra_reshaped], axis=0) #4x3xNt
    carry = jnp.concatenate([Cab, Cab0_x, ra_reshaped], axis=0)
    return carry, y

def integrate_strains_t(ra_0n, Cab_0n, X3, sol, fem):

    ds = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab  # 3x3xNn
    tn, _, num_nodes = X3.shape
    # TODO: make as fori loop
    Cab = jnp.zeros((tn, 3, 3, num_nodes))
    ra = jnp.zeros((tn, 3, num_nodes))
    comp_nodes = jnp.array(fem.component_nodes[fem.component_names[0]])[1:]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.concatenate([Cab_0n.transpose((1,2,0)),
                            jnp.broadcast_to(Cab0_init, (tn, 3, 3)).transpose(1, 2, 0),
                            ra_0n.reshape((tn, 3, 1)).T], axis=0)
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i, (tn, 3, numcomp_nodes)).T.reshape((numcomp_nodes, 3, 1, tn))
    strains_i = X3[:, :3, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
    kappas_i = X3[:, 3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
    C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
    C0ab_i = jnp.broadcast_to(C0ab_i, (tn, numcomp_nodes, 3, 3)).transpose(1, 2, 3, 0)
    xs = jnp.concatenate([C0ab_i, strains_i, kappas_i, ds_i], axis=2)
    last_carry, Cra = jax.lax.scan(integrate_X3_t, init, xs)  # Ncnx4x3xNt
    ra = ra.at[:, :, 0].set(ra_0n)
    Cab = Cab.at[:, :, :, 0].set(Cab_0n)
    ra = ra.at[:, :, comp_nodes].set(Cra[:, 3].transpose((2, 1, 0)))
    Cab = Cab.at[:, :, :, comp_nodes].set(Cra[:, :3].transpose((3, 1, 2, 0)))

    for ci in fem.component_names[1:]:

        comp_father = fem.component_father[ci]
        comp_nodes = jnp.array(fem.component_nodes[ci])
        numcomp_nodes = len(comp_nodes)
        if comp_father is None:
            node_father = 0
        else:
            node_father = fem.component_nodes[comp_father][-1]
        Cab_init = Cab[:, :, :, node_father]
        Cab0_init = C0ab[:, :, node_father]
        ra_init = ra[:, :, node_father]
        init = jnp.concatenate([Cab_init.transpose((1,2,0)),
                                jnp.broadcast_to(Cab0_init, (tn, 3, 3)).transpose(1, 2, 0),
                                ra_init.reshape((tn, 3, 1)).T], axis=0)
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i, (tn, 3, numcomp_nodes)).T.reshape((numcomp_nodes, 3, 1, tn))
        strains_i = X3[:, :3, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
        kappas_i = X3[:, 3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        C0ab_i = jnp.broadcast_to(C0ab_i, (tn, numcomp_nodes, 3, 3)).transpose(1, 2, 3, 0)
        xs = jnp.concatenate([C0ab_i, strains_i, kappas_i, ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3_t, init, xs)  # Ncnx4x3xNt
        ra = ra.at[:, :, comp_nodes].set(Cra[:, 3].transpose((2, 1, 0)))
        Cab = Cab.at[:, :, :, comp_nodes].set(Cra[:, :3].transpose((3, 1, 2, 0)))

    return Cab, ra
