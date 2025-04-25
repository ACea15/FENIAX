import jax.numpy as jnp
import jax
from functools import partial
from feniax.intrinsic.functions import H0, H1, H0_t, H1_t, H0l, H1l, H0l_t, H1l_t
import feniax.intrinsic.quaternions as quaternions


def compute_velocities(phi1l: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:
    X1 = jnp.tensordot(phi1l, q1, axes=(0, 1))  # 6xNnxNt
    return X1.transpose((2, 0, 1))


def compute_internalforces(phi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    X2 = jnp.tensordot(phi2l, q2, axes=(0, 1))  # 6xNnxNt
    return X2.transpose((2, 0, 1))


def compute_strains(cphi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    X3 = jnp.tensordot(cphi2l, q2, axes=(0, 1))  # 6xNnxNt
    return X3.transpose((2, 0, 1))


def compute_velocities_t(phi1l: jnp.ndarray, q1: jnp.ndarray) -> jnp.ndarray:
    X1 = jnp.tensordot(phi1l, q1, axes=(0, 0))  # 6xNnxNt
    return X1


def compute_internalforces_t(phi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    X2 = jnp.tensordot(phi2l, q2, axes=(0, 0))  # 6xNnxNt
    return X2


def compute_strains_t(psi2l: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    X3 = jnp.tensordot(psi2l, q2, axes=(0, 0))  # 6xNnxNt
    return X3


def velocity_Rab(): ...


def strains_Rab(): ...


def velocity_ra(): ...


def strains_ra(): ...

@jax.jit
def integrate_node0(X1, dt, ra_n0, Rab_n0):
    """Integrate velocities in the first node of the model to get the the Rigid Body dynamics
    Parameters:
        data (list): Input data.
        verbose (bool): Show verbose output.

    This function uses the averge of velocities between time-steps,
    assume the average constant and integrate using the the
    exponential map similar to the integration of strains (more
    efficient algorithms can be employed)

    Parameters
    ----------
    X1 : jnp.ndarray
        Velocity field
    dt : jnp.ndarray
        Delta time
    ra_n0 : jnp.ndarray
        Initial position of integrated node (0)
    Rab_n0 : jnp.ndarray
        Initial rotation matrix (set to the indentity)

    Examples
    --------
    FIXME: Add docs.


    """
    
    v_average = (X1[:-1, :3] + X1[1:, :3]) / 2
    theta_average = (X1[:-1, 3:6] + X1[1:, 3:6]) / 2 * dt
    theta_norm = jnp.linalg.norm(theta_average, axis=1)
    theta = jnp.hstack(
        [theta_average, theta_norm.reshape((theta_norm.shape[0], 1)), v_average]
    )
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
    Rab = jnp.vstack([Rab_n0.reshape((1, 3, 3)), y[:, :3]])
    ra = jnp.vstack([ra_n0, y[:, 3]])
    return Rab, ra

@jax.jit
def integrate_node0q(X1, dt, ra_n0, Rab_n0, quaternion_n0):
    """Integrate velocities in the first node of the model to get the the Rigid Body dynamics
    Parameters:
        data (list): Input data.
        verbose (bool): Show verbose output.

    Similar to the previous function but now using the quaternion variable to get
    the rotation matrix

    Parameters
    ----------
    X1 : jnp.ndarray
        Velocity field
    dt : jnp.ndarray
        Delta time
    ra_n0 : jnp.ndarray
        Initial position of integrated node (0)
    Rab_n0 : jnp.ndarray
        Initial rotation matrix (set to the indentity)
    quaternion_n0 : jnp.ndarray
    Examples
    --------
    FIXME: Add docs.


    """
        
    v_average = (X1[:-1, :3] + X1[1:, :3]) / 2
    theta_average = (X1[:-1, 3:6] + X1[1:, 3:6]) / 2 * dt
    theta_norm = jnp.linalg.norm(theta_average, axis=1)
    theta = jnp.hstack(
        [theta_average, theta_norm.reshape((theta_norm.shape[0], 1)), v_average]
    )
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
    Rab = jnp.vstack([Rab_n0.reshape((1, 3, 3)), y[:, :3]])
    ra = jnp.vstack([ra_n0, y[:, 3]])
    return Rab, ra


def integrate_X3Cab(carry, x):
    """Integrate from curvatures the rotation matrix. To be used within integrate_strainsCab
    Parameters:
        data (list): Input data.
        verbose (bool): Show verbose output.

    Parameters
    ----------
    carry : jnp.ndarray
    x : jnp.ndarray

    Examples
    --------
    FIXME: Add docs.


    """
    
    Cab0_x = x[:, :3]
    # strain = x[:, 3]
    kappa = x[:, 3]
    ds = x[0, 4]
    Cab_carry = carry[:, :3]
    Cab0_carry = carry[:, 3:6]
    # ra0 = carry[:, 6]
    Ipsi = kappa * ds
    Itheta = jnp.linalg.norm(Ipsi)
    Cab = Cab_carry @ Cab0_carry.T @ Cab0_x @ H0(Itheta, Ipsi)
    # ra = ra0 + Cab_carry @ Cab0_carry.T @ Cab0_x @ (
    #     H1(Itheta, Ipsi, ds) @ (strain + jnp.array([1, 0, 0])))
    # y = jnp.hstack([Cab, ra.reshape((3, 1))])
    carry = jnp.hstack([Cab, Cab0_x])
    return carry, Cab


def integrate_X3lCab(carry, x):
    """Integrate from curvatures the rotation matrix using linear assumptions
    Parameters:
        data (list): Input data.
        verbose (bool): Show verbose output.

    Parameters
    ----------
    carry : jnp.ndarray
    x : jnp.ndarray

    Examples
    --------
    FIXME: Add docs.


    """    
    Cab0_x = x[:, :3]
    # strain = x[:, 3]
    kappa = x[:, 3]
    ds = x[0, 4]
    Cab_carry = carry[:, :3]
    Cab0_carry = carry[:, 3:6]
    # ra0 = carry[:, 6]
    Ipsi = kappa * ds
    Itheta = jnp.linalg.norm(Ipsi)
    Cab = Cab_carry @ Cab0_carry.T @ Cab0_x @ H0l(Itheta, Ipsi)
    # ra = ra0 + Cab_carry @ Cab0_carry.T @ Cab0_x @ (
    #     H1(Itheta, Ipsi, ds) @ (strain + jnp.array([1, 0, 0])))
    # y = jnp.hstack([Cab, ra.reshape((3, 1))])
    carry = jnp.hstack([Cab, Cab0_x])
    return carry, Cab

def integrate_strainslCab(
    Cab_0n,
    X3t,
    X_xdelta,
    C0ab,
    component_names,
    num_nodes,
    component_nodes,
    component_father,
):
    ds = X_xdelta
    C0ab = C0ab  # 3x3xNn
    # TODO: make as fori loop
    Cab = jnp.zeros((3, 3, num_nodes))
    # ra = jnp.zeros((3, num_nodes))

    comp_nodes = jnp.array(component_nodes[component_names[0]])[1:]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.hstack([Cab_0n, Cab0_init])
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i, (3, ds_i.shape[0])).T.reshape((numcomp_nodes, 3, 1))
    # strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    # import pdb; pdb.set_trace()
    C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
    xs = jnp.concatenate([C0ab_i, kappas_i, ds_i], axis=2)
    last_carry, Cra = jax.lax.scan(integrate_X3lCab, init, xs)
    # ra = ra.at[:, 0].set(ra_0n)
    Cab = Cab.at[:, :, 0].set(Cab_0n)
    # ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
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
        # ra_init = ra[:, node_father]
        init = jnp.hstack([Cab_init, Cab0_init])
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i, (3, ds_i.shape[0])).T.reshape(
            (numcomp_nodes, 3, 1)
        )
        # strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        xs = jnp.concatenate([C0ab_i, kappas_i, ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3lCab, init, xs)
        # ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
        Cab = Cab.at[:, :, comp_nodes].set(Cra.transpose((1, 2, 0)))

    return Cab


def integrate_X3_t(carry, x):
    """
    Function for jax.scan to integrate strains into postions and rotations
    """
    
    f_Cab = jax.vmap(lambda u1, u2, u3, u4: u1 @ u2.T @ u3 @ u4, in_axes=2, out_axes=2)
    f_ra = jax.vmap(
        lambda ra0, Cab_carry, Cab0_carry, Cab0_x, H1, strain1: ra0
        + Cab_carry @ Cab0_carry.T @ Cab0_x @ H1 @ strain1,
        in_axes=1,
        out_axes=1,
    )
    Cab0_x = x[:, :3]  # 3x3xNt
    strain = x[:, 3]  # 3xNt
    kappa = x[:, 4]  # 3xNt
    ds = x[0, 5, 0]  # 1
    Cab_carry = carry[:3]  # 3x3xNt
    Cab0_carry = carry[3:6]  # 3x3xNt
    ra0 = carry[6]  # 3xNt
    Ipsi = kappa * ds  # 3xNt
    Itheta = jnp.linalg.norm(Ipsi, axis=0)  # Nt
    Cab = f_Cab(Cab_carry, Cab0_carry, Cab0_x, H0_t(Itheta, Ipsi))  # 3x3xNt
    ra = f_ra(
        ra0,
        Cab_carry.transpose((0, 2, 1)),
        Cab0_carry.transpose((0, 2, 1)),
        Cab0_x.transpose((0, 2, 1)),
        H1_t(Itheta, Ipsi, ds).transpose((0, 2, 1)),
        (strain + jnp.array([1, 0, 0]).reshape(3, 1)),
    )  # 3xNt
    # y = jnp.hstack([Cab, ra.reshape((3, 1))])
    ra_reshaped = ra.reshape((1,) + ra.shape)
    y = jnp.concatenate([Cab, ra_reshaped], axis=0)  # 4x3xNt
    carry = jnp.concatenate([Cab, Cab0_x, ra_reshaped], axis=0)
    return carry, y

def integrate_X3l_t(carry, x):
    """
    Function for jax.scan to integrate strains into 
    """
    
    f_Cab = jax.vmap(lambda u1, u2, u3, u4: u1 @ u2.T @ u3 @ u4, in_axes=2, out_axes=2)
    f_ra = jax.vmap(
        lambda ra0, Cab_carry, Cab0_carry, Cab0_x, H1, strain1: ra0
        + Cab_carry @ Cab0_carry.T @ Cab0_x @ H1 @ strain1,
        in_axes=1,
        out_axes=1,
    )
    Cab0_x = x[:, :3]  # 3x3xNt
    strain = x[:, 3]  # 3xNt
    kappa = x[:, 4]  # 3xNt
    ds = x[0, 5, 0]  # 1
    Cab_carry = carry[:3]  # 3x3xNt
    Cab0_carry = carry[3:6]  # 3x3xNt
    ra0 = carry[6]  # 3xNt
    Ipsi = kappa * ds  # 3xNt
    Itheta = jnp.linalg.norm(Ipsi, axis=0)  # Nt
    Cab = f_Cab(Cab_carry, Cab0_carry, Cab0_x, H0l_t(Itheta, Ipsi))  # 3x3xNt
    ra = f_ra(
        ra0,
        Cab_carry.transpose((0, 2, 1)),
        Cab0_carry.transpose((0, 2, 1)),
        Cab0_x.transpose((0, 2, 1)),
        H1l_t(Itheta, Ipsi, ds).transpose((0, 2, 1)),
        (strain + jnp.array([1, 0, 0]).reshape(3, 1)),
    )  # 3xNt
    # y = jnp.hstack([Cab, ra.reshape((3, 1))])
    ra_reshaped = ra.reshape((1,) + ra.shape)
    y = jnp.concatenate([Cab, ra_reshaped], axis=0)  # 4x3xNt
    carry = jnp.concatenate([Cab, Cab0_x, ra_reshaped], axis=0)
    return carry, y


@partial(jax.jit, static_argnames=["config"])
def integrate_strains_t(ra_0n, Cab_0n, X3, X_xdelta, C0ab, config):
    """Integration of strains and curvatures to get the position and rotational fields
    Parameters:
        data (list): Input data.
        verbose (bool): Show verbose output.

    Parameters
    ----------
    ra_0n : Initial position of node 0
    Cab_0n : jnp.ndarray
        Initial rotation matrix (identity)
    X3 : jnp.ndarray
        Strain field
    X_xdelta : jnp.ndarray
        Load paths increment lengths (initial pre-stressed
        configutation)
    C0ab : jnp.ndarray
        Local to global initial reference system (x-component in the
        direction of the load-paths)
    config : Config
        Configuration object

    Examples
    --------
    FIXME: Add docs.


    """


    
    # ds = sol.data.modes.X_xdelta
    # C0ab = sol.data.modes.C0ab  # 3x3xNn
    ds = X_xdelta
    tn, _, num_nodes = X3.shape
    # TODO: make as fori loop (or not cause it is not back differentiated)
    Cab = jnp.zeros((tn, 3, 3, num_nodes))
    ra = jnp.zeros((tn, 3, num_nodes))
    comp_nodes = jnp.array(config.fem.component_nodes[config.fem.component_names[0]])[
        1:
    ]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.concatenate(
        [
            Cab_0n.transpose((1, 2, 0)),
            jnp.broadcast_to(Cab0_init, (tn, 3, 3)).transpose(1, 2, 0),
            ra_0n.reshape((tn, 3, 1)).T,
        ],
        axis=0,
    )
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i, (tn, 3, numcomp_nodes)).T.reshape(
        (numcomp_nodes, 3, 1, tn)
    )
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

    for ci in config.fem.component_names[1:]:
        comp_father = config.fem.component_father[ci]
        comp_nodes = jnp.array(config.fem.component_nodes[ci])
        numcomp_nodes = len(comp_nodes)
        if comp_father is None:
            node_father = 0
        else:
            node_father = config.fem.component_nodes[comp_father][-1]
        Cab_init = Cab[:, :, :, node_father]
        Cab0_init = C0ab[:, :, node_father]
        ra_init = ra[:, :, node_father]
        init = jnp.concatenate(
            [
                Cab_init.transpose((1, 2, 0)),
                jnp.broadcast_to(Cab0_init, (tn, 3, 3)).transpose(1, 2, 0),
                ra_init.reshape((tn, 3, 1)).T,
            ],
            axis=0,
        )
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i, (tn, 3, numcomp_nodes)).T.reshape(
            (numcomp_nodes, 3, 1, tn)
        )
        strains_i = X3[:, :3, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
        kappas_i = X3[:, 3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        C0ab_i = jnp.broadcast_to(C0ab_i, (tn, numcomp_nodes, 3, 3)).transpose(
            1, 2, 3, 0
        )
        xs = jnp.concatenate([C0ab_i, strains_i, kappas_i, ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3_t, init, xs)  # Ncnx4x3xNt
        ra = ra.at[:, :, comp_nodes].set(Cra[:, 3].transpose((2, 1, 0)))
        Cab = Cab.at[:, :, :, comp_nodes].set(Cra[:, :3].transpose((3, 1, 2, 0)))

    return Cab, ra


@partial(jax.jit, static_argnames=["config"])
def integrate_strainsl_t(ra_0n, Cab_0n, X3, X_xdelta, C0ab, config):
    """Linear Integration of strains and curvatures to get the position and rotational fields
    Parameters:
        data (list): Input data.
        verbose (bool): Show verbose output.

    Parameters
    ----------
    ra_0n : Initial position of node 0
    Cab_0n : jnp.ndarray
        Initial rotation matrix (identity)
    X3 : jnp.ndarray
        Strain field
    X_xdelta : jnp.ndarray
        Load paths increment lengths (initial pre-stressed
        configutation)
    C0ab : jnp.ndarray
        Local to global initial reference system (x-component in the
        direction of the load-paths)
    config : Config
        Configuration object

    Examples
    --------
    FIXME: Add docs.


    """
    
    # ds = sol.data.modes.X_xdelta
    # C0ab = sol.data.modes.C0ab  # 3x3xNn
    ds = X_xdelta
    tn, _, num_nodes = X3.shape
    # TODO: make as fori loop (or not cause it is not back differentiated)
    Cab = jnp.zeros((tn, 3, 3, num_nodes))
    ra = jnp.zeros((tn, 3, num_nodes))
    comp_nodes = jnp.array(config.fem.component_nodes[config.fem.component_names[0]])[
        1:
    ]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.concatenate(
        [
            Cab_0n.transpose((1, 2, 0)),
            jnp.broadcast_to(Cab0_init, (tn, 3, 3)).transpose(1, 2, 0),
            ra_0n.reshape((tn, 3, 1)).T,
        ],
        axis=0,
    )
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i, (tn, 3, numcomp_nodes)).T.reshape(
        (numcomp_nodes, 3, 1, tn)
    )
    strains_i = X3[:, :3, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
    kappas_i = X3[:, 3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
    C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
    C0ab_i = jnp.broadcast_to(C0ab_i, (tn, numcomp_nodes, 3, 3)).transpose(1, 2, 3, 0)
    xs = jnp.concatenate([C0ab_i, strains_i, kappas_i, ds_i], axis=2)
    last_carry, Cra = jax.lax.scan(integrate_X3l_t, init, xs)  # Ncnx4x3xNt
    ra = ra.at[:, :, 0].set(ra_0n)
    Cab = Cab.at[:, :, :, 0].set(Cab_0n)
    ra = ra.at[:, :, comp_nodes].set(Cra[:, 3].transpose((2, 1, 0)))
    Cab = Cab.at[:, :, :, comp_nodes].set(Cra[:, :3].transpose((3, 1, 2, 0)))

    for ci in config.fem.component_names[1:]:
        comp_father = config.fem.component_father[ci]
        comp_nodes = jnp.array(config.fem.component_nodes[ci])
        numcomp_nodes = len(comp_nodes)
        if comp_father is None:
            node_father = 0
        else:
            node_father = config.fem.component_nodes[comp_father][-1]
        Cab_init = Cab[:, :, :, node_father]
        Cab0_init = C0ab[:, :, node_father]
        ra_init = ra[:, :, node_father]
        init = jnp.concatenate(
            [
                Cab_init.transpose((1, 2, 0)),
                jnp.broadcast_to(Cab0_init, (tn, 3, 3)).transpose(1, 2, 0),
                ra_init.reshape((tn, 3, 1)).T,
            ],
            axis=0,
        )
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i, (tn, 3, numcomp_nodes)).T.reshape(
            (numcomp_nodes, 3, 1, tn)
        )
        strains_i = X3[:, :3, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
        kappas_i = X3[:, 3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1, tn))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        C0ab_i = jnp.broadcast_to(C0ab_i, (tn, numcomp_nodes, 3, 3)).transpose(
            1, 2, 3, 0
        )
        xs = jnp.concatenate([C0ab_i, strains_i, kappas_i, ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3l_t, init, xs)  # Ncnx4x3xNt
        ra = ra.at[:, :, comp_nodes].set(Cra[:, 3].transpose((2, 1, 0)))
        Cab = Cab.at[:, :, :, comp_nodes].set(Cra[:, :3].transpose((3, 1, 2, 0)))

    return Cab, ra

def integrate_strainsCab(
    Cab_0n,
    X3t,
    X_xdelta,
    C0ab,
    component_names,
    num_nodes,
    component_nodes,
    component_father,
):
    """Integrates curvatures to get rotational matrix field at one step

    Parameters
    ----------
    Cab_0n : jnp.ndarray
        Initial rotation matrix (identity)    
    X3t : jnp.ndarray
        Strain field at time t
    X_xdelta : jnp.ndarray
    ra_0n : Initial position of node 0
    X_xdelta : jnp.ndarray
        Load paths increment lengths (initial pre-stressed
        configutation)
    C0ab : jnp.ndarray
        Local to global initial reference system (x-component in the
        direction of the load-paths)    
    component_names : list[str]
        List of component names
    num_nodes : int
    component_nodes : dict
        Map from component name to node ids
    component_father : dict
        Component the current one is attached to (flow given as first node being a source)

    Examples
    --------
    FIXME: Add docs.


    """
    
    ds = X_xdelta
    C0ab = C0ab  # 3x3xNn
    # TODO: make as fori loop
    Cab = jnp.zeros((3, 3, num_nodes))
    # ra = jnp.zeros((3, num_nodes))

    comp_nodes = jnp.array(component_nodes[component_names[0]])[1:]
    numcomp_nodes = len(comp_nodes)
    Cab0_init = C0ab[:, :, 0]
    init = jnp.hstack([Cab_0n, Cab0_init])
    ds_i = ds[comp_nodes]
    ds_i = jnp.broadcast_to(ds_i, (3, ds_i.shape[0])).T.reshape((numcomp_nodes, 3, 1))
    # strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
    # import pdb; pdb.set_trace()
    C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
    xs = jnp.concatenate([C0ab_i, kappas_i, ds_i], axis=2)
    last_carry, Cra = jax.lax.scan(integrate_X3Cab, init, xs)
    # ra = ra.at[:, 0].set(ra_0n)
    Cab = Cab.at[:, :, 0].set(Cab_0n)
    # ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
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
        # ra_init = ra[:, node_father]
        init = jnp.hstack([Cab_init, Cab0_init])
        ds_i = ds[comp_nodes]
        ds_i = jnp.broadcast_to(ds_i, (3, ds_i.shape[0])).T.reshape(
            (numcomp_nodes, 3, 1)
        )
        # strains_i = X3t[:3, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        kappas_i = X3t[3:, comp_nodes].T.reshape((numcomp_nodes, 3, 1))
        C0ab_i = C0ab[:, :, comp_nodes].transpose((2, 0, 1))
        xs = jnp.concatenate([C0ab_i, kappas_i, ds_i], axis=2)
        last_carry, Cra = jax.lax.scan(integrate_X3Cab, init, xs)
        # ra = ra.at[:, comp_nodes].set(Cra[:, :, 3].T)
        Cab = Cab.at[:, :, comp_nodes].set(Cra.transpose((1, 2, 0)))

    return Cab
