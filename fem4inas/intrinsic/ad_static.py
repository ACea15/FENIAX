import fem4inas.intrinsic.ad_common as ad_common

@partial(jax.jit, static_argnames=['config', 'f_obj'])
def main_10g11(t,
               #t_array,
               q0,
               Ka,
               Ma,
               config,
               f_obj,
               obj_args=None
               ):

    if obj_args is None:
        obj_args = dict()

    #t_loads = jnp.hstack([t_array, t])
    t_loads = jnp.hstack([config.system.t, t])
    tn = len(t_loads)
    config.system.build_states(config.fem.num_modes)
    q2_index = config.system.states['q2']
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = _compute_modes(X,
                                                    Ka,
                                                    Ma,
                                                    reduced_eigenvals,
                                                    reduced_eigenvecs,
                                                    config)

    # gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )
    config.system.xloads.build_point_follower(
                config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = config.system.xloads.force_follower
    dq_args = (gamma2, omega, phi1l, x_forceinterpol,
               y_forceinterpol)
    
    #q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q = _solve2(newton, dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q2 = q[:, q2_index]
    # X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
                                           phi2l, psi2l, X_xdelta, C0ab, config.fem)
    
    objective = f_obj(X2=X2, X3=X3, ra=ra, Cab=Cab, **obj_args)
    return objective
