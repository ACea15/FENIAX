import jax.numpy as jnp
import jax
from functools import partial
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.ad_common as adcommon
import fem4inas.intrinsic.gust as igust
import fem4inas.intrinsic.couplings as couplings
import fem4inas.intrinsic.dq_dynamic as dq_dynamic
import fem4inas.systems.intrinsic_system as isys

@partial(jax.jit, static_argnames=['config', 'f_obj', 'obj_args'])
def main_40g11_1(inputs, # alpha
                 q0,
                 config,
                 f_obj,
                 obj_args,
                 *args,
                 **kwargs                 
                 ):

    alpha = inputs["alpha"]
    Ka = config.fem.Ka
    Ma = config.fem.Ma    

    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = config.system.states['q2']
    q1_index = config.system.states['q1']
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = adcommon._compute_modes(X,
                                                             Ka,
                                                             Ma,
                                                             reduced_eigenvals,
                                                             reduced_eigenvecs,
                                                             config)

    gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )
    config.system.xloads.build_point_follower(
                config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = alpha * config.system.xloads.force_follower
    states = config.system.states
    eta0 = jnp.zeros(config.fem.num_modes)
    dq_args = (eta0, gamma1, gamma2, omega, phi1,
               x_forceinterpol,
               y_forceinterpol, states)

    states_puller, eqsolver = sollibs.factory(config.system.solver_library,
                                              config.system.solver_function)

    sol = eqsolver(dq_dynamic.dq_20g11,
                   dq_args,
                   config.system.solver_settings,
                   q0=q0,
                   t0=config.system.t0,
                   t1=config.system.t1,
                   tn=config.system.tn,
                   dt=config.system.dt,
                   t=config.system.t)
    q = states_puller(sol)
    
    #q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    # X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    tn = len(q)
    X1, X2, X3, ra, Cab = isys.recover_fields(q1,q2,
                                              tn, X,
                                              phi1l, phi2l,
                                              psi2l, X_xdelta,
                                              C0ab, config)
    # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config)
    return adcommon._objective_output(q=q, X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab,
                                      f_obj=f_obj, nodes=jnp.array(obj_args.nodes),
                                      components=jnp.array(obj_args.components),
                                      t=jnp.array(obj_args.t),axis=obj_args.axis)

@partial(jax.jit, static_argnames=['config', 'f_obj', 'obj_args'])
def main_40g11_3(inputs, # alpha
                 q0,
                 config,
                 f_obj,
                 obj_args,
                 *args,
                 **kwargs                 
                 ):

    #alpha = inputs["alpha"]
    Ka = inputs["Ka"]
    Ma = inputs["Ka"]
    eigenvals = inputs["eigenvals"] #jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs["eigenvecs"] #jnp.load(config.fem.folder / config.fem.eig_names[1])

    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = config.system.states['q2']
    q1_index = config.system.states['q1']
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = adcommon._compute_modes(X,
                                                             Ka,
                                                             Ma,
                                                             reduced_eigenvals,
                                                             reduced_eigenvecs,
                                                             config)

    gamma1 = couplings.f_gamma1(phi1, psi1)
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
    states = config.system.states
    eta0 = jnp.zeros(config.fem.num_modes)
    dq_args = (eta0, gamma1, gamma2, omega, phi1,
               x_forceinterpol,
               y_forceinterpol, states)

    states_puller, eqsolver = sollibs.factory(config.system.solver_library,
                                              config.system.solver_function)

    sol = eqsolver(dq_dynamic.dq_20g11,
                   dq_args,
                   config.system.solver_settings,
                   q0=q0,
                   t0=config.system.t0,
                   t1=config.system.t1,
                   tn=config.system.tn,
                   dt=config.system.dt,
                   t=config.system.t)
    q = states_puller(sol)
    
    #q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    # X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    tn = len(q)
    X1, X2, X3, ra, Cab = isys.recover_fields(q1,q2,
                                              tn, X,
                                              phi1l, phi2l,
                                              psi2l, X_xdelta,
                                              C0ab, config)
    # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config)
    return adcommon._objective_output(q=q, X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab,
                                      f_obj=f_obj, nodes=jnp.array(obj_args.nodes),
                                      components=jnp.array(obj_args.components),
                                      t=jnp.array(obj_args.t),axis=obj_args.axis)


@partial(jax.jit, static_argnames=['config', 'f_obj', 'obj_args'])
def main_40g21_2(inputs, #gust_intensity, gust_length, u_inf, rho_inf,
                 q0,
                 config,
                 f_obj,
                 obj_args,
                 *args,
                 **kwargs
                 ):
    
    gust_intensity = inputs["intensity"]
    gust_length = inputs["length"]
    u_inf = inputs["u_inf"]
    rho_inf = inputs["rho_inf"]
    # u_inf, rho_inf, gust_length, gust_intensity = input1
    # if obj_args is None:
    #     obj_args = dict()

    Ka = config.fem.Ka
    Ma = config.fem.Ma    
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = jnp.array(range(config.fem.num_modes, 2 * config.fem.num_modes)) #config.system.states['q2']
    q1_index = jnp.array(range(config.fem.num_modes)) #config.system.states['q1']
    #eigenvals, eigenvecs = scipy.linalg.eigh(Ka, Ma)
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0]).T
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1]).T
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = adcommon._compute_modes(X,
                                                             Ka,
                                                             Ma,
                                                             reduced_eigenvals,
                                                             reduced_eigenvecs,
                                                             config)
    #################
    gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )

    #################
    A0 = config.system.aero.A[0]
    A1 = config.system.aero.A[1]
    A2 = config.system.aero.A[2]
    A3 = config.system.aero.A[3:]
    D0 = config.system.aero.D[0]
    D1 = config.system.aero.D[1]
    D2 = config.system.aero.D[2]
    D3 = config.system.aero.D[3:]    
    #u_inf = config.system.aero.u_inf
    #rho_inf = config.system.aero.rho_inf
    q_inf = 0.5 * rho_inf * u_inf ** 2 # config.system.aero.q_inf
    c_ref = config.system.aero.c_ref
    A0hat = q_inf * A0
    A1hat = c_ref * rho_inf * u_inf / 4 * A1
    A2hat = c_ref**2 * rho_inf / 8 * A2
    A3hat = q_inf * A3
    A2hatinv = jnp.linalg.inv(jnp.eye(len(A2hat)) -
                              A2hat)
    D0hat = q_inf * D0
    D1hat = c_ref * rho_inf * u_inf / 4 * D1
    D2hat = c_ref**2 * rho_inf / 8 * D2
    D3hat = q_inf * D3    
    # gust_intensity = config.system.aero.gust.intensity
    # gust_length = config.system.aero.gust.length
    gust_shift = config.system.aero.gust.shift
    gust_step = config.system.aero.gust.step
    dihedral = config.system.aero.gust.panels_dihedral
    time = config.system.t
    collocation_points = config.system.aero.gust.collocation_points
    gust_totaltime = config.system.aero.gust.totaltime
    xgust = config.system.aero.gust.x
    time = config.system.aero.gust.time
    ntime = config.system.aero.gust.ntime
    # gust_totaltime, xgust, time, ntime, npanels = igust._get_gustRogerMc(
    #     gust_intensity,
    #     dihedral,
    #     gust_shift,
    #     gust_step,
    #     time,
    #     collocation_points,
    #     gust_length,
    #     u_inf)
    # gust_totaltime, xgust, time, ntime = igust._get_gustRogerMc(
    #     config.system.aero.gust.intensity,
    #     config.system.aero.gust.panels_dihedral,
    #     config.system.aero.gust.shift,
    #     config.system.aero.gust.step,
    #     time,
    #     config.system.aero.gust.length,
    #     config.system.aero.u_inf,
    #     jnp.min(collocation_points[:,0]),
    #     jnp.max(collocation_points[:,0])
    # )
    npanels = len(collocation_points)
    fshape_span = igust._get_spanshape(config.system.aero.gust.shape)
    gust, gust_dot, gust_ddot = igust._downwashRogerMc(u_inf,
                                                       gust_length,
                                                       gust_intensity,
                                                       gust_shift,
                                                       collocation_points,
                                                       dihedral, #normals,
                                                       time,
                                                       gust_totaltime,
                                                       fshape_span
                                                       )
    Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot = igust._getGAFs(D0hat,  # NbxNm
             D1hat,         
             D2hat,         
             D3hat,
             gust,
             gust_dot,
             gust_ddot
             )
    poles = config.system.aero.poles
    num_poles = config.system.aero.num_poles
    num_modes = config.fem.num_modes
    states = config.system.states
    eta0 = jnp.zeros(num_modes)
    dq_args = (eta0, gamma1, gamma2, omega, states,
               num_modes, num_poles,
               A0hat, A1hat, A2hatinv, A3hat,
               u_inf, c_ref, poles,
               time, Q_wsum, Ql_wdot)

    #################
    # import pdb;pdb.set_trace()
    states_puller, eqsolver = sollibs.factory(config.system.solver_library,
                                              config.system.solver_function)

    sol = eqsolver(dq_dynamic.dq_20g21,
                   dq_args,
                   config.system.solver_settings,
                   q0=q0,
                   t0=config.system.t0,
                   t1=config.system.t1,
                   tn=config.system.tn,
                   dt=config.system.dt,
                   t=config.system.t)
    q = states_puller(sol)
    
    #q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    tn = len(q)
    # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    X1, X2, X3, ra, Cab = isys.recover_fields(q1,q2,
                                              tn, X,
                                              phi1l, phi2l,
                                              psi2l, X_xdelta,
                                              C0ab, config)
    # X1 = jnp.zeros_like(X2)
    return adcommon._objective_output(q=q, X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab,
                                      f_obj=f_obj,
                                      nodes=jnp.array(obj_args.nodes),
                                      components=jnp.array(obj_args.components),
                                                           t=jnp.array(obj_args.t),axis=obj_args.axis)

@partial(jax.jit, static_argnames=['config', 'f_obj', 'obj_args'])
def main_40g21_3(inputs, #gust_intensity, gust_length, u_inf, rho_inf,
                 q0,
                 config,
                 f_obj,
                 obj_args,
                 *args,
                 **kwargs
                 ):
    
    Ka = inputs["Ka"]
    Ma = inputs["Ka"]
    eigenvals = inputs["eigenvals"] #jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs["eigenvecs"] #jnp.load(config.fem.folder / config.fem.eig_names[1])

    gust_intensity = config.system.aero.gust.intensity #inputs["gust_intensity"]
    gust_length = config.system.aero.gust.length
    u_inf = config.system.aero.u_inf
    rho_inf = config.system.aero.rho_inf
    # u_inf, rho_inf, gust_length, gust_intensity = input1
    # if obj_args is None:
    #     obj_args = dict()

    Ka = config.fem.Ka
    Ma = config.fem.Ma    
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = jnp.array(range(config.fem.num_modes, 2 * config.fem.num_modes)) #config.system.states['q2']
    q1_index = jnp.array(range(config.fem.num_modes)) #config.system.states['q1']
    #eigenvals, eigenvecs = scipy.linalg.eigh(Ka, Ma)
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0]).T
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1]).T
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = adcommon._compute_modes(X,
                                                             Ka,
                                                             Ma,
                                                             reduced_eigenvals,
                                                             reduced_eigenvecs,
                                                             config)
    #################
    gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )

    #################
    A0 = config.system.aero.A[0]
    A1 = config.system.aero.A[1]
    A2 = config.system.aero.A[2]
    A3 = config.system.aero.A[3:]
    D0 = config.system.aero.D[0]
    D1 = config.system.aero.D[1]
    D2 = config.system.aero.D[2]
    D3 = config.system.aero.D[3:]    
    #u_inf = config.system.aero.u_inf
    #rho_inf = config.system.aero.rho_inf
    q_inf = 0.5 * rho_inf * u_inf ** 2 # config.system.aero.q_inf
    c_ref = config.system.aero.c_ref
    A0hat = q_inf * A0
    A1hat = c_ref * rho_inf * u_inf / 4 * A1
    A2hat = c_ref**2 * rho_inf / 8 * A2
    A3hat = q_inf * A3
    A2hatinv = jnp.linalg.inv(jnp.eye(len(A2hat)) -
                              A2hat)
    D0hat = q_inf * D0
    D1hat = c_ref * rho_inf * u_inf / 4 * D1
    D2hat = c_ref**2 * rho_inf / 8 * D2
    D3hat = q_inf * D3    
    # gust_intensity = config.system.aero.gust.intensity
    # gust_length = config.system.aero.gust.length
    gust_shift = config.system.aero.gust.shift
    gust_step = config.system.aero.gust.step
    dihedral = config.system.aero.gust.panels_dihedral
    time = config.system.t
    collocation_points = config.system.aero.gust.collocation_points
    gust_totaltime = config.system.aero.gust.totaltime
    xgust = config.system.aero.gust.x
    time = config.system.aero.gust.time
    ntime = config.system.aero.gust.ntime
    # gust_totaltime, xgust, time, ntime, npanels = igust._get_gustRogerMc(
    #     gust_intensity,
    #     dihedral,
    #     gust_shift,
    #     gust_step,
    #     time,
    #     collocation_points,
    #     gust_length,
    #     u_inf)
    # gust_totaltime, xgust, time, ntime = igust._get_gustRogerMc(
    #     config.system.aero.gust.intensity,
    #     config.system.aero.gust.panels_dihedral,
    #     config.system.aero.gust.shift,
    #     config.system.aero.gust.step,
    #     time,
    #     config.system.aero.gust.length,
    #     config.system.aero.u_inf,
    #     jnp.min(collocation_points[:,0]),
    #     jnp.max(collocation_points[:,0])
    # )
    npanels = len(collocation_points)
    fshape_span = igust._get_spanshape(config.system.aero.gust.shape)
    gust, gust_dot, gust_ddot = igust._downwashRogerMc(u_inf,
                                                       gust_length,
                                                       gust_intensity,
                                                       gust_shift,
                                                       collocation_points,
                                                       dihedral, #normals,
                                                       time,
                                                       gust_totaltime,
                                                       fshape_span
                                                       )
    Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot = igust._getGAFs(D0hat,  # NbxNm
             D1hat,         
             D2hat,         
             D3hat,
             gust,
             gust_dot,
             gust_ddot
             )
    poles = config.system.aero.poles
    num_poles = config.system.aero.num_poles
    num_modes = config.fem.num_modes
    states = config.system.states
    eta0 = jnp.zeros(num_modes)
    dq_args = (eta0, gamma1, gamma2, omega, states,
               num_modes, num_poles,
               A0hat, A1hat, A2hatinv, A3hat,
               u_inf, c_ref, poles,
               time, Q_wsum, Ql_wdot)

    #################
    # import pdb;pdb.set_trace()
    states_puller, eqsolver = sollibs.factory(config.system.solver_library,
                                              config.system.solver_function)

    sol = eqsolver(dq_dynamic.dq_20g21,
                   dq_args,
                   config.system.solver_settings,
                   q0=q0,
                   t0=config.system.t0,
                   t1=config.system.t1,
                   tn=config.system.tn,
                   dt=config.system.dt,
                   t=config.system.t)
    q = states_puller(sol)
    
    #q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    tn = len(q)
    # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    X1, X2, X3, ra, Cab = isys.recover_fields(q1,q2,
                                              tn, X,
                                              phi1l, phi2l,
                                              psi2l, X_xdelta,
                                              C0ab, config)
    # X1 = jnp.zeros_like(X2)
    return adcommon._objective_output(q=q, X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab,
                                      f_obj=f_obj,
                                      nodes=jnp.array(obj_args.nodes),
                                      components=jnp.array(obj_args.components),
                                                           t=jnp.array(obj_args.t),axis=obj_args.axis)
