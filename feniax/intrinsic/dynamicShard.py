import jax.numpy as jnp
import jax
from functools import partial
import feniax.intrinsic.xloads as xloads
import feniax.intrinsic.postprocess as postprocess
import feniax.intrinsic.dq_common as common
import feniax.systems.intrinsic_system as isys
import feniax.systems.sollibs.diffrax as libdiffrax
import feniax.intrinsic.gust as igust
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal


# @partial(jax.jit, static_argnames=["config"])
def main_20g21_3(
    inputs,  # 
    q0,
    config,
    args,
    **kwargs,
):

    X = config.fem.X
    phi1l, phi2l, psi2l, X_xdelta, C0ab, A, D, c_ref, _dqargs = args
    states = _dqargs[4]
    q1_index = states["q1"]
    q2_index = states["q2"]
    collocation_points = config.system.aero.gust.collocation_points
    #xcollocation_points = collocation_points[:, 0]
    #xcollocation_min = min(xcollocation_points)
    #xcollocation_max = max(xcollocation_points)
    gust_shift = config.system.aero.gust.shift
    dihedral = config.system.aero.gust.panels_dihedral
    fshape_span = igust._get_spanshape(config.system.aero.gust.shape)        
    # gust_totaltime = config.system.aero.gust.totaltime
    time_gust = config.system.aero.gust.time
    # @jax.jit
    def _main_20g21_3(inp):

        rho_inf = inp[0]
        u_inf = inp[1]
        gust_length = inp[2]
        gust_intensity = inp[3]
        q_inf = 0.5 * rho_inf * u_inf**2
        gust_totaltime = gust_length / u_inf
        
        A0hat = q_inf * A[0]
        A1hat = c_ref * rho_inf * u_inf / 4 * A[1]
        A2hat = c_ref**2 * rho_inf / 8 * A[2]
        A3hat = q_inf * A[3:]
        A2hatinv = jnp.linalg.inv(jnp.eye(len(A2hat)) - A2hat)
        D0hat = q_inf * D[0]
        D1hat = c_ref * rho_inf * u_inf / 4 * D[1]
        D2hat = c_ref**2 * rho_inf / 8 * D[2]
        D3hat = q_inf * D[3:]

        # gust_totaltime, xgust, time_gust, ntime = intrinsicmodal.gust_discretisation(
        #     gust_intensity,
        #     config.system.aero.gust.panels_dihedral,
        #     config.system.aero.gust.shift,
        #     config.system.aero.gust.step,
        #     simulation_time,
        #     gust_length,
        #     u_inf,
        #     xcollocation_min,
        #     xcollocation_max,
        # )
        
        #gust_totaltime = config.system.aero.gust.totaltime
        gust, gust_dot, gust_ddot = igust._downwashRogerMc(
            u_inf,
            gust_length,
            gust_intensity,
            gust_shift,
            collocation_points,
            dihedral,  # normals,
            time_gust,
            gust_totaltime,
            fshape_span,
        )
        Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot = igust._getGAFs(
            D0hat,  # NbxNm
            D1hat,
            D2hat,
            D3hat,
            gust,
            gust_dot,
            gust_ddot,
        )

        # (
        #     eta_0,
        #     gamma1,
        #     gamma2,
        #     omega,
        #     states,
        #     num_modes,
        #     num_poles,
        #     A0hat,
        #     A1hat,
        #     A2hatinv,
        #     A3hat,
        #     u_inf,
        #     c_ref,
        #     poles,
        #     xgust,
        #     F1gust,
        #     Flgust,
        # ) = args[0]

        args_inp = (c_ref,
                    A0hat,
                    A1hat,
                    A2hatinv,
                    A3hat,
                    u_inf,
                    Q_wsum,
                    Ql_wdot
                    )
        
        dq_args = _dqargs + args_inp
        states_puller, eqsolver = sollibs.factory(
            config.system.solver_library, config.system.solver_function
        )

        sol = eqsolver(
            dq_dynamic.dq_20g21,
            dq_args,
            config.system.solver_settings,
            q0=q0,
            t0=config.system.t0,
            t1=config.system.t1,
            tn=config.system.tn,
            dt=config.system.dt,
            t=config.system.t,
        )
        # jax.debug.breakpoint()
        q = states_puller(sol)
        q1 = q[:, q1_index]
        q2 = q[:, q2_index]
        tn = len(q)
        # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
        #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
        # X1, X2, X3, ra, Cab = isys.recover_fields(
        #     q1, q2, tn, X, phi1l, phi2l, psi2l, X_xdelta, C0ab, config
        #     )

        return dict(q=q, X2=q) #dict(q=q, X1=X1,X2=X2, X3=X3, ra=ra, Cab=Cab)

    main_vmap = jax.vmap(_main_20g21_3)
    results = main_vmap(inputs)
    return results

# @partial(jax.jit, static_argnames=["config"])
def main_20g546_3(
    inputs,  # 
    q0,
    config,
    args,
    **kwargs,
):

    X = config.fem.X
    phi1l, phi2l, psi2l, X_xdelta, C0ab, A, D, c_ref,  _dqargs = args
    states = _dqargs[5]
    q1_index = states["q1"]
    q2_index = states["q2"]
    # q1_index = states["q1"]
    # q2_index = states["q2"]
    
    collocation_points = config.system.aero.gust.collocation_points
    gust_shift = config.system.aero.gust.shift
    dihedral = config.system.aero.gust.panels_dihedral
    fshape_span = igust._get_spanshape(config.system.aero.gust.shape)        
    # gust_totaltime = config.system.aero.gust.totaltime
    time_gust = config.system.aero.gust.time
    # @jax.jit
    def _main_20g546_3(inp):

        rho_inf = inp[0]
        u_inf = inp[1]
        gust_length = inp[2]
        gust_intensity = inp[3]
        q_inf = 0.5 * rho_inf * u_inf**2
        gust_totaltime = gust_length / u_inf
        
        A0hat = q_inf * A[0]
        A1hat = c_ref * rho_inf * u_inf / 4 * A[1]
        A2hat = c_ref**2 * rho_inf / 8 * A[2]
        A3hat = q_inf * A[3:]
        A2hatinv = jnp.linalg.inv(jnp.eye(len(A2hat)) - A2hat)
        D0hat = q_inf * D[0]
        D1hat = c_ref * rho_inf * u_inf / 4 * D[1]
        D2hat = c_ref**2 * rho_inf / 8 * D[2]
        D3hat = q_inf * D[3:]

        gust, gust_dot, gust_ddot = igust._downwashRogerMc(
            u_inf,
            gust_length,
            gust_intensity,
            gust_shift,
            collocation_points,
            dihedral,  # normals,
            time_gust,
            gust_totaltime,
            fshape_span,
        )
        Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot = igust._getGAFs(
            D0hat,  # NbxNm
            D1hat,
            D2hat,
            D3hat,
            gust,
            gust_dot,
            gust_ddot,
        )

        args_inp = (c_ref,
                    A0hat,
                    A1hat,
                    A2hatinv,
                    A3hat,
                    u_inf,
                    Q_wsum,
                    Ql_wdot
                    )
        
        dq_args = _dqargs + args_inp
        states_puller, eqsolver = sollibs.factory(
            config.system.solver_library, config.system.solver_function
        )

        sol = eqsolver(
            dq_dynamic.dq_20g546,
            dq_args,
            config.system.solver_settings,
            q0=q0,
            t0=config.system.t0,
            t1=config.system.t1,
            tn=config.system.tn,
            dt=config.system.dt,
            t=config.system.t,
        )
        # jax.debug.breakpoint()
        q = states_puller(sol)
        q1 = q[:, q1_index]
        q2 = q[:, q2_index]
        tn = len(q)
        # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
        #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
        X1, X2, X3, ra, Cab = isys.recover_fieldsRB(
            q1, q2, tn, config.system.dt, X, phi1l, phi2l, psi2l, X_xdelta, C0ab, config
            )

        return dict(q=q, X1=X1,X2=X2, X3=X3, ra=ra, Cab=Cab)

    main_vmap = jax.vmap(_main_20g546_3)
    results = main_vmap(inputs)
    return results
