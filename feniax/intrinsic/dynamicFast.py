import jax.numpy as jnp
import jax
from functools import partial
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.couplings as couplings
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.systems.intrinsic_system as isys


# @partial(jax.jit, static_argnames=["config"])
def main_20g1(
    q0,
    config,
    *args,
    **kwargs,
):
    """
    Dynamic response free vibrations
    """

    output, input_dict = adcommon._build_intrinsic(adcommon._get_inputs, config, **kwargs)

    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    states = config.system.states
    eta0 = jnp.zeros(config.fem.num_modes)
    dq_args = (
        eta0,
        output["gamma1"],
        output["gamma2"],
        output["omega"],
        states,
    )

    states_puller, eqsolver = sollibs.factory(
        config.system.solver_library, config.system.solver_function
    )

    sol = eqsolver(
        dq_dynamic.dq_20g1,
        dq_args,
        config.system.solver_settings,
        q0=q0,
        t0=config.system.t0,
        t1=config.system.t1,
        tn=config.system.tn,
        dt=config.system.dt,
        t=config.system.t,
    )
    q = states_puller(sol)
    adcommon._build_solution(q, output, config)
    return output


# @partial(jax.jit, static_argnames=["config"])
def main_20g11(
    q0,
    config,
    *args,
    **kwargs,
):
    """
    Dynamic response to Follower load
    """

    output, input_dict = adcommon._build_intrinsic(adcommon._get_inputs, config, **kwargs)

    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    config.system.xloads.build_point_follower(config.fem.num_nodes, output["C06ab"])

    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = input_dict["alpha"] * config.system.xloads.force_follower
    states = config.system.states
    eta0 = jnp.zeros(config.fem.num_modes)
    dq_args = (
        eta0,
        output["gamma1"],
        output["gamma2"],
        output["omega"],
        output["phi1l"],
        x_forceinterpol,
        y_forceinterpol,
        states,
    )

    states_puller, eqsolver = sollibs.factory(
        config.system.solver_library, config.system.solver_function
    )

    sol = eqsolver(
        dq_dynamic.dq_20g11,
        dq_args,
        config.system.solver_settings,
        q0=q0,
        t0=config.system.t0,
        t1=config.system.t1,
        tn=config.system.tn,
        dt=config.system.dt,
        t=config.system.t,
    )
    q = states_puller(sol)
    adcommon._build_solution(q, output, config)
    return output


# @partial(jax.jit, static_argnames=["config"])
def main_20g21(
    q0,
    config,
    *args,
    **kwargs,
):
    """
    Gust response, clamped model
    """

    output, input_dict = adcommon._build_intrinsic(adcommon._get_inputs_aero, config, **kwargs)
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)

    #################
    (q_inf, c_ref, poles, A0hat, A1hat, A2hatinv, A3hat) = adcommon._get_aero(
        input_dict["u_inf"], input_dict["rho_inf"], config
    )

    timegust, Q_wsum, Ql_wdot = adcommon._get_gust(input_dict, q_inf, c_ref, config)
    num_poles = config.system.aero.num_poles
    num_modes = config.fem.num_modes
    states = config.system.states
    eta0 = jnp.zeros(num_modes)
    dq_args = (
        eta0,
        output["gamma1"],
        output["gamma2"],
        output["omega"],
        states,
        poles,
        num_modes,
        num_poles,
        timegust,
        c_ref,
        A0hat,
        A1hat,
        A2hatinv,
        A3hat,
        input_dict["u_inf"],
        Q_wsum,
        Ql_wdot,
    )

    #################
    # import pdb;pdb.set_trace()
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
    q = states_puller(sol)
    adcommon._build_solution(q, output, config)
    return output


def main_20g546(q0, config, *args, **kwargs):
    """Gust response free flight, q0 obtained via integrator q1."""

    output, input_dict = adcommon._build_intrinsic(adcommon._get_inputs_aero, config, **kwargs)
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)

    #################
    (q_inf, c_ref, poles, A0hat, A1hat, A2hatinv, A3hat) = adcommon._get_aero(
        input_dict["u_inf"], input_dict["rho_inf"], config
    )

    timegust, Q_wsum, Ql_wdot = adcommon._get_gust(input_dict, q_inf, c_ref, config)
    num_poles = config.system.aero.num_poles
    num_modes = config.fem.num_modes
    states = config.system.states
    eta0 = jnp.zeros(num_modes)
    dq_args = (
        eta0,
        output["gamma1"],
        output["gamma2"],
        output["omega"],
        output["phi1l"],
        states,
        poles,
        num_modes,
        num_poles,
        timegust,
        c_ref,
        A0hat,
        A1hat,
        A2hatinv,
        A3hat,
        input_dict["u_inf"],
        Q_wsum,
        Ql_wdot,
    )

    #################
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
    q = states_puller(sol)
    adcommon._build_solutionRB(q, output, config)
    return output
