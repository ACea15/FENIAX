import jax.numpy as jnp
import feniax.intrinsic.dq_dynamic as dq_dynamic

def dq_20G2(t, q, *args):
    """Free Structural dynamic gravity forces."""

    (
        eta_0,
        gamma1,
        gamma2,
        omega,
        phi1l,
        psi2l,
        force_gravity,
        states,
        X_xdelta,
        C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father,
    ) = args[0]

    q1 = q[states["q1"]]
    q2 = q[states["q2"]]
    qr = q[states["qr"]]

    Rab = common.computeRab_node0(
        psi2l,
        q2,
        qr,
        X_xdelta,
        C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father,
    )

    # no interpolation of gravity in dynamic case
    eta = xloads.eta_pointdead_const(phi1l, force_gravity[-1], Rab)
    F1, F2 = common.f_12(omega, gamma1, gamma2, q1, q2)
    F1 += eta
    F1 += eta_0
    Fr = common.f_quaternion(phi1l, q1, qr)
    F = jnp.hstack([F1, F2, Fr])
    return F


def main_20G2(t, q, kwargs):

    Fall = jnp.zeros_like(q)
    for i, ni in enumerate(name_body):
        
        F = dq_dynamic.dq_20G2(t, q[ni], kwargs[ni])
        ci = constraints[ni]
        citype = ci.type_name
        F[states[ni]["q1"]] += 0 #f(citype)
        Fall = Fall.at[states_global[ni]["q1"]].set(F)
    return 

