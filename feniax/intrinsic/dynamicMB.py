import jax.numpy as jnp
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.intrinsic.constraintsMB as constraintsMB
import feniax.intrinsic.commonMB as commonMB

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
    states_body = 0

    for i, ni in enumerate(num_body):
        q_ni = q[states_body[ni]]
        F = dq_dynamic.dq_20G2(t, q_ni, kwargs[ni])
        Fall = Fall.at[states_global[ni]["q1"]].set(F)
    
    for i, ni in enumerate(num_constrains):
        ci = constraints[ni]
        constraint_fun = CONSTRAINT_DICT[Ci.name]
        Phi1_b1 = Phi1[ci.body_father]
        Phi1_b2 = Phi1[ci.body]
        q1_b1 = q[states_body[ci.body_father]["q1"]]
        q1_b2 = q[states_body[ci.body]["q1"]]
        qz_b1 = q[states_body[ci.body_father]["qz"]][:4]
        qz_b2 = q[states_body[ci.body]["qz"]][:4]
        F1_b1 = Fall[states_body[ci.body_father]["q1"]]
        F1_b2 = Fall[states_body[ci.body]["q1"]]
        Fz_b1 = Fall[states_body[ci.body_father]["qz"]][:4]
        Fz_b2 = Fall[states_body[ci.body]["qz"]][:4]
        
        G1, G2 = constraint_fun()
        lagrange_multiplier = commonMB.lambda12(F11, Fz1, F12, Fz2, G1, G1T, G2, G2T)
        Glagrange
        q_ni = q[states_body[ni]]
        citype = ci.type_name
        F[states[ni]["q1"]] += 0 #f(citype)
        Fall = Fall.at[states_global[ni]["q1"]].set(F)
        
    return Fall

