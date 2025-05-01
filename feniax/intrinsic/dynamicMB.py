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


def main_20G2(t, q, *args):

    (
        eta_0,
        gamma1,
        gamma2,
        omega,
        phi1l,
        psi2l,
        force_gravity,
        states_body,
        X_xdelta,
        C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father,
        states_global,
        multibody
    ) = args[0]

    Fall = jnp.zeros_like(q)
    for i, ni in enumerate(multibody.name_bodies):
        states_global_ni = states_global[ni]
        q_ni = q[states_global_ni]        
        F = dq_dynamic.dq_20G2(t, q_ni, (ai[ni] for ai in args[0][:-2]))
        Fall = Fall.at[states_global_ni].set(F)
        
    Fall_lm = Fall.copy()
    for cnamei, ci in multibody.constraints:
        constraint_fun = constraintsMB.CONSTRAINT_DICT[ci.type_name]
        body_father = ci.body_father
        body = ci.body
        states_b1 = states_global[body_father]
        states_b2 = states_global[body]
        states_qb1 = states_b1[states_body[body_father]["q1"]]
        states_qb2 = states_b2[states_body[body]["q1"]]
        states_rb1 = states_b1[states_body[body_father]["qr"]]
        states_rb2 = states_b2[states_body[body]["qr"]]
        Phi1_b1 = phi1l[body_father]
        Phi1_b2 = phi1l[body]
        q1_b1 = q[states_qb1]
        q1_b2 = q[states_qb2]
        qr_b1 = q[states_rb1]
        qr_b2 = q[states_rb1]        
        F1_b1 = Fall[states_qb1]
        F1_b2 = Fall[states_qb2]
        Fr_b1 = Fall[states_rb1]
        Fr_b2 = Fall[states_rb1]
        Gq, Gr  = constraint_fun(Phi1_b1,
                                 Phi1_b2,
                                 q1_b1,
                                 q1_b2,
                                 qr_b1,
                                 qr_b2)
        GqT = Gq.T
        GrT = Gr.T
        lagrange_multiplier = commonMB.lambda12(F1_b1,
                                                F1_b2,
                                                Fr_b1,
                                                Fr_b2,
                                                Gq,
                                                GqT,
                                                Gr,
                                                GrT)
        Gq_lagrange = GqT @ lagrange_multiplier
        Gr_lagrange = GrT @ lagrange_multiplier
        Gqb1_lagrange = Gq_lagrange[states_qb1]
        Gqb2_lagrange = Gq_lagrange[states_qb2]
        Grb1_lagrange = Gr_lagrange[states_rb1]
        Grb2_lagrange = Gr_lagrange[states_rb2]
        Fall_lm = Fall_lm.at[states_qb1].add(-Gqb1_lagrange)
        Fall_lm = Fall_lm.at[states_qb2].add(-Gqb2_lagrange)
        Fall_lm = Fall_lm.at[states_rb1].add(-Grb1_lagrange)
        Fall_lm = Fall_lm.at[states_rb2].add(-Grb2_lagrange)
        
    return Fall_lm

