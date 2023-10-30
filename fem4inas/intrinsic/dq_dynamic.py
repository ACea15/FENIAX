import jax.numpy as jnp
import numpy as np
import jax
import fem4inas.intrinsic.xloads as xloads
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.intrinsic.dq_common as common

#@jax.jit
def dq_20g1(t, q, *args):

    gamma1, gamma2, omega, states = args[0]
    q1 = q[states['q1']]
    q2 = q[states['q2']]
    F1, F2 = common.f_12(omega, gamma1, gamma2, q1, q2)
    F = jnp.hstack([F1, F2])
    return F

#@jax.jit
def dq_20g11(t, q, *args):

    (gamma1, gamma2, omega, phi1, x,
     force_follower, states) = args[0]

    q1 = q[states['q1']]
    q2 = q[states['q2']]
    eta = xloads.eta_pointfollower(t,
                                   phi1,
                                   x,
                                   force_follower)
    F1, F2 = common.f_12(omega, gamma1, gamma2, q1, q2)
    F1 += eta
    F = jnp.hstack([F1, F2])
    return F

def dq_20g121(t, q, *args):

    (gamma1, gamma2, omega, phi1l, psi2l,
     x, force_dead,
     states,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father) = args[0]
    q1i = q[states['q1']]
    q2i = q[states['q2']]
    #@jax.jit
    def _dq_20g121(t, q1, q2):
        X3t = postprocess.compute_strains_t(
            psi2l, q2)
        Rab = postprocess.integrate_strainsCab(
            jnp.eye(3), X3t,
            X_xdelta, C0ab,
            component_names,
            num_nodes,
            component_nodes,
            component_father)
        eta = xloads.eta_pointdead(t,
                                   phi1l,
                                   x,
                                   force_dead,
                                   Rab)
        F1, F2 = common.f_12(omega, gamma1, gamma2, q1, q2)
        F1 += eta
        F = jnp.hstack([F1, F2])
        return F
    F = _dq_20g121(t, q1i, q2i)
    return F

#@jax.jit
def dq_20g21(t, q, *args):

    (gamma1, gamma2, omega, states,
     num_modes, num_poles,
     A0hat, A1hat, A2hat, Aphat,
     u_inf, c_ref, poles,
     xgust, F1g, Flg) = args[0]

    q1 = q[states['q1']]
    q2 = q[states['q2']]
    q0 = -q2 / omega
    ql = q[states['ql']]
    ql_tensor = ql.reshape((num_modes, num_poles))
    eta_s = xloads.eta_rogerstruct(q0, q1, ql_tensor,
                                   A0hat, A1hat, A2hat)
    eta_g = xloads.linear_interpolation2(t, xgust, F1g)
    F1, F2 = common.f_12(omega, gamma1, gamma2, q1, q2, A2hat)
    F1 += eta_s + eta_g
    F1 = A2hat @ F1
    Fl_tensor = (jnp.tensordot(Aphat, q1, axis=(1,0)) -
                 2 * u_inf /c_ref * (poles * ql_tensor)) #NmxNp
    Flg_tensor = xloads.linear_interpolation2(t, xgust, Flg)
    Fl_tensor += Flg_tensor
    Fl = Fl_tensor.reshape(num_modes * num_poles)
    F = jnp.hstack([F1, F2, Fl])

    return F
