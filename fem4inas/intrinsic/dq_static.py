import jax.numpy as jnp
import numpy as np
import jax
import fem4inas.intrinsic.xloads as xloads
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.intrinsic.dq_common as common


def dq_10g11(q, *args):

    (gamma2, omega, phi1, x,
     force_follower, t) = args[0]
    #@jax.jit
    def _dq_10g11(q):
        F = omega * q - common.contraction_gamma2(gamma2, q)
        F += xloads.eta_pointfollower(t,
                                      phi1,
                                      x,
                                      force_follower)
        return F

    F = _dq_10g11(q)
    return F

def dq_10g121(q, *args):

    (gamma2, omega, phi1l, psi2l,
     x, force_dead,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father, t) = args[0]
    #@jax.jit
    def _dq_10g121(q2):
        X3t = postprocess.compute_strains_t(psi2l, q2)
        Rab = postprocess.integrate_strainsCab(
            jnp.eye(3), X3t,
            X_xdelta, C0ab,
            component_names,
            num_nodes,
            component_nodes,
            component_father)
        F = omega * q2 - common.contraction_gamma2(gamma2, q2)
        F += xloads.eta_pointdead(t, phi1l, x, force_dead, Rab)
        return F
    F = _dq_10g121(q)
    return F


def dq_10g15(q, *args):

    (gamma2, omega,
     u_inf, rho_inf,
     qalpha, A0, C0) = args[0]
    q0 = -q / omega
    F = omega * q - common.contraction_gamma2(gamma2, q)
    F += xloads.eta_manoeuvre(q0, qalpha,
                              u_inf, rho_inf,
                              A0, C0)
    return F
