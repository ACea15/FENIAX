import jax.numpy as jnp
import numpy as np
import jax
import fem4inas.intrinsic.xloads as xloads
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.intrinsic.dq_common as common

def dq_10g11(q, *args):
    """Structural static with follower point forces."""
    
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
    """Structural static with dead point forces."""
    
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
    # jax.debug.breakpoint()
    return F

def dq_10g15(q, *args):
    """Manoeuvre under qalpha."""
    
    (gamma2, omega,
     qalpha, A0, C0) = args[0]
    q0 = -q / omega
    F = omega * q - common.contraction_gamma2(gamma2, q)
    F += xloads.eta_steadyaero(q0, A0)
    F += xloads.eta_manoeuvre(qalpha, A0)
    return F

def dq_10g150(q, *args):
    """Static trim"""
    
    (gamma2, omega,
     qalpha, A0, B0) = args[0]
    qalpha = q[states['qalpha']]
    qplunged = q[states['qplunged']]
    qh = q[states['qh']]
    qe = q[states['qe']]
    q2 = q[states['q2']]
    q0 = -q2 / omega
    F = omega * q2 - common.contraction_gamma2(gamma2, q2)
    F += xloads.eta_steadyaero(q0, A0)
    F += xloads.eta_control(qe, B0)
    return F
