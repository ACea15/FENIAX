import jax.numpy as jnp
import numpy as np
import jax
import fem4inas.intrinsic.xloads as xloads
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.intrinsic.dq_common as common

def dq_10G1(q, *args):
    """Structural static under gravity."""

    (eta_0, gamma2, omega, phi1l, psi2l,
     x, force_gravity,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father, t) = args[0]
    X3t = postprocess.compute_strains_t(psi2l, q)
    Rab = postprocess.integrate_strainsCab(
        jnp.eye(3), X3t,
        X_xdelta, C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father)
    F = omega * q - common.contraction_gamma2(gamma2, q)
    F += xloads.eta_pointdead(t, phi1l, x, force_gravity, Rab)
    F += eta_0
    return F

def dq_10g11(q, *args):
    """Structural static with follower point forces."""
    
    (eta_0, gamma2, omega, phi1, x,
     force_follower, t) = args[0]
    #@jax.jit
    def _dq_10g11(q):
        F = omega * q - common.contraction_gamma2(gamma2, q)
        F += xloads.eta_pointfollower(t,
                                      phi1,
                                      x,
                                      force_follower)
        F += eta_0
        return F

    F = _dq_10g11(q)
    return F

def dq_10g121(q, *args):
    """Structural static with dead point forces."""
    
    (eta_0, gamma2, omega, phi1l, psi2l,
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
        F += eta_0
        return F
    F = _dq_10g121(q)
    return F

def dq_10G121(q, *args):
    """Structural static with dead point forces and gravity."""
    
    (eta_0, gamma2, omega, phi1l, psi2l,
     x, force_dead, force_gravity,
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
        F += xloads.eta_pointdead(t, phi1l, x, force_dead + force_gravity, Rab)
        F += eta_0
        return F
    F = _dq_10g121(q)
    return F

def dq_10g15(q, *args):
    """Manoeuvre under qalpha."""
    
    (eta_0, gamma2, omega,
     qalpha, A0, C0) = args[0]
    q0 = -q / omega
    F = omega * q - common.contraction_gamma2(gamma2, q)
    F += xloads.eta_steadyaero(q0, A0)
    F += xloads.eta_manoeuvre(qalpha, A0)
    F += eta_0
    return F

def dq_11G6(q, *args):
    """Static trim1 """

    (eta_0, gamma2, omega, phi1, phi1l, psi2l,
     x, force_gravity,
     states,
     A0hat, B0hat, elevator_index, elevator_link,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father, t) = args[0]

    # qe = q[states['qe']]
    q2 = q[states['q2']]
    q0i = -q2[2:] / omega[2:]
    q0 = jnp.hstack([q2[:2], q0i])
    qx = q[states['qx']]
    X3t = postprocess.compute_strains_t(psi2l, q2)
    Rab = postprocess.integrate_strainsCab(
        jnp.eye(3), X3t,
        X_xdelta, C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father)

    eta_gravity = xloads.eta_pointdead(t, phi1l, x, force_gravity, Rab)
    eta_aero = xloads.eta_steadyaero(q0, A0hat)
    eta_elevator = xloads.eta_controls(qx, B0hat, elevator_index, elevator_link)
    F1 = omega * q2 - common.contraction_gamma2(gamma2, q2)
    F1 += (eta_gravity + eta_aero + eta_elevator)
    F1 += eta_0
    Fh = phi1[:, 2, 0].dot(q0)  # added eq: 0 vertical position of first node
    #import jax.debug; jax.debug.breakpoint()
    F = jnp.hstack([F1, Fh])
    return F

def dq_11G6l(q, *args):
    """Static trim1 linear"""

    (eta_0, gamma2, omega, phi1, phi1l, psi2l,
     x, force_gravity,
     states,
     A0hat, B0hat, elevator_index, elevator_link,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father, t) = args[0]

    # qe = q[states['qe']]
    q2 = q[states['q2']]
    q0i = -q2[2:] / omega[2:]
    q0 = jnp.hstack([q2[:2], q0i])
    qx = q[states['qx']]
    X3t = postprocess.compute_strains_t(psi2l, q2)
    Rab = postprocess.integrate_strainsCab(
        jnp.eye(3), X3t,
        X_xdelta, C0ab,
        component_names,
        num_nodes,
        component_nodes,
        component_father)

    eta_gravity = xloads.eta_pointdead(t, phi1l, x, force_gravity, Rab)
    eta_aero = xloads.eta_steadyaero(q0, A0hat)
    eta_elevator = xloads.eta_controls(qx, B0hat, elevator_index, elevator_link)
    F1 = omega * q2 #- common.contraction_gamma2(gamma2, q2)
    F1 += (eta_gravity + eta_aero + eta_elevator)
    F1 += eta_0
    Fh = phi1[:, 2, 0].dot(q0)  # added eq: 0 vertical position of first node
    #import jax.debug; jax.debug.breakpoint()
    F = jnp.hstack([F1, Fh])
    return F


def dq_12G2(q, *args):
    """Static trim2

    TODO: FINALISE!
    """

    (eta_0, gamma2, omega, phi1l,
     x, force_gravity,
     states,
     A0hat, B0hat, C0hat,
     X_xdelta,
     C0ab,
     component_names, num_nodes,
     component_nodes, component_father, t) = args[0]

    # qalpha = q[states['qalpha']]
    # qe = q[states['qe']]
    q2i = q[states['q2']]
    q0i = -q2i / omega[2:]
    qalpha = q[states['qalpha']]
    qx = q[states['qx']]
    q2 = jnp.hstack([0., 0., q[states['q2']]])
    q0 = jnp.hstack([0., 0., q0i])
    jnp.hstack([0., 0., qalpha])
    qm = jnp.hstack([0., 0., qalpha])
    eta_gravity = xloads.eta_pointdead(t, phi1l, x, force_gravity, C0ab)
    eta_aero = xloads.eta_steadyaero(q0, A0hat)
    eta_aoa = xloads.eta_manoeuvre(qm, C0hat)
    eta_elevator = xloads.eta_control(qx, B0hat)
    F = omega * q2 - common.contraction_gamma2(gamma2, q2)
    F += (eta_gravity + eta_aero + eta_elevator)
    F += eta_0
    return F
