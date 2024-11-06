import jax.numpy as jnp
import feniax.preprocessor.solution as solution
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal

def arg_10g11(
    sol: solution.IntrinsicSolution,
    system: intrinsicmodal.Dsystem,
    fem: intrinsicmodal.Dfem,
    *args,
    **kwargs,
):
    pointforces = getattr(sol.data, f"pointforces_{system.name}")
    eta_0 = kwargs["eta_0"]
    gamma2 = sol.data.couplings.gamma2
    phi1l = sol.data.modes.phi1l
    phi2l = sol.data.modes.phi2l
    psi2l = sol.data.modes.psi2l
    X_xdelta = sol.data.modes.X_xdelta
    omega = sol.data.modes.omega
    x = pointforces.x
    # force_follower = pointforces.force_follower
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    
    return (phi2l, psi2l, X_xdelta, C0ab, (eta_0, gamma2, omega, phi1l, x,))

def arg_10g15(
    sol: solution.IntrinsicSolution,
    system: intrinsicmodal.Dsystem,
    fem: intrinsicmodal.Dfem,
    *args,
    **kwargs,
):
    
    eta_0 = kwargs["eta_0"]
    gamma2 = sol.data.couplings.gamma2
    phi1l = sol.data.modes.phi1l
    phi2l = sol.data.modes.phi2l
    psi2l = sol.data.modes.psi2l
    X_xdelta = sol.data.modes.X_xdelta
    omega = sol.data.modes.omega
    x = system.xloads.x
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    qalpha = system.aero.qalpha
    A0 = system.aero.A[0]
    C0 = system.aero.Q0_rigid
    
    return (phi2l, psi2l, X_xdelta, C0ab, A0, C0, (eta_0, gamma2, omega, x, qalpha))
