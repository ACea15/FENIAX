import diffrax
from diffrax.solution import Solution
import jax.numpy as jnp

def ode(F: callable, solver_name: str, q0, t0, t1, tn, dt, **kwargs) -> Solution:
    term = diffrax.ODETerm(F)
    saveat = jnp.linspace(t0, t1, tn)
    _solver = getattr(diffrax, solver_name)
    solver = _solver()
    sol = diffrax.diffeqsolve(term, solver,saveat=saveat,
                              t0=t0, t1=t1, dt0=dt, y0=q0)
    return sol

def newton_raphson(F, q0, args, rtol, atol, max_steps, kappa, norm, jac=None, **kwargs):

    solver = diffrax.NewtonNonlinearSolver(rtol=rtol,
                                           atol=atol,
                                           max_steps=max_steps,
                                           kappa=kappa,
                                           norm=norm,
                                           tolerate_nonconvergence=False)
    sol = solver(F, q0, args, jac)
    return sol

def pull_ode(sol):

    qs = jnp.array(sol.ys)
    return qs

def pull_newton_raphson(sol):

    qs = jnp.array(sol.root)
    return qs

#__init__(self, rtol: Optional[Scalar] = None, atol: Optional[Scalar] = None, max_steps: Optional[int] = 10, kappa: Scalar = 0.01, norm: Callable = <function rms_norm>, tolerate_nonconvergence: bool = False)
#__call__(self, fn: Callable, x: PyTree, args: PyTree, jac: Optional[~LU_Jacobian] = None)
