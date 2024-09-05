import diffrax
import scipy.optimize
import jax.numpy as jnp


def ode(F: callable, solver_name: str, q0, t0, t1, tn, dt, **kwargs):
    term = diffrax.ODETerm(F)
    saveat = jnp.linspace(t0, t1, tn)
    _solver = getattr(diffrax, solver_name)
    solver = _solver()
    sol = diffrax.diffeqsolve(term, solver, saveat=saveat, t0=t0, t1=t1, dt0=dt, y0=q0)
    return sol


def root(F, q0, args, method="hybr", tolerance=1e-6, jac=None, **kwargs):
    sol = scipy.optimize.root(F, q0, args=args, method=method, tol=tolerance)
    return sol


def pull_ode(sol):
    qs = jnp.array(sol.ys)
    return qs


def pull_root(sol):
    qs = jnp.array(sol.x)
    return qs


# __init__(self, rtol: Optional[Scalar] = None, atol: Optional[Scalar] = None, max_steps: Optional[int] = 10, kappa: Scalar = 0.01, norm: Callable = <function rms_norm>, tolerate_nonconvergence: bool = False)
# __call__(self, fn: Callable, x: PyTree, args: PyTree, jac: Optional[~LU_Jacobian] = None)
