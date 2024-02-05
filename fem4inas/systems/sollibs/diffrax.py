import diffrax
import optimistix as optx
import jax.numpy as jnp

DICT_NORM = dict(linalg_norm=jnp.linalg.norm)

def ode(F: callable,
        args,
        sett,
        q0,
        t0,
        t1,
        tn,
        dt,
        **kwargs) -> diffrax.Solution:
    
    term = diffrax.ODETerm(F)
    if sett.save_at is None:
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, tn))#diffrax.SaveAt(steps=True) #
        #saveat = diffrax.SaveAt(steps=True)
    else:
        saveat = sett.save_at
    _solver = getattr(diffrax, sett.solver_name)
    solver = _solver()
    sol = diffrax.diffeqsolve(term,
                              solver,
                              t0=t0,
                              t1=t1,
                              dt0=dt,
                              y0=q0,
                              args=args,
                              #throw=False,
                              #max_steps=sett.max_steps,
                              saveat=saveat
                              )
    return sol

def newton(F, q0, args, sett, jac=None, **kwargs):

    solver = optx.Newton(rtol=sett.rtol,
                         atol=sett.atol,
                         kappa=sett.kappa,
                         norm=DICT_NORM[sett.norm])
    sol = optx.root_find(F, solver, q0, args=args, max_steps=sett.max_steps)
    #sol = solver(F, q0, args, jac)
    return sol

def pull_ode(sol):

    qs = jnp.array(sol.ys)
    return qs

def pull_newton(sol):

    qs = jnp.array(sol.value)
    return qs

#__init__(self, rtol: Optional[Scalar] = None, atol: Optional[Scalar] = None, max_steps: Optional[int] = 10, kappa: Scalar = 0.01, norm: Callable = <function rms_norm>, tolerate_nonconvergence: bool = False)
#__call__(self, fn: Callable, x: PyTree, args: PyTree, jac: Optional[~LU_Jacobian] = None)
