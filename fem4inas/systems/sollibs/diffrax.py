import diffrax
import optimistix as optx
import jax.numpy as jnp

dict_norm = dict(linalg_norm=jnp.linalg.norm)

def ode(F: callable,
        args,
        solver_name: str,
        q0,
        t0,
        t1,
        tn,
        dt,
        save_at=None,
        **kwargs) -> diffrax.Solution:

    solver_sett = dict()
    diffeqsolve_sett = dict()
    term = diffrax.ODETerm(F)
    if save_at is None:
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, tn))#diffrax.SaveAt(steps=True) #
    else:
        saveat = save_at
    _solver = getattr(diffrax, solver_name)
    if (root := "root_finder") in kwargs.keys():
        _root_finder = getattr(optx, list(kwargs[root].keys())[0])
        root_finder = _root_finder(**list(kwargs[root].values())[0])
        solver_sett["root_finder"] = root_finder

    solver = _solver(**solver_sett)
    if (controller := "stepsize_controller") in kwargs.keys():
        _stepsize_controller = getattr(diffrax, list(kwargs[controller].keys())[0])
        stepsize_controller = _stepsize_controller(**list(kwargs[controller].values())[0])
        diffeqsolve_sett["stepsize_controller"] = stepsize_controller
    sol = diffrax.diffeqsolve(term,
                              solver,
                              t0=t0,
                              t1=t1,
                              dt0=dt,
                              y0=q0,
                              args=args,
                              #throw=False,
                              max_steps=20000,
                              saveat=saveat,
                              **diffeqsolve_sett
                              )
    return sol

def ode2(F: callable,
        args,
        solver_name: str,
        q0,
        t0,
        t1,
        tn,
        dt,
        save_at=None,
        **kwargs) -> diffrax.Solution:
    
    term = diffrax.ODETerm(F)
    if save_at is None:
        saveat = diffrax.SaveAt(steps=True) #
    else:
        saveat = save_at
    _solver = getattr(diffrax, solver_name)
    solver = _solver()
    sol = diffrax.diffeqsolve(term,
                              solver,
                              t0=t0,
                              t1=t1,
                              dt0=dt,
                              y0=q0,
                              args=args,
                              #throw=False,
                              max_steps=20000,
                              saveat=saveat
                              )
    return sol


def newton_raphson(F, q0, args, rtol, atol, max_steps, kappa, norm, jac=None, **kwargs):

    solver = optx.Newton(rtol=rtol,
                         atol=atol,
                         kappa=kappa,
                         norm=dict_norm[norm])
    sol = optx.root_find(F, solver, q0, args=args, max_steps=max_steps)
    #sol = solver(F, q0, args, jac)
    return sol

def pull_ode(sol):

    qs = jnp.array(sol.ys)
    return qs

def pull_newton_raphson(sol):

    qs = jnp.array(sol.value)
    return qs

#__init__(self, rtol: Optional[Scalar] = None, atol: Optional[Scalar] = None, max_steps: Optional[int] = 10, kappa: Scalar = 0.01, norm: Callable = <function rms_norm>, tolerate_nonconvergence: bool = False)
#__call__(self, fn: Callable, x: PyTree, args: PyTree, jac: Optional[~LU_Jacobian] = None)
