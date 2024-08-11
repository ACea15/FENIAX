import diffrax
import optimistix as optx
import jax.numpy as jnp
import jax

dict_norm = dict(linalg_norm=jnp.linalg.norm)

def ode(F: callable,
        args,
        sett,
        #solver_name: str,
        q0,
        t0,
        t1,
        tn,
        dt,
        #save_at=None,
        **kwargs) -> diffrax.Solution:

    solver_sett = dict()
    diffeqsolve_sett = dict()
    term = diffrax.ODETerm(F)
    if sett.save_at is None:
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, tn))#diffrax.SaveAt(steps=True) #
    else:
        saveat = sett.save_at
    _solver = getattr(diffrax, sett.solver_name)
    if (root := sett.root_finder) is not None:
        _root_finder = getattr(optx, list(root.keys())[0])
        root_finder = _root_finder(**list(root.values())[0])
        solver_sett["root_finder"] = root_finder        
    solver = _solver(**solver_sett)
    
    if (stepsize := sett.stepsize_controller) is not None:
        _stepsize_controller = getattr(optx, list(stepsize.keys())[0])
        stepsize_controller = _stepsize_controller(**list(stepsize.values())[0])
        diffeqsolve_sett["stepsize_controller"] = stepsize_controller
    
    sol = diffrax.diffeqsolve(term,
                              solver,
                              t0=t0,
                              t1=t1,
                              dt0=dt,
                              y0=q0,
                              args=args,
                              #throw=False,
                              max_steps=sett.max_steps,
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


def newton(F, q0, args, sett, jac=None, **kwargs):

    solver = optx.Newton(rtol=sett.rtol,
                         atol=sett.atol,
                         kappa=sett.kappa,
                         norm=dict_norm[sett.norm]
                         )
    # jax.debug.breakpoint()
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
