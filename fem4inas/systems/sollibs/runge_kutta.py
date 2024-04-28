import jax.numpy as jnp
from functools import partial
import jax

#@partial(jax.jit, static_argnames=['f'])
def rk4(ys, dt, N, f, args):

    @jax.jit
    def step(i, ys):
        h = dt
        t = dt * i
        ys0 = ys[i - 1]
        k1 = f(t, ys0, args)
        #jax.debug.print("k1: {}", k1)        
        k2 = f(t + 0.5 * h, ys0 + 0.5 * k1 * h, args)
        #jax.debug.print("k2: {}", k2)        
        k3 = f(t + 0.5 * h, ys0 + 0.5 * k2 * h, args)
        #jax.debug.print("k3: {}", k3)        
        k4 = f(t + h, ys0 + k3 * h, args)
        #jax.debug.print("k4: {}", k4)        
        ysi = ys0 + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        ys = ys.at[i].set(ysi)
        #jax.debug.breakpoint()
        return ys
  
    return jax.lax.fori_loop(1, N, step, ys)

@partial(jax.jit, static_argnames=['f'])
def rk2(ys, dt, N, f, args):

    @jax.jit
    def step(i, ys):
        h = dt
        t = dt * i
        ys0 = ys[i - 1]
        k1 = f(t, ys0, args)
        ysi = ys0 + k1 * h
        ys = ys.at[i].set(ysi)
        #jax.debug.breakpoint()
        return ys
  
    return jax.lax.fori_loop(1, N, step, ys)


def ode(F: callable,
        args,
        q0,
        dt,
        tn,
        solver_name: str,
        **kwargs) -> jnp.ndarray:
  
    ys = jnp.zeros((tn, len(q0)))
    ys  = ys.at[0].set(q0)
    _solver = globals()[solver_name]
    sol = _solver(ys, dt, tn, F, args)
    return sol

def pull_ode(sol):

    return sol
