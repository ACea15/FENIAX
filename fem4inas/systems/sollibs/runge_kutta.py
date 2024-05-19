import jax.numpy as jnp
from functools import partial
import jax

#@partial(jax.jit, static_argnames=['f'])
def rk4old(ys, dt, N, f, args):

    @jax.jit
    def step(i, ys):
        h = dt
        t = dt * i
        ys0 = ys[i - 1]
        k1 = f(t, ys0, args)
        #jax.debug.print("k1: {}", k1)        
        k2 = f(t + 0.5 * h, ys0 + 0.5 * k1 * h, args)
        k3 = f(t + 0.5 * h, ys0 + 0.5 * k2 * h, args)
        k4 = f(t + h, ys0 + k3 * h, args)
        ysi = ys0 + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
        ys = ys.at[i].set(ysi)
        #jax.debug.breakpoint()
        return ys
  
    return jax.lax.fori_loop(1, N, step, ys)

@partial(jax.jit, static_argnames=['N', 'f'])
def rk4(ys0, dt, N, f, args):

    @jax.jit
    def step(carry, ys):
        i, ys0 = carry
        t = dt * i
        k1 = f(t, ys0, args)
        #jax.debug.print("k1: {}", k1)        
        k2 = f(t + 0.5 * dt, ys0 + 0.5 * k1 * dt, args)
        k3 = f(t + 0.5 * dt, ys0 + 0.5 * k2 * dt, args)
        k4 = f(t + dt, ys0 + k3 * dt, args)
        ysi = ys0 + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt
        #ys = ys.at[i].set(ysi)
        new_carry = (i + 1, ysi)
        # jax.debug.breakpoint()
        return new_carry, ysi

    last_carry, yt  = jax.lax.scan(step, (0, ys0), None, length = N - 1)
    return jnp.vstack([ys0, yt])

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
        sett,
        q0,
        dt,
        tn,
        **kwargs) -> jnp.ndarray:
  
    #ys = jnp.zeros((tn, len(q0)))
    #ys  = ys.at[0].set(q0)
    _solver = globals()[sett.solver_name]
    sol = _solver(q0, dt, tn, F, args)
    return sol

def odeold(F: callable,
        args,
        sett,
        q0,
        dt,
        tn,
        **kwargs) -> jnp.ndarray:
  
    ys = jnp.zeros((tn, len(q0)))
    ys  = ys.at[0].set(q0)
    _solver = globals()[sett.solver_name]
    sol = _solver(ys, dt, tn, F, args)
    return sol

def pull_ode(sol):

    return sol
