from jax.experimental.ode import odeint
import jax.numpy as jnp
from functools import partial
import jax

@jax.jit
def rk4(y0, dt, N, f, args):
  @jax.jit
  def step(i, ys0):
    h = dt
    t = dt * i
    k1 = f(t, ys0, *args)
    k2 = f(t + 0.5 * h, ys0 + 0.5 * k1 * h)
    k3 = f(t + 0.5 * h, ys0 + 0.5 * k2 * h)
    k4 = f(t + h, ys0 + k3)
    ysi = ys0 + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4) * h
    return ysi

  return jax.lax.fori_loop(1, N, step, y0)
 

def ode(F: callable,
        args,
        q0,
        dt,
        tn,
        solver_name: str,
        **kwargs) -> jnp.ndarray:
    
    _solver = locals()[solver_name]
    sol = _solver(q0, dt, tn, F, args)
    return sol

def pull_ode(sol):

    return sol
