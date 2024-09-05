from jax.experimental.ode import odeint
import jax.numpy as jnp


def ode(F: callable, args, q0, t, rtol=1.4e-8, atol=1.4e-8, **kwargs) -> jnp.ndarray:
    sol = odeint(F, q0, t, args, rtol=rtol, atol=atol)
    return sol


def pull_ode(sol):
    return sol
