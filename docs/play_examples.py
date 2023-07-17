import jax.numpy as jnp
from jax import jit
import jax
from functools import partial

@jit
def reshape_arrays(x: jnp.ndarray):

    x2 = jnp.reshape(x, (int(x.shape[0] / 6), 6))
    return x2

@jit
def insert_arrays(x: jnp.ndarray, n):

    x2 = jnp.reshape(x, (int(x.shape[0] / 6), 6))
    x3 = jnp.insert(x2, n, jnp.zeros(6), axis=0)
    return x3

@partial(jit, static_argnames=['n'])
#@jit
def return_arrays(x: jnp.ndarray, n):

    x2 = x[:, :n]
    x3 = jnp.reshape(x2, (int(x2.shape[0] / 6), 6, n))

    return x3


a = jnp.array([i for i in range(6*3)])
x = reshape_arrays(a)
x2= insert_arrays(a,2)
#x3= return_arrays(jnp.array([i for i in range(6*4)]).reshape(), 3)


#f = jax.vmap(lambda u, v: jnp.matmul(u, v.T).T, in_axes=(0,1), out_axes=1)
f = jax.vmap(lambda u, v: jnp.matmul(v, u.T), in_axes=(0,1), out_axes=1)
u = jnp.array([jnp.eye(6) for i in range(3)])
v = jnp.arange(4*6*3).reshape((4, 3, 6))
fuv = f(u, v)

##########################################

def tilde(vector):
  """ Finds the matrix that yields the vectorial product when multiplied by another vector """
    
  tilde = jnp.array([[0.,        -vector[2], vector[1]],
                     [vector[2],  0.,       -vector[0]],
                     [-vector[1], vector[0], 0.]])
  return tilde

def tilde0010(vector):
    a = jnp.vstack([jnp.zeros((3,6)),
                    jnp.hstack([tilde(vector), jnp.zeros((3,3))])
                    ])
    return a

####################################################
def some_func(a,r1,r2):
    return a + r1 + r2

a = 0 
r1 = jnp.arange(0,3)
r2 = jnp.arange(0,3)
s = 0 
for i in range(len(r1)): 
    for j in range(len(r2)): 
        s+= some_func(a, r1[i], r2[j])
    
print(s)

func1= jax.vmap(some_func, (None, 0, None))
func2 = jax.vmap(func1, (None, None, 0))
func2(a, r1, r2).sum()

######################################################

a1 = jnp.arange(3*2*4).reshape((3,4,2))
f1 = jax.vmap(tilde0010, in_axes=1, out_axes=2)
f2 = jax.vmap(f1, in_axes=2, out_axes=3)
res1 = f2(a1)

for i in range(a1.shape[1]):
    for j in range(a1.shape[2]):
        assert (res1[3:,:3, i, j] == tilde(a1[:,i,j])).all()


# f = jax.vmap(jax.vmap(lambda u, v: jnp.matmul(v, u.T), in_axes=(0,1), out_axes=1)
# u = jnp.array([jnp.eye(6) for i in range(3)])
# v = jnp.arange(4*6*3).reshape((4, 3, 6))
# fuv = f(u, v)

###############################################
from multipledispatch import dispatch

@dispatch(str, ss=list)
def Func(s, ss=[]):
    return s


@dispatch(list, list)
def Func(l, ss=[]):
    return Func(l[0], ss=ss)

Func(["string"])  # output: string
#Func("string", [])  # calling this will raise error
Func("string", ss=[])  # output: string

#################################################

def contraction(u, v):
    
    f = jax.vmap(lambda u, v: jnp.dot(u, v),
                 in_axes=(3,1), out_axes=2)
    fuv = f(u, v)
    return fuv

def moment_force(u, v):

    f1 = jax.vmap(lambda u, v: jnp.matmul(u, v), in_axes=(2,2), out_axes=2)
    f2 = jax.vmap(f1, in_axes=(None, 3), out_axes=3)
    fuv = f2(u, v)

    return fuv
    

x6 = jnp.arange(6*6*5*5).reshape((6, 6, 5, 5))
phi = jnp.arange(4*6*5).reshape((4,6,5))
M = jnp.arange(5*5).reshape((5,5))

mf = moment_force(phi, x6)
for i in range(x6.shape[3]):
    for j in range(phi.shape[2]):
        assert (mf[:,:,j,i] == jnp.matmul(phi[:,:,j], x6[:,:,j,i])).all()



fuv = contraction(mf, M)




##################################
from functools import partial
from jax import jit
import jax
import jax.numpy as jnp

import diffrax
import numpy as np
import jaxopt
jax.config.update("jax_enable_x64", True)

def newton_raphson(F, q0, args, rtol, atol, max_steps, kappa, norm, jac=None, **kwargs):

    solver = diffrax.NewtonNonlinearSolver(rtol=rtol, atol=atol,
                                           max_steps=max_steps, kappa=kappa,
                                           norm=norm, tolerate_nonconvergence=True)
    sol = solver(F, q0, args, jac)
    return sol

@partial(jit, static_argnames=['args'])
def F(x, args):

    y = args[0] + args[1] * x +args[2] * x**2
    return y


sol = newton_raphson(F, q0=jnp.array([100.]),  args=(0,-4,2), rtol=1e-7, atol=1e-7, max_steps=100, kappa=0.0001, norm=jnp.linalg.norm)


jopt = jaxopt.ScipyRootFinding('hybr', optimality_fun=F)
sol = jopt.run(jnp.array([-1.]),  args=(0,-4,2))


import scipy.optimize
def F2(x, *args):

    y = args[0] + args[1] * x +args[2] * x**2
    return y

sol2 = scipy.optimize.root(F2, x0=np.array([1.]),  args=(0,-4,2))


#########################

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import jax.numpy as jnp


def vector_field(t, y, args):
    prey, predator = y
    α, β, γ, δ = args
    d_prey = α * prey - β * prey * predator
    d_predator = -γ * predator + δ * prey * predator
    d_y = d_prey, d_predator
    return d_y


term = ODETerm(vector_field)
solver = Tsit5()
t0 = 0
t1 = 140
dt0 = 0.1
y0 = (10.0, 10.0)
args = (0.1, 0.02, 0.4, 0.02)
saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))
sol = diffeqsolve(term, solver, t0, t1, dt0, y0,saveat=saveat, args=args)


# scan
import numpy as np
wealth_record = []
starting_wealth = 100.0
interest_factor = 1.01
num_timesteps = 100
prev_wealth = starting_wealth

for t in range(num_timesteps):
    new_wealth = prev_wealth * interest_factor
    wealth_record.append(prev_wealth)
    prev_wealth = new_wealth

wealth_record = np.array(wealth_record)

from functools import partial
import jax.lax

def wealth_at_time(prev_wealth, time, interest_factor):
    # The lax.scannable function to compute wealth at a given time.
    # your answer here
    return  (interest_factor)*prev_wealth, prev_wealth


# Comment out the import to test your answer
#from dl_workshop.jax_idioms import lax_scan_ex_1 as wealth_at_time

wealth_func = partial(wealth_at_time, interest_factor=interest_factor)
timesteps = np.arange(num_timesteps)
final, result = jax.lax.scan(wealth_func, init=starting_wealth, xs=timesteps)

assert np.allclose(wealth_record, result)
