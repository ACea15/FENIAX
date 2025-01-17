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
##################################
import feniax.intrinsic.functions as functions
import numpy as np
import time
num_modes = 80
phi = jnp.arange(num_modes*6*5).reshape((num_modes,6,5)) #Nmx6xNn
psi = jnp.arange(0,num_modes*6*5*2,2).reshape((num_modes,6,5))
phil = np.arange(num_modes*6*5).reshape((num_modes,6,5)) #Nmx6xNn
psil = np.arange(0,num_modes*6*5*2,2).reshape((num_modes,6,5))


f1 = jax.vmap(lambda u, v: jnp.tensordot(functions.L1(u), v, axes=(1, 1)),
              in_axes=(1,2), out_axes=2) #iterate nodes
#f1 = jax.vmap(lambda u, v: functions.L1(u), in_axes=(1, 2), out_axes=1)
f2 = jax.vmap(f1, in_axes=(0, None), out_axes=0) # iterate modes first tensor
st1 = time.time()
L1 = f2(phi, psi) # Nmx6xNmxNm
gamma1 = jnp.einsum('isn,jskn->ijk', phi, L1+1)
time1 = time.time() - st1

st2 = time.time()
gamma1l = np.zeros((num_modes,num_modes,num_modes))
for i in range(num_modes):
    for j in range(num_modes):
        for k in range(num_modes):
            for n in range(5):
                gamma1l[i, j, k] += phil[i,:,n].dot(functions.L1_np(phil[j, :, n]).dot(psil[k, :, n]) +1)
time2 = time.time() - st2




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
#############################

def fibonacci(n, init):

    array = jnp.arange(n-2)
    #init = jnp.array(init)
    @jax.jit
    def loop(carry, xi):

        out = carry[-1] + carry[-2]
        carry_new = jnp.array([carry[-1], out])
        return carry_new, out

    carry, out = jax.lax.scan(loop, init, array)
    return jnp.hstack([init, out])

fibo = fibonacci(8, jnp.array([1,1]))

###############################
# sliding windows
# https://github.com/google/jax/issues/3171

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

a = jnp.arange(10)
print(moving_window(a, 4))


from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window(matrix, window_shape):
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = window_shape[0]
    window_height = window_shape[1]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(-1, 2) # cartesian product => [[x,y], [x,y], ...]

    return vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)

matrix = jnp.asarray([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(moving_window(matrix, (2, 3))) # window width = 2, window height = 3

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window(matrix, window_shape):
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = window_shape[0]
    window_height = window_shape[1]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(-1, 2) # cartesian product => [[x,y], [x,y], ...]

    return vmap(lambda start: jax.lax.dynamic_slice(matrix, (start[1], start[0]), (window_height, window_width)))(starts_xy)

matrix = jnp.asarray([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(moving_window(matrix, (3, 3))) # window width = 2, window height = 3


# https://github.com/google/jax/issues/1646#issuecomment-551955947
@jax.jit
def toeplitz(x):
  if len(x.shape) == 1:
    return toeplitz(x[:,None])
  n = x.shape[-2]
  m = x.shape[-1]
  if m == n:
    return x
  if m > n:
    return x[...,:n]
  r  = jnp.roll(x,m,axis=-2)
  return toeplitz(jnp.concatenate([x,r], axis=-1))

@jax.jit
def toeplitz2(x):
  n = x.shape[-1]
  iota = jnp.arange(n)
  def roll(x,i):
    return jnp.roll(x, i, axis=-1)
  return jax.vmap(roll,in_axes=(None,0),out_axes=1)(x, iota)



###########################

class passing_class:

    def __init__(self, inp1):

        self.inp1  = inp1

    def add_attr(self, k, v):
        setattr(self, k, v)

class to_thisclass:

    def __init__(self, obj1):

        self.obj1  = obj1

    def  set_attr(self, k, v):

        self.obj1.add_attr(k, v)

pc1 = passing_class('inp')
tc1 = to_thisclass(pc1)
tc1.set_attr('new', 4)
# new also pc1


########################################################

from dataclasses import dataclass, fields, Field, replace
from typing import Type

@dataclass
class MyDataClass:
    field1: int
    field2: str

def create_frozen_dataclass(obj: MyDataClass) -> Type:
    # Extract fields from the original class
    original_fields = fields(obj)

    # Create a dictionary for the new class
    new_class_dict = {
        '__annotations__': {f.name: f.type for f in original_fields},
        '__hash__': lambda self: hash(tuple(getattr(self, f.name) for f in original_fields)),
    }

    # Add the original class as a base class
    bases = (obj.__class__,)

    # Create the new class
    new_cls = type(obj.__class__.__name__ + 'Frozen', bases, new_class_dict)

    # Set frozen attribute to True for all fields
    for field in original_fields:
        if not hasattr(field, 'frozen') or not field.frozen:
            setattr(new_cls, field.name, replace(field, frozen=True))

    return new_cls

# Create an instance of MyDataClass
obj = MyDataClass(42, "Hello")

# Use the create_frozen_dataclass function to generate FrozenMyDataClass
FrozenMyDataClass = create_frozen_dataclass(obj)

# Create an instance of the frozen dataclass
frozen_obj = FrozenMyDataClass(42, "Hello")

# Attempting to modify a frozen attribute will raise an error
try:
    frozen_obj.field1 = 10
except AttributeError as e:
    print(f"AttributeError: {e}")

# Verify that the frozen object is hashable
hash_value = hash(frozen_obj)
print(f"Hash value of frozen_obj: {hash_value}")



@dataclass(frozen=True)
class FrozenDataClass:
    a: int
    b: int
    
    def __post_init__(self):
        object.__setattr__(self, 'c', self.a + self.b)
        #self.c = self.a + self.b

d1 = FrozenDataClass(3,4)

import re

class MyClass:
    """
    MyClass is an example class.

    Attributes:
    attr1 (int): Description of attr1.
    attr2 (str): Description of attr2.
    """

    attributes = []
    # attributes = MyClass.extract_attributes(MyClass.__doc__)
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
        MyClass.extract_attributes()
        
    @classmethod
    def extract_attributes(cls):
        docstring = cls.__doc__
        attributes = []
        pattern = re.compile(r"(\w+) \((\w+)\): (.+)")
        lines = docstring.splitlines()
        in_attributes_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("Attributes:"):
                in_attributes_section = True
                continue
            if in_attributes_section:
                match = pattern.match(line)
                if match:
                    attribute_name = match.group(1)
                    attribute_type = match.group(2)
                    attribute_description = match.group(3)
                    attributes.append(f"{attribute_name} ({attribute_type}): {attribute_description}")
                elif line == "":  # Empty line signals end of attributes section
                    in_attributes_section = False
        MyClass.attributes = attributes

MyClass.extract_attributes()
for attribute in MyClass.attributes:
    print(attribute)

class MyClass:
    """
    MyClass is an example class.

    Attributes:
    attr1 (int): Description of attr1.
    attr2 (str): Description of attr2.
    """
    attributes = []
    _attributes_initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._attributes_initialized:
            cls._initialize_attributes()
        return super().__new__(cls)

    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2

    @classmethod
    def _initialize_attributes(cls):
        docstring = cls.__doc__
        attributes = []
        pattern = re.compile(r"(\w+) \((\w+)\): (.+)")
        lines = docstring.splitlines()
        in_attributes_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("Attributes:"):
                in_attributes_section = True
                continue
            if in_attributes_section:
                match = pattern.match(line)
                if match:
                    attribute_name = match.group(1)
                    attribute_type = match.group(2)
                    attribute_description = match.group(3)
                    attributes.append(f"{attribute_name} ({attribute_type}): {attribute_description}")
                elif line == "":  # Empty line signals end of attributes section
                    in_attributes_section = False

        cls.attributes = attributes
        cls._attributes_initialized = True

# Create an instance to trigger the initialization logic
instance = MyClass(attr1=10, attr2="example")

# Print the extracted attributes
for attribute in MyClass.attributes:
    print(attribute)


import re
from dataclasses import dataclass, field

class DataC:
    attributes = {}
    _attributes_initialized = False    
    def __new__(cls, *args, **kwargs):
        if not cls._attributes_initialized:
            cls._initialize_attributes()
        return super().__new__(cls)

    @classmethod
    def _initialize_attributes(cls):
        docstring = cls.__doc__
        attributes = {}
        in_attributes_section = False

        lines = docstring.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "Attributes":
                in_attributes_section = True
                continue
            if in_attributes_section and line.startswith('----'):
                continue
            if in_attributes_section:
                if line == "":  # Exit section once we reach an empty line
                    break
                # Look ahead to pick the description
                if i + 1 < len(lines) and re.match(r".+:", lines[i]):
                    name_type = line.split(':')
                    attribute_name = name_type[0].strip()
                    attribute_type = name_type[1].strip()
                    attribute_description = lines[i + 1].strip()
                    #attributes.append(f"{attribute_name} ({attribute_type}): {attribute_description}")
                    attributes[attribute_name] = attribute_description
        cls.attributes = attributes
        cls._attributes_initialized = True


@dataclass
class MyClass(DataC):
    """
    MyClass is an example class.

    Parameters
    ----------
    attr1 : int
        Description of attr1.
    attr2 : str
        Description of attr2.

    Attributes
    ----------
    attr1 : int
        Description of attr1.
    attr2 : str
        Description of attr2.
    """
    
    attr1: int #= field(metadata=MyClass.attributes['attr1'])
    attr2: str #= field(metadata=MyClass.attributes['attr2'])
    # attributes: list = field(default_factory=list, init=False)
    # _attributes_initialized: bool = field(default=False, init=False, repr=False)

# Create an instance to trigger the initialization logic
instance = MyClass(attr1=10, attr2="example")

# Print the extracted attributes
for attribute in MyClass.attributes:
    print(attribute)

##########################

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Create a Sharding object to distribute a value across devices:
mesh = Mesh(devices=mesh_utils.create_device_mesh((4, 2)),
            axis_names=('x', 'y'))

# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
jax.debug.visualize_array_sharding(y)


@jax.jit
def f(x):

    # y = jnp.ones((2, 3))
    y = x * 2 #y.at[0, 2].set(x*2)
    dy = dict(x=x, y=y)
    return dy

@jax.jit
def fshard(x):

    fvmap = jax.vmap(f)
    y = fvmap(x)
    return y

mesh = Mesh(devices=mesh_utils.create_device_mesh((8,)),
            axis_names=('x'))

# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (80,2,3))
# and use jax.device_put to distribute it across devices:
xshard = jax.device_put(x, NamedSharding(mesh, P('x')))

y = fshard(xshard)
jax.debug.visualize_array_sharding(y[:,0])

#print(y)

#################################

import jax
from jax import jit
from functools import partial


def inner_function(x, static_val):
    # The behavior of `static_val` in JIT context depends on how it was passed
    return x + static_val

def outer_function(x, static_val):
    return inner_function(x, static_val)

# JIT-compile the outer function where the second argument is treated as static
compiled_function = jit(outer_function, static_argnums=(1,))

# Call the compiled function with a static argument
result = compiled_function(3, 10)
print(result)  # Expected to print 13
