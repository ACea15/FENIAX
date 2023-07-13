import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, lax
from fem4inas.preprocessor import config
from functools import partial

###############################
# Exponential map integration #
###############################

def H0(Itheta,Ipsi):

  I3=np.eye(3)
  if Itheta==0.0:

   return I3

  else:

   return I3+np.sin(Itheta)/Itheta*tilde(Ipsi)+(1-np.cos(Itheta))/(Itheta**2)*(tilde(Ipsi).dot(tilde(Ipsi)))


def H1(Itheta,Ipsi,ds):

  I3=np.eye(3)
  if Itheta==0.0:

   return I3*ds

  else:

   return ds*(I3+(1-np.cos(Itheta))/(Itheta**2)*tilde(Ipsi)+(Itheta-np.sin(Itheta))/(Itheta**3)*(tilde(Ipsi).dot(tilde(Ipsi))))


def tilde(vector):
  """ Finds the matrix that yields the vectorial product when multiplied by another vector """
    
  tilde = jnp.array([[0.,        -vector[2], vector[1]],
                     [vector[2],  0.,       -vector[0]],
                     [-vector[1], vector[0], 0.]])
  return tilde


def L1(x1):

   L1= np.zeros((6,6))
   #v=np.zeros(3)
   #w=np.zeros(3)
   v=x1[0:3]
   w=x1[3:6]

   L1[0:3,0:3]=  [[ 0    ,-w[2]  , w[1]],\
                 [ w[2]  , 0     ,-w[0]],\
                 [-w[1]  ,  w[0] ,  0 ]]


   L1[3:6,3:6]=L1[0:3,0:3]

   L1[3:6,0:3]= [[ 0    ,-v[2]  , v[1]],\
                [ v[2]  , 0     ,-v[0]],\
                [-v[1]  ,  v[0] ,  0 ]]

   return np.asarray(L1)

@jit
def L2fun(x2):

   L2 = jnp.zeros((6,6))
   #f=np.zeros(3)
   #m=np.zeros(3)
   f = x2[0:3]
   m = x2[3:6]
   L2 = L2.at[3:6,3:6].set(jnp.array([[ 0    ,-m[2]  , m[1]],
                                      [ m[2]  , 0     ,-m[0]],
                                      [-m[1]  ,  m[0] ,  0 ]]))

   L2 = L2.at[0:3,3:6].set(jnp.array([[ 0    ,-f[2] , f[1]],
                                      [ f[2] , 0    ,-f[0]],
                                      [-f[1],  f[0] ,  0 ]]))

   L2 = L2.at[3:6,0:3].set(L2[0:3,3:6])

   return L2

@partial(jit, static_argnames=['config'])
def compute_C0ab(X_diff: jnp.ndarray, X_xdelta: jnp.ndarray,
                 config: config.Config) -> jnp.ndarray:

    x = X_diff / X_xdelta
    x = x.at[:, 0].set(jnp.array([1, 0, 0])) # WARNING: this says the first node FoR at time 0
    # aligns with the global reference frame.
    cond = jnp.linalg.norm(x - jnp.array([0,0,1])) > config.ex.Cab_xtol # if not true,
    # local-x is almost parallel to global-z, z direction parallel to global y
    y = lax.select(cond,
                   jnp.cross(jnp.array([0, 0, 1]), x, axisb=0, axisc=0),
                   jnp.cross(jnp.array([0, 1, 0]), x, axisb=0, axisc=0))
    z = jnp.cross(x, y, axisa=0, axisb=0, axisc=0)
    C0ab = jnp.stack([x, y, z], axis=1)
    return C0ab

