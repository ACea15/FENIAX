import jax.numpy as jnp
import numpy as np
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsicmodal
import jax

def init_X1(modes, X1i: jnp.ndarray,
            *args, **kwags) -> jnp.ndarray:

    phi1 = modes.phi1
    coord_length = np.prod(X1i.shape)
    X1_reshaped = X1i.reshape(coord_length)
    num_modes = len(phi1)
    phi1_reshaped = phi1.reshape((num_modes, coord_length))
    q1 = jnp.linalg.lstsq(phi1_reshaped.T, X1_reshaped)
    return q1

def init_X2(modes, X2i: jnp.ndarray,
            *args, **kwags) -> jnp.ndarray:

    phi2 = modes.phi2
    coord_length = np.prod(X2i.shape)
    X2_reshaped = X2i.reshape(coord_length)
    num_modes = len(phi2)
    phi2_reshaped = phi2.reshape((num_modes, coord_length))
    q2 = jnp.linalg.lstsq(phi2_reshaped.T, X2_reshaped)
    return q2



mapper = dict(velocity=init_X1, # given a velocity field, and phi1, output q10s
              force=init_X2,
              displacement=None) #Placeholder


class Container:

    @staticmethod
    def axial_parabolic(x0: jnp.array,
                        L: float,
                        fem:intrinsicmodal.Dfem,
                        *args, **kwags):

        X0 = fem.X[0]
        f = jax.vmap(lambda Xi: (jnp.linalg.norm(Xi - X0) / L)**2,
                     in_axes=0, out_axes=0)
        s = f(fem.X)
        s = s.reshape((len(s), 1))
        x0 = jnp.array(x0).reshape((6, 1)) 
        return jnp.tensordot(x0, s, axes=(1, 1)) #6xNn


        
    

    

