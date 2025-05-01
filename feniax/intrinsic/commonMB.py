import jax.numpy as jnp

def lambda12(F11, F12, Fr1, Fr2, Gq, GqT, Gr, GrT):

    Ginv = jnp.linalg.inv(Gq.dot(GqT)+Gr.dot(GrT))
    lagrange_multiplier = Ginv @ (Gq @ jnp.hstack([F11, F12]) +
                   Gr @ jnp.hstack([Fr1, Fr2]))
    return lagrange_multiplier

def rotation_quaternion(quaternion):

    l0,l1,l2,l3 = quaternion

    return jnp.array([[l0**2+l1**2-l2**2-l3**2, -2*l0*l3+2*l1*l2, 2*l0*l2+2*l1*l3],
                      [2*l0*l3+2*l1*l2, l0**2-l1**2+l2**2-l3**2, -2*l0*l1+2*l2*l3],
                      [-2*l0*l2+2*l1*l3, 2*l0*l1+2*l2*l3, l0**2-l1**2-l2**2+l3**2]])
    
