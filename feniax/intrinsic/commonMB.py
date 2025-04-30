def lambda12(F11, Fz1, F12, Fz2, G1, G1T, G2, G2T):

    Ginv = jnp.linalg.inv(G1.dot(G1T)+G2.dot(G2T))
    lagrange_multiplier = Ginv @ (G1 @ jnp.hstack([F11, Fz1]) +
                   G2 @ jnp.hstack([F11, Fz2]))
    return lagrange_multiplier

def rotation_quaternion(quaternion):

    l0,l1,l2,l3 = quaternion

    return jnp.array([[l0**2+l1**2-l2**2-l3**2, -2*l0*l3+2*l1*l2, 2*l0*l2+2*l1*l3],
                      [2*l0*l3+2*l1*l2, l0**2-l1**2+l2**2-l3**2, -2*l0*l1+2*l2*l3],
                      [-2*l0*l2+2*l1*l3, 2*l0*l1+2*l2*l3, l0**2-l1**2-l2**2+l3**2]])
    
