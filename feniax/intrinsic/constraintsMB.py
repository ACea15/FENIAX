import jax.numpy as jnp
import feniax.intrinsic.commonMB as commonMB


def Cvelocity():

    

def Gvz(self,l,v):
    l0,l1,l2,l3 = l
    v1,v2,v3 = v

    return jnp.array([[2*l0*v1+2*l2*v3-2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v3+2*l1*v2-2*l2*v1, -2*l0*v2+2*l1*v3-2*l3*v1],
                      [2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v3-2*l1*v2+2*l2*v1, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v1+2*l2*v3-2*l3*v2],
                      [2*l0*v3+2*l1*v2-2*l2*v1, 2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v1-2*l2*v3+2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3]])


def Gspherical(q1_b1, q1_b2, qz_b1, qz_b2,
               Phi1_b1, Phi1_b2) -> list[jnp.ndarray, jnp.ndarray]:

    velocity_b1, velocity_b2 = 0
    Phi1u_b1 = 0
    Phi1u_b2 = 0
    
    R1 = commonMB.rotation_quaternion(quaternion_b1)
    R2 = commonMB.rotation_quaternion(quaternion_b2)
    Gvq1 = R1.dot(Phi1u_b1)
    Gvq2 = -R2.dot(Phi1u_b2)
    Gvz2 = Gvz(quaternion_b1,
               velocity_b1)
    Gvz2 = -Gvz(quaternion_b2,
                velocity_b2)
    G1 = jnp.hstack([Gvq1, Gvz1])
    G2 = jnp.hstack([Gvq2, Gvz2])
    return [G1, G2]

def Ghinge():
    #G1 = jnp.vstack([jnp.hstack([Gvq1,Gvq2]),
    #                 jnp.hstack([Gxq1, Gxq2])])
    ...

    
CONSTRAINT_DICT = dict(spherical=Gspherical,
                       hinge=Ghinge)
    
