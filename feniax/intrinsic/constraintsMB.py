import jax.numpy as jnp
import feniax.intrinsic.commonMB as commonMB


def Cvelocity():
    ...
    

def Gvz(quaternion, linear_velocity):
    l0,l1,l2,l3 = quaternion
    v1,v2,v3 = linear_velocity

    return jnp.array([[2*l0*v1+2*l2*v3-2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v3+2*l1*v2-2*l2*v1, -2*l0*v2+2*l1*v3-2*l3*v1],
                      [2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v3-2*l1*v2+2*l2*v1, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v1+2*l2*v3-2*l3*v2],
                      [2*l0*v3+2*l1*v2-2*l2*v1, 2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v1-2*l2*v3+2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3]])


def Gspherical(q1_b1,
               q1_b2,
               qr_b1,
               qr_b2,
               Phi1_b1,
               Phi1_b2) -> list[jnp.ndarray, jnp.ndarray]:

    velocity_b1 = jnp.tensordot(Phi1_b1[:,:,0], q1_b1, axes=(0, 0)) # (6)
    velocity_b2 = jnp.tensordot(Phi1_b2[:,:,-1], q1_b2, axes=(0, 0))     
    Phi1u_b1 = Phi1_b1[:,:3, 0]
    Phi1u_b2 = Phi1_b1[:,:3, -1]    
    R1 = commonMB.rotation_quaternion(qr_b1)
    R2 = commonMB.rotation_quaternion(qr_b2)
    Gvq1 = jnp.tensordot(R1,  Phi1u_b1, axes=(1, 0)) #(3, Nm1)
    Gvq2 = jnp.tensordot(-R2, Phi1u_b2, axes=(1, 0)) #(3, Nm2) -R2.dot(Phi1u_b2)
    Gvr1 = Gvz(qr_b1,
               velocity_b1[:3])
    Gvr2 = -Gvz(qr_b2,
                velocity_b2[:3])
    Gq = jnp.hstack([Gvq1, Gvq2])
    Gr = jnp.hstack([Gvr1, Gvr2])
    return [Gq, Gr]

def Ghinge():
    #G1 = jnp.vstack([jnp.hstack([Gvq1,Gvq2]),
    #                 jnp.hstack([Gxq1, Gxq2])])
    ...

    
CONSTRAINT_DICT = dict(spherical=Gspherical,
                       hinge=Ghinge)
    
