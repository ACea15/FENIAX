import numpy as np

def transform_Q(q,theta):
  """
  calculate rotation matrix for composite material
  theta : (nparam,)
  """
  t=np.zeros((theta.shape[0],3,3))
  t[:,0,0]=np.cos(theta)**2
  t[:,0,1]=np.sin(theta)**2
  t[:,1,0]=np.sin(theta)**2
  t[:,0,2]=2*np.cos(theta)*np.sin(theta)
  t[:,1,1]=np.cos(theta)**2
  t[:,1,2]=-2*np.cos(theta)*np.sin(theta)
  t[:,2,0]=-np.cos(theta)*np.sin(theta)
  t[:,2,1]=np.cos(theta)*np.sin(theta) 
  t[:,2,2]=np.cos(theta)**2-np.sin(theta)**2
  invT=np.linalg.inv(t) # (nparam,3,3)
  q_new=invT@q@invT.transpose(0,2,1) # (nparam,3,3)
  return q_new

def gmat_from_alpha_and_theta(alpha,theta,qmat_ref):
  """
  qmat_ref : (nparam,3,3)
  alpha : (nparam,)
  theta : (nparam,)
  """
  _pi_2=np.array([np.pi/2]).reshape(1,1)
  q_temp=0.25*(qmat_ref+transform_Q(qmat_ref,_pi_2)+
               transform_Q(qmat_ref,alpha)+
               transform_Q(qmat_ref,-alpha))
  gmat=transform_Q(q_temp,theta)
  return gmat
  