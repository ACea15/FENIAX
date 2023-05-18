import numpy as np
from functools import wraps
#===================================================================================================================
def beamsectioncoeff(totals,compmat):


    # Feed in compliance matrix and total length of beam section
    global EMAT
    EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])

    # ems=[0 0 00 0 -10 1 0]

    # test [010001]
    # testv=[010001]
    # total s =2
    # Goes to pure shear stress at mid s=1

    # Define force at BASE
    # x2'+Ex2 = 0
    # COLUMN vectors representing x2_0
    # Unity disturbance in each variable
    x20=np.eye(6)
    # COLUMN vectors representing x2_d
    # x2=x20+s*x2s
    x2s=-EMAT.dot(x20)
    #
    # compmat=eye(6)
    # compmat=diag([10.00010.0001111])
    # gm=cx2=cx20+s*cx2s
    cx20=compmat.dot(x20)
    cx2s=compmat.dot(x2s)
    # COLUMN vectors
    # rotation=int(kp)ds
    # rt=cx20*s+cx2d*s*s/2
    # rt=s*rt0+s*s*rts
    # wrt s=0
    rt0=cx20[3:6,:]
    rts=cx2s[3:6,:]/2
    # disp=int(gm)ds+int(rt)ds
    # disp_gm=s*cx20++cx2d*s*s/2
    # disp_rt=rt0~*s*s/2+rts~*s*s*s/3
    # disp=s*cx20+s*s*(cx2d/2+rt0~/2)+s*s*s*(rts~/3)
    dp0=cx20[0:3,:]
    # dps=cx2s(1:3,:)/2+[zeros(1,6)-rt0(3,:)rt0(2,:)]/2
    # dpss=[zeros(1,6)-rts(3,:)rts(2,:)]/3
    dps=cx2s[0:3,:]/2+np.vstack(((np.zeros((1,6)),rt0[2,:],-rt0[1,:])))/2
    dpss=np.vstack((np.zeros((1,6)),rts[2,:],-rts[1,:]))/3

    dpt=totals*dp0+totals*totals*dps+totals*totals*totals*dpss
    rtt=totals*rt0+totals*totals*rts
    totaldpmats=np.vstack((dpt,rtt))
    # tinv defines displacement to force
    tinv=np.linalg.inv(totaldpmats)
    # tinv*tipdisp=baseforce
    # However base used to balance tip, therefore minus
    f0mat=-tinv
    # tip force uses base force and gradient E
    fsmat=tinv-EMAT.dot(tinv).dot(totals)
    # These are EXTERNAL APPLIED FORCES which will cause the SAID DISPLACEMENT
    # at the TIP

    # for a negative totalS, in case displacement is applied at the other side,
    # ie s is -total
    # while still fixed at s=0
    dptr=-totals*dp0+totals*totals*dps-totals*totals*totals*dpss
    rttr=-totals*rt0+totals*totals*rts
    totaldpmatsr=np.vstack((dptr,rttr))
    # tinv defines displacement to force
    tinvr=np.linalg.inv(totaldpmatsr)
    # Note this time the reversed order of start and finish (ie start at -total
    # and finish at 0)
    fsmatr=tinvr
    f0matr=-(tinvr+EMAT.dot(tinvr).dot(totals))

    return f0mat,fsmat,f0matr,fsmatr
#===============================================================================================================================================================

def Rot6(R):
  """ Applies the rotation of a 3by3 rotation matrix to to a 6-component vector """
  z=np.zeros((3,3))
  R6=np.vstack((np.hstack((R,z)),np.hstack((z,R))))

  return R6
#==============================================================================================================================================================

def Matrix_rotation6(v,R):

    return(Rot6(R).dot(v))

def L1fun(x1):

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
#===========================================================================================================================================================

def L2fun(x2):

   L2= np.zeros((6,6))
   #v=np.zeros(3)
   #w=np.zeros(3)
   f=x2[0:3]
   m=x2[3:6]

   L2[3:6,3:6]=  [[ 0    ,-m[2]  , m[1]],\
                 [ m[2]  , 0     ,-m[0]],\
                 [-m[1]  ,  m[0] ,  0 ]]


   L2[0:3,3:6]= [[ 0    ,-f[2] , f[1]],\
                [ f[2] , 0    ,-f[0]],\
                [-f[1],  f[0] ,  0 ]]

   L2[3:6,0:3]=L2[0:3,3:6]

   return np.asarray(L2)
#============================================================================================================================================================

def NormaltoPlane(a,b):
  " Finds the unit vector perpendicular to 2 given vectors"
  c=np.cross(a,b)
  if np.linalg.norm(c)/(np.linalg.norm(a)*np.linalg.norm(b)) < 1e-5:
      raise ValueError('NormaltoPlane function applied to parallel vectors')
  else:
      c=c/np.linalg.norm(c)
  return c
#============================================================================================================================================================

#============================================================================================================================================================
def BaseO(a,b):
  """ Finds the 3D orthogonal base for 2 perpendicular vectors given"""

  ax=a/np.linalg.norm(a)
  ay=b/np.linalg.norm(b)
  az=np.cross(ax,ay)
  G=np.asarray([ax,ay,az]).T
  return G

#============================================================================================================================================================
def Base(a,b):
  """ Finds the 3D orthogonal base for 2 non-parallel vectors given"""
  ax=a/np.linalg.norm(a)
  az=NormaltoPlane(a,b)
  ay=np.cross(az,ax)
  G=np.asarray([ax,ay,az]).T
  return G

def Base2(a,b):
  """ Finds the 3D orthogonal base for 2 non-parallel vectors given"""
  ax=a/np.linalg.norm(a)
  az=-NormaltoPlane(a,b)
  ay=np.cross(az,ax)
  G=np.asarray([ax,ay,az]).T
  return G

def tilde(a):
  """ Finds the matrix that yields the vectorial product when multiplied by another vector """
  at=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
  return at


# Exponential map approach
#=========================================================================================================================================

#H functions
#=======================================================

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


print('Read Functions')


# ======================================================================================================
# Time Discretization
#======================================================================================================

def time_def(t0,tf,tn):
    dt=(tf-t0)/(tn-1)
    time=np.linspace(t0,tf,tn)

    return dt,time




def mode_classification(Phi0,BeamSeg,Omega,NumBeams,NumNodes,NumModes,err):

    #import pdb
    Mbendxy=[]
    Mbendxz=[]
    Mtorsion=[]
    Maxial=[]
    for k in range(NumModes):
        bxy=0;bxz=0;btor=0;baxial=0
        for i in range(NumBeams):
         for nodex in range(BeamSeg[i].EnumNodes):
           #pdb.set_trace()
           bxy=bxy+abs(Phi0[i][k][nodex].dot(np.array([0,1,0,0,0,0])))+ abs(Phi0[i][k][nodex].dot(np.array([0,0,0,0,0,1])))
           bxz=bxz+abs(Phi0[i][k][nodex].dot(np.array([0,0,1,0,0,0])))+ abs(Phi0[i][k][nodex].dot(np.array([0,0,0,0,1,0])))
           btor=btor+abs(Phi0[i][k][nodex].dot(np.array([0,0,0,1,0,0])))
           baxial=baxial+abs(Phi0[i][k][nodex].dot(np.array([1,0,0,0,0,0])))

        bxy=bxy/NumNodes
        bxz=bxz/NumNodes
        btor=btor/NumNodes
        baxial=baxial/NumNodes
        X=[0,0,0,0]
        if bxy>err:
            Mbendxy.append([k,Omega[k]])
            X[0]=1
        if bxz > err:
            Mbendxz.append([k,Omega[k]])
            X[1]=1
        if btor > err:
            Mtorsion.append([k,Omega[k]])
            X[2]=1
        if baxial > err:
            Maxial.append([k,Omega[k]])
            X[3] = 1
     #pdb.set_trace()
     #print(X)

        if sum(X)==0:
         print('Not selection found for mode %s'%k)
        elif sum(X)>1:
         print('More than one selection for mode %s'%k)

    return(Mbendxy,Mbendxz,Mtorsion,Maxial)


def parameters(param,q0,BeamSeg,NumBeams,BeamsClamped,Clamped,Omega,gamma1,gamma2,eta,t0,dt,tf,tn,integrator,jacobian):

  param['BeamSeg'] = BeamSeg
  param['NumBeams'] = NumBeams
  param['BeamsClamped'] = BeamsClamped
  param['Clamped'] = Clamped
  param['args']=[Omega,gamma1,gamma2,eta]
  param['t0']=t0
  param['q0']=q0
  param['dt']=dt
  param['tf']=tf
  param['tn']=tn
  param['integrator']=integrator
  param['jacobian']=jacobian


def dic(**kwargs):
    return kwargs


def fx(nm):
  comp=2*nm**3
  comp_opt=nm**2*(nm+1)/2+nm*(nm+1)/2

  return float(comp_opt)/float(comp)

def my_timer(originalF):
    import time
    #@wraps(originalF)
    def wrapper(*args,**kwargs):
        t1 = time.time()
        result = originalF(*args,**kwargs)
        t2 = time.time() - t1
        print('{} ran in {} sec.'.format(originalF.__name__,t2))
        return result
    return wrapper
