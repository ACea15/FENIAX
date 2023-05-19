import pdb
import numpy as np
from scipy.optimize import fsolve


import intrinsic.geometry
from intrinsic.functions import tilde,H0,H1
from intrinsic.functions import my_timer
import intrinsic.qsolvers
reload(intrinsic.qsolvers)
from intrinsic.qsolvers import Qsol,Qsol2
from intrinsic.Tools.transformations import rotation_matrix,rotation_from_matrix
from Tools.ODE import RK4
import importlib
import Runs.Torun
V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)

BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)





#====================================================================================================================================================
  # Recovery of displacements
#====================================================================================================================================================


# Curvatures Calculations
#============================================================================================================================================


def strain_def(curvature,q2,CPhi2,V=V,BeamSeg=BeamSeg):

    strain=[np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)]
    for k in range(V.NumModes-V.NumModes_res):
      for i in range(V.NumBeams):

            for j in range(BeamSeg[i].EnumNodes-1):
                #strain[i][j] = strain[i][j] + BeamSeg[i].GlobalAxes.dot(CPhi2[i][k][j][curvature*3:curvature*3+3])*q2[k]
                strain[i][j] = strain[i][j] + CPhi2[i][k][j][curvature*3:curvature*3+3]*q2[k]
    return strain



def integration_strains2(kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn,rai=[],Rabi=[]):
    I3=np.eye(3)
    e_1=np.array([1,0,0])
    def IRab(Rab,Itheta,Ipsi):
      return Rab.dot(H0(Itheta,Ipsi))

    def Ira(ra,Rab,Itheta,Ipsi,DL,strain):
      return ra + Rab.dot(H1(Itheta,Ipsi,DL)).dot(strain)

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]

    for i in range(V.NumBeams):

        ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if V.Clamped and i in V.BeamsClamped:
            ra[i][0]=BeamSeg[i].NodeX[0]
            Rab[i][0]=BeamSeg[i].GlobalAxes
        elif not V.Clamped and i==0:
            ra[i][0] = rai
            Rab[i][0]=Rabi
        else:
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            Rab[i][0] = BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T.dot(Rab[k][-1]))
            BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T.dot(Rab[k][-1]))
            #Rab[i][0] = Rab[k][-1].dot(BeamSeg[k].GlobalAxes.T.dot(BeamSeg[i].GlobalAxes))


        for j in range(1,BeamSeg[i].EnumNodes):
          Ipsi=(kappa[i][j-1])*BeamSeg[i].NodeDL[j-1]
          Itheta=np.linalg.norm(Ipsi)
          Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)
          ra[i][j] = Ira(ra[i][j-1],Rab[i][j-1],Itheta,Ipsi,BeamSeg[i].NodeDL[j-1],strain[i][j-1]+e_1)

    return(ra,Rab)

def integration_strains(kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn,rai=[],Rabi=[]):
    I3=np.eye(3)
    e_1=np.array([1,0,0])
    def IRab(Rab,Itheta,Ipsi):
      return Rab.dot(H0(Itheta,Ipsi))

    def Ira(ra,Rab,Itheta,Ipsi,DL,strain):
      return ra + Rab.dot(H1(Itheta,Ipsi,DL)).dot(strain)

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]

    for i in range(V.NumBeams):
        #pdb.set_trace()
        ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if V.Clamped and i in V.BeamsClamped:
            ra[i][0]=BeamSeg[i].NodeX[0]
            Rab[i][0]=BeamSeg[i].GlobalAxes
        elif not V.Clamped and i==0:
            ra[i][0] = rai
            Rab[i][0]=Rabi
        else:
            #pdb.set_trace()
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            angle, direction = rotation_from_matrix(BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T))
            #direction_new = (Rab[k][-1].T).dot(BeamSeg[k].GlobalAxes.dot(direction))
            direction_new = (Rab[k][-1]).dot(BeamSeg[k].GlobalAxes.T.dot(direction))
            Rab[i][0] = rotation_matrix(angle,direction_new)[:3,:3].dot(Rab[k][-1])
            #Rab[i][0] = BeamSeg[i].GlobalAxes.T.dot(BeamSeg[k].GlobalAxes).dot(Rab[k][-1])

        for j in range(1,BeamSeg[i].EnumNodes):
          Ipsi=(kappa[i][j-1])*BeamSeg[i].NodeDL[j-1]
          Itheta=np.linalg.norm(Ipsi)
          Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)
          ra[i][j] = Ira(ra[i][j-1],Rab[i][j-1],Itheta,Ipsi,BeamSeg[i].NodeDL[j-1],strain[i][j-1]+e_1)

    return(ra,Rab)


@my_timer
def integration_strain_time(q2,CPhi2,V,BeamSeg,inverseconn,X1=[]):

    strain = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    kappa = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    ra = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    Rab = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3)) for i in range(V.NumBeams)]

    if not V.Clamped:
        rai,Rabi = integration_velocities_i(X1,V,0,0)
    for ti in range(V.tn):

        strainx = strain_def(0,q2[ti],CPhi2,V,BeamSeg)
        kappax = strain_def(1,q2[ti],CPhi2,V,BeamSeg)
        if V.Clamped:
            rax,Rabx = integration_strains(kappax,strainx,V,BeamSeg,inverseconn)
        else:
            rax,Rabx = integration_strains(kappax,strainx,V,BeamSeg,inverseconn,rai=rai[ti],Rabi=Rabi[ti])
        for i in range(V.NumBeams):
          strain[i][:,ti] = strainx[i]
          kappa[i][:,ti] = kappax[i]
          ra[i][:,ti] = rax[i]
          Rab[i][:,ti] = Rabx[i]

    return(strain,kappa,ra,Rab)



def integration_strainsrem(kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn,rai=[],Rabi=[]):
    I3=np.eye(3)
    e_1=np.array([1,0,0])
    def IRab(Rab,Itheta,Ipsi):
      return Rab.dot(H0(Itheta,Ipsi))

    def Ira(ra,Rab,Itheta,Ipsi,DL,strain):
      return ra + Rab.dot(H1(Itheta,Ipsi,DL)).dot(strain)

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]
    di=0
    for i in range(V.NumBeams):
        #pdb.set_trace()
        ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if  i in V.BeamsClamped:
            ra[i][0]=BeamSeg[i].NodeX[0]
            Rab[i][0]=BeamSeg[i].GlobalAxes
        elif (i in V.initialbeams) and  (i not in V.MBbeams):
            ra[i][0] = rai[di]
            Rab[i][0] = Rabi[di]
            di+=1
        elif (i not in V.initialbeams) and  (i in V.MBbeams):
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            Rab[i][0] = Rabi[di]
            di+=1
        elif (i in V.initialbeams) and  (i  in V.MBbeams):
            ra[i][0] = rai[di]
            Rab[i][0] = Rabi[di]
            di+=1
        else:
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            angle, direction = rotation_from_matrix(BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T))
            #direction_new = (Rab[k][-1].T).dot(BeamSeg[k].GlobalAxes.dot(direction))
            direction_new = (Rab[k][-1]).dot(BeamSeg[k].GlobalAxes.T.dot(direction))
            Rab[i][0] = rotation_matrix(angle,direction_new)[:3,:3].dot(Rab[k][-1])
            #Rab[i][0] = BeamSeg[i].GlobalAxes.T.dot(BeamSeg[k].GlobalAxes).dot(Rab[k][-1])

        for j in range(1,BeamSeg[i].EnumNodes):
          Ipsi=(kappa[i][j-1])*BeamSeg[i].NodeDL[j-1]
          Itheta=np.linalg.norm(Ipsi)
          Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)
          ra[i][j] = Ira(ra[i][j-1],Rab[i][j-1],Itheta,Ipsi,BeamSeg[i].NodeDL[j-1],strain[i][j-1]+e_1)

    return(ra,Rab)


def integration_strains_rot(kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn,Rabi=[]):
    
    I3=np.eye(3)
    e_1=np.array([1,0,0])
    def IRab(Rab,Itheta,Ipsi):
      return Rab.dot(H0(Itheta,Ipsi))

    Rab=[[] for i in range(V.NumBeams)]
    di=0
    for i in range(V.NumBeams):

        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if  i in V.BeamsClamped:
            Rab[i][0]=BeamSeg[i].GlobalAxes
        elif (i in V.initialbeams) or  (i in V.MBbeams):
            Rab[i][0] = Rabi[di]
            di+=1
        else:
            k=inverseconn[i]
            angle, direction = rotation_from_matrix(BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T))
            #direction_new = (Rab[k][-1].T).dot(BeamSeg[k].GlobalAxes.dot(direction))
            direction_new = (Rab[k][-1]).dot(BeamSeg[k].GlobalAxes.T.dot(direction))
            Rab[i][0] = rotation_matrix(angle,direction_new)[:3,:3].dot(Rab[k][-1])
            #Rab[i][0] = BeamSeg[i].GlobalAxes.T.dot(BeamSeg[k].GlobalAxes).dot(Rab[k][-1])

        for j in range(1,BeamSeg[i].EnumNodes):
          Ipsi=(kappa[i][j-1])*BeamSeg[i].NodeDL[j-1]
          Itheta=np.linalg.norm(Ipsi)
          Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)

    return(Rab)


def integration_strain_timerm(q2,CPhi2,V,BeamSeg,inverseconn,X1=[]):

    strain = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    kappa = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    ra = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    Rab = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3)) for i in range(V.NumBeams)]
    rai = [] ; Rabi = []
    for i in range(V.NumBeams):
        if (i not in V.BeamsClamped and i in V.initialbeams):
            raix,Rabix = integration_velocities_i(X1,V,i,0)
            rai.append(raix) ; Rabi.append(Rabix)
        elif  (i in V.MBbeams):
            #raix,Rabix = integration_velocities_i(X1,V,inverseconn[i],-1)
            raix,Rabix = integration_velocities_i(X1,V,i,0)
            raix2 = inverseconn[i]
            rai.append(raix) ; Rabi.append(Rabix)
    rai = np.array(rai)
    Rabi = np.array(Rabi)
    for ti in range(V.tn):

        strainx = strain_def(0,q2[ti],CPhi2,V,BeamSeg)
        kappax = strain_def(1,q2[ti],CPhi2,V,BeamSeg)

        rax,Rabx = integration_strainsrem(kappax,strainx,V,BeamSeg,inverseconn,rai=rai[:,ti],Rabi=Rabi[:,ti])
        for i in range(V.NumBeams):
          strain[i][:,ti] = strainx[i]
          kappa[i][:,ti] = kappax[i]
          ra[i][:,ti] = rax[i]
          Rab[i][:,ti] = Rabx[i]

    return(strain,kappa,ra,Rab)



# Rotation matrix derivative in space

def dRab_s(s,Rab,args1):

    kappa=args1['kappa']
    Rab_sh = np.reshape(Rab,(3,3))
    dRab_sh = Rab_sh.dot(tilde(kappa))
    dRab=np.reshape(dRab_sh,9)
    return(dRab)


#  Displacements derivative (space)
def dra_s(s,ra,args1):
  strain = args1['strain']
  Rab = args1['Rab']
  Rab1 = args1['Rab1']
  L = args1['L']
  dL = args1['dL']
  e_1 = np.asarray([1,0,0])
  dra = ((Rab*(L+dL-s)+Rab1*(s-L))/dL).dot(strain + e_1)
  return dra

def Matintegration_strains(kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn):

    param={}
    I3=np.eye(3)
    e_1=np.array([1,0,0])

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]

    for i in range(V.NumBeams):

        ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if V.Clamped and i in V.BeamsClamped:
            ra[i][0]=BeamSeg[i].NodeX[0]
            Rab[i][0]=BeamSeg[i].GlobalAxes
        elif not V.Clamped and i==0:
            ra[i][0] = BeamSeg[0].NodeX[0]
            Rab[i][0]=BeamSeg[i].GlobalAxes
        else:
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            Rab[i][0] = BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T.dot(Rab[k][-1]))
            #Rab[i][0] = Rab[k][-1].dot(BeamSeg[k].GlobalAxes.T.dot(BeamSeg[i].GlobalAxes))

        for j in range(1,BeamSeg[i].EnumNodes):
          param['kappa'] = kappa[i][j]
          param['strain'] = strain[i][j]
          Rab[i][j] = np.reshape(RK4(dRab_s,BeamSeg[i].NodeL[j-1],np.reshape(Rab[i][j-1],9),BeamSeg[i].NodeDL[j-1],args1=param),(3,3))
          param['Rab'] = Rab[i][j-1]
          param['Rab1'] = Rab[i][j]
          param['L'] = BeamSeg[i].NodeL[j-1]
          param['dL'] = BeamSeg[i].NodeDL[j-1]
          ra[i][j] = RK4(dra_s,BeamSeg[i].NodeL[j-1],ra[i][j-1],BeamSeg[i].NodeDL[j-1],args1=param)

    return(ra,Rab)
@my_timer
def Matintegration_strain_time(q2,CPhi2,V,BeamSeg,inverseconn):

    strain = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    kappa = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    ra = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
    Rab = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3)) for i in range(V.NumBeams)]

    for ti in range(V.tn):

        strainx = strain_def(0,q2[ti],CPhi2,V,BeamSeg)
        kappax = strain_def(1,q2[ti],CPhi2,V,BeamSeg)
        rax,Rabx = Matintegration_strains(kappax,strainx,V,BeamSeg,inverseconn)
        for i in range(V.NumBeams):
          strain[i][:,ti] = strainx[i]
          kappa[i][:,ti] = kappax[i]
          ra[i][:,ti] = rax[i]
          Rab[i][:,ti] = Rabx[i]

    return(strain,kappa,ra,Rab)


def solX(Phi1,Phi2,q1,q2,V,BeamSeg):

    X1=[[] for i in range(V.NumBeams)]
    X2=[[] for i in range(V.NumBeams)]


    for i in range(V.NumBeams):
            X1[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,6))
            X2[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,6))
            for ti in range(V.tn):
                    #for j in range(BeamSeg[i].EnumNodes):
                        for k in range((V.NumModes-V.NumModes_res)):

                            X1[i][:,ti]= X1[i][:,ti] + Phi1[i][k]*q1[ti][k]
                            X2[i][:,ti]= X2[i][:,ti] + Phi2[i][k]*q2[ti][k]

    return(X1,X2)

# @my_timer
# def integration_velocities(X1,V):

#     ra=[[] for i in range(V.NumBeams)]
#     Rab=[[] for i in range(V.NumBeams)]

#     for i in range(V.NumBeams):
#         ra[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,3))
#         Rab[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3))
#         ra[i][:,0] = BeamSeg[i].NodeX
#         Rab[i][:,0] = BeamSeg[i].GlobalAxes
#         for ti in range(1,V.tn):
#           for j in range(BeamSeg[i].EnumNodes):
#               Ipsi=((X1[i][j][ti][3:6]+X1[i][j][ti-1][3:6])/2)*V.dt
#               Itheta=np.linalg.norm(Ipsi)
#               Rab[i][j][ti] = Rab[i][j][ti-1].dot(H0(Itheta,Ipsi))
#               #pdb.set_trace()
#               ra[i][j][ti] = ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)

#     return  (ra,Rab)

def integration_velocities_i(X1,V,i,j):

        rai=np.zeros((V.tn,3))
        Rabi=np.zeros((V.tn,3,3))
        rai[0] = BeamSeg[i].NodeX[j]
        Rabi[0] = BeamSeg[i].GlobalAxes
        for ti in range(1,V.tn):
              Ipsi=((X1[i][j][ti][3:6]+X1[i][j][ti-1][3:6])/2)*V.dt
              Itheta=np.linalg.norm(Ipsi)
              Rabi[ti] = Rabi[ti-1].dot(H0(Itheta,Ipsi))
              rai[ti] = rai[ti-1]+Rabi[ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)

        return(rai,Rabi)
def integration_velocities(X1,V,BeamSeg):

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]

    for i in range(V.NumBeams):
        ra[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3))
        ra[i][:,0] = BeamSeg[i].NodeX
        Rab[i][:,0] = BeamSeg[i].GlobalAxes
        for ti in range(1,V.tn):
          for j in range(BeamSeg[i].EnumNodes):
              Ipsi=((X1[i][j][ti][3:6]+X1[i][j][ti-1][3:6])/2)*V.dt
              Itheta=np.linalg.norm(Ipsi)
              Rab[i][j][ti] = Rab[i][j][ti-1].dot(H0(Itheta,Ipsi))
              #pdb.set_trace()
              if (i in V.MBbeams) and (i not in V.initialbeams) and (j==0):
                  
                  diff_ra = ra[inverseconn[i]][-1][ti]- (ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2))
                  ra[i][j][ti] = diff_ra + ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)
              elif (i in V.MBbeams) and (i not in V.initialbeams) and (j!=0):
                  ra[i][j][ti] = diff_ra + ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)
              else:
                  ra[i][j][ti] = ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)

    return  (ra,Rab)

def integration_velocities2(X1,V,BeamSeg):

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]

    for i in range(V.NumBeams):
        ra[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3))
        ra[i][:,0] = BeamSeg[i].NodeX
        Rab[i][:,0] = BeamSeg[i].GlobalAxes
        for ti in range(1,V.tn):
          for j in range(BeamSeg[i].EnumNodes):
              Ipsi=((X1[i][j][ti][3:6]+X1[i][j][ti-1][3:6])/2)*V.dt
              Itheta=np.linalg.norm(Ipsi)
              Rab[i][j][ti] = Rab[i][j][ti-1].dot(H0(Itheta,Ipsi))
              #pdb.set_trace()
              if (i in V.MBbeams) and (i not in V.initialbeams) and (j==0):
                  ra[i][j][ti] = ra[inverseconn[i]][-1][ti-1]+Rab[inverseconn[i]][-1][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[inverseconn[i]][-1][ti-1][0:3]+X1[inverseconn[i]][-1][ti][0:3])/2)

                  inv = inverseconn[i]
                  Rab[i][j][ti] = Rab[i][0][ti].dot(Rab[inv][-1][ti].T).dot(H0(Itheta,Ipsi))
                  ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)
              else:
                  ra[i][j][ti] = ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,V.dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)

    return  (ra,Rab)



def current_config(rai0,Rabi0,q,Phi1,CPhi2,V,BeamSeg,inverseconn):

    #pdb.set_trace()
    q1 = q[1][0:V.NumModes]
    q10 = q[0][0:V.NumModes]
    q2 = q[1][V.NumModes:2*V.NumModes]
    X1n = np.zeros(6)
    X10 = np.zeros(6)
    for k in range(V.NumModes):
        X10 = X10 + Phi1[0][k][0]*q10[k]
        X1n = X1n + Phi1[0][k][0]*q1[k]

    X1 = (X10+X1n)/2
    Ipsi=(X1[3:6])*V.dt
    Itheta=np.linalg.norm(Ipsi)
    Rabi= Rabi0.dot(H0(Itheta,Ipsi))
    rai = rai0+Rabi0.dot(H1(Itheta,Ipsi,V.dt)).dot((X1[0:3]))
    strainx = strain_def(0,q2,CPhi2,V,BeamSeg)
    kappax = strain_def(1,q2,CPhi2,V,BeamSeg)
    rax,Rabx = integration_strains(kappax,strainx,V,BeamSeg,inverseconn,rai=rai,Rabi=Rabi)

    return(strainx,kappax,rax,Rabx)


def dQuater(t,Q,args1):

    i = args1['i']; j = args1['j']; ti = int(round(t/args1['dt'],5))+1
    #print ti
    if ti>=args1['tn']:
        #print ti
        ti=args1['tn']-1


    X1=args1['X1']

    w1=(X1[i][j][ti][3]+X1[i][j][ti-1][3])/2
    w2=(X1[i][j][ti][4]+X1[i][j][ti-1][4])/2
    w3=(X1[i][j][ti][5]+X1[i][j][ti-1][5])/2

    dQ = np.zeros(4)
    dQ[0] = -0.5*(w1*Q[1]+w2*Q[2]+w3*Q[3])
    dQ[1] = 0.5*(w1*Q[0] + Q[2]*w3-Q[3]*w2)
    dQ[2] = 0.5*(w2*Q[0] + Q[3]*w1-Q[1]*w3)
    dQ[3] = 0.5*(w3*Q[0] + Q[1]*w2-Q[2]*w1)

    return dQ


@my_timer
def Quatintegration_velocities(X1,**kwargs):
    from Tools.transformations import quaternion_from_matrix,quaternion_matrix

    BeamSeg=kwargs['BeamSeg'];NumBeams=kwargs['NumBeams']; tn= kwargs['tn']; dt = kwargs['dt']
    kwargs['X1'] = X1
    #pdb.set_trace()
    ra=[[] for i in range(NumBeams)]
    Rab=[[] for i in range(NumBeams)]
    Quater=[[] for i in range(NumBeams)]

    for i in range(NumBeams):
       kwargs['i'] = i
       ra[i]=np.zeros((BeamSeg[i].EnumNodes,tn,3))
       Quater[i]=np.zeros((BeamSeg[i].EnumNodes,tn,4))
       Rab[i]=np.zeros((BeamSeg[i].EnumNodes,tn,3,3))
       kwargs['q0'] = quaternion_from_matrix(BeamSeg[i].GlobalAxes)
       for j in range(BeamSeg[i].EnumNodes):
           kwargs['j'] = j
           Rab[i][j][0]=BeamSeg[i].GlobalAxes
           ra[i][j][0] = BeamSeg[i].NodeX[j]
           #pdb.set_trace()
           Quater[i][j] = Qsol2(RK4,dQuater,None,kwargs)
           for ti in range(1,tn):
             #H = np.vstack((-Quater[i][j][ti][1:],Quater[i][j][ti][0]*np.eye(3)+tilde(Quater[i][j][ti][1:]))).T
             #Gt = np.vstack((-Quater[i][j][ti][1:],Quater[i][j][ti][0]*np.eye(3)-tilde(Quater[i][j][ti][1:])))
             #Rab[i][j][ti]=Gt.T.dot(H.T)
             Rab[i][j][ti]=quaternion_matrix(Quater[i][j][ti])[:3,:3]
             Ipsi=((X1[i][j][ti][0:3]+X1[i][j][ti-1][0:3])/2)*dt
             Itheta=np.linalg.norm(Ipsi)
             ra[i][j][ti] = ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)

    return ra,Rab,Quater




# Rotation matrix derivative with time
def dRab_t(t,Rab,kwargs):

    i = kwargs['i']; j = kwargs['j']; ti = int(round(t/kwargs['dt'],5))+1
    #print ti
    if ti>=kwargs['tn']:
        #print ti
        ti=kwargs['tn']-1

    X1=kwargs['X1']

    w=(X1[i][j][ti][3:6]+X1[i][j][ti-1][3:6])/2

    Rab_sh = np.reshape(Rab,(3,3))
    dRab_sh = Rab_sh.dot(tilde(w))
    dRab = np.reshape(dRab_sh,9)
    return(dRab)


def dra_t(t,ra,args):

    Rab=args[0]
    va=args[1]
    dra=Rab.dot(va)
@my_timer
def Matintegration_velocities(X1,**kwargs):

    BeamSeg=kwargs['BeamSeg'];NumBeams=kwargs['NumBeams']; tn= kwargs['tn']; dt = kwargs['dt']
    kwargs['X1'] = X1
    #pdb.set_trace()
    ra=[[] for i in range(NumBeams)]
    Rab=[[] for i in range(NumBeams)]
    Rab9=[[] for i in range(NumBeams)]

    for i in range(NumBeams):
       kwargs['i'] = i
       ra[i]=np.zeros((BeamSeg[i].EnumNodes,tn,3))
       Rab[i]=np.zeros((BeamSeg[i].EnumNodes,tn,3,3))
       Rab9[i]=np.zeros((BeamSeg[i].EnumNodes,tn,9))
       kwargs['q0'] = np.reshape(BeamSeg[i].GlobalAxes,9)
       for j in range(BeamSeg[i].EnumNodes):
           kwargs['j'] = j
           Rab[i][j][0]=BeamSeg[i].GlobalAxes
           ra[i][j][0] = BeamSeg[i].NodeX[j]
           Rab9[i][j][0] = kwargs['q0']
           Rab9[i][j] = Qsol2(RK4,dRab_t,None,kwargs)
           for ti in range(1,tn):

             Rab[i][j][ti] = np.reshape(Rab9[i][j][ti],(3,3))
             Ipsi=((X1[i][j][ti][0:3]+X1[i][j][ti-1][0:3])/2)*dt
             Itheta=np.linalg.norm(Ipsi)
             ra[i][j][ti] = ra[i][j][ti-1]+Rab[i][j][ti-1].dot(H1(Itheta,Ipsi,dt)).dot((X1[i][j][ti-1][0:3]+X1[i][j][ti][0:3])/2)

    return ra,Rab


"""
def integration_strains(kappa,strain,k0,NumBeams,BeamsClamped,Clamped,inverseconn,H0,H1,ra0=np.zeros(3),Rab0=np.eye(3)):
    I3=np.eye(3)
    e_1=np.array([1.,0.,0.])
    def IRab(Rab,Itheta,Ipsi):
      return Rab.dot(H0(Itheta,Ipsi))


    def Ira(ra,Rab,Itheta,Ipsi,DL,strain):
      return ra + Rab.dot(H1(Itheta,Ipsi,DL)).dot(strain)

    ra=[[] for i in range(NumBeams)]
    Rab=[[] for i in range(NumBeams)]

    for i in range(NumBeams):

        ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if Clamped and i in BeamsClamped:
            ra[i][0]=ra0
            Rab[i][0]=Rab0
        elif not Clamped and i==0:
            ra[i][0]=ra0
            Rab[i][0]=Rab0
        else:
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            Rab[i][0] = Rab[k][-1]

        for j in range(1,BeamSeg[i].EnumNodes):
          Ipsi=(kappa[i][j-1]+k0[i][j-1])*BeamSeg[i].NodeDL[j-1]
          Itheta=np.linalg.norm(Ipsi)
          ra[i][j] = Ira(ra[i][j-1],Rab[i][j-1],Itheta,Ipsi,BeamSeg[i].NodeDL[j-1],strain[i][j-1]+e_1)
          Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)
    return(np.asarray(ra),np.asarray(Rab))
"""


def disp_sol(BeamSeg,ra,dis):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i in range(V.NumBeams):

      x = BeamSeg[i].NodeX[:,0]
      y = BeamSeg[i].NodeX[:,1]
      z = BeamSeg[i].NodeX[:,2]
      rx= ra[i][:,0]
      ry= ra[i][:,1]
      rz= ra[i][:,2]
      #ax.scatter(x, y, z, c='r', marker='o')
      ax.plot(x, y, z, c='r', marker='o')
      if dis:
       ax.plot(rx+x,ry+y,rz+z,c='b')
      else:
       ax.plot(rx,ry,rz,c='b')

    #plt.axis('off')

    #fig.suptitle('Mode'+str(modeplot)+'_Phi1:'+str(r))
    #plt.axis([0,80,-1,1,-3,3])
  plt.show()


def sol_deadstatic(BeamSeg,Phi1l,CPhi2xl,gamma2,Omega,Fa,dic_deadF):
    from intrinsic.functions import Rot6
    tn=dic_deadF['tn'];inverseconn=dic_deadF['inverseconn'];k0=dic_deadF['k0'];BeamForce=dic_deadF['Beamforce']; NodeForce=dic_deadF['NodeForce']
    NumModes=dic_deadF['NumModes']; NumBeams=dic_deadF['Numbeams'];BeamsClamped=dic_deadF['Beamsclamped']; Clamped = dic_deadF['Clamped']
    ra=[]
    Fai=Fa[0]
    q20=np.zeros(NumModes)
    for ti in range(tn):
        etai = intrinsic.integrals.integral_etan(Fai,NumModes,NumBeams,BeamSeg,Phi1l)
        q2i = intrinsic.qt.qstatic_soln(intrinsic.qt.qstatic,intrinsic.qt.Jqstatic,NumModes,Omega,gamma2,etai,q20)
        q20=q2i
        straini=strain_def(BeamSeg,NumModes,NumBeams,0,q2i,CPhi2xl)
        kappai=strain_def(BeamSeg,NumModes,NumBeams,1,q2i,CPhi2xl)
        rai,Rabi=integration_strains(kappai,straini,k0,NumBeams,BeamsClamped,Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))

        #Fai=np.zeros((NumBeams,BeamSeg[i].EnumNodes,6))
        Fai=[np.zeros((BeamSeg[i].EnumNodes,6)) for i in range(V.NumBeams)]
        for i in range(len(BeamForce)):
          for j in range(len(NodeForce[i])):

            Fai[BeamForce[i]][NodeForce[i][j]] = Rot6(Rabi[i][j].T).dot(Fa[BeamForce[i]][ti][NodeForce[i][j]])

        ra.append(rai)
    return(ra)


def sol_deadstatic2(BeamSeg,Phi1l,CPhi2xl,gamma2,Omega,Fa,dic_deadF):
    from intrinsic.functions import Rot6
    tn=dic_deadF['tn'];inverseconn=dic_deadF['inverseconn'];k0=dic_deadF['k0'];BeamForce=dic_deadF['Beamforce']; NodeForce=dic_deadF['NodeForce']
    NumModes=dic_deadF['NumModes']; NumBeams=dic_deadF['Numbeams'];BeamsClamped=dic_deadF['Beamsclamped']; Clamped = dic_deadF['Clamped']
    ra=[]
    Fai=Fa[0]
    q20=np.zeros(NumModes)
    for ti in range(tn):
        etai = intrinsic.integrals.integral_etan(Fai,NumModes,NumBeams,BeamSeg,Phi1l)
        q2i = intrinsic.qt.qstatic_soln(intrinsic.qt.qstatic,intrinsic.qt.Jqstatic,NumModes,Omega,gamma2,etai,q20)
        q20=q2i
        straini=strain_def(BeamSeg,NumModes,NumBeams,0,q2i,CPhi2xl)
        kappai=strain_def(BeamSeg,NumModes,NumBeams,1,q2i,CPhi2xl)
        rai,Rabi=integration_strains(kappai,straini,k0,NumBeams,BeamsClamped,Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))

        #Fai=np.zeros((NumBeams,BeamSeg[i].EnumNodes,6))
        Fai=[np.zeros((BeamSeg[i].EnumNodes,6)) for i in range(V.NumBeams)]
        for i in range(len(BeamForce)):
          for j in range(len(NodeForce[i])):
            Fai[BeamForce[i]][NodeForce[i][j]] = Rot6(Rabi[i][j].T).dot(Fa[BeamForce[i]][ti][NodeForce[i][j]])

        ra.append(rai)
    return(ra)

@my_timer
def Qsolstrains_im(solver,dq,Jdq,**kwargs):

   V = kwargs['V']; F = kwargs['F']; BeamSeg = kwargs['BeamSeg'];inverseconn = kwargs['inverseconn']
   q0=kwargs['q0']
   Phi1 = kwargs['Phi1'];CPhi2 = kwargs['CPhi2']

   strain = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
   kappa = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
   ra = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3)) for i in range(V.NumBeams)]
   Rab = [np.zeros((BeamSeg[i].EnumNodes,V.tn,3,3)) for i in range(V.NumBeams)]
   Rabx = [np.zeros((BeamSeg[i].EnumNodes,3,3)) for i in range(V.NumBeams)]

   for i in range(V.NumBeams):
      ra[i][:,0] = BeamSeg[i].NodeX
      Rab[i][:,0] = Rabx[i][:] = BeamSeg[i].GlobalAxes

   #pdb.set_trace()
   qsol=[]
   qsol.append(q0)
   tni = 1
   if type(solver).__name__ == 'function':

      tfi = V.t0

      while tni<V.tn:

        rotation = [Rabx[F.Dead_points_app[ix][0]][F.Dead_points_app[ix][1]] for ix in range(F.NumDLoads)]
        kwargs['rotation'] = rotation
        qsol.append(solver(dq,tfi,qsol[-1],V.dt,kwargs))
        strainx,kappax,rax,Rabx = current_config(ra[0][0,tni-1],Rab[0][0,tni-1],qsol[-2:],Phi1,CPhi2,V,BeamSeg,inverseconn)
        for i in range(V.NumBeams):
           strain[i][:,tni] = strainx[i]
           kappa[i][:,tni] = kappax[i]
           ra[i][:,tni] = rax[i]
           Rab[i][:,tni] = Rabx[i]

        tni=tni+1
        tfi=tfi+V.dt
        if kwargs['printx']:
         print(tfi,tni)
        #print tni

      return(np.array(qsol),strain,kappa,ra,Rab)


   elif type(solver).__name__ == 'str':

     qs=ode(dq,Jdq)

     qs.set_initial_value(q0,t0)
     qs.set_f_params(kwargs)
     if Jdq is not None:
      qs.set_integrator(solver,with_jacobian=Jdq)
      qs.set_jac_params(kwargs)
     else:
      qs.set_integrator(solver)


     while qs.successful() and qs.t <= kwargs['tf']:

       if kwargs['printx']:
        print(qs.t,tni)
       qs.integrate(qs.t+kwargs['dt'])
       qsol.append(qs.y)
       tni=tni+1

     return(np.array(qsol))



'''
# Integration of velocities
#=======================================================================================================================================




#  Rotation function
def Tt(T0,t0,tf,dt,n):

 Tsol=ode(dTdt)
 Tsol.set_integrator('dopri5')
 Tsol.set_initial_value(np.reshape(T0,9),t0)
 Tsol.set_f_params(n)
 Ti=[T0.tolist()]; t=[t0];

 while  Tsol.successful() and Tsol.t+dt/2<tf:
   Tsol.integrate(Tsol.t+dt)
   Tid=np.reshape(Tsol.y,(3,3))
   Ti.append(Tid); t.append(Tsol.t)

 return Ti,t


# Displacements derivative
def dRdt(t,R,args1):


  dR=args1[1][int(t/dt)].dot(X1[int(t/dt)][args1[0]][0:3])
  return dR


# Displacement function
def Rt(R0,t0,tf,dt,n,T):

 Rsol=ode(dRdt)
 Rsol.set_integrator('dopri5')
 Rsol.set_initial_value(R0,t0)
 Rsol.set_f_params([n,T])
 Ri=[R0.tolist()]; t=[t0];

 while  Rsol.successful() and Rsol.t+dt/2<tf:
   Rsol.integrate(Rsol.t+dt)
   #Rid=np.reshape(Rsol.y,(3,3))
   Ri.append(Rsol.y); t.append(Rsol.t)
 return Ri,t



# Integration of equations
#=====================================================
for i in range(NumBeams):

  BeamSeg[i].solTvelocity=np.zeros((tn,NumNodes,3,3))
  BeamSeg[i].solRvelocity=np.zeros((tn,NumNodes,3))

for i in range(NumBeams):
  for n in range(BeamSeg[i].EnumNode):

    T0=np.eye(3)
    R0=np.array([0.0,0.0,0.])
    BeamSeg[i].solTvelocity[:,n],t= Tt(T0,t0,tf,dt,n)
    #pdb.set_trace()
    BeamSeg[i].solRvelocity[:,n],t= Rt(R0,t0,tf,dt,n,BeamSeg[i].solTvelocity[:,n])




# Integration of Strains
#===========================================================================================================================================




# Exponential map approach
#===========================================================================================================


# Integration of equations
#=============================================================

for i in range(NumBeams):
  R0=np.zeros(3)
  BeamSeg[i].solRa_dyn=np.zeros((tn,BeamSeg[i].EnumNode,3))
  BeamSeg[i].solRa_dyn[:,0]=R0
  BeamSeg[i].solCab_dyn=np.zeros((tn,BeamSeg[i].EnumNode,3,3))
  Cab0=np.eye(3)
  BeamSeg[i].solCab_dyn[:,0]=Cab0


for ti in range(tn):
  for i in range(NumBeams):
    for j in range(BeamSeg[i].EnumNode-1):
        #pdb.set_trace()
        Ipsi=(kappa_dyn[ti,BeamSeg[i].NodeOrder[j]]+kappa0[BeamSeg[i].NodeOrder[j]])*BeamSeg[i].NodeDL[j]
        Itheta=np.linalg.norm(Ipsi)
        BeamSeg[i].solRa_dyn[ti,j+1] = BeamSeg[i].solRa_dyn[ti,j]+BeamSeg[i].solCab_dyn[ti,j].dot(H1(Itheta,Ipsi,BeamSeg[i].NodeDL[j])).dot(strain_dyn[ti,BeamSeg[i].NodeOrder[j]]+e_1)
        BeamSeg[i].solCab_dyn[ti,j+1] = BeamSeg[i].solCab_dyn[ti,j].dot(H0(Itheta,Ipsi))



# Direct solution of equation with Runge-Kutta
#==========================================================================================================
# Rotation matrix derivative with space

def dTds(s,T,ts):

  Tsh=np.reshape(T,(3,3))

  dT=Tsh.dot(tilde(kappa_dyn[ts[0],ts[1]]+kappa0[ts[1]]))

  dTsh=np.reshape(dT,9)
  return dTsh


#  Rotation function (space)
def Ts(T0,s0,sf,ti,sn,dsn):
    Tsol=ode(dTds)
    Tsol.set_integrator('dopri5')
    Tsol.set_initial_value(np.reshape(T0,9),t0)
    Ti=[T0]; s=[s0];
    k=0
    while Tsol.successful() and Tsol.t<sf:
      sni=sn[k]
      Tsol.set_f_params([ti,sni])
      Tsol.integrate(Tsol.t+dsn[k])
      Tish=np.reshape(Tsol.y,(3,3))
      Ti.append(Tish); s.append(Tsol.t)
      k=k+1
    return Ti


#  Displacements derivative (space)
def dRds(s,R,ts):

  dR=ts[2].dot(strain_dyn[ts[0],ts[1]]+e_1)
  return dR

# Displacements function
def Rs(R0,T,s0,sf,ti,sn,dsn):

  Rsol=ode(dRds)
  Rsol.set_integrator('dopri5')
  Rsol.set_initial_value(R0,s0)
  Ri=[R0];s=[s0]
  k=0
  while Rsol.successful() and Rsol.t+0.05<sf:
    sni=sn[k]
    T0=T[k]
    Rsol.set_f_params([ti,sni,T0])
    k=k+1
    Rsol.integrate(Rsol.t+dsn[k-1])
    Ri.append(Rsol.y); s.append(Rsol.t)

  return Ri


# Integration of equations
#===============================================================

for i in range(NumBeams):
  BeamSeg[i].solTstrain_dyn=np.zeros((tn,BeamSeg[i].EnumNode,3,3))
  BeamSeg[i].solRstrain_dyn=np.zeros((tn,BeamSeg[i].EnumNode,3))


for ti in range(tn):
  T0=np.eye(3)
  R0=np.array([0.,0.,0.])
  s0=0
  for i in range(NumBeams):
    sn=BeamSeg[i].NodeOrder
    dsn=BeamSeg[i].NodeDL
    sf=BeamSeg[i].NodeL[-1]
    BeamSeg[i].solTstrain_dyn[ti]= np.asarray(Ts(T0,s0,sf,ti,sn,dsn))


for ti in range(tn):

  for i in range(NumBeams):
    sn=BeamSeg[i].NodeOrder
    dsn=BeamSeg[i].NodeDL
    sf=BeamSeg[i].NodeL[-1]
    BeamSeg[i].solRstrain_dyn[ti] = np.asarray(Rs(R0,BeamSeg[i].solTstrain_dyn[ti],s0,sf,ti,sn,dsn))

'''


if (__name__ == '__main__'):

    Fa=np.load(V.feminas_dir+'/Runs/'+V.model+'/Fa1.npy')  # BeamSeg,tn,NumNodes,6
    multi=1
    from intrinsic.modes import Omega,Phi1,Phi2,CPhi2xl,Phi1l
    from intrinsic.integrals import integral2_gammas,integral_eta,solve_integrals
    gamma1,gamma2=solve_integrals(multi,'gammas',V.NumModes)
    eta=integral_eta(Fa,V.tn,V.NumModes,V.NumBeams,BeamSeg=BeamSeg,Phi1=Phi1l)

    import intrinsic.qt
    import intrinsic.functions


    q2=intrinsic.qt.Jqstatic_sol(intrinsic.qt.qstatic,intrinsic.qt.Jqstatic,V.NumModes,Omega,gamma2,eta,V.tn)
    #q1,q2=qsol(ODE.RK4,dq12,time,0.)
    #q1_lin,q2_lin=qsol(ODE.RK4,dq12_lin,time,0.)

    #q2fix,q2fix_lin=qstatic_solfix(eta,Omega,gamma2)

    q2_lin=intrinsic.qt.qstatic_sollin(eta,Omega,V.NumModes,V.tn)

    #intrinsic.qt.qstatic(q2,eta[-1])
    '''
    strain=strain_def(BeamSeg,0,q2,CPhi2xl)
    kappa=strain_def(BeamSeg,1,q2,CPhi2xl)
    ra,Rab=integration_strains(kappa,strain)
    '''
    strain=[[] for i in range(V.tn)]
    kappa=[[] for i in range(V.tn)]
    ra=[[] for i in range(V.tn)]
    Rab=[[] for i in range(V.tn)]
    k0=[np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)]
    strain0=np.asarray([np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)])
    #k0=np.asarray([[np.array([0,0,1./50]) for j in range(BeamSeg[i].EnumNodes)] for i in range(V.NumBeams)])
    ra20,Rab20=integration_strains2(BeamSeg,strain0,strain0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1)
    for ti in range(V.tn):
        strain[ti] = strain_def(BeamSeg,V.NumModes,V.NumBeams,0,q2[ti],CPhi2xl)
        kappa[ti] = strain_def(BeamSeg,V.NumModes,V.NumBeams,1,q2[ti],CPhi2xl)
        ra[ti],Rab[ti] = integration_strains(kappa[ti],strain[ti],k0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))

    print('sol')




















'''
def strain_def(BeamSeg,NumBeams,curvature,q2):

    for k in range(V.NumModes):
      for i in range(NumBeams):
        if static:
            strain=np.zeros((NumBeams,BeamSeg[i].EnumNodes,3))
            for j in range(BeamSeg[i].EnumNodes-1):
                strain[i][j] = strain[i][j] + BeamSeg[i].GlobalAxes.dot(CPhi2x[i][k][j][curvature*3:curvature*3+3])q2[k]
        else:
            strain=np.zeros((V.tn,NumBeams,BeamSeg[i].EnumNodes,3))
            for ti in range(V.tn):
              for j in range(BeamSeg[i].EnumNodes-1):
                strain[ti][i][j] = strain[ti][i][j] + BeamSeg[i].GlobalAxes.dot(CPhi2x[i][k][j][curvature*3:curvature*3+3])q2[ti][k]
    return strain


def strain_def(BeamSeg,curvature,q2,CPhi2):

    strain=[np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)]
    for k in range(V.NumModes):
      for i in range(V.NumBeams):

            for j in range(BeamSeg[i].EnumNodes-1):
                strain[i][j] = strain[i][j] + BeamSeg[i].GlobalAxes.dot(CPhi2[i][k][j][curvature*3:curvature*3+3])*q2[k]

    return strain


def integration_strains(kappa,strain,k0,ra0=np.zeros(3),Rab0=np.eye(3)):
    I3=np.eye(3)
    e_1=np.array([1,0,0])
    def IRab(Rab,Itheta,Ipsi):
      return Rab.dot(intrinsic.functions.H0(Itheta,Ipsi))


    def Ira(ra,Rab,Itheta,Ipsi,DL,strain):
      return ra + Rab.dot(intrinsic.functions.H1(Itheta,Ipsi,DL)).dot(strain)

    ra=[[] for i in range(V.NumBeams)]
    Rab=[[] for i in range(V.NumBeams)]

    for i in range(V.NumBeams):

        ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
        Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
        if V.Clamped and i in V.BeamsClamped:
            ra[i][0]=ra0
            Rab[i][0]=Rab0
        elif not V.Clamped and i==0:
            ra[i][0]=ra0
            Rab[i][0]=Rab0
        else:
            k=inverseconn[i]
            ra[i][0] = ra[k][-1]
            Rab[i][0] = Rab[k][-1]

        for j in range(1,BeamSeg[i].EnumNodes):
          Ipsi=(kappa[i][j-1]+k0[i][j-1])*BeamSeg[i].NodeDL[j-1]
          Itheta=np.linalg.norm(Ipsi)
          ra[i][j] = Ira(ra[i][j-1],Rab[i][j-1],Itheta,Ipsi,BeamSeg[i].NodeDL[j-1],strain[i][j-1]+e_1)
          Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)
    return(np.asarray(ra),np.asarray(Rab))
'''
