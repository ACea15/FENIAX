from scipy.interpolate import interp1d
import numpy as np
import pdb
import importlib
import multiprocessing
import time
from Tools.transformations import quaternion_rotation6,quaternion_conjugate

import Runs.Torun
#Runs.Torun.torun = 'ArgyrisFrame_20'
#V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')
#V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
from intrinsic.functions import Matrix_rotation6
#import intrinsic.geometry
#BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = intrinsic.geometry.geometry_def(
#self.V.Grid,self.V.NumBeams,self.V.BeamConn,self.V.start_reading,self.V.beam_start,self.V.nodeorder_start,
#self.V.node_start,self.V.Clamped,self.V.ClampX,self.V.BeamsClamped)

class Force:
    """Class to define any type of external force."""


    def __init__(self,Phi1,V,Gravity=None,Phig0=None,BeamSeg=None,NumFLoads=0,NumDLoads=0,NumALoads=0,
                 Follower_points_app=None,Follower_interpol=None,
                 Dead_points_app=None,Dead_interpol=None):

        
        self.Phi1 = Phi1
        self.V = V
        self.Gravity = Gravity
        self.Phig0 = Phig0
        self.BeamSeg = BeamSeg
        # self.NumSLoads = NumSLoads
        # if NumSLoads>0:

        #     if len(Static_points_app)!=NumSLoads:
        #        raise ValueError('Number of point loads not equal to application points')
        #     else:
        #        self.Static_points_app = Static_points_app
        #     if len(Static_interpol)!=NumSLoads:
        #        raise ValueError('Number of interpolation array not equal to Number of loads')
        #     else:
        #        self.Static_interpol = Static_interpol
        #     self.NumSDimen = [len(Static_points_app[i][2]) for i in range(NumSLoads)]
        self.NumDLoads = NumDLoads
        if NumDLoads>0:

            if len(Dead_points_app)!=NumDLoads:
               raise ValueError('Number of point loads not equal to application points')
            else:
               self.Dead_points_app = Dead_points_app
            if len(Dead_interpol)!=NumDLoads:
               raise ValueError('Number of interpolation array not equal to Number of loads')
            else:
               self.Dead_interpol = Dead_interpol

            self.NumDDimen = [len(Dead_points_app[i][2]) for i in range(NumDLoads)]

        self.NumFLoads = NumFLoads
        if NumFLoads>0:

            if len(Follower_points_app)!=NumFLoads:
               raise ValueError('Number of point loads not equal to application points')
            else:
               self.Follower_points_app = Follower_points_app
            if len(Follower_interpol)!=NumFLoads:
               raise ValueError('Number of interpolation array not equal to Number of loads')
            else:
               self.Follower_interpol = Follower_interpol

            self.NumFDimen = [len(Follower_points_app[i][2]) for i in range(NumFLoads)]

        self.NumALoads = NumALoads
        if NumALoads>0:
            self.A = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.aero)
            self.AICs = np.load(self.A.Amatrix)
            if self.A.Axmatrix:
                self.AICsx = np.load(self.A.Axmatrix)
    def interpolation(self):

        # if self.NumSLoads>0:
        #     self.Staticinterpol = [[[] for d in range(self.NumSDimen[i]) ] for i in range(self.NumSLoads)]
        #     for i in range(self.NumSLoads):
        #       for d in range(self.NumSDimen[i]):
        #         self.Staticinterpol[i][d] = interp1d(self.Static_interpol[i][d][0],self.Static_interpol[i][d][1])

        if self.NumFLoads>0:
            self.Followerinterpol = [[[] for d in range(self.NumFDimen[i]) ] for i in range(self.NumFLoads)]
            for i in range(self.NumFLoads):
              for d in range(self.NumFDimen[i]):
                self.Followerinterpol[i][d] = interp1d(self.Follower_interpol[i][d][0],self.Follower_interpol[i][d][1])

        if self.NumDLoads>0:
            self.Deadinterpol = [[[] for d in range(self.NumDDimen[i]) ] for i in range(self.NumDLoads)]
            for i in range(self.NumDLoads):
              for d in range(self.NumDDimen[i]):
                self.Deadinterpol[i][d] = interp1d(self.Dead_interpol[i][d][0],self.Dead_interpol[i][d][1])


    #Finterpol = interpolation()
    #Finterpol = [[[] for d in range(self.NumDimen[i]) ] for i in range(self.NumLoads)]
    #for i in range(self.NumLoads):
    #   for d in range(self.NumDimen[i]):
    #      Finterpol[i][d] = interp1d(self.interpol[i][d][0],self.interpol[i][d][1])

    def follower_interpol(self,ix,t):
        #pdb.set_trace()
        if 'Followerinterpol'not in dir(self):
            self.interpolation()

        F = [0. for i in range(6)]
        for d in range(len(self.Follower_points_app[ix][2])):
            F[self.Follower_points_app[ix][2][d]] = self.Followerinterpol[ix][d](t)

        return np.asarray(F)

    def dead_interpol(self,ix,t):

        if 'Deadinterpol'not in dir(self):
            self.interpolation()

        F = [0. for i in range(6)]
        for d in range(len(self.Dead_points_app[ix][2])):
            F[self.Dead_points_app[ix][2][d]] = self.Deadinterpol[ix][d](t)

        return np.asarray(F)


    def force_follower(self,t):
        F=[[] for i in range(self.NumFLoads)]
        for i in range(self.NumFLoads):
            F[i] = self.follower_interpol(i,t)
        return F

    def force_dead(self,t,rotation):
        #pdb.set_trace()
        F = [[] for i in range(self.NumDLoads)]
        for i in range(self.NumDLoads):
            F[i] = self.dead_interpol(i,t)

        Fr=[[] for i in range(self.NumDLoads)]
        if np.shape(rotation) == (self.NumDLoads,4): # Rotation given by the quaternions at the point of application of the dead force
            for i in range(self.NumDLoads):
                Fr[i] = quaternion_rotation6(F[i],quaternion_conjugate(rotation[i]))
                #Fr[i] = quaternion_rotation6(F[i],rotation[i])
        elif np.shape(rotation) == (self.NumDLoads,3): # Rotation  given by the rotation matrix
            # (only calculated at the points of application of the dead force)
            for i in range(self.NumDLoads):
                Fr[i] = Matrix_rotation6(F[i],rotation[i].T)
                #Fr[i] = Matrix_rotation6(F[i],rotation[i])
        elif np.shape(rotation)[2] == 3: # Rotation  given by the rotation matrix
            #(calculated along the whole structure, for example for gravity forces
            for i in range(self.NumDLoads):
                Fr[i] = Matrix_rotation6(F[i],rotation[self.Dead_points_app[i][0]][self.Dead_points_app[i][1]].T)
                #Fr[i] = Matrix_rotation6(F[i],rotation[i])

        return Fr

    def forceFollower_eta(self,t):

        F = self.force_follower(t)
        eta = np.zeros(self.V.NumModes)
        for k in range(self.V.NumModes):
          for i in range(self.NumFLoads):

            eta[k] = eta[k]+ self.Phi1[self.Follower_points_app[i][0]][k][self.Follower_points_app[i][1]].dot(F[i])
            #eta[k] =  self.Phi1[self.Follower_points_app[i][0]][k][self.Follower_points_app[i][1]].dot(F[i])
        return eta

    def forceDead_eta(self,t,rotation):

        F = self.force_dead(t,rotation)
        eta = np.zeros(self.V.NumModes)
        for k in range(self.V.NumModes):
          for i in range(self.NumDLoads):
            eta[k] = eta[k]+self.Phi1[self.Dead_points_app[i][0]][k][self.Dead_points_app[i][1]].dot(F[i])
        return eta

    def forceGravity_eta(self,t,rotation):

        eta = np.zeros(self.V.NumModes-self.V.NumModes_res)
        for k in range(self.V.NumModes-self.V.NumModes_res):
          for i in range(self.V.NumBeams):
              for j in range(self.BeamSeg[i].EnumNodes):
                  eta[k] += self.Phi1[i][k][j].dot(Matrix_rotation6(self.Phig0[i][j],rotation[i][j].T))
        return eta


    def forceAeroTotal_eta(self,q1,q0,ql,dq1):

        # qinf = 0.5*self.A.rho_inf*self.A.u_inf**2
        # eta = qinf*self.AICs[0,:,:].dot(q[2*self.V.NumModes:3*self.V.NumModes]) +\
        #       self.A.c*qinf/(2*self.A.u_inf)*self.AICs[1,:,:].dot(q[:self.V.NumModes]) +\
        #       qinf*(self.A.c/(2*self.A.u_inf))**2*self.AICs[2,:,:].dot(q[1*self.V.NumModes:2*self.V.NumModes])
        # for i in range(self.A.NumPoles):
        #     eta += qinf*self.AICs[i+3,:,:].dot(q[(i+3)*self.V.NumModes:(i+4)*self.V.NumModes])

        eta = 0.5*self.A.rho_inf*(self.A.u_inf**2*self.AICs[0,:,:].dot(q0) +
                                  self.A.u_inf*self.A.c/2*self.AICs[1,:,:].dot(q1) +
                                  (self.A.c/2)**2*self.AICs[2,:,:].dot(dq1))
        for i in range(self.A.NumPoles):
            eta += 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICs[i+3,:,:].dot(ql[i*self.V.NumModes:(i+1)*self.V.NumModes])

        return eta

    def forceAero_eta(self,q1,q0,ql=[]):
        modesfull = np.shape(self.AICs[0])[0]
        eta = 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICs[0,:,:].dot(q0) +\
              0.5*self.A.rho_inf*self.A.u_inf*self.A.c/2*self.AICs[1,:,:].dot(q1)
        for i in range(self.A.NumPoles):
            eta += 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICs[i+3,:,:].dot(ql[i*modesfull:(i+1)*modesfull])
        return eta

    def forceAero_eta_rbd(self,q):

        eta = 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICs[0,:,:].dot(q[2*self.V.NumModes+self.A.rbd:3*self.V.NumModes+2*self.A.rbd]) +\
              0.5*self.A.rho_inf*self.A.u_inf*self.A.c/2*self.AICs[1,:,:].dot(q[:self.V.NumModes+self.A.rbd])
        for i in range(self.A.NumPoles):
            eta += 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICs[i+3,:,:].dot(q[(i+3)*self.V.NumModes+2*self.A.rbd+(i)*self.A.rbd:(i+4)*self.V.NumModes+2*self.A.rbd+(i+1)*self.A.rbd])
        return eta

    def forceAero_eta_x(self):
        #pdb.set_trace()
        eta = 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICsx[:,0]*self.A.qx[0]
        for i in range(1,np.shape(self.AICsx)[1]):
            eta += 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICsx[:,i]*self.A.qx[i]
        return eta

    def etaAero(self,q1,q0,ql,dq1=[]):

        if dq1:
            eta = self.forceAeroTotal_eta(q1,q0,ql,dq1)
        else:
            eta = self.forceAero_eta(q1,q0,ql)
        if self.A.rbd:
            eta += self.forceAero_eta_rbd(np.hstack([q1,np.zeros(self.V.NumModes),q0,ql]))
        elif self.A.rbdx:
            eta += self.forceAero_eta_x()

        return eta

    def eta(self,t,q,rotation=None):

        eta = np.zeros(self.V.NumModes)
        if self.NumFLoads>0:
            eta += self.forceFollower_eta(t)

        if self.Gravity:
            eta += self.forceGravity_eta(t,rotation)
            #print eta
            if self.NumDLoads>0:
               rotation =[]
               eta += self.forceDead_eta(t,rotation)
        elif self.NumDLoads>0 and (not self.Gravity):

            if rotation is not None:
                rotation_dead = rotation

            else:
                rotation_dead = [[q[2*self.V.NumModes+l*4],q[2*self.V.NumModes+l*4+1],q[2*self.V.NumModes+l*4+2],q[2*self.V.NumModes+l*4+3]] for l in range(self.NumDLoads)]

            eta += self.forceDead_eta(t,rotation_dead)

        # if self.NumALoads>0:
        #     if self.A.rbd:
        #         eta += self.forceAero_eta_rbd(q)
        #     else:
        #         eta += self.forceAero_etax(q)

        return eta

#self.Ma[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6].dot(Rab[i][j].T.dot(np.array([0.,0.,self.V.g,0.,0.,0.])))

# def Rotation4mStrain(Rab0,kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn):
#     I3=np.eye(3)
#     e_1=np.array([1,0,0])
#     def IRab(Rab,Itheta,Ipsi):
#       return Rab.dot(H0(Itheta,Ipsi))

#     Rab=[[] for i in range(self.V.NumBeams)]

#     for i in range(self.V.NumBeams):

#         ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
#         Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
#         if self.V.Clamped and i in self.V.BeamsClamped:
#             Rab[i][0]=BeamSeg[i].GlobalAxes
#         elif not self.V.Clamped and i in self.V.BeamsInit:
#             Rab[i][0]=Rab0
#         else:
#             k=inverseconn[i]
#             Rab[i][0] = BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T.dot(Rab[k][-1]))
#             #Rab[i][0] = Rab[k][-1].dot(BeamSeg[k].GlobalAxes.T.dot(BeamSeg[i].GlobalAxes))

#         for j in range(1,BeamSeg[i].EnumNodes):
#           Ipsi=(kappa[i][j-1])*BeamSeg[i].NodeDL[j-1]
#           Itheta=np.linalg.norm(Ipsi)
#           Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)

#     return(ra,Rab)


if __name__ == "__main__":
    # Phi1,Gravity=None,NumFLoads=0,NumDLoads=0,
    # Follower_points_app=None,Follower_interpol=None,
    # Dead_points_app=None,Dead_interpol=None
    NL=2
    X=[[0,1,[1,3]],[0,3,[0]]]
    I=[[[[0,5],[1,1]],[[0,5],[0,1]]],[[[0,5],[5,0]]]]
    t1 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)
    t2 = Force(Phi1l,Dead_points_app=X,Dead_interpol=I,NumDLoads=NL)

    NL=1
    X=[[0,-1,[1]]]
    I=[[[[0,1,2],[0,1,0]]]]
    t3 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)

    NL=1
    X=[[0,-1,[0,5]]]
    I=[[[[0.,2.5,2.5,10.],[8.,8.,0.,0.]],[[0.,2.5,2.5,10.],[80.,80.,0.,0.]]]]
    t3 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)

    NL = 1
    X = [[0,-1,[0,4,5]]]
    I = [[[[0.,2.5,5.,10.],[0.,20.,0.,0.]],[[0.,2.5,5.,10.],[0.,100.,0.,0.]],[[0.,2.5,5.,10.],[0.,200.,0.,0.]]]]
    t4 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)
