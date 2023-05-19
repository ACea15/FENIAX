import pdb
import numpy as np
import importlib
import Runs.Torun
#import intrinsic.sol
import intrinsic.solrb
from Tools.transformations import quaternion_from_matrix,quaternion_matrix
#V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
#F = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.force)
# try:
#     A = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.aero)
# except KeyError:
#     A=None

class StaticEqs:
    """Class"""

    def __init__(self,Omega,gamma2,force1,V,F):
        self.Omega = Omega
        self.gamma2 = gamma2
        self.force1 = force1
        self.V = V
        self.F = F
    ####################################################################################
    # Static Problem: Algebraic Eqs.                                                   #
    ####################################################################################

    ######################
    # Fix point function #
    ######################
    def qstatic_solfix(self,eta,Omega,gamma2):
        import copy
        NumModes=len(Omega)
        tn=np.shape(eta)[0]
        q2_old = np.zeros((tn,NumModes))
        q2_new=np.zeros((tn,NumModes))
        q2_lin=np.zeros((tn,NumModes))

        for fi in range(tn):
            #pdb.set_trace()
            count=0
            err=1
            while err > 1e-6 and count<100:
                for k in range(NumModes):
                    sx=0
                    for k1 in range(NumModes):
                        for k2 in range(NumModes):
                            if count==0:
                              sx=sx+gamma2[k][k1][k2]*q2_old[fi-1][k1]*q2_old[fi-1][k2]
                            else:
                              sx=sx+gamma2[k][k1][k2]*q2_old[fi][k1]*q2_old[fi][k2]
                    #print sx
                    q2_new[fi][k] = (sx-eta[fi][k])/Omega[k]
                    q2_lin[fi][k] = (-eta[fi][k])/Omega[k]

                err = max(abs((q2_new[fi]-q2_old[fi])/q2_new[fi]))
                q2_old=q2_new
                count=count+1

        return(q2_new,q2_lin)

    ###########################################
    # Structural problem with follower forces #
    ###########################################
    def qstatic(self,q2st,coeff):

       F=np.zeros(self.V.NumModes)
       for i in range(self.V.NumModes):
        for k in range(self.V.NumModes):
          for l in range(self.V.NumModes):
            F[i]=F[i]-self.gamma2[i,k,l]*(q2st[k]*q2st[l])
        F[i]=F[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],q2st[i])[i]
       return F

    def qstatic_opt(self,q2st, coeff):

        F = np.zeros(self.V.NumModes)
        Q2 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        for k in range(self.V.NumModes):
         for l in range(k+1,self.V.NumModes):
           Q2[k][l]=q2st[k]*q2st[l]
           for i in range(self.V.NumModes):
            F[i]=F[i]-(self.gamma2[i,k,l] + self.gamma2[i,l,k])*Q2[k][l]

        for i in range(self.V.NumModes):
            F[i]=F[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],q2st[i])[i]
            for k in range(self.V.NumModes):
                Q2[k][k]=q2st[k]*q2st[k]
                F[i]=F[i]-(self.gamma2[i,k,k])*Q2[k][k]

        return F


    def qstatic_opt2(self,q2st,coeff):


        F = np.zeros(self.V.NumModes)
        Q2 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        for k in range(self.V.NumModes):
         for l in range(self.V.NumModes):
           Q2[k][l]=q2st[k]*q2st[l]
           for i in range(self.V.NumModes):
            F[i]=F[i]-(self.gamma2[i,k,l])*Q2[k][l]

        for i in range(self.V.NumModes):
            F[i]=F[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],q2st[i])[i]

        return F


    def qstatic_opt3(self,q2st,coeff):


        F = np.zeros(self.V.NumModes)
        Q2 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        for k in range(self.V.NumModes):
         for l in range(k,self.V.NumModes):
           Q2[k][l]=q2st[k]*q2st[l]
           for i in range(self.V.NumModes):
            if l==k:
                F[i]=F[i]-(self.gamma2[i,k,l])*Q2[k][l]
            else:
                F[i]=F[i]-(self.gamma2[i,k,l] + self.gamma2[i,l,k])*Q2[k][l]

        for i in range(self.V.NumModes):
            F[i]=F[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],q2st[i])[i]

        return F

    def qstatic_opt4(self,q2st,coeff):

        F = np.zeros(self.V.NumModes)
        Q2 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        for k in range(self.V.NumModes):
         for l in range(k+1,self.V.NumModes):
           Q2[k][l]=q2st[k]*q2st[l]
           for i in range(self.V.NumModes):
            F[i]=F[i]-(self.gamma2[i,k,l] + self.gamma2[i,l,k])*Q2[k][l]

        for k in range(self.V.NumModes):
           Q2[k][k]=q2st[k]*q2st[k]
           for i in range(self.V.NumModes):
             F[i]=F[i]-(self.gamma2[i,k,k])*Q2[k][k]

        for i in range(self.V.NumModes):

               F[i]=F[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],q2st[i])[i]

        return F


    ########################################
    # Jacobian function for the structural #
    # problem(without dead forces)         #
    ########################################

    def Jqstatic(self,q2st,coeff):

       Jf = np.zeros((self.V.NumModes,self.V.NumModes))

       for i in range(self.V.NumModes):
        for j in range(self.V.NumModes):
          for k in range(self.V.NumModes):
            Jf[i][j]=Jf[i][j]-(self.gamma2[i,j,k]+self.gamma2[i,k,j])*q2st[k]
        Jf[i][i]=Jf[i][i]+self.Omega[i]
       return Jf



    #######################################
    # Structural problem with dead forces #
    #######################################
    def qstatic_dead(self,q2st,coeff):
        import intrinsic.sol

        F0 = np.zeros(self.V.NumModes)
        Q2 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        for k in range(self.V.NumModes):
         for l in range(k+1,self.V.NumModes):
           Q2[k][l]=q2st[k]*q2st[l]
           for i in range(self.V.NumModes):
            F0[i]=F0[i]-(self.gamma2[i,k,l] + self.gamma2[i,l,k])*Q2[k][l]

        for k in range(self.V.NumModes):
           Q2[k][k]=q2st[k]*q2st[k]
           for i in range(self.V.NumModes):
             F0[i]=F0[i]-(self.gamma2[i,k,k])*Q2[k][k]


        strainx = intrinsic.sol.strain_def(0,q2st,coeff['CPhi2'],V,coeff['BeamSeg'])
        kappax = intrinsic.sol.strain_def(1,q2st,coeff['CPhi2'],V,coeff['BeamSeg'])
        ra,Rab = intrinsic.sol.integration_strains(kappax,strainx,V=V,
                 BeamSeg=coeff['BeamSeg'],inverseconn=coeff['inverseconn'])
        rotation = [Rab[self.F.Dead_points_app[ix][0]][self.F.Dead_points_app[ix][1]] for ix in range(self.F.NumDLoads)]
        for i in range(self.V.NumModes):

               F0[i]=F0[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],rotation)[i]

        return F0

    def qstatic_dead_lin(self,q2st,coeff):
        import intrinsic.sol

        F0 = np.zeros(self.V.NumModes)
        strainx = intrinsic.sol.strain_def(0,q2st,coeff['CPhi2'],V,coeff['BeamSeg'])
        kappax = intrinsic.sol.strain_def(1,q2st,coeff['CPhi2'],V,coeff['BeamSeg'])
        ra,Rab = intrinsic.sol.integration_strains(kappax,strainx,V=V,
                 BeamSeg=coeff['BeamSeg'],inverseconn=coeff['inverseconn'])
        rotation = [Rab[self.F.Dead_points_app[ix][0]][self.F.Dead_points_app[ix][1]] for ix in range(self.F.NumDLoads)]
        for i in range(self.V.NumModes):

               F0[i]=F0[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],rotation)[i]

        return F0

class DynamicODE:
    """Class"""

    def __init__(self,Omega,Phi1,gamma1,gamma2,force1,V,F):
        self.Omega = Omega
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Phi1 = Phi1
        self.force1 = force1
        self.V = V
        self.F = F

    def q_gamma(self,q1,q2):
        dq1 = np.zeros(self.V.NumModes)
        dq2 = np.zeros(self.V.NumModes)
        Q12 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        Q22 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        Q11 = [[[] for lx in range(self.V.NumModes)] for kx in range(self.V.NumModes)]
        for k in range(self.V.NumModes):
            for l in range(k+1,self.V.NumModes):
              Q12[k][l] = q1[k]*q2[l] ; Q12[l][k] = q1[l]*q2[k]
              Q11[k][l]=q1[k]*q1[l];Q22[k][l]=q2[k]*q2[l]
              for j in range(self.V.NumModes):
               dq1[j]=dq1[j]-(self.gamma1[j,k,l]+self.gamma1[j,l,k])*Q11[k][l]-(self.gamma2[j,k,l]+self.gamma2[j,l,k])*Q22[k][l]
               dq2[j]=dq2[j]+self.gamma2[k,j,l]*Q12[k][l] + self.gamma2[l,j,k]*Q12[l][k]

        for k in range(self.V.NumModes):
            Q12[k][k] = q1[k]*q2[k]
            Q11[k][k]=q1[k]*q1[k];Q22[k][k]=q2[k]*q2[k]
            for j in range(self.V.NumModes):
               dq1[j]=dq1[j]-(self.gamma1[j,k,k])*Q11[k][k]-(self.gamma2[j,k,k])*Q22[k][k]
               dq2[j]=dq2[j]+self.gamma2[k,j,k]*Q12[k][k]

        return dq1,dq2

    def q_rotation_quaternion(self,q1,qr,args):
        if self.F.Gravity and self.V.rotation_quaternions:
            #dqr = np.zeros(4*NumNodes)
            #pdb.set_trace()
            dqr=[]
            for ix in range(self.V.NumBeams):
                for jx in range(args['BeamSeg'][ix].EnumNodes):
                    dqrx = np.zeros(4)
                    for j in range(self.V.NumModes):
                        dqrx[0] = dqrx[0] - 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qr[1]
                                + self.Phi1[ix][j][jx][4]*qr[2] + self.Phi1[ix][j][jx][5]*qr[3] )
                        dqrx[1] = dqrx[1] + 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qr[0]
                                + self.Phi1[ix][j][jx][5]*qr[2] - self.Phi1[ix][j][jx][4]*qr[3] )
                        dqrx[2] = dqrx[2] + 0.5*q1[j]*(self.Phi1[ix][j][jx][4]*qr[0]
                                - self.Phi1[ix][j][jx][5]*qr[1] + self.Phi1[ix][j][jx][3]*qr[3] )
                        dqrx[3] = dqrx[3] + 0.5*q1[j]*(self.Phi1[ix][j][jx][5]*qr[0]
                                + self.Phi1[ix][j][jx][4]*qr[1] - self.Phi1[ix][j][jx][3]*qr[2] )
                    dqr.append(dqrx)
            dqr=np.hstack(dqr)

        elif self.F.Gravity and self.V.rotation_strains:
            #dqr = np.zeros(4*NumNodes)
            #pdb.set_trace()
            dqr=[]
            for ix in range(self.V.NumBeams):
                jx = 0
                if ((ix in self.V.initialbeams and ix not in self.V.BeamsClamped) or ix in self.V.MBbeams):

                    dqrx = np.zeros(4)
                    for j in range(self.V.NumModes):
                        dqrx[0] = dqrx[0] - 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qr[1]
                                + self.Phi1[ix][j][jx][4]*qr[2] + self.Phi1[ix][j][jx][5]*qr[3] )
                        dqrx[1] = dqrx[1] + 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qr[0]
                                + self.Phi1[ix][j][jx][5]*qr[2] - self.Phi1[ix][j][jx][4]*qr[3] )
                        dqrx[2] = dqrx[2] + 0.5*q1[j]*(self.Phi1[ix][j][jx][4]*qr[0]
                                - self.Phi1[ix][j][jx][5]*qr[1] + self.Phi1[ix][j][jx][3]*qr[3] )
                        dqrx[3] = dqrx[3] + 0.5*q1[j]*(self.Phi1[ix][j][jx][5]*qr[0]
                                + self.Phi1[ix][j][jx][4]*qr[1] - self.Phi1[ix][j][jx][3]*qr[2] )
                    dqr.append(dqrx)
            dqr=np.hstack(dqr)

        elif self.F.NumDLoads>0:
            for j in range(self.V.NumModes-self.V.NumModes_res):
                if j == 0:
                    dqr = np.zeros(4*self.F.NumDLoads)
                for l in range(self.F.NumDLoads):
                    dqr[4*l+0] = dqr[4*l+0] - 0.5*q1[j]*(self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][3]*qr[4*l+1]
                               + self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][4]*qr[4*l+2]
                               + self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][5]*qr[4*l+3] )
                    dqr[4*l+1] = dqr[4*l+1] + 0.5*q1[j]*(self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][3]*qr[4*l+0]
                                + self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][5]*qr[4*l+2]
                                - self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][4]*qr[4*l+3] )
                    dqr[4*l+2] = dqr[4*l+2] + 0.5*q1[j]*(self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][4]*qr[4*l+0]
                               - self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][5]*qr[4*l+1]
                               + self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][3]*qr[4*l+3] )
                    dqr[4*l+3] = dqr[4*l+3] + 0.5*q1[j]*(self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][5]*qr[4*l+0]
                               + self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][4]*qr[4*l+1]
                               - self.Phi1[self.F.Dead_points_app[l][0]][j][self.F.Dead_points_app[l][1]][3]*qr[4*l+2] )

        else:
            dqr = []

        return dqr

    def q_rotation_matrix(self,q2,qr,args):
        strainx = intrinsic.solrb.strain_def(0,q2,args['CPhi2'],V,args['BeamSeg'])
        kappax = intrinsic.solrb.strain_def(1,q2,args['CPhi2'],V,args['BeamSeg'])
        Rabi=[]
        count=0
        for i in range(self.V.NumBeams):
            if ((i in self.V.initialbeams and i not in self.V.BeamsClamped) or i in self.V.MBbeams):
                Rabi.append(quaternion_matrix(qr[4*count:4*count+4])[:3,:3])
                count+=1
        Rab = intrinsic.solrb.integration_strains_rot(kappax,strainx,V=V,Rabi=Rabi)
        return Rab
    def q_aerostates(self,qa,q1,args):

        dqa = np.zeros((A.NumPoles,self.V.NumModes+A.rbd))
        for j in range(self.V.NumModes+A.rbd):
            for p in range(A.NumPoles):
                if j<A.rbd:
                    pass
                else:
                    dqa[p][j] = q1[j-A.rbd]-(2*A.u_inf*args['poles'][p]/A.c)*qa[p*(self.V.NumModes+A.rbd)+j]
        return np.hstack(dqa)

    def dq_12(self,t,q,args):

        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        dq1,dq2 = self.q_gamma(q1,q2)
        eta = self.force1.eta(t,q)
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+eta[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]

        return (np.hstack((dq1,dq2)))

    def dJq_12(self,t,q,args):
        dq11 = np.zeros((self.V.NumModes,self.V.NumModes))
        dq12 = np.zeros((self.V.NumModes,self.V.NumModes))
        dq21 = np.zeros((self.V.NumModes,self.V.NumModes))
        dq22 = np.zeros((self.V.NumModes,self.V.NumModes))
        for h in range(self.V.NumModes):
            for j in range(self.V.NumModes):

                dq11[j,h]=dq11[j,h]
                if j==h:
                    dq12[j,h]=dq12[j,h]+self.Omega[j]
                else:
                    dq12[j,h]=dq12[j,h]
                if j==h:
                    dq21[j,h]=dq21[j,h]-self.Omega[j]
                else:
                    dq21[j,h]=dq21[j,h]

                dq22[j,h]=dq22[j,h]

        return np.vstack((np.hstack((dq11,dq12)),np.hstack((dq21,dq22))))


    def dq_12lin(self,t,q,args):
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+self.force1.eta(t,q)[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]

        return (np.hstack((dq1,dq2)))

    def dq_12_rot(self,t,q,args):

        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        qr = q[2*self.V.NumModes:]
        if self.V.rotation_quaternions:
            Rab=[[] for i in range(self.V.NumBeams)]
            count=0
            for i in range(self.V.NumBeams):
                for j in range(args['BeamSeg'][i].EnumNodes):
                    Rab[i].append(quaternion_matrix(qr[4*count:4*count+4])[:3,:3])
                    count+=1
            dqr = self.q_rotation_quaternion(q1,qr,args)
        elif self.V.rotation_strains:
            dqr = self.q_rotation_quaternion(q1,qr,args)
            Rab=self.q_rotation_matrix(q2,qr,args)
        else:
            Rab=None
            dqr = self.q_rotation_quaternion(q1,qr,args)
        dq1,dq2 = self.q_gamma(q1,q2)
        #pdb.set_trace()
        eta = self.force1.eta(t,q,rotation=Rab)
        #pdb.set_trace()
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+eta[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]

        return np.hstack((dq1,dq2,dqr))

    def dq_aero(self,t,q,args):
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        q0 = q[2*self.V.NumModes:3*self.V.NumModes]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = args['Aqinv'].dot(dq1+self.Omega*q2+self.force1.eta(t,q))
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        if A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes:])
            #dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = q_aerostates(qa)
        else:
            dqa=[]
        return np.hstack((dq1,dq2,dq0,dqa))

    def dq_aero_rbd(self,t,q,args):
        q1 = q[A.rbd:self.V.NumModes+A.rbd]
        q2 = q[self.V.NumModes+A.rbd:2*self.V.NumModes+A.rbd]
        #q0 = q[2*self.V.NumModes+2*A.rbd:3*self.V.NumModes+2*A.rbd]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = np.insert(dq1,0,np.zeros(A.rbd))
        #pdb.set_trace()
        dq1 = args['Aqinv'].dot(dq1+np.hstack([np.zeros(A.rbd),self.Omega*q2])+self.force1.forceAero_eta_rbd(q))
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        dq0 = np.insert(dq0,0,np.zeros(A.rbd))
        if A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes+2*A.rbd:])
            #dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(qa,q1,args)
            #for i in range(A.NumPoles):
               # dqa = np.insert(dqa,i*(A.rbd+self.V.NumModes),np.zeros(A.rbd))
        else:
            dqa=[]
        return np.hstack((dq1,dq2,dq0,dqa))

    def dq_aerolin(self,t,q,args):
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        q0 = q[2*self.V.NumModes:3*self.V.NumModes]
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+args['force1'].eta(t,q)[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]
        dq1 = args['Aqinv'].dot(self.Omega*q2+force1.eta(t,q))
        dq2 = -self.Omega*q1
        dq0 = q1
        if NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes:])
            #dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = q_aerostates(qa)
        else:
            dqa=[]
        return np.hstack((dq1,dq2,dqa))

    def dq_aero_rot(self,t,q,args):

        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        q0 = q[2*self.V.NumModes:3*self.V.NumModes]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = args['Aqinv'].dot(dq1+self.Omega*q2+force1.eta(t,q))
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        if NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes:3*self.V.NumModes+NumPoles*self.V.NumModes])
            dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = q_aerostates(qa)
            qr = q[3*self.V.NumModes+NumPoles*self.V.NumModes:]
            dqr = self.q_rotation(qr)
        else:
            dqa=[]
            qr = q[3*self.V.NumModes:]
            dqr = self.q_rotation(qr)
        return np.hstack((dq1,dq2,dqa,dqr))




class MultibodyODE:
    """Class"""

    def __init__(self,Omega,Phi1,gamma1,gamma2,force1,V,F,NumBodies):
        self.Omega = Omega
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Phi1 = Phi1
        self.Phi11 = Phi1[0][0][:,-1,0:3].T
        self.Phi12 = Phi1[1][0][:,0,0:3].T
        self.force1 = force1
        self.V = V
        self.F = F
        self.m = [[] for i in range(NumBodies)]
        for i in range(NumBodies):
            self.m[i] = DynamicODE(Omega[i],Phi1[i],gamma1[i],gamma2[i],force1[i],V[i],F[i])

    def Lv(self,l):
        l0,l1,l2,l3 = l
        
        return np.array([[l0**2+l1**2-l2**2-l3**2, -2*l0*l3+2*l1*l2, 2*l0*l2+2*l1*l3],
                         [2*l0*l3+2*l1*l2, l0**2-l1**2+l2**2-l3**2, -2*l0*l1+2*l2*l3],
                         [-2*l0*l2+2*l1*l3, 2*l0*l1+2*l2*l3, l0**2-l1**2-l2**2+l3**2]])
        # return np.array([[-l3**3+l0**2+l1**2-l2**2, -2*l0*l3+2*l1*l2, 2*l0*l2+2*l1*l3],
        #                  [2*l0*l3+2*l1*l2, -l3**3+l0**2-l1**2+l2**2, -2*l0*l1+2*l2*l3],
        #                  [-2*l0*l2+2*l1*l3, 2*l0*l1+2*l2*l3, -l3**3+l0**2-l1**2-l2**2+2*l3**2]])
    def L0(self,l,v):
        l0,l1,l2,l3 = l
        v1,v2,v3 = v

        return np.array([[2*l0*v1+2*l2*v3-2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v3+2*l1*v2-2*l2*v1, -2*l0*v2+2*l1*v3-2*l3*v1],
                         [2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v3-2*l1*v2+2*l2*v1, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v1+2*l2*v3-2*l3*v2],
                         [2*l0*v3+2*l1*v2-2*l2*v1, 2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v1-2*l2*v3+2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3]])
        # return np.array([[2*l0*v1+2*l2*v3-2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v3+2*l1*v2-2*l2*v1,
        #     -3*l3**2*v1-2*l0*v2+2*l1*v3], [2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v3-2*l1*v2+2*l2*v1,
        #     2*l1*v1+2*l2*v2+2*l3*v3, -3*l3**2*v2+2*l0*v1+2*l2*v3], [2*l0*v3+2*l1*v2-2*l2*v1,
        #     2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v1-2*l2*v3+2*l3*v2, -3*l3**2*v3+2*l1*v1+2*l2*v2+4*l3*v3]])

    def dq_12_rot(self,t,q,args):
        q11 = q[:self.V[0].NumModes]
        q21 = q[self.V[0].NumModes:2*self.V[0].NumModes]
        q12 = q[2*self.V[0].NumModes:2*self.V[0].NumModes+self.V[1].NumModes]
        q22 = q[2*self.V[0].NumModes+self.V[1].NumModes:2*self.V[0].NumModes+2*self.V[1].NumModes]
        q01 = q[2*self.V[0].NumModes+2*self.V[1].NumModes:2*self.V[0].NumModes+2*self.V[1].NumModes+4*2]
        q02 = q[2*self.V[0].NumModes+2*self.V[1].NumModes+4*2:2*self.V[0].NumModes+2*self.V[1].NumModes+8*2]
        v1=v2=0.
        for im in range(self.V[0].NumModes):
            v1+=self.Phi1[0][0][im][-1,:3]*q11[im]
        for im in range(self.V[1].NumModes):
            v2+=self.Phi1[1][0][im][0,:3]*q12[im]

        #pdb.set_trace()    
        args['BeamSeg'] = args['BeamSeg1']
        args['inverseconn'] = args['inverseconn1']
        dq1=self.m[0].dq_12_rot(t,np.hstack((q11,q21,q01)),args)
        args['BeamSeg'] = args['BeamSeg2']
        args['inverseconn'] = args['inverseconn2']
        
        dq2=self.m[1].dq_12_rot(t,np.hstack((q12,q22,q02)),args)
        dq11 = dq1[:self.V[0].NumModes]
        dq21 = dq1[self.V[0].NumModes:2*self.V[0].NumModes]
        dq12 = dq2[:self.V[1].NumModes]
        dq22 = dq2[self.V[1].NumModes:2*self.V[1].NumModes]
        dq01 = dq1[2*self.V[0].NumModes:2*self.V[0].NumModes+4*2]
        dq02 = dq2[2*self.V[1].NumModes:2*self.V[1].NumModes+4*2]
        L1 = self.Lv(q01[4:]); L2 = -self.Lv(q02[:4])
        L01 = self.L0(q01[4:],v1); L02 = -self.L0(q02[:4],v2)
        lambda12 = np.linalg.inv(L1.dot(self.Phi11.dot(self.Phi11.T)).dot(L1.T)+L2.dot(self.Phi12.dot(self.Phi12.T)).dot(L2.T)).dot(L01.dot(dq01[4:])+L02.dot(dq02[:4])+L1.dot(self.Phi11).dot(dq11)+L2.dot(self.Phi12).dot(dq12))
        #lambda12 = np.zeros(3)
        dq11+=-(L1.dot(self.Phi11)).T.dot(lambda12)
        dq12+=-(L2.dot(self.Phi12)).T.dot(lambda12)
        return np.hstack((dq11,dq21,dq12,dq22,dq01,dq02))

    # def dq_12_rot(self,t,q,args):

    #     q1 = q[0:self.V.NumModes]
    #     q2 = q[self.V.NumModes:2*self.V.NumModes]
    #     qr = q[2*self.V.NumModes:]
    #     if self.V.rotation_quaternions:
    #         Rab=[[] for i in range(self.V.NumBeams)]
    #         count=0
    #         for i in range(self.V.NumBeams):
    #             for j in range(args['BeamSeg'][i].EnumNodes):
    #                 Rab[i].append(quaternion_matrix(qr[4*count:4*count+4])[:3,:3])
    #                 count+=1
    #         dqr = self.q_rotation_quaternion(q1,qr,args)
    #     elif self.V.rotation_strains:
    #         dqr = self.q_rotation_quaternion(q1,qr,args)
    #         Rab=self.q_rotation_matrix(q2,qr,args)
    #     else:
    #         Rab=None
    #         dqr = self.q_rotation_quaternion(q1,qr,args)
    #     dq1,dq2 = self.q_gamma(q1,q2)
    #     #pdb.set_trace()
    #     eta = self.force1.eta(t,q,rotation=Rab)
    #     #pdb.set_trace()
    #     for j in range(self.V.NumModes):
    #         dq1[j]=dq1[j]+self.Omega[j]*q2[j]+eta[j]
    #         dq2[j]=dq2[j]-self.Omega[j]*q1[j]

    #     return np.hstack((dq1,dq2,dqr))
