import pdb
import numpy as np
import importlib
import Runs.Torun
#import intrinsic.sol
import intrinsic.solrb
from Tools.transformations import quaternion_from_matrix,quaternion_matrix
# V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
# F = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.force)
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


        strainx = intrinsic.sol.strain_def(0,q2st,coeff['CPhi2'],self.V,coeff['BeamSeg'])
        kappax = intrinsic.sol.strain_def(1,q2st,coeff['CPhi2'],self.V,coeff['BeamSeg'])
        ra,Rab = intrinsic.sol.integration_strains(kappax,strainx,V=self.V,
                 BeamSeg=coeff['BeamSeg'],inverseconn=coeff['inverseconn'])
        rotation = [Rab[self.F.Dead_points_app[ix][0]][self.F.Dead_points_app[ix][1]] for ix in range(self.F.NumDLoads)]
        for i in range(self.V.NumModes):

               F0[i]=F0[i]+self.Omega[i]*(q2st[i])+self.force1.eta(coeff['tix'],[],rotation)[i]

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

    def __init__(self,Omega,Phi1,gamma1,gamma2,force1,V,F,A=None,mb_constrains=None):
        self.Omega = Omega
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Phi1 = Phi1
        self.force1 = force1
        self.V = V
        self.F = F
        self.A = A
        self.mb_constrains = mb_constrains
    def q_gamma(self,q1,q2):
        """Function to calculate the nonlinear quadratic terms of the intrinsic Eqs., Gamma1*q1q1
           Gamma2*q2q2,Gamma12*q1*q2."""
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
        """Function to calculate the deravative of rotational states, qr, from quaternions."""
        dqr=[]
        if self.F.Gravity and self.V.rotation_quaternions:
            #dqr = np.zeros(4*NumNodes)
            #pdb.set_trace()
            count = 0
            for ix in range(self.V.NumBeams):
                for jx in range(args['BeamSeg'][ix].EnumNodes):
                    dqrx = np.zeros(4)
                    qrx = qr[count*4:(count+1)*4]
                    for j in range(self.V.NumModes):
                        dqrx[0] = dqrx[0] - 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[1]
                                + self.Phi1[ix][j][jx][4]*qrx[2] + self.Phi1[ix][j][jx][5]*qrx[3] )
                        dqrx[1] = dqrx[1] + 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[0]
                                + self.Phi1[ix][j][jx][5]*qrx[2] - self.Phi1[ix][j][jx][4]*qrx[3] )
                        dqrx[2] = dqrx[2] + 0.5*q1[j]*(self.Phi1[ix][j][jx][4]*qrx[0]
                                - self.Phi1[ix][j][jx][5]*qrx[1] + self.Phi1[ix][j][jx][3]*qrx[3] )
                        dqrx[3] = dqrx[3] + 0.5*q1[j]*(self.Phi1[ix][j][jx][5]*qrx[0]
                                + self.Phi1[ix][j][jx][4]*qrx[1] - self.Phi1[ix][j][jx][3]*qrx[2] )
                    count+=1
                    dqr.append(dqrx)

        elif self.F.Gravity and self.V.rotation_strains:
            #dqr = np.zeros(4*NumNodes)
            #pdb.set_trace()
            count = 0
            for ix in range(self.V.NumBeams):
                jx = 0
                if ((ix in self.V.initialbeams and ix not in self.V.BeamsClamped) or ix in self.V.MBbeams):
                        dqrx = np.zeros(4)
                        qrx = qr[count*4:(count+1)*4]
                        for j in range(self.V.NumModes):
                            dqrx[0] = dqrx[0] - 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[1]
                                    + self.Phi1[ix][j][jx][4]*qrx[2] + self.Phi1[ix][j][jx][5]*qrx[3] )
                            dqrx[1] = dqrx[1] + 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[0]
                                    + self.Phi1[ix][j][jx][5]*qrx[2] - self.Phi1[ix][j][jx][4]*qrx[3] )
                            dqrx[2] = dqrx[2] + 0.5*q1[j]*(self.Phi1[ix][j][jx][4]*qrx[0]
                                    - self.Phi1[ix][j][jx][5]*qrx[1] + self.Phi1[ix][j][jx][3]*qrx[3] )
                            dqrx[3] = dqrx[3] + 0.5*q1[j]*(self.Phi1[ix][j][jx][5]*qrx[0]
                                    + self.Phi1[ix][j][jx][4]*qrx[1] - self.Phi1[ix][j][jx][3]*qrx[2] )
                        count+=1
                        dqr.append(dqrx)

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

        if self.mb_constrains is not None:
            #dqr = np.zeros(4*NumNodes)
            #pdb.set_trace()
            count_cons=0
            for ci in range(len(self.mb_constrains.keys())):
                    if self.mb_constrains['c%s'%ci][0][0] == args['mb_body']:

                        dqrx = np.zeros(4)
                        ix = self.mb_constrains['c%s'%ci][1][0]
                        jx = -1
                        qrx  = qr[4*count_cons:4*(count_cons+1)]
                        for j in range(self.V.NumModes):
                            dqrx[0] = dqrx[0] - 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[1]
                                    + self.Phi1[ix][j][jx][4]*qrx[2] + self.Phi1[ix][j][jx][5]*qrx[3] )
                            dqrx[1] = dqrx[1] + 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[0]
                                    + self.Phi1[ix][j][jx][5]*qrx[2] - self.Phi1[ix][j][jx][4]*qrx[3] )
                            dqrx[2] = dqrx[2] + 0.5*q1[j]*(self.Phi1[ix][j][jx][4]*qrx[0]
                                    - self.Phi1[ix][j][jx][5]*qrx[1] + self.Phi1[ix][j][jx][3]*qrx[3] )
                            dqrx[3] = dqrx[3] + 0.5*q1[j]*(self.Phi1[ix][j][jx][5]*qrx[0]
                                    + self.Phi1[ix][j][jx][4]*qrx[1] - self.Phi1[ix][j][jx][3]*qrx[2] )
                        dqr.append(dqrx)
                        count_cons +=1

                    elif self.mb_constrains['c%s'%ci][0][1] == args['mb_body']:
                        dqrx = np.zeros(4)
                        ix = self.mb_constrains['c%s'%ci][1][1]
                        jx = 0
                        qrx  = qr[4*count_cons:4*(count_cons+1)]
                        for j in range(self.V.NumModes):
                            dqrx[0] = dqrx[0] - 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[1]
                                    + self.Phi1[ix][j][jx][4]*qrx[2] + self.Phi1[ix][j][jx][5]*qrx[3] )
                            dqrx[1] = dqrx[1] + 0.5*q1[j]*(self.Phi1[ix][j][jx][3]*qrx[0]
                                    + self.Phi1[ix][j][jx][5]*qrx[2] - self.Phi1[ix][j][jx][4]*qrx[3] )
                            dqrx[2] = dqrx[2] + 0.5*q1[j]*(self.Phi1[ix][j][jx][4]*qrx[0]
                                    - self.Phi1[ix][j][jx][5]*qrx[1] + self.Phi1[ix][j][jx][3]*qrx[3] )
                            dqrx[3] = dqrx[3] + 0.5*q1[j]*(self.Phi1[ix][j][jx][5]*qrx[0]
                                    + self.Phi1[ix][j][jx][4]*qrx[1] - self.Phi1[ix][j][jx][3]*qrx[2] )
                        dqr.append(dqrx)
                        count_cons +=1

        if dqr !=[]:
            dqr=np.hstack(dqr)

        return dqr

    def q_rotation_matrix(self,q2,qr,args):
        """Function to give the rotation matrix at each node after integration from strains."""
        strainx = intrinsic.solrb.strain_def(0,q2,args['CPhi2'],self.V,args['BeamSeg'])
        kappax = intrinsic.solrb.strain_def(1,q2,args['CPhi2'],self.V,args['BeamSeg'])
        Rabi=[]
        count=0
        for i in range(self.V.NumBeams):
            if ((i in self.V.initialbeams and i not in self.V.BeamsClamped) or i in self.V.MBbeams):
                Rabi.append(quaternion_matrix(qr[4*count:4*count+4])[:3,:3])
                count+=1
        #pdb.set_trace()
        Rab = intrinsic.solrb.integration_strains_rot(kappax,strainx,V=self.V,BeamSeg=args['BeamSeg'],inverseconn=args['inverseconn'],Rabi=Rabi)
        return Rab

    def q_rotation(self,q1,q2,qr,args):
        """Returns rotational states and matrices as required for the dynamic system"""
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

        return dqr,Rab

    def q_aerostates(self,t,qa,q1,args):
        """Return the derivative of aerodynamic states."""
        if self.A.GustOn:
            dqa = np.zeros((self.A.NumPoles,self.V.NumModes+self.A.rbd))
            force_lags = self.force1.forceAeroStates(q1,t)
            for j in range(self.V.NumModes+self.A.rbd):
                for p in range(self.A.NumPoles):
                    if j<self.A.rbd:
                        pass
                    else:
                        dqa[p][j] = -(2*self.A.u_inf*args['poles'][p]/self.A.c)*qa[p*(self.V.NumModes+self.A.rbd)+j]+force_lags[p][j-self.A.rbd]

        else:
            dqa = np.zeros((self.A.NumPoles,self.V.NumModes+self.A.rbd))
            for j in range(self.V.NumModes+self.A.rbd):
                for p in range(self.A.NumPoles):
                    if j<self.A.rbd:
                        pass
                    else:
                        dqa[p][j] = q1[j-self.A.rbd]-(2*self.A.u_inf*args['poles'][p]/self.A.c)*qa[p*(self.V.NumModes+self.A.rbd)+j]
        return np.hstack(dqa)

    def dq_12(self,t,q,args):
        """Solver for structural dynamics with follower forces."""
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        dq1,dq2 = self.q_gamma(q1,q2)
        eta = self.force1.eta(t,q)
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+eta[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]

        return (np.hstack((dq1,dq2)))

    def dJq_12(self,t,q,args):
        """Returns the Jacobian of the previous function."""
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
        """Solver for linear structural dynamics (No Gammas) with follower forces."""
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+self.force1.eta(t,q)[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]

        return (np.hstack((dq1,dq2)))

    def dq_12_rot(self,t,q,args):
        """Solver for structural dynamics with follower, dead and gravity forces."""
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        qr = q[2*self.V.NumModes:]
        # if self.V.rotation_quaternions:
        #     Rab=[[] for i in range(self.V.NumBeams)]
        #     count=0
        #     for i in range(self.V.NumBeams):
        #         for j in range(args['BeamSeg'][i].EnumNodes):
        #             Rab[i].append(quaternion_matrix(qr[4*count:4*count+4])[:3,:3])
        #             count+=1
        #     dqr = self.q_rotation_quaternion(q1,qr,args)
        # elif self.V.rotation_strains:
        #     dqr = self.q_rotation_quaternion(q1,qr,args)
        #     Rab=self.q_rotation_matrix(q2,qr,args)
        # else:
        #     Rab=None
        #     dqr = self.q_rotation_quaternion(q1,qr,args)
        dq1,dq2 = self.q_gamma(q1,q2)
        #pdb.set_trace()
        dqr,Rab = self.q_rotation(q1,q2,qr,args)
        eta = self.force1.eta(t,q,rotation=Rab)
        for j in range(self.V.NumModes):
            dq1[j]=dq1[j]+self.Omega[j]*q2[j]+eta[j]
            dq2[j]=dq2[j]-self.Omega[j]*q1[j]

        return np.hstack((dq1,dq2,dqr))

    def dq_aero(self,t,q,args):
        """Solver for aeroelastic system (aero+follower forces)."""
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        q0 = q[2*self.V.NumModes:3*self.V.NumModes]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = args['Aqinv'].dot(dq1+self.Omega*q2+self.force1.eta(t,q,args=args))
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes:])
            #dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
        else:
            dqa=[]
        return np.hstack((dq1,dq2,dq0,dqa))

    # def dq_aero_lin(self,t,q,args):
    #     """Solver for linear aeroelastic system (aero+follower forces)."""
    #     q1 = q[0:self.V.NumModes]
    #     q2 = q[self.V.NumModes:2*self.V.NumModes]
    #     q0 = q[2*self.V.NumModes:3*self.V.NumModes]
    #     dq1 = np.zeros(self.V.NumModes)
    #     dq2 = np.zeros(self.V.NumModes)
    #     eta1 = self.force1.eta(t,q,args=args)
    #     for j in range(self.V.NumModes):
    #         dq1[j]=dq1[j]+self.Omega[j]*q2[j]+eta1[j]
    #         dq2[j]=dq2[j]-self.Omega[j]*q1[j]
    #     dq1 = args['Aqinv'].dot(dq1)
    #     dq0 = q1
    #     if self.A.NumPoles > 0:
    #         qa = np.array(q[3*self.V.NumModes:])
    #         #dqa = np.zeros((NumPoles,self.V.NumModes))
    #         dqa = self.q_aerostates(t,qa,q1,args)
    #     else:
    #         dqa=[]
    #     return np.hstack((dq1,dq2,dq0,dqa))

    def dq_aero_lin(self,t,q,args):
        """Solver for linear aeroelastic system (aero+follower forces)."""
        q1 = q[self.A.rbd:self.V.NumModes+self.A.rbd]
        q2 = q[self.V.NumModes+self.A.rbd:2*self.V.NumModes+self.A.rbd]
        dq1 = np.zeros(self.V.NumModes+self.A.rbd)
        dq2 = np.zeros(self.V.NumModes)
        dq1 = args['Aqinv'].dot(dq1+np.hstack([np.zeros(self.A.rbd),self.Omega*q2])+self.force1.eta(t,q,args=args))
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        dq0 = np.insert(dq0,0,np.zeros(self.A.rbd))
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes+2*self.A.rbd:])
            #dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
            #for i in range(self.A.NumPoles):
               # dqa = np.insert(dqa,i*(self.A.rbd+self.V.NumModes),np.zeros(self.A.rbd))
        else:
            dqa=[]
        return np.hstack((dq1,dq2,dq0,dqa))


    def dq_aero_rbd(self,t,q,args):
        """Solver for aeroelastic system with a rigid-body fixed and follower forces"""
        q1 = q[self.A.rbd:self.V.NumModes+self.A.rbd]
        q2 = q[self.V.NumModes+self.A.rbd:2*self.V.NumModes+self.A.rbd]
        #q0 = q[2*self.V.NumModes+2*self.A.rbd:3*self.V.NumModes+2*self.A.rbd]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = np.insert(dq1,0,np.zeros(self.A.rbd))
        #pdb.set_trace()
        dq1 = args['Aqinv'].dot(dq1+np.hstack([np.zeros(self.A.rbd),self.Omega*q2])+self.force1.eta(t,q,args=args))
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        dq0 = np.insert(dq0,0,np.zeros(self.A.rbd))
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes+2*self.A.rbd:])
            #dqa = np.zeros((NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
            #for i in range(self.A.NumPoles):
               # dqa = np.insert(dqa,i*(self.A.rbd+self.V.NumModes),np.zeros(self.A.rbd))
        else:
            dqa=[]
        return np.hstack((dq1,dq2,dq0,dqa))

    def dq_aero_rot(self,t,q,args):
        """Solver for aeroelastic system with follower, dead, and gravity forces"""
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        #q0 = q[2*self.V.NumModes:3*self.V.NumModes]
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes:3*self.V.NumModes+self.A.NumPoles*self.V.NumModes])
            #dqa = np.zeros((self.A.NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
            qr = q[3*self.V.NumModes+self.A.NumPoles*self.V.NumModes:]
        else:
            dqa=[]
            qr = q[3*self.V.NumModes:]

        dq1,dq2 = self.q_gamma(q1,q2)
        dqr,Rab = self.q_rotation(q1,q2,qr,args)
        dq1 = args['Aqinv'].dot(dq1+self.Omega*q2+self.force1.eta(t,q,rotation=Rab,args=args))
        dq2 = dq2-self.Omega*q1
        dq0 = q1

        return np.hstack((dq1,dq2,dq0,dqa,dqr))

    def dq_aero_rot_lin(self,t,q,args):
        """Solver for linear aeroelastic system without Gammas but with  follower, dead, and gravity forces"""
        q1 = q[0:self.V.NumModes]
        q2 = q[self.V.NumModes:2*self.V.NumModes]
        #q0 = q[2*self.V.NumModes:3*self.V.NumModes]
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes:3*self.V.NumModes+self.A.NumPoles*self.V.NumModes])
            #dqa = np.zeros((self.A.NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
            qr = q[3*self.V.NumModes+self.A.NumPoles*self.V.NumModes:]
        else:
            dqa=[]
            qr = q[3*self.V.NumModes:]

        dq1 = np.zeros(self.V.NumModes+self.A.rbd)
        dq2 = np.zeros(self.V.NumModes)
        dqr,Rab = self.q_rotation(q1,q2,qr,args)
        dq1 = args['Aqinv'].dot(dq1+self.Omega*q2+self.force1.eta(t,q,rotation=Rab,args=args))
        dq2 = dq2-self.Omega*q1
        dq0 = q1

        return np.hstack((dq1,dq2,dq0,dqa,dqr))

    def dq_aero_rot_rbd(self,t,q,args):
        """Solver for aeroelastic system with follower, dead, gravity forces, and rigid-body fixed"""
        q1 = q[self.A.rbd:self.V.NumModes+self.A.rbd]
        q2 = q[self.V.NumModes+self.A.rbd:2*self.V.NumModes+self.A.rbd]
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes+2*self.A.rbd:3*self.V.NumModes+self.A.NumPoles*self.V.NumModes+(2+self.A.NumPoles)*self.A.rbd])
            #dqa = np.zeros((self.A.NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
            qr = q[3*self.V.NumModes+self.A.NumPoles*self.V.NumModes+(2+self.A.NumPoles)*self.A.rbd:]
        else:
            dqa=[]
            qr = q[3*self.V.NumModes+2*self.A.rbd:]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = np.insert(dq1,0,np.zeros(self.A.rbd))
        dqr,Rab = self.q_rotation(q1,q2,qr,args)
        #pdb.set_trace()
        #eta = self.force1.forceAero_eta_rbd(q)
        #eta = np.hstack([np.zeros(self.A.rbd),self.force1.eta(t,q,rotation=Rab)])
        eta = self.force1.eta(t,q,rotation=Rab,args=args)
        dq1 = args['Aqinv'].dot(dq1+np.hstack([np.zeros(self.A.rbd),self.Omega*q2])+eta)
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        dq0 = np.insert(dq0,0,np.zeros(self.A.rbd))

        return np.hstack((dq1,dq2,dq0,dqa,dqr))

# NumBodies=2
# NumConstrains=1
# Constrains={'c1':[[bodies],[beams],[nodes],[qrot],[dof]]}
class MultibodyODE:
    """Class"""

    def __init__(self,Omega,Phi1,CPhi2,gamma1,gamma2,force1,BeamSeg,inverseconn,V,F,A=None):
        self.Omega = Omega
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Phi1 = Phi1
        self.CPhi2 = CPhi2
        self.force1 = force1
        self.BeamSeg = BeamSeg
        self.inverseconn = inverseconn
        self.V = V
        self.F = F
        self.A = A

        self.NumBodies = self.V[0].NumBodies
        self.m = [[] for i in range(self.NumBodies)]
        if self.A==None:
            for i in range(self.NumBodies):
                self.m[i] = DynamicODE(Omega[i],Phi1[i],gamma1[i],gamma2[i],force1[i],V[i],F[i],mb_constrains=V[0].Constrains)

        else:
            for i in range(self.NumBodies):
                self.m[i] = DynamicODE(Omega[i],Phi1[i],gamma1[i],gamma2[i],force1[i],V[i],F[i],A[i],mb_constrains=V[0].Constrains)
        self.Phi11=[];self.Phi12=[]
        self.Phi11o=[];self.Phi12o=[]
        self.b1 = [] ; self.b2 = []
        self.quat1 = [] ; self.quat2 = []
        #self.node1 = [] ; self.node2 = []
        self.beam1 = [] ; self.beam2 = []
        #self.rot_node1 = [] ; self.rot_node2 = []
        self.hinge=[]
        self.Gxq1=[]; self.Gxq2=[];self.Gxq1T=[]; self.Gxq2T=[]
        for ci in range(self.V[0].NumConstrains):
            self.b1.append(self.V[0].Constrains['c%s'%ci][0][0]); self.b2.append(self.V[0].Constrains['c%s'%ci][0][1])
            self.beam1.append(self.V[0].Constrains['c%s'%ci][1][0]); self.beam2.append(self.V[0].Constrains['c%s'%ci][1][1])
            self.quat1.append(self.V[0].Constrains['c%s'%ci][4][0]); self.quat2.append(self.V[0].Constrains['c%s'%ci][4][1])
            #self.node1.append(Constrains['c%s'%ci][2][0]); self.node2.append(Constrains['c%s'%ci][2][1])
            #self.rot_node1.append(Constrains['c%s'%ci][3][0]); self.rot_node2.append(Constrains['c%s'%ci][3][1])
            self.Phi11.append(Phi1[self.b1[ci]][self.beam1[ci]][:,-1,:3])
            self.Phi12.append(Phi1[self.b2[ci]][self.beam2[ci]][:,0,:3])
            self.Phi11o.append(Phi1[self.b1[ci]][self.beam1[ci]][:,-1,3:])
            self.Phi12o.append(Phi1[self.b2[ci]][self.beam2[ci]][:,0,3:])
            if '1' in self.V[0].Constrains['c%s'%ci][2]:
                self.hinge.append(1)
                self.Gxq1.append(np.cross(self.Phi11o[ci],self.V[0].Constrains['c%s'%ci][3][0]).T)
                self.Gxq2.append(-np.cross(self.Phi12o[ci],self.V[0].Constrains['c%s'%ci][3][1]).T)
                removeik =[]
                #pdb.set_trace()
                for ik in range(3):
                    if np.allclose(self.Gxq1[ci][ik,:],np.zeros(len(self.Gxq1[ci][ik,:]))) and np.allclose(self.Gxq2[ci][ik,:],np.zeros(len(self.Gxq2[ci][ik,:]))):
                       removeik.append(ik)
                if removeik:
                    self.Gxq1[ci] = np.delete(self.Gxq1[ci],removeik,0)
                    self.Gxq2[ci] = np.delete(self.Gxq2[ci],removeik,0)
                self.Gxq1T.append(self.Gxq1[ci].T)
                self.Gxq2T.append(self.Gxq2[ci].T)

            else:
                self.hinge.append(0)
                self.Gxq1.append([])
                self.Gxq2.append([])
                self.Gxq1T.append([])
                self.Gxq2T.append([])
    def R(self,l):
        l0,l1,l2,l3 = l

        return np.array([[l0**2+l1**2-l2**2-l3**2, -2*l0*l3+2*l1*l2, 2*l0*l2+2*l1*l3],
                         [2*l0*l3+2*l1*l2, l0**2-l1**2+l2**2-l3**2, -2*l0*l1+2*l2*l3],
                         [-2*l0*l2+2*l1*l3, 2*l0*l1+2*l2*l3, l0**2-l1**2-l2**2+l3**2]])
    def Gvz(self,l,v):
        l0,l1,l2,l3 = l
        v1,v2,v3 = v

        return np.array([[2*l0*v1+2*l2*v3-2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v3+2*l1*v2-2*l2*v1, -2*l0*v2+2*l1*v3-2*l3*v1],
                         [2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v3-2*l1*v2+2*l2*v1, 2*l1*v1+2*l2*v2+2*l3*v3, 2*l0*v1+2*l2*v3-2*l3*v2],
                         [2*l0*v3+2*l1*v2-2*l2*v1, 2*l0*v2-2*l1*v3+2*l3*v1, -2*l0*v1-2*l2*v3+2*l3*v2, 2*l1*v1+2*l2*v2+2*l3*v3]])

    def q_12_rot(self,t,qb,args):
        dqb = []
        for bi in range(self.NumBodies):
            args['BeamSeg'] = self.BeamSeg[bi]
            args['CPhi2'] = self.CPhi2[bi]
            args['inverseconn'] = self.inverseconn[bi]
            args['mb_body'] = bi
            dqb.append(self.m[bi].dq_12_rot(t,qb[bi],args))
        vc1=[]
        vc2=[]
        lambdav = []
        lambdavx = []
        for ci in range(self.V[0].NumConstrains):
            v1=v2=0.
            omega1=0.; omega2=0.
            for im in range(self.V[0].NumModes):
                v1+=self.Phi11[ci][im]*qb[self.b1[ci]][im]
                omega1+=self.Phi11o[ci][im]*qb[self.b1[ci]][im]
            for im in range(self.V[1].NumModes):
                v2+=self.Phi12[ci][im]*qb[self.b2[ci]][im]
                omega2+=self.Phi12o[ci][im]*qb[self.b2[ci]][im]
            vc1.append(v1)
            vc2.append(v2)
            qrot1 = np.hstack([qb[self.b1[ci]][-4*self.quat1[ci]:-4*self.quat1[ci]+3],qb[self.b1[ci]][-4*self.quat1[ci]+3]])
            qrot2 = np.hstack([qb[self.b2[ci]][-4*self.quat2[ci]:-4*self.quat2[ci]+3],qb[self.b2[ci]][-4*self.quat2[ci]+3]])
            dqrot1 = np.hstack([dqb[self.b1[ci]][-4*self.quat1[ci]:-4*self.quat1[ci]+3],dqb[self.b1[ci]][-4*self.quat1[ci]+3]])
            dqrot2 = np.hstack([dqb[self.b2[ci]][-4*self.quat2[ci]:-4*self.quat2[ci]+3],dqb[self.b2[ci]][-4*self.quat2[ci]+3]])

            R1 = self.R(qrot1); R2 = self.R(qrot2)
            #pdb.set_trace()
            Gvq1 = R1.dot(self.Phi11[ci].T); Gvq2 = -R2.dot(self.Phi12[ci].T)
            Gvq1T = R1.dot(self.Phi11[ci].T).T; Gvq2T = -R2.dot(self.Phi12[ci].T).T
            Gvz1 = self.Gvz(qrot1,vc1[ci]); Gvz2 = -self.Gvz(qrot2,vc2[ci])

            if self.hinge[ci]:

                G = np.vstack([np.hstack([Gvq1,Gvq2]),np.hstack([self.Gxq1[ci],self.Gxq2[ci]])])
                S = np.hstack([Gvq1.dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+Gvz1.dot(dqrot1)+Gvq2.dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])+Gvz2.dot(dqrot2),self.Gxq1[ci].dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+self.Gxq2[ci].dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])])
                lambdavx.append(np.linalg.inv(G.dot(G.T)).dot(S))
                lambdav.append([])
                dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes]-Gvq1T.dot(lambdavx[ci][:3])-self.Gxq1T[ci].dot(lambdavx[ci][3:])
                dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes]-Gvq2T.dot(lambdavx[ci][:3])-self.Gxq2T[ci].dot(lambdavx[ci][3:])

            #L1 = self.Lv(qrot1); L2 = -self.Lv(qrot2)
            #L01 = self.L0(qrot1,vc1[ci]); L02 = -self.L0(qrot2,vc2[ci])
            else:
                lambdavx.append([])
                lambdav.append(np.linalg.inv(Gvq1.dot(Gvq1T)+Gvq2.dot(Gvq2T)).dot(Gvz1.dot(dqrot1)+
                       Gvz2.dot(dqrot2)+Gvq1.dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+
                       Gvq2.dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])))
                #lambda12 = np.zeros(3)
                dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes]-Gvq1T.dot(lambdav[ci])
                dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes]-Gvq2T.dot(lambdav[ci])
        return dqb

    def dq_12_rot(self,t,q,args):

        qb = []
        #pdb.set_trace()
        dqb = []
        qb_i=0
        for bi in range(self.NumBodies):
            qb.append(q[qb_i:qb_i+2*self.V[bi].NumModes+self.V[0].rotation_states[bi]])
            qb_i+=len(qb[bi])
        #pdb.set_trace()
        dqb = self.q_12_rot(t,qb,args)
        return np.hstack([dqb[bi] for bi in range(self.NumBodies)])

    def dq_aero(self,t,q,args):

        qb1=[]
        qb0=[]
        qba=[]
        qb = []
        dqb = []
        qb_i=0
        for bi in range(self.NumBodies):
            qb1.append(q[qb_i:qb_i+self.V[bi].NumModes])
            #qb2.append(q[qb_i+self.V[bi].NumModes:qb_i+2*self.V[bi].NumModes])
            qb0.append(q[qb_i+2*self.V[bi].NumModes:qb_i+3*self.V[bi].NumModes])
            qba.append(q[qb_i+3*self.V[bi].NumModes:qb_i+(3+self.A[bi].NumPoles)*self.V[bi].NumModes])
            #qbr.append()
            qb.append(np.hstack([q[qb_i:qb_i+2*self.V[bi].NumModes],q[qb_i+self.V[bi].NumModes*(3+self.A[bi].NumPoles):qb_i+self.V[bi].NumModes*(3+self.A[bi].NumPoles)+self.V[0].rotation_states[bi]]]))
            qb_i+=len(qb[bi])+self.V[bi].NumModes*(1+self.A[bi].NumPoles)

        #pdb.set_trace()
        #dqb0 = qb1
        eta = self.force1[0].etaAero(np.hstack(qb1),np.hstack(qb0),np.hstack(qba))
        dqb = self.q_12_rot(t,qb,args)
        dq1 = args['Aqinv'].dot(np.hstack([dqb[bx][0:self.V[bx].NumModes] for bx  in range(self.NumBodies)])+eta)

        dqb1 = []
        counter = 0
        for bi in range(self.NumBodies):
            dqb1.append(dq1[counter:counter+self.V[bi].NumModes])
            counter += self.V[bi].NumModes

        dqb2 = [dqb[bi][self.V[bi].NumModes:2*self.V[bi].NumModes] for bi in range(self.NumBodies)]
        dqbr= [dqb[bi][2*self.V[bi].NumModes:] for bi in range(self.NumBodies)]
        dqba=[]
        for bi in range(self.NumBodies):
            if self.A[bi].NumPoles > 0:
                dqba.append(self.m[bi].q_aerostates(t,qba[bi],qb1[bi],args))

        return np.hstack([np.hstack([dqb1[bi],dqb2[bi],qb1[bi],dqba[bi],dqbr[bi]]) for bi in range(self.NumBodies)])

    def dq_aero_rbd(self,t,q,args):

        qb = []
        dqb = []
        qb_i=0
        for bi in range(self.NumBodies):
            qb1.append(q[qb_i:qb_i+self.V[bi].NumModes])
            #qb2.append(q[qb_i+self.V[bi].NumModes:qb_i+2*self.V[bi].NumModes])
            qb0.append(q[qb_i+2*self.V[bi].NumModes:qb_i+3*self.V[bi].NumModes])
            qba.append(q[qb_i+3*self.V[bi].NumModes:qb_i+(3+self.A[bi].NumPoles)*self.V[bi].NumModes])
            #qbr.append()
            qb.append(q[qb_i:qb_i+2*self.V[bi].NumModes]+q[qb_i+self.V[bi].NumModes*(3+self.A[bi].NumPoles):qb_i+self.V[bi].NumModes*(3+self.A[bi].NumPoles)+self.V[0].rotation_states[bi]])
            qb_i+=len(qb[bi])+self.V[bi].NumModes*(1+self.A[bi].NumPoles)

        #dqb0 = qb1
        dqb = self.q_12_rot(t,qb,args)
        dq1 = args['Aqinv'].dot(np.hstack([dqb[bx][0:self.V[bx].NumModes] for bx  in range(self.NumBodies)])+self.force1.etaaero(t,qb1,qb0,qbr))

        dqb1 = []
        counter = 0
        for bi in range(self.NumBodies):
            dqb1.append(dq1[counter:counter+self.V[bi].NumModes])
            counter += self.V[bi].NumModes

        dqb2 = [dqb[bi][self.V[bi].NumModes:2*self.V[bi].NumModes] for bi in range(self.NumBodies)]
        dqbr= [dqb[bi][2*self.V[bi].NumModes:] for bi in range(self.NumBodies)]
        dqba=[]
        for bi in range(self.NumBodies):
            if self.A[bi].NumPoles > 0:
                dqba.append(self.m[bi].q_aerostates(t,qba[bi],qb1[bi],args))

        #return np.hstack([np.hstack([dqb1[bi],dqb2[bi],qb1[bi],dqba[bi],dqbr[bi]]) for bi in range(self.NumBodies)])

        q1 = q[self.A.rbd:self.V.NumModes+self.A.rbd]
        q2 = q[self.V.NumModes+self.A.rbd:2*self.V.NumModes+self.A.rbd]
        if self.A.NumPoles > 0:
            qa = np.array(q[3*self.V.NumModes+2*self.A.rbd:3*self.V.NumModes+self.A.NumPoles*self.V.NumModes+(2+self.A.NumPoles)*self.A.rbd])
            #dqa = np.zeros((self.A.NumPoles,self.V.NumModes))
            dqa = self.q_aerostates(t,qa,q1,args)
            qr = q[3*self.V.NumModes+self.A.NumPoles*self.V.NumModes+(2+self.A.NumPoles)*self.A.rbd:]
        else:
            dqa=[]
            qr = q[3*self.V.NumModes+2*self.A.rbd:]
        dq1,dq2 = self.q_gamma(q1,q2)
        dq1 = np.insert(dq1,0,np.zeros(self.A.rbd))
        dqr,Rab = self.q_rotation(q1,q2,qr,args)
        eta = self.force1.forceAero_eta_rbd(q)
        eta += np.hstack([np.zeros(self.A.rbd),self.force1.forceGravity_eta(t,rotation=Rab)])
        dq1 = args['Aqinv'].dot(dq1+np.hstack([np.zeros(self.A.rbd),self.Omega*q2])+eta)
        dq2 = dq2-self.Omega*q1
        dq0 = q1
        dq0 = np.insert(dq0,0,np.zeros(self.A.rbd))

        return np.hstack((dq1,dq2,dq0,dqa,dqr))


























    # def q_12_rot(self,t,qb,args):
    #     dqb = []
    #     for bi in range(self.NumBodies):
    #         args['BeamSeg'] = self.BeamSeg[bi]
    #         args['inverseconn'] = self.inverseconn[bi]
    #         dqb.append(self.m[bi].dq_12_rot(t,qb[bi],args))
    #     vc1=[]
    #     vc2=[]
    #     lambdav = []
    #     lambdavx = []
    #     for ci in range(self.V[0].NumConstrains):
    #         v1=v2=0.
    #         for im in range(self.V[0].NumModes):
    #             v1+=self.Phi11[ci][im]*qb[self.b1[ci]][im]
    #         for im in range(self.V[1].NumModes):
    #             v2+=self.Phi12[ci][im]*qb[self.b2[ci]][im]
    #         vc1.append(v1)
    #         vc2.append(v2)
    #         qrot1 = qb[self.b1[ci]][-4:]
    #         qrot2 = qb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4]
    #         dqrot1 = dqb[self.b1[ci]][-4:]
    #         dqrot2 = dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4]
    #         R1 = self.R(qrot1); R2 = self.R(qrot2)
    #         #pdb.set_trace()
    #         Gvq1 = R1.dot(self.Phi11[ci].T); Gvq2 = -R2.dot(self.Phi12[ci].T)
    #         Gvq1T = R1.dot(self.Phi11[ci].T).T; Gvq2T = -R2.dot(self.Phi12[ci].T).T
    #         Gvz1 = self.Gvz(qrot1,vc1[ci]); Gvz2 = -self.Gvz(qrot2,vc2[ci])
    #         omega1=0.; omega2=0.
    #         for im in range(self.V[0].NumModes):
    #             omega1+=self.Phi11o[ci][im]*qb[self.b1[ci]][im]
    #         for im in range(self.V[1].NumModes):
    #             omega2+=self.Phi12o[ci][im]*qb[self.b2[ci]][im]

    #         if self.hinge[ci] and not (np.allclose(vc1[ci],np.zeros(3)) and np.allclose(vc2[ci],np.zeros(3)) and np.allclose(omega1,np.zeros(3)) and np.allclose(omega2,np.zeros(3))):
    #             R12 = R1.T.dot(R2)

    #             if self.hinge_xyz[ci] == 0:
    #                 Gxq1 = -np.cross(self.Phi11o[ci],R12[:,0]).T
    #                 Gxq2 =  R12.dot(np.array([[0.,self.Phi12o[ci][k][2],-self.Phi12o[ci][k][1]] for k in range(len(self.Phi12o[ci]))]).T)
    #                 Gxz1 = self.Giz1(qrot1,qrot2,omega1,omega2)
    #                 Gxz2 = self.Giz2(qrot1,qrot2,omega1,omega2)

    #             elif self.hinge_xyz[ci] == 1:
    #                 Gxq1 = -np.cross(self.Phi11o[ci],R12[:,1])
    #                 Gxq2 =  R12.dot([[0.,self.Phi12[ci][k][3],-self.Phi12[ci][k][2]] for k in range()].T)
    #                 Gxz1 = self.Gjz1(qrot1,qrot2,omega1,omega2)
    #                 Gxz2 = self.Gjz2(qrot1,qrot2,omega1,omega2)

    #             elif self.hinge_xyz[ci] == 2:
    #                 Gxq1 = -np.cross(self.Phi11o[ci],R12[:,2])
    #                 Gxq2 =  R12.dot([[0.,self.Phi12[ci][k][3],-self.Phi12[ci][k][2]] for k in range()].T)
    #                 Gxz1 = self.Gkz1(qrot1,qrot2,omega1,omega2)
    #                 Gxz2 = self.Gkz2(qrot1,qrot2,omega1,omega2)

    #                 lambdavx1 = np.vstack([np.hstack([Gvq1.dot(Gvq1T),Gvz1.dot(Gxz1.T)]),np.hstack([Gxq1.dot(Gvq1T),Gxz1.dot(Gxz1.T)])]) + np.vstack([np.hstack([Gvq2.dot(Gvq2T),Gvz2.dot(Gxz2.T)]),np.hstack([Gxq2.dot(Gvq2T),Gxz2.dot(Gxz2.T)])])
    #                 lambdavx2 = np.vstack([Gvq1.dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+Gvz1.dot(dqrot1)+Gvq2.dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])+Gvz2.dot(dqrot2),Gxq1.dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+Gxz1.dot(dqrot1)+Gxq2.dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])+Gxz2.dot(dqrot2)])
    #                 lambdavx.append(np.linalg.inv(lambdavx1).dot(lambdavx2))

    #                 dqrot1 = dqrot1 - Gxz1.T.dot(lambdavx[ci][3:])
    #                 dqrot2 = dqrot2 - Gxz2.T.dot(lambdavx[ci][3:])
    #                 dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes]-Gvq1T.dot(lambdavx[ci][:3])
    #                 dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes]-Gvq2T.dot(lambdavx[ci][:3])
    #                 dqb[self.b1[ci]][-4:] = dqrot1
    #                 dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4] = dqrot2

    #         #L1 = self.Lv(qrot1); L2 = -self.Lv(qrot2)
    #         #L01 = self.L0(qrot1,vc1[ci]); L02 = -self.L0(qrot2,vc2[ci])
    #         else:
    #             lambdavx.append([])
    #             lambdav.append(np.linalg.inv(Gvq1.dot(Gvq1T)+Gvq2.dot(Gvq2T)).dot(Gvz1.dot(dqrot1)+
    #                    Gvz2.dot(dqrot2)+Gvq1.dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+
    #                    Gvq2.dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])))
    #             #lambda12 = np.zeros(3)
    #             dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes]-Gvq1T.dot(lambdav[ci])
    #             dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes]-Gvq2T.dot(lambdav[ci])
    #     return dqb

    # def dq_12_rot(self,t,q,args):
    #     qb = []
    #     dqb = []
    #     qb_i=0
    #     for bi in range(self.V[0].NumBodies):
    #         qb.append(q[qb_i:qb_i+2*self.V[bi].NumModes+])
    #         qb_i+=len(qb[bi])
    #         args['BeamSeg'] = args['BeamSeg%s'%bi]
    #         args['inverseconn'] = args['inverseconn%s'%bi]
    #         dqb.append(self.m[bi].dq_12_rot(t,qb[bi],args))

    #     # q11 = q[:self.V[0].NumModes]
    #     # q21 = q[self.V[0].NumModes:2*self.V[0].NumModes]
    #     # q12 = q[2*self.V[0].NumModes:2*self.V[0].NumModes+self.V[1].NumModes]
    #     # q22 = q[2*self.V[0].NumModes+self.V[1].NumModes:2*self.V[0].NumModes+2*self.V[1].NumModes]
    #     # q01 = q[2*self.V[0].NumModes+2*self.V[1].NumModes:2*self.V[0].NumModes+2*self.V[1].NumModes+4*2]
    #     # q02 = q[2*self.V[0].NumModes+2*self.V[1].NumModes+4*2:2*self.V[0].NumModes+2*self.V[1].NumModes+8*2]
    #     vc1=[]
    #     vc2=[]
    #     for ci in range(self.V[0].NumConstrains):
    #         v1=v2=0.
    #         for im in range(self.V[0].NumModes):
    #             v1+=self.Phi11[ci]*qb[b1[ci]][im]
    #         for im in range(self.V[1].NumModes):
    #             v2+=self.Phi12[ci]*qb[b2[ci]][im]
    #         vc1.append(v1)
    #         vc2.append(v2)
    #         # dq11 = dq1[:self.V[0].NumModes]
    #         # dq21 = dq1[self.V[0].NumModes:2*self.V[0].NumModes]
    #         # dq12 = dq2[:self.V[1].NumModes]
    #         # dq22 = dq2[self.V[1].NumModes:2*self.V[1].NumModes]
    #         # dq01 = dq1[2*self.V[0].NumModes:2*self.V[0].NumModes+4*2]
    #         # dq02 = dq2[2*self.V[1].NumModes:2*self.V[1].NumModes+4*2]
    #         qrot1 = qb[self.b1[ci]][2*self.V[self.b1[ci]].NumModes+4*self.rot_node1[ci]:2*self.V[self.b1[ci]].NumModes+4*self.rot_node1[ci]+4]
    #         qrot2 = qb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes+4*self.rot_node2[ci]:2*self.V[self.b2[ci]].NumModes+4*self.rot_node2[ci]+4]
    #         dqrot1 = dqb[self.b1[ci]][2*self.V[self.b1[ci]].NumModes+4*self.rot_node1[ci]:2*self.V[self.b1[ci]].NumModes+4*self.rot_node1[ci]+4]
    #         dqrot2 = dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes+4*self.rot_node2[ci]:2*self.V[self.b2[ci]].NumModes+4*self.rot_node2[ci]+4]

    #         L1 = self.Lv(qrot1); L2 = -self.Lv(qrot2)
    #         L01 = self.L0(qrot1,vc1[ci]); L02 = -self.L0(qrot2,vc2[ci])
    #         lambda12.append(np.linalg.inv(L1.dot(self.Phi11[ci].dot(self.Phi11[ci].T)).dot(L1.T)+
    #                    L2.dot(self.Phi12[ci].dot(self.Phi12[ci].T)).dot(L2.T)).dot(L01.dot(dqrot1)+
    #                    L02.dot(dqrot2)+L1.dot(self.Phi11[ci]).dot(dqb[b1][0:self.V[b1].NumModes])+
    #                    L2.dot(self.Phi12[ci]).dot(dqb[b2][0:self.V[b2].NumModes])))
    #         #lambda12 = np.zeros(3)
    #         dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[b1][0:self.V[self.b1[ci]].NumModes]-(L1.dot(self.Phi11[ci])).T.dot(lambda12[ci])
    #         dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[b2][0:self.V[self.b2[ci]].NumModes] -(L2.dot(self.Phi12[ci])).T.dot(lambda12[ci])
    #     return np.hstack(dqb)






# def Lqx(self,qrot1,qrot2):
#     l10,l11,l12,l13 = qrot1
#     l20,l21,l22,l23 = qrot2
#     return np.array([[2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23),
#                2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23),
#                -2*l12*(l20**2+l21**2-l22**2-l23**2)+2*l11*(2*l20*l23+2*l21*l22)-2*l10*(-2*l20*l22+2*l21*l23),
#                -2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23),
#                2*(l10**2+l11**2-l12**2-l13**2)*l20+2*(2*l10*l13+2*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22,
#                2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23,
#                -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l22+2*(2*l10*l13+2*l11*l12)*l21-(-4*l10*l12+4*l11*l13)*l20,
#                -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+2*(2*l10*l13+2*l11*l12)*l20+2*(-2*l10*l12+2*l11*l13)*l21],
#               [-2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23),
#                2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23),
#                2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23),
#                -2*l10*(l20**2+l21**2-l22**2-l23**2)-2*l13*(2*l20*l23+2*l21*l22)+2*l12*(-2*l20*l22+2*l21*l23),
#                2*(-2*l10*l13+2*l11*l12)*l20+2*(l10**2-l11**2+l12**2-l13**2)*l23-(4*l10*l11+4*l12*l13)*l22,
#                2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23,
#                -(-4*l10*l13+4*l11*l12)*l22+2*(l10**2-l11**2+l12**2-l13**2)*l21-(4*l10*l11+4*l12*l13)*l20,
#                -(-4*l10*l13+4*l11*l12)*l23+2*(l10**2-l11**2+l12**2-l13**2)*l20+2*(2*l10*l11+2*l12*l13)*l21],
#               [2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23),
#                2*l13*(l20**2+l21**2-l22**2-l23**2)-2*l10*(2*l20*l23+2*l21*l22)-2*l11*(-2*l20*l22+2*l21*l23),
#                2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23),
#                2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23),
#                2*(2*l10*l12+2*l11*l13)*l20+2*(-2*l10*l11+2*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22,
#                2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23,
#                -(4*l10*l12+4*l11*l13)*l22+2*(-2*l10*l11+2*l12*l13)*l21-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l20,
#                -(4*l10*l12+4*l11*l13)*l23+2*(-2*l10*l11+2*l12*l13)*l20+2*(l10**2-l11**2-l12**2+l13**2)*l21]])

# def Lqy(self,qrot1,qrot2):
#     l10,l11,l12,l13 = qrot1
#     l20,l21,l22,l23 = qrot2
#     return np.array([[2*l10*(-2*l20*l23+2*l21*l22)+2*l13*(l20**2-l21**2+l22**2-l23**2)-2*l12*(2*l20*l21+2*l22*l23), 2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23), -2*l12*(-2*l20*l23+2*l21*l22)+2*l11*(l20**2-l21**2+l22**2-l23**2)-2*l10*(2*l20*l21+2*l22*l23), -2*l13*(-2*l20*l23+2*l21*l22)+2*l10*(l20**2-l21**2+l22**2-l23**2)+2*l11*(2*l20*l21+2*l22*l23), -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+2*(2*l10*l13+2*l11*l12)*l20+2*(-2*l10*l12+2*l11*l13)*l21, 2*(l10**2+l11**2-l12**2-l13**2)*l22-(4*l10*l13+4*l11*l12)*l21+2*(-2*l10*l12+2*l11*l13)*l20, 2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23, -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l20-(4*l10*l13+4*l11*l12)*l23+2*(-2*l10*l12+2*l11*l13)*l22], [-2*l13*(-2*l20*l23+2*l21*l22)+2*l10*(l20**2-l21**2+l22**2-l23**2)+2*l11*(2*l20*l21+2*l22*l23), 2*l12*(-2*l20*l23+2*l21*l22)-2*l11*(l20**2-l21**2+l22**2-l23**2)+2*l10*(2*l20*l21+2*l22*l23), 2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23), -2*l10*(-2*l20*l23+2*l21*l22)-2*l13*(l20**2-l21**2+l22**2-l23**2)+2*l12*(2*l20*l21+2*l22*l23), -(-4*l10*l13+4*l11*l12)*l23+2*(l10**2-l11**2+l12**2-l13**2)*l20+2*(2*l10*l11+2*l12*l13)*l21, 2*(-2*l10*l13+2*l11*l12)*l22-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21+2*(2*l10*l11+2*l12*l13)*l20, 2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23, -(-4*l10*l13+4*l11*l12)*l20-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l23+2*(2*l10*l11+2*l12*l13)*l22], [2*l12*(-2*l20*l23+2*l21*l22)-2*l11*(l20**2-l21**2+l22**2-l23**2)+2*l10*(2*l20*l21+2*l22*l23), 2*l13*(-2*l20*l23+2*l21*l22)-2*l10*(l20**2-l21**2+l22**2-l23**2)-2*l11*(2*l20*l21+2*l22*l23), 2*l10*(-2*l20*l23+2*l21*l22)+2*l13*(l20**2-l21**2+l22**2-l23**2)-2*l12*(2*l20*l21+2*l22*l23), 2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23), -(4*l10*l12+4*l11*l13)*l23+2*(-2*l10*l11+2*l12*l13)*l20+2*(l10**2-l11**2-l12**2+l13**2)*l21, 2*(2*l10*l12+2*l11*l13)*l22-(-4*l10*l11+4*l12*l13)*l21+2*(l10**2-l11**2-l12**2+l13**2)*l20, 2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23, -(4*l10*l12+4*l11*l13)*l20-(-4*l10*l11+4*l12*l13)*l23+2*(l10**2-l11**2-l12**2+l13**2)*l22]])

# def Lqz(self,qrot1,qrot2):
#     l10,l11,l12,l13 = qrot1
#     l20,l21,l22,l23 = qrot2
#     return np.array([[2*l10*(2*l20*l22+2*l21*l23)+2*l13*(-2*l20*l21+2*l22*l23)-2*l12*(l20**2-l21**2-l22**2+l23**2), 2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2), -2*l12*(2*l20*l22+2*l21*l23)+2*l11*(-2*l20*l21+2*l22*l23)-2*l10*(l20**2-l21**2-l22**2+l23**2), -2*l13*(2*l20*l22+2*l21*l23)+2*l10*(-2*l20*l21+2*l22*l23)+2*l11*(l20**2-l21**2-l22**2+l23**2), 2*(l10**2+l11**2-l12**2-l13**2)*l22-(4*l10*l13+4*l11*l12)*l21+2*(-2*l10*l12+2*l11*l13)*l20, 2*(l10**2+l11**2-l12**2-l13**2)*l23-(4*l10*l13+4*l11*l12)*l20-(-4*l10*l12+4*l11*l13)*l21, 2*(l10**2+l11**2-l12**2-l13**2)*l20+2*(2*l10*l13+2*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22, 2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23], [-2*l13*(2*l20*l22+2*l21*l23)+2*l10*(-2*l20*l21+2*l22*l23)+2*l11*(l20**2-l21**2-l22**2+l23**2), 2*l12*(2*l20*l22+2*l21*l23)-2*l11*(-2*l20*l21+2*l22*l23)+2*l10*(l20**2-l21**2-l22**2+l23**2), 2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2), -2*l10*(2*l20*l22+2*l21*l23)-2*l13*(-2*l20*l21+2*l22*l23)+2*l12*(l20**2-l21**2-l22**2+l23**2), 2*(-2*l10*l13+2*l11*l12)*l22-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21+2*(2*l10*l11+2*l12*l13)*l20, 2*(-2*l10*l13+2*l11*l12)*l23-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l20-(4*l10*l11+4*l12*l13)*l21, 2*(-2*l10*l13+2*l11*l12)*l20+2*(l10**2-l11**2+l12**2-l13**2)*l23-(4*l10*l11+4*l12*l13)*l22, 2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23], [2*l12*(2*l20*l22+2*l21*l23)-2*l11*(-2*l20*l21+2*l22*l23)+2*l10*(l20**2-l21**2-l22**2+l23**2), 2*l13*(2*l20*l22+2*l21*l23)-2*l10*(-2*l20*l21+2*l22*l23)-2*l11*(l20**2-l21**2-l22**2+l23**2), 2*l10*(2*l20*l22+2*l21*l23)+2*l13*(-2*l20*l21+2*l22*l23)-2*l12*(l20**2-l21**2-l22**2+l23**2), 2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2), 2*(2*l10*l12+2*l11*l13)*l22-(-4*l10*l11+4*l12*l13)*l21+2*(l10**2-l11**2-l12**2+l13**2)*l20, 2*(2*l10*l12+2*l11*l13)*l23-(-4*l10*l11+4*l12*l13)*l20-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l21, 2*(2*l10*l12+2*l11*l13)*l20+2*(-2*l10*l11+2*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22, 2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23]])


# def q_12_rot(self,t,qb,args):
#     dqb = []
#     for bi in range(self.NumBodies):
#         args['BeamSeg'] = self.BeamSeg[bi]
#         args['inverseconn'] = self.inverseconn[bi]
#         dqb.append(self.m[bi].dq_12_rot(t,qb[bi],args))
#     vc1=[]
#     vc2=[]
#     lambda12 = []
#     for ci in range(self.V[0].NumConstrains):
#         v1=v2=0.
#         for im in range(self.V[0].NumModes):
#             v1+=self.Phi11[ci][im]*qb[self.b1[ci]][im]
#         for im in range(self.V[1].NumModes):
#             v2+=self.Phi12[ci][im]*qb[self.b2[ci]][im]
#         vc1.append(v1)
#         vc2.append(v2)
#         # dq11 = dq1[:self.V[0].NumModes]
#         # dq21 = dq1[self.V[0].NumModes:2*self.V[0].NumModes]
#         # dq12 = dq2[:self.V[1].NumModes]
#         # dq22 = dq2[self.V[1].NumModes:2*self.V[1].NumModes]
#         # dq01 = dq1[2*self.V[0].NumModes:2*self.V[0].NumModes+4*2]
#         # dq02 = dq2[2*self.V[1].NumModes:2*self.V[1].NumModes+4*2]
#         qrot1 = qb[self.b1[ci]][-4:]
#         qrot2 = qb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4]
#         dqrot1 = dqb[self.b1[ci]][-4:]
#         dqrot2 = dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4]
#         #pdb.set_trace()
#         if self.hinge[ci]:
#             if self.hinge_xyz[ci] == 0:
#                 Gh = self.Lqx(qrot1,qrot2)
#             elif self.hinge_xyz[ci] == 1:
#                 Gh = self.Lqy(qrot1,qrot2)
#             elif self.hinge_xyz[ci] == 2:
#                 Gh = self.Lqz(qrot1,qrot2)
#             lambdah12 = np.linalg.inv(Gh.dot(Gh.T)).dot(Gh).dot(np.hstack([dqrot1,dqrot2]))
#             dqrot1 = dqrot1 - Gh[:,:4].T.dot(lambdah12)
#             dqrot2 = dqrot2 - Gh[:,4:].T.dot(lambdah12)
#             dqb[self.b1[ci]][-4:] = dqrot1
#             dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4] = dqrot2
#         L1 = self.Lv(qrot1); L2 = -self.Lv(qrot2)
#         L01 = self.L0(qrot1,vc1[ci]); L02 = -self.L0(qrot2,vc2[ci])
#         lambda12.append(np.linalg.inv(L1.dot(self.Phi11[ci].T.dot(self.Phi11[ci])).dot(L1.T)+
#                    L2.dot(self.Phi12[ci].T.dot(self.Phi12[ci])).dot(L2.T)).dot(L01.dot(dqrot1)+
#                    L02.dot(dqrot2)+L1.dot(self.Phi11[ci].T).dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+L2.dot(self.Phi12[ci].T).dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])))
#         #lambda12 = np.zeros(3)
#         dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes]-(self.Phi11[ci].dot(L1.T)).dot(lambda12[ci])
#         dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes]-(self.Phi12[ci].dot(L2.T)).dot(lambda12[ci])
#     return dqb






#     def Giz1(self,qrot1,qrot2,omega1,omega2):
#         l10,l11,l12,l13 = qrot1
#         l20,l21,l22,l23 = qrot2
#         o11,o12,o13 = omega1
#         o21,o22,o23 = omega2
#         return np.array([[(2*l10*(-2*l20*l23+2*l21*l22)+2*l13*(l20**2-l21**2+l22**2-l23**2)-2*l12*(2*l20*l21+2*l22*l23))*o23-(2*l10*(2*l20*l22+2*l21*l23)+2*l13*(-2*l20*l21+2*l22*l23)-2*l12*(l20**2-l21**2-l22**2+l23**2))*o22+o13*(-2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23))-o12*(2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23)), (2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23))*o23-(2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2))*o22+o13*(2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23))-o12*(2*l13*(l20**2+l21**2-l22**2-l23**2)-2*l10*(2*l20*l23+2*l21*l22)-2*l11*(-2*l20*l22+2*l21*l23)), (-2*l12*(-2*l20*l23+2*l21*l22)+2*l11*(l20**2-l21**2+l22**2-l23**2)-2*l10*(2*l20*l21+2*l22*l23))*o23-(-2*l12*(2*l20*l22+2*l21*l23)+2*l11*(-2*l20*l21+2*l22*l23)-2*l10*(l20**2-l21**2-l22**2+l23**2))*o22+o13*(2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23))-o12*(2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23)), (-2*l13*(-2*l20*l23+2*l21*l22)+2*l10*(l20**2-l21**2+l22**2-l23**2)+2*l11*(2*l20*l21+2*l22*l23))*o23-(-2*l13*(2*l20*l22+2*l21*l23)+2*l10*(-2*l20*l21+2*l22*l23)+2*l11*(l20**2-l21**2-l22**2+l23**2))*o22+o13*(-2*l10*(l20**2+l21**2-l22**2-l23**2)-2*l13*(2*l20*l23+2*l21*l22)+2*l12*(-2*l20*l22+2*l21*l23))-o12*(2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23))], [(-2*l13*(-2*l20*l23+2*l21*l22)+2*l10*(l20**2-l21**2+l22**2-l23**2)+2*l11*(2*l20*l21+2*l22*l23))*o23-(-2*l13*(2*l20*l22+2*l21*l23)+2*l10*(-2*l20*l21+2*l22*l23)+2*l11*(l20**2-l21**2-l22**2+l23**2))*o22-o13*(2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23))+o11*(2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23)), (2*l12*(-2*l20*l23+2*l21*l22)-2*l11*(l20**2-l21**2+l22**2-l23**2)+2*l10*(2*l20*l21+2*l22*l23))*o23-(2*l12*(2*l20*l22+2*l21*l23)-2*l11*(-2*l20*l21+2*l22*l23)+2*l10*(l20**2-l21**2-l22**2+l23**2))*o22-o13*(2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23))+o11*(2*l13*(l20**2+l21**2-l22**2-l23**2)-2*l10*(2*l20*l23+2*l21*l22)-2*l11*(-2*l20*l22+2*l21*l23)), (2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23))*o23-(2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2))*o22-o13*(-2*l12*(l20**2+l21**2-l22**2-l23**2)+2*l11*(2*l20*l23+2*l21*l22)-2*l10*(-2*l20*l22+2*l21*l23))+o11*(2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23)), (-2*l10*(-2*l20*l23+2*l21*l22)-2*l13*(l20**2-l21**2+l22**2-l23**2)+2*l12*(2*l20*l21+2*l22*l23))*o23-(-2*l10*(2*l20*l22+2*l21*l23)-2*l13*(-2*l20*l21+2*l22*l23)+2*l12*(l20**2-l21**2-l22**2+l23**2))*o22-o13*(-2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23))+o11*(2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23))], [(2*l12*(-2*l20*l23+2*l21*l22)-2*l11*(l20**2-l21**2+l22**2-l23**2)+2*l10*(2*l20*l21+2*l22*l23))*o23-(2*l12*(2*l20*l22+2*l21*l23)-2*l11*(-2*l20*l21+2*l22*l23)+2*l10*(l20**2-l21**2-l22**2+l23**2))*o22+o12*(2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23))-o11*(-2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23)), (2*l13*(-2*l20*l23+2*l21*l22)-2*l10*(l20**2-l21**2+l22**2-l23**2)-2*l11*(2*l20*l21+2*l22*l23))*o23-(2*l13*(2*l20*l22+2*l21*l23)-2*l10*(-2*l20*l21+2*l22*l23)-2*l11*(l20**2-l21**2-l22**2+l23**2))*o22+o12*(2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23))-o11*(2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23)), (2*l10*(-2*l20*l23+2*l21*l22)+2*l13*(l20**2-l21**2+l22**2-l23**2)-2*l12*(2*l20*l21+2*l22*l23))*o23-(2*l10*(2*l20*l22+2*l21*l23)+2*l13*(-2*l20*l21+2*l22*l23)-2*l12*(l20**2-l21**2-l22**2+l23**2))*o22+o12*(-2*l12*(l20**2+l21**2-l22**2-l23**2)+2*l11*(2*l20*l23+2*l21*l22)-2*l10*(-2*l20*l22+2*l21*l23))-o11*(2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23)), (2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23))*o23-(2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2))*o22+o12*(-2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23))-o11*(-2*l10*(l20**2+l21**2-l22**2-l23**2)-2*l13*(2*l20*l23+2*l21*l22)+2*l12*(-2*l20*l22+2*l21*l23))]])

#     def Giz2(self,qrot1,qrot2,omega1,omega2):
#         l10,l11,l12,l13 = qrot1
#         l20,l21,l22,l23 = qrot2
#         o11,o12,o13 = omega1
#         o21,o22,o23 = omega2
#         return np.array([[(-(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+2*(2*l10*l13+2*l11*l12)*l20+2*(-2*l10*l12+2*l11*l13)*l21)*o23-((2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l22-(4*l10*l13+4*l11*l12)*l21+(-4*l10*l12+4*l11*l13)*l20)*o22+o13*(2*(-2*l10*l13+2*l11*l12)*l20+2*(l10**2-l11**2+l12**2-l13**2)*l23-(4*l10*l11+4*l12*l13)*l22)-o12*((4*l10*l12+4*l11*l13)*l20+(-4*l10*l11+4*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22), (2*(l10**2+l11**2-l12**2-l13**2)*l22-(4*l10*l13+4*l11*l12)*l21+2*(-2*l10*l12+2*l11*l13)*l20)*o23-((2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23-(4*l10*l13+4*l11*l12)*l20-(-4*l10*l12+4*l11*l13)*l21)*o22+o13*(2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23)-o12*((4*l10*l12+4*l11*l13)*l21+(-4*l10*l11+4*l12*l13)*l22+(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l23), (2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23)*o23-((2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l20+(4*l10*l13+4*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22)*o22+o13*(-(-4*l10*l13+4*l11*l12)*l22+2*(l10**2-l11**2+l12**2-l13**2)*l21-(4*l10*l11+4*l12*l13)*l20)-o12*(-(4*l10*l12+4*l11*l13)*l22+(-4*l10*l11+4*l12*l13)*l21-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l20), (-(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l20-(4*l10*l13+4*l11*l12)*l23+2*(-2*l10*l12+2*l11*l13)*l22)*o23-((2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l21+(4*l10*l13+4*l11*l12)*l22+(-4*l10*l12+4*l11*l13)*l23)*o22+o13*(-(-4*l10*l13+4*l11*l12)*l23+2*(l10**2-l11**2+l12**2-l13**2)*l20+2*(2*l10*l11+2*l12*l13)*l21)-o12*(-(4*l10*l12+4*l11*l13)*l23+(-4*l10*l11+4*l12*l13)*l20+(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l21)], [(-(-4*l10*l13+4*l11*l12)*l23+2*(l10**2-l11**2+l12**2-l13**2)*l20+2*(2*l10*l11+2*l12*l13)*l21)*o23-((-4*l10*l13+4*l11*l12)*l22-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21+(4*l10*l11+4*l12*l13)*l20)*o22-o13*((2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l20+(4*l10*l13+4*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22)+o11*(2*(2*l10*l12+2*l11*l13)*l20+2*(-2*l10*l11+2*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22), (2*(-2*l10*l13+2*l11*l12)*l22-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21+2*(2*l10*l11+2*l12*l13)*l20)*o23-((-4*l10*l13+4*l11*l12)*l23-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l20-(4*l10*l11+4*l12*l13)*l21)*o22-o13*((2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l21+(4*l10*l13+4*l11*l12)*l22+(-4*l10*l12+4*l11*l13)*l23)+o11*(2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23), (2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23)*o23-((-4*l10*l13+4*l11*l12)*l20+(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l23-(4*l10*l11+4*l12*l13)*l22)*o22-o13*(-(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l22+(4*l10*l13+4*l11*l12)*l21-(-4*l10*l12+4*l11*l13)*l20)+o11*(-(4*l10*l12+4*l11*l13)*l22+2*(-2*l10*l11+2*l12*l13)*l21-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l20), (-(-4*l10*l13+4*l11*l12)*l20-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l23+2*(2*l10*l11+2*l12*l13)*l22)*o23-((-4*l10*l13+4*l11*l12)*l21+(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l22+(4*l10*l11+4*l12*l13)*l23)*o22-o13*(-(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+(4*l10*l13+4*l11*l12)*l20+(-4*l10*l12+4*l11*l13)*l21)+o11*(-(4*l10*l12+4*l11*l13)*l23+2*(-2*l10*l11+2*l12*l13)*l20+2*(l10**2-l11**2-l12**2+l13**2)*l21)], [(-(4*l10*l12+4*l11*l13)*l23+2*(-2*l10*l11+2*l12*l13)*l20+2*(l10**2-l11**2-l12**2+l13**2)*l21)*o23-((4*l10*l12+4*l11*l13)*l22-(-4*l10*l11+4*l12*l13)*l21+(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l20)*o22+o12*(2*(l10**2+l11**2-l12**2-l13**2)*l20+2*(2*l10*l13+2*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22)-o11*((-4*l10*l13+4*l11*l12)*l20+(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l23-(4*l10*l11+4*l12*l13)*l22), (2*(2*l10*l12+2*l11*l13)*l22-(-4*l10*l11+4*l12*l13)*l21+2*(l10**2-l11**2-l12**2+l13**2)*l20)*o23-((4*l10*l12+4*l11*l13)*l23-(-4*l10*l11+4*l12*l13)*l20-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l21)*o22+o12*(2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23)-o11*((-4*l10*l13+4*l11*l12)*l21+(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l22+(4*l10*l11+4*l12*l13)*l23), (2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23)*o23-((4*l10*l12+4*l11*l13)*l20+(-4*l10*l11+4*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22)*o22+o12*(-(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l22+2*(2*l10*l13+2*l11*l12)*l21-(-4*l10*l12+4*l11*l13)*l20)-o11*(-(-4*l10*l13+4*l11*l12)*l22+(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21-(4*l10*l11+4*l12*l13)*l20), (-(4*l10*l12+4*l11*l13)*l20-(-4*l10*l11+4*l12*l13)*l23+2*(l10**2-l11**2-l12**2+l13**2)*l22)*o23-((4*l10*l12+4*l11*l13)*l21+(-4*l10*l11+4*l12*l13)*l22+(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l23)*o22+o12*(-(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+2*(2*l10*l13+2*l11*l12)*l20+2*(-2*l10*l12+2*l11*l13)*l21)-o11*(-(-4*l10*l13+4*l11*l12)*l23+(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l20+(4*l10*l11+4*l12*l13)*l21)]])

#     def Lqx(self,qrot1,qrot2):
#         l10,l11,l12,l13 = qrot1
#         l20,l21,l22,l23 = qrot2
#         return np.array([[2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23),
#                    2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23),
#                    -2*l12*(l20**2+l21**2-l22**2-l23**2)+2*l11*(2*l20*l23+2*l21*l22)-2*l10*(-2*l20*l22+2*l21*l23),
#                    -2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23),
#                    2*(l10**2+l11**2-l12**2-l13**2)*l20+2*(2*l10*l13+2*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22,
#                    2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23,
#                    -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l22+2*(2*l10*l13+2*l11*l12)*l21-(-4*l10*l12+4*l11*l13)*l20,
#                    -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+2*(2*l10*l13+2*l11*l12)*l20+2*(-2*l10*l12+2*l11*l13)*l21],
#                   [-2*l13*(l20**2+l21**2-l22**2-l23**2)+2*l10*(2*l20*l23+2*l21*l22)+2*l11*(-2*l20*l22+2*l21*l23),
#                    2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23),
#                    2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23),
#                    -2*l10*(l20**2+l21**2-l22**2-l23**2)-2*l13*(2*l20*l23+2*l21*l22)+2*l12*(-2*l20*l22+2*l21*l23),
#                    2*(-2*l10*l13+2*l11*l12)*l20+2*(l10**2-l11**2+l12**2-l13**2)*l23-(4*l10*l11+4*l12*l13)*l22,
#                    2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23,
#                    -(-4*l10*l13+4*l11*l12)*l22+2*(l10**2-l11**2+l12**2-l13**2)*l21-(4*l10*l11+4*l12*l13)*l20,
#                    -(-4*l10*l13+4*l11*l12)*l23+2*(l10**2-l11**2+l12**2-l13**2)*l20+2*(2*l10*l11+2*l12*l13)*l21],
#                   [2*l12*(l20**2+l21**2-l22**2-l23**2)-2*l11*(2*l20*l23+2*l21*l22)+2*l10*(-2*l20*l22+2*l21*l23),
#                    2*l13*(l20**2+l21**2-l22**2-l23**2)-2*l10*(2*l20*l23+2*l21*l22)-2*l11*(-2*l20*l22+2*l21*l23),
#                    2*l10*(l20**2+l21**2-l22**2-l23**2)+2*l13*(2*l20*l23+2*l21*l22)-2*l12*(-2*l20*l22+2*l21*l23),
#                    2*l11*(l20**2+l21**2-l22**2-l23**2)+2*l12*(2*l20*l23+2*l21*l22)+2*l13*(-2*l20*l22+2*l21*l23),
#                    2*(2*l10*l12+2*l11*l13)*l20+2*(-2*l10*l11+2*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22,
#                    2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23,
#                    -(4*l10*l12+4*l11*l13)*l22+2*(-2*l10*l11+2*l12*l13)*l21-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l20,
#                    -(4*l10*l12+4*l11*l13)*l23+2*(-2*l10*l11+2*l12*l13)*l20+2*(l10**2-l11**2-l12**2+l13**2)*l21]])

#     def Lqy(self,qrot1,qrot2):
#         l10,l11,l12,l13 = qrot1
#         l20,l21,l22,l23 = qrot2
#         return np.array([[2*l10*(-2*l20*l23+2*l21*l22)+2*l13*(l20**2-l21**2+l22**2-l23**2)-2*l12*(2*l20*l21+2*l22*l23), 2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23), -2*l12*(-2*l20*l23+2*l21*l22)+2*l11*(l20**2-l21**2+l22**2-l23**2)-2*l10*(2*l20*l21+2*l22*l23), -2*l13*(-2*l20*l23+2*l21*l22)+2*l10*(l20**2-l21**2+l22**2-l23**2)+2*l11*(2*l20*l21+2*l22*l23), -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l23+2*(2*l10*l13+2*l11*l12)*l20+2*(-2*l10*l12+2*l11*l13)*l21, 2*(l10**2+l11**2-l12**2-l13**2)*l22-(4*l10*l13+4*l11*l12)*l21+2*(-2*l10*l12+2*l11*l13)*l20, 2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23, -(2*l10**2+2*l11**2-2*l12**2-2*l13**2)*l20-(4*l10*l13+4*l11*l12)*l23+2*(-2*l10*l12+2*l11*l13)*l22], [-2*l13*(-2*l20*l23+2*l21*l22)+2*l10*(l20**2-l21**2+l22**2-l23**2)+2*l11*(2*l20*l21+2*l22*l23), 2*l12*(-2*l20*l23+2*l21*l22)-2*l11*(l20**2-l21**2+l22**2-l23**2)+2*l10*(2*l20*l21+2*l22*l23), 2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23), -2*l10*(-2*l20*l23+2*l21*l22)-2*l13*(l20**2-l21**2+l22**2-l23**2)+2*l12*(2*l20*l21+2*l22*l23), -(-4*l10*l13+4*l11*l12)*l23+2*(l10**2-l11**2+l12**2-l13**2)*l20+2*(2*l10*l11+2*l12*l13)*l21, 2*(-2*l10*l13+2*l11*l12)*l22-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21+2*(2*l10*l11+2*l12*l13)*l20, 2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23, -(-4*l10*l13+4*l11*l12)*l20-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l23+2*(2*l10*l11+2*l12*l13)*l22], [2*l12*(-2*l20*l23+2*l21*l22)-2*l11*(l20**2-l21**2+l22**2-l23**2)+2*l10*(2*l20*l21+2*l22*l23), 2*l13*(-2*l20*l23+2*l21*l22)-2*l10*(l20**2-l21**2+l22**2-l23**2)-2*l11*(2*l20*l21+2*l22*l23), 2*l10*(-2*l20*l23+2*l21*l22)+2*l13*(l20**2-l21**2+l22**2-l23**2)-2*l12*(2*l20*l21+2*l22*l23), 2*l11*(-2*l20*l23+2*l21*l22)+2*l12*(l20**2-l21**2+l22**2-l23**2)+2*l13*(2*l20*l21+2*l22*l23), -(4*l10*l12+4*l11*l13)*l23+2*(-2*l10*l11+2*l12*l13)*l20+2*(l10**2-l11**2-l12**2+l13**2)*l21, 2*(2*l10*l12+2*l11*l13)*l22-(-4*l10*l11+4*l12*l13)*l21+2*(l10**2-l11**2-l12**2+l13**2)*l20, 2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23, -(4*l10*l12+4*l11*l13)*l20-(-4*l10*l11+4*l12*l13)*l23+2*(l10**2-l11**2-l12**2+l13**2)*l22]])

#     def Lqz(self,qrot1,qrot2):
#         l10,l11,l12,l13 = qrot1
#         l20,l21,l22,l23 = qrot2
#         return np.array([[2*l10*(2*l20*l22+2*l21*l23)+2*l13*(-2*l20*l21+2*l22*l23)-2*l12*(l20**2-l21**2-l22**2+l23**2), 2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2), -2*l12*(2*l20*l22+2*l21*l23)+2*l11*(-2*l20*l21+2*l22*l23)-2*l10*(l20**2-l21**2-l22**2+l23**2), -2*l13*(2*l20*l22+2*l21*l23)+2*l10*(-2*l20*l21+2*l22*l23)+2*l11*(l20**2-l21**2-l22**2+l23**2), 2*(l10**2+l11**2-l12**2-l13**2)*l22-(4*l10*l13+4*l11*l12)*l21+2*(-2*l10*l12+2*l11*l13)*l20, 2*(l10**2+l11**2-l12**2-l13**2)*l23-(4*l10*l13+4*l11*l12)*l20-(-4*l10*l12+4*l11*l13)*l21, 2*(l10**2+l11**2-l12**2-l13**2)*l20+2*(2*l10*l13+2*l11*l12)*l23-(-4*l10*l12+4*l11*l13)*l22, 2*(l10**2+l11**2-l12**2-l13**2)*l21+2*(2*l10*l13+2*l11*l12)*l22+2*(-2*l10*l12+2*l11*l13)*l23], [-2*l13*(2*l20*l22+2*l21*l23)+2*l10*(-2*l20*l21+2*l22*l23)+2*l11*(l20**2-l21**2-l22**2+l23**2), 2*l12*(2*l20*l22+2*l21*l23)-2*l11*(-2*l20*l21+2*l22*l23)+2*l10*(l20**2-l21**2-l22**2+l23**2), 2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2), -2*l10*(2*l20*l22+2*l21*l23)-2*l13*(-2*l20*l21+2*l22*l23)+2*l12*(l20**2-l21**2-l22**2+l23**2), 2*(-2*l10*l13+2*l11*l12)*l22-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l21+2*(2*l10*l11+2*l12*l13)*l20, 2*(-2*l10*l13+2*l11*l12)*l23-(2*l10**2-2*l11**2+2*l12**2-2*l13**2)*l20-(4*l10*l11+4*l12*l13)*l21, 2*(-2*l10*l13+2*l11*l12)*l20+2*(l10**2-l11**2+l12**2-l13**2)*l23-(4*l10*l11+4*l12*l13)*l22, 2*(-2*l10*l13+2*l11*l12)*l21+2*(l10**2-l11**2+l12**2-l13**2)*l22+2*(2*l10*l11+2*l12*l13)*l23], [2*l12*(2*l20*l22+2*l21*l23)-2*l11*(-2*l20*l21+2*l22*l23)+2*l10*(l20**2-l21**2-l22**2+l23**2), 2*l13*(2*l20*l22+2*l21*l23)-2*l10*(-2*l20*l21+2*l22*l23)-2*l11*(l20**2-l21**2-l22**2+l23**2), 2*l10*(2*l20*l22+2*l21*l23)+2*l13*(-2*l20*l21+2*l22*l23)-2*l12*(l20**2-l21**2-l22**2+l23**2), 2*l11*(2*l20*l22+2*l21*l23)+2*l12*(-2*l20*l21+2*l22*l23)+2*l13*(l20**2-l21**2-l22**2+l23**2), 2*(2*l10*l12+2*l11*l13)*l22-(-4*l10*l11+4*l12*l13)*l21+2*(l10**2-l11**2-l12**2+l13**2)*l20, 2*(2*l10*l12+2*l11*l13)*l23-(-4*l10*l11+4*l12*l13)*l20-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l21, 2*(2*l10*l12+2*l11*l13)*l20+2*(-2*l10*l11+2*l12*l13)*l23-(2*l10**2-2*l11**2-2*l12**2+2*l13**2)*l22, 2*(2*l10*l12+2*l11*l13)*l21+2*(-2*l10*l11+2*l12*l13)*l22+2*(l10**2-l11**2-l12**2+l13**2)*l23]])


#     # def q_12_rot(self,t,qb,args):
#     #     dqb = []
#     #     for bi in range(self.NumBodies):
#     #         args['BeamSeg'] = self.BeamSeg[bi]
#     #         args['inverseconn'] = self.inverseconn[bi]
#     #         dqb.append(self.m[bi].dq_12_rot(t,qb[bi],args))
#     #     vc1=[]
#     #     vc2=[]
#     #     lambda12 = []
#     #     for ci in range(self.V[0].NumConstrains):
#     #         v1=v2=0.
#     #         for im in range(self.V[0].NumModes):
#     #             v1+=self.Phi11[ci][im]*qb[self.b1[ci]][im]
#     #         for im in range(self.V[1].NumModes):
#     #             v2+=self.Phi12[ci][im]*qb[self.b2[ci]][im]
#     #         vc1.append(v1)
#     #         vc2.append(v2)
#     #         # dq11 = dq1[:self.V[0].NumModes]
#     #         # dq21 = dq1[self.V[0].NumModes:2*self.V[0].NumModes]
#     #         # dq12 = dq2[:self.V[1].NumModes]
#     #         # dq22 = dq2[self.V[1].NumModes:2*self.V[1].NumModes]
#     #         # dq01 = dq1[2*self.V[0].NumModes:2*self.V[0].NumModes+4*2]
#     #         # dq02 = dq2[2*self.V[1].NumModes:2*self.V[1].NumModes+4*2]
#     #         qrot1 = qb[self.b1[ci]][-4:]
#     #         qrot2 = qb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4]
#     #         dqrot1 = dqb[self.b1[ci]][-4:]
#     #         dqrot2 = dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4]
#     #         #pdb.set_trace()
#     #         if self.hinge[ci]:
#     #             if self.hinge_xyz[ci] == 0:
#     #                 Gh = self.Lqx(qrot1,qrot2)
#     #             elif self.hinge_xyz[ci] == 1:
#     #                 Gh = self.Lqy(qrot1,qrot2)
#     #             elif self.hinge_xyz[ci] == 2:
#     #                 Gh = self.Lqz(qrot1,qrot2)
#     #             lambdah12 = np.linalg.inv(Gh.dot(Gh.T)).dot(Gh).dot(np.hstack([dqrot1,dqrot2]))
#     #             dqrot1 = dqrot1 - Gh[:,:4].T.dot(lambdah12)
#     #             dqrot2 = dqrot2 - Gh[:,4:].T.dot(lambdah12)
#     #             dqb[self.b1[ci]][-4:] = dqrot1
#     #             dqb[self.b2[ci]][2*self.V[self.b2[ci]].NumModes:2*self.V[self.b2[ci]].NumModes+4] = dqrot2
#     #         L1 = self.Lv(qrot1); L2 = -self.Lv(qrot2)
#     #         L01 = self.L0(qrot1,vc1[ci]); L02 = -self.L0(qrot2,vc2[ci])
#     #         lambda12.append(np.linalg.inv(L1.dot(self.Phi11[ci].T.dot(self.Phi11[ci])).dot(L1.T)+
#     #                    L2.dot(self.Phi12[ci].T.dot(self.Phi12[ci])).dot(L2.T)).dot(L01.dot(dqrot1)+
#     #                    L02.dot(dqrot2)+L1.dot(self.Phi11[ci].T).dot(dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes])+L2.dot(self.Phi12[ci].T).dot(dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes])))
#     #         #lambda12 = np.zeros(3)
#     #         dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes] = dqb[self.b1[ci]][0:self.V[self.b1[ci]].NumModes]-(self.Phi11[ci].dot(L1.T)).dot(lambda12[ci])
#     #         dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes] = dqb[self.b2[ci]][0:self.V[self.b2[ci]].NumModes]-(self.Phi12[ci].dot(L2.T)).dot(lambda12[ci])
#     #     return dqb
