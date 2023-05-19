import numpy as np
import pdb
import os
import scipy.linalg, scipy.sparse.linalg
import copy
import Utils.FEM_MatrixBuilder as fmb


class Condense:
    """Class to condense full FE matrices """

    def __init__(self,Kf_file,Mf_file,aset,rbe3s=1,K_name=['KAA'],M_name=['MAA'],Ka_file=None,Ma_file=None):

        if Ka_file is not None:
            self.Ka = np.load(Ka_file)
            self.Ma = np.load(Ma_file)
        if Kf_file.split('.')[-1] == 'npy':
            self.Kf = np.load(Kf_file)
        if Kf_file.split('.')[-1] == 'op4':
            self.Kf=fmb.readOP4_matrices(readKM=[Kf_file],nameKM=K_name,saveKM=[])[0]
        if Mf_file.split('.')[-1] == 'npy':
            self.Mf = np.load(Mf_file)
        if Mf_file.split('.')[-1] == 'op4':
            self.Mf=fmb.readOP4_matrices(readKM=[Mf_file],nameKM=M_name,saveKM=[])[0]

        #Da,Va = scipy.linalg.eigh(Ka,Ma)

        self.dof = np.shape(self.Kf)[0]
        self.NumNode = self.dof/6
        self.aset = aset
        self.ommit = self.NumNode-self.aset

        #Koo1,Koa1,Kao1,Kaa1 = rem_zeros(Ka,25)
        #Moo1x,Moa1x,Mao1x,Maa1x,Aim,Ajm = rem_zeros(Ma,aset)
        if rbe3s:
            self.Koo1,self.Koa1,self.Kao1,self.Kaa1,self.Moo1,self.Moa1,self.Mao1,self.Maa1,self.Ai,self.Aj = self.rem_zeros2(self.Kf,self.aset,self.Mf)
        else:
            self.Koo1 = self.Kf[:ommit*6,:ommit*6]
            self.Koa1 = self.Kf[:ommit*6,ommit*6:]
            self.Kao1 = self.Kf[ommit*6:,:ommit*6]
            self.Kaa1 = self.Kf[ommit*6:,ommit*6:]
            self.Moo1 = self.Mf[:ommit*6,:ommit*6]
            self.Moa1 = self.Mf[:ommit*6,ommit*6:]
            self.Mao1 = self.Mf[ommit*6:,:ommit*6]
            self.Maa1 = self.Mf[ommit*6:,ommit*6:]
        self.IKoo1 = np.linalg.inv(self.Koo1)
        self.K = np.hstack((np.vstack((self.Koo1,self.Kao1)),np.vstack((self.Koa1,self.Kaa1))))
        self.M = np.hstack((np.vstack((self.Moo1,self.Mao1)),np.vstack((self.Moa1,self.Maa1))))
        #np.linalg.cholesky(K)

        self.dof2 = np.shape(self.K)[0]
        self.NumNode2 = self.dof2/6
        self.ommit2 = self.NumNode2-self.aset
        self.cond_guyan()

    def save_matrices(self,directory,name):
        for i in name.keys():
            np.save(directory+'/'+i+'.npy',eval('self.'+name[i]))

    def reorder_matrices(self):

        pass

    def rem_zeros(self,Ka,aset):


        dof = np.shape(Ka)[0]
        NumNode = dof/6
        #self.aset = 25
        ommit = NumNode-aset
        ai=[];aj=[]

        Koo = Ka[:ommit*6,:ommit*6]
        Koa = Ka[:ommit*6,ommit*6:]
        Kao = Ka[ommit*6:,:ommit*6]
        Kaa = Ka[ommit*6:,ommit*6:]

        Koo1 =np.copy(Koo)
        Koa1 = np.copy(Koa)
        Kao1 = np.copy(Kao)
        Kaa1 = np.copy(Kaa)

        count=0
        for i in range(dof):
            if np.allclose(Ka[i,:],np.zeros((dof,1))):
                #pdb.set_trace()
                if i<=ommit*6:
                    Koo1 = np.delete(Koo1,i-count,0)
                    Koa1 = np.delete(Koa1,i-count,0)
                else:
                    Kao1 = np.delete(Kao1,i-count,0)
                    Kaa1 = np.delete(Kaa1,i-count,0)
                count+=1
            else:
                ai.append(i)

        count2=0
        for i in range(dof):
            if np.allclose(Ka[:,i],np.zeros((dof,1))):

                if i<=ommit*6:
                    Koo1 = np.delete(Koo1,i-count2,1)
                    Kao1 = np.delete(Kao1,i-count2,1)

                else:

                    Koa1 = np.delete(Koa1,i-count2,1)
                    Kaa1 = np.delete(Kaa1,i-count2,1)
                count2+=1
            else:
                aj.append(i)

        return(Koo1,Koa1,Kao1,Kaa1,ai,aj)


    @staticmethod
    def rem_zeros2(Ka,aset,Ma=None):


        dof = np.shape(Ka)[0]
        NumNode = dof/6
        #self.aset = 25
        ommit = NumNode-aset

        ai=[];aj=[]

        Koo = Ka[:ommit*6,:ommit*6]
        Koa = Ka[:ommit*6,ommit*6:]
        Kao = Ka[ommit*6:,:ommit*6]
        Kaa = Ka[ommit*6:,ommit*6:]

        Koo1 =np.copy(Koo)
        Koa1 = np.copy(Koa)
        Kao1 = np.copy(Kao)
        Kaa1 = np.copy(Kaa)

        if Ma is not None:
            Moo = Ma[:ommit*6,:ommit*6]
            Moa = Ma[:ommit*6,ommit*6:]
            Mao = Ma[ommit*6:,:ommit*6]
            Maa = Ma[ommit*6:,ommit*6:]

            Moo1 =np.copy(Moo)
            Moa1 = np.copy(Moa)
            Mao1 = np.copy(Mao)
            Maa1 = np.copy(Maa)


        count=0
        print 'start'
        for i in range(dof):
            if np.allclose(Ka[i,:],np.zeros((dof,1))):
                #pdb.set_trace()
                if i == ommit*6:
                    print 'yess'
                if i<ommit*6:
                    Koo1 = np.delete(Koo1,i-count,0)
                    Koa1 = np.delete(Koa1,i-count,0)
                    if Ma is not None:
                        if np.allclose(Ma[i,:],np.zeros((dof,1))):
                            pass
                        else:
                            print i
                        Moo1 = np.delete(Moo1,i-count,0)
                        Moa1 = np.delete(Moa1,i-count,0)

                else:
                    Kao1 = np.delete(Kao1,i-count,0)
                    Kaa1 = np.delete(Kaa1,i-count,0)
                    if Ma is not None:
                        if np.allclose(Ma[i,:],np.zeros((dof,1))):
                            pass
                        else:
                            print i
                        Mao1 = np.delete(Mao1,i-count,0)
                        Maa1 = np.delete(Maa1,i-count,0)
                count+=1
            else:
                ai.append(i)


        count2=0
        for i in range(dof):
            if np.allclose(Ka[:,i],np.zeros((dof,1))):
                if i == ommit*6:
                   print 'yess'
               
                if i<ommit*6:
                    Koo1 = np.delete(Koo1,i-count2,1)
                    Kao1 = np.delete(Kao1,i-count2,1)
                    if Ma is not None:
                        Moo1 = np.delete(Moo1,i-count2,1)
                        Mao1 = np.delete(Mao1,i-count2,1)
                        if np.allclose(Ma[:,i],np.zeros((dof,1))):
                            pass
                        else:
                            print i
                else:

                    Koa1 = np.delete(Koa1,i-count2,1)
                    Kaa1 = np.delete(Kaa1,i-count2,1)
                    if Ma is not None:
                        Moa1 = np.delete(Moa1,i-count2,1)
                        Maa1 = np.delete(Maa1,i-count2,1)
                        if np.allclose(Ma[:,i],np.zeros((dof,1))):
                            pass
                        else:
                            print i
                count2+=1
            else:
                aj.append(i)

        if Ma is not None:
            return(Koo1,Koa1,Kao1,Kaa1,Moo1,Moa1,Mao1,Maa1,ai,aj)
        else:
            return(Koo1,Koa1,Kao1,Kaa1,ai,aj)

    def cond_guyan(self):

        self.Ka_g2 = self.Kaa1-self.Kao1.dot(self.IKoo1.dot(self.Koa1))
        self.Ma_g2 = self.Maa1+self.Kao1.dot(self.IKoo1).dot(self.Moo1).dot(self.IKoo1).dot(self.Koa1)
        #self.Kao1.dot(self.IKoo1).dot(self.Moa1)-self.Mao1.dot(self.IKoo1.dot(self.Koa1))

        Rg = -self.IKoo1.dot(self.Koa1)
        self.Tg=np.vstack((Rg,np.eye(self.aset*6)))
        self.Ka_g = self.Tg.T.dot(self.K.dot(self.Tg))
        self.Ma_g = self.Tg.T.dot(self.M.dot(self.Tg))
        self.Dg,self.Vg = scipy.linalg.eigh(self.Ka_g,self.Ma_g)
        self.Dg2,self.Vg2 = scipy.linalg.eigh(self.Ka_g2,self.Ma_g2)
    def check_guyan(self,r_tol=1e-3,a_tol=1e-2):
            np.allclose(self.Ka,self.Ka_g2,rtol=r_tol, atol=a_tol)
            np.allclose(self.Ma,self.Ma_g2,rtol=r_tol, atol=a_tol)

    def cond_classic(self,w0):
        #w0=np.sqrt(147)
        #w0=1.8
        self.T_w0=np.vstack((-np.linalg.inv(self.Koo1-w0**2*self.Moo1).dot(self.Koa1-
        w0**2*self.Moa1),np.eye(self.aset*6)))
        self.Ka_w0 = self.T_w0.T.dot(self.K.dot(self.T_w0))
        self.Ma_w0 = self.T_w0.T.dot(self.M.dot(self.T_w0))
        self.D_w0,self.V_w0 = scipy.linalg.eigh(self.Ka_w0,self.Ma_w0)

    def cond_irs(self):
        # T_irs0 = np.vstack((self.IKoo1.dot((self.Moa1+
        # self.Moo1.dot(Rs)).dot(np.linalg.inv(self.Ma_g).dot(self.Ka_g))),np.eye(self.aset*6)))
        # self.T_irs = T_irs0 + self.Tg
        # self.Ka_irs = self.T_irs.T.dot(self.K.dot(self.T_irs))
        # self.Ma_irs = self.T_irs.T.dot(self.M.dot(self.T_irs))
        # self.D_irs,self.V_irs = scipy.linalg.eigh(self.Ka_irs,self.Ma_irs)

        Kox=np.zeros(np.shape(self.K))
        Kox[:len(self.K)-6*self.aset,:len(self.K)-6*self.aset] = self.Koo1
        T_irs0 = Kox.dot(self.M.dot(self.Tg.dot(np.linalg.inv(self.Maa1).dot(self.Kaa1))))
        self.T_irs = T_irs0 + self.Tg
        self.Ka_irs = self.T_irs.T.dot(self.K.dot(self.T_irs))
        self.Ma_irs = self.T_irs.T.dot(self.M.dot(self.T_irs))
        self.D_irs,self.V_irs = scipy.linalg.eigh(self.Ka_irs,self.Ma_irs)

    def fcond(self,w,v):
        phi_s = -np.linalg.inv(self.Koo1-w*self.Moo1).dot(self.Koa1-w*self.Moa1).dot(v)
        return phi_s

    def fcond2(self,w,v):
        phi_s = -(self.IKoo1+w*self.IKoo1.dot(self.Moo1.dot(self.IKoo1))).dot(self.Koa1-w*self.Moa1).dot(v)
        return phi_s

    def fcond3(self,w,v):
        phi_s = -(self.IKoo1.dot(self.Koa1)+w*(-self.IKoo1.dot(self.Moa1)+
        self.IKoo1.dot(self.Moo1).dot(self.IKoo1).dot(self.Koa1))).dot(v)
        return phi_s

    def cond_iter(self,iteration):

        for iter in range(iteration):
            if iter == 0:
                self.D_i=copy.deepcopy(self.Dg); self.V_i = copy.deepcopy(self.Vg)
            Phis=np.zeros((self.ommit2*6,self.aset*6))
            for i in range(6*self.aset):
                Phis[:,i] = self.fcond3(self.D_i[i],self.V_i[:,i])
            self.T_i = np.vstack((Phis.dot(np.linalg.inv(self.V_i)),np.eye(6*self.aset)))
            self.K_i = self.T_i.T.dot(self.K.dot(self.T_i))
            self.M_i = self.T_i.T.dot(self.M.dot(self.T_i))
            self.D_i,self.V_i = scipy.linalg.eigh(self.K_i,self.M_i)


# Ti,Ki,Mi,Di,Vi = iterCond(1,aset,ommit2)
# Ti2,Ki2,Mi2,Di2,Vi2 = iterCond(1,aset,ommit2)

# #np.save('../../FEM/Ki',Ki)
# #np.save('../../FEM/Mi',Mi)
# np.save('../../FEM/Kw12',Ka_w0)
# np.save('../../FEM/Mw12',Ma_w0)

# from numpy import linalg as la

# def nearestPD(A):
#     """Find the nearest positive-definite matrix to input

#     A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
#     credits [2].

#     [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

#     [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
#     matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
#     """

#     B = (A + A.T) / 2
#     _, s, V = la.svd(B)

#     H = np.dot(V.T, np.dot(np.diag(s), V))

#     A2 = (B + H) / 2

#     A3 = (A2 + A2.T) / 2

#     if isPD(A3):
#         return A3

#     spacing = np.spacing(la.norm(A))
#     # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
#     # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
#     # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
#     # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
#     # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
#     # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
#     # `spacing` will, for Gaussian random matrixes of small dimension, be on
#     # othe order of 1e-16. In practice, both ways converge, as the unit test
#     # below suggests.
#     I = np.eye(A.shape[0])
#     k = 1
#     while not isPD(A3):
#         mineig = np.min(np.real(la.eigvals(A3)))
#         A3 += I * (-mineig * k**2 + spacing)
#         k += 1

#     return A3

# def isPD(B):
#     """Returns true when input is positive-definite, via Cholesky"""
#     try:
#         _ = la.cholesky(B)
#         return True
#     except la.LinAlgError:
#         return False


# class try1:
#     def __init__(self,K):
#         self.v = self.tryf(K)

#     #@staticmethod
#     def tryf(self,x):
#         return x**2+1
