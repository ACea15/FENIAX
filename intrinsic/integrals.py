import numpy as np
import pdb
import importlib
import multiprocessing
import time

import importlib
#from Runs.Torun import torun
#torun = ''
import Runs.Torun
V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')

import  intrinsic.functions
import intrinsic.geometry
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)
quadratic=0
# results_modes=V.feminas_dir+V.model_name+'/Results_modes'
# nm='_'+str(V.NumModes)

# try:
#    raise IOError
#    Phi0l =np.load(results_modes+'/Phi0%s.npy'%nm)
#    Phi1l =np.load(results_modes+'/Phi1%s.npy'%nm)
#    Phi1ml =np.load(results_modes+'/Phi1m%s.npy'%nm)
#    MPhi1l =np.load(results_modes+'/MPhi1l%s.npy'%nm)
#    Phi2l =np.load(results_modes+'/Phi2%s.npy'%nm)
#    CPhi2xl =np.load(results_modes+'/CPhi2x%s.npy'%nm)

# except IOError:
#    from intrinsic.modes import Phi0l,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl,Omega,Phi1,Phi1m,Phi2,MPhi1

#from pathos.multiprocessing import ProcessingPool as Pool
#Integral to compute intrinsic coefficients. Cycle through nodes.
#================================================================

if quadratic:
    def integral_gammas(L1fun,L2fun,BeamSeg,Clamped,NumModes,NumBeams,Phi1,Phi1l,Phi1m,MPhi1,Phi2,CPhi2):

        gamma1=np.zeros((NumModes,NumModes,NumModes))
        gamma2=np.zeros((NumModes,NumModes,NumModes))
        for i in range(NumBeams):

            for j in range(0,BeamSeg[i].EnumNodes-1):
              quadterm=np.array([0., 1./12*(Phi1l[i][k1][j,5] - Phi1l[i][k1][j+1,5]), 1./12*(Phi1l[i][k1][j,4] - Phi1l[i][k1][j+1,4]), 0., 0., 0.])*BeamSeg[i].NodeDL[j]
              gm1=np.zeros((NumModes,NumModes,NumModes))
              gm1x=np.zeros((NumModes,NumModes,NumModes))
              gm2=np.zeros((NumModes,NumModes,NumModes))
              for k1 in range(NumModes):
                   for k2 in range(NumModes):
                        for k3 in range(NumModes):

                            gm1[k1,k2,k3] = Phi1[i][k1][j+1,:].T.dot(L1fun(Phi1[i][k2][j+1,:].T)).dot(MPhi1[i][k3][j+1,:])
                            if not Clamped and i==0 and j==0:
                                gm1x[k1,k2,k3] = Phi1[i][k1][0,:].T.dot(L1fun(Phi1[i][k2][0,:].T)).dot(MPhi1[i][k3][0,:])

                            gm2[k1,k2,k3] = (Phi1m[i][k1][j,:]+quadterm).T.dot(L2fun(Phi2[i][k2][j,:].T)).dot(CPhi2[i][k3][j,:])*BeamSeg[i].NodeDL[j]

              gamma1 = gamma1 + gm1
              if not Clamped and i==0 and j==0:
                  gamma1=gamma1 + gm1x
              gamma2 = gamma2 + gm2

        return gamma1,gamma2

else:
    def integral_gammas(L1fun,L2fun,BeamSeg,Clamped,NumModes,NumBeams,Phi1,Phi1m,MPhi1,Phi2,CPhi2):

        gamma1=np.zeros((NumModes,NumModes,NumModes))
        gamma2=np.zeros((NumModes,NumModes,NumModes))
        for i in range(NumBeams):

            for j in range(0,BeamSeg[i].EnumNodes-1):

              gm1=np.zeros((NumModes,NumModes,NumModes))
              gm1x=np.zeros((NumModes,NumModes,NumModes))
              gm2=np.zeros((NumModes,NumModes,NumModes))
              for k1 in range(NumModes):
                   for k2 in range(NumModes):
                        for k3 in range(NumModes):

                            gm1[k1,k2,k3] = Phi1[i][k1][j+1,:].T.dot(L1fun(Phi1[i][k2][j+1,:].T)).dot(MPhi1[i][k3][j+1,:])
                            if not Clamped and i==0 and j==0:
                                gm1x[k1,k2,k3] = Phi1[i][k1][0,:].T.dot(L1fun(Phi1[i][k2][0,:].T)).dot(MPhi1[i][k3][0,:])

                            gm2[k1,k2,k3] = Phi1m[i][k1][j,:].T.dot(L2fun(Phi2[i][k2][j,:].T)).dot(CPhi2[i][k3][j,:])*BeamSeg[i].NodeDL[j]

              gamma1 = gamma1 + gm1
              if not Clamped and i==0 and j==0:
                  gamma1=gamma1 + gm1x
              gamma2 = gamma2 + gm2

        return gamma1,gamma2

'''
def integral2_gammas(L1fun,L2fun,k,BeamSeg=BeamSeg,Clamped=1,NumBeams=V.NumBeams,Phi1=Phi1l,Phi1m=Phi1ml,MPhi1=MPhi1l,Phi2=Phi2l,CPhi2=CPhi2xl):

    k1=k[0]; k2=k[1];  k3=k[2]
    gamma1=0.
    gamma2=0.
    for i in range(NumBeams):
        for j in range(0,BeamSeg[i].EnumNodes-1):

          gamma1 = gamma1 + Phi1[i][k1][j+1,:].T.dot(L1fun(Phi1[i][k2][j+1,:].T)).dot(MPhi1[i][k3][j+1,:])
          if not Clamped and i==0 and j==0:
              gamma1=gamma1 +  Phi1[i][k1][0,:].T.dot(L1fun(Phi1[i][k2][0,:].T)).dot(MPhi1[i][k3][0,:])
          gamma2 = gamma2 + Phi1m[i][k1][j,:].T.dot(L2fun(Phi2[i][k2][j,:].T)).dot(CPhi2[i][k3][j,:])*BeamSeg[i].NodeDL[j]

    return gamma1,gamma2
'''

#[list(inter.repeat(range(n)[i],n)) for i in range(n)]


#r2=np.isclose(alpha2i,np.eye(V.NumModes))
def integral_alphas(BeamSeg,Clamped,NumModes,NumBeams,Phi1,Phi1m,MPhi1,Phi2,CPhi2):

    alpha1=np.zeros((NumModes,NumModes))
    alpha2=np.zeros((NumModes,NumModes))

    for i in range(NumBeams):

        for j in range(0,BeamSeg[i].EnumNodes-1):

          ap1=np.zeros((NumModes,NumModes))
          ap2=np.zeros((NumModes,NumModes))
          ap1x=np.zeros((NumModes,NumModes))

          for k1 in range(NumModes):
               for k2 in range(NumModes):
                    ap1[k1,k2] = Phi1[i][k1][j+1,:].T.dot(MPhi1[i][k2][j+1,:])
                    if not Clamped and i==0 and j==0:
                     ap1x[k1,k2] = Phi1[i][k1][0,:].T.dot(MPhi1[i][k2][0,:])
                    ap2[k1,k2] = Phi2[i][k1][j,:].T.dot(CPhi2[i][k2][j,:])*BeamSeg[i].NodeDL[j]

          alpha1 = alpha1+ap1
          if not Clamped and i==0 and j==0:
              alpha1=alpha1 + ap1x
          alpha2 = alpha2+ap2
    return alpha1,alpha2
'''
def integral2_alphas(k,BeamSeg=BeamSeg,Clamped=1,NumBeams=V.NumBeams,Phi1=Phi1l,Phi1m=Phi1ml,MPhi1=MPhi1l,Phi2=Phi2l,CPhi2=CPhi2xl):

    k1=k[0]; k2=k[1]
    alpha1=0.
    alpha2=0.

    for i in range(NumBeams):

        for j in range(0,BeamSeg[i].EnumNodes-1):

                    alpha1 = alpha1+ Phi1[i][k1][j+1,:].T.dot(MPhi1[i][k2][j+1,:])
                    if not Clamped and i==0 and j==0:
                     alpha1 = alpha1 + Phi1[i][k1][0,:].T.dot(MPhi1[i][k2][0,:])
                    alpha2 = alpha2 + Phi2[i][k1][j,:].T.dot(CPhi2[i][k2][j,:])*BeamSeg[i].NodeDL[j]


    return alpha1,alpha2

'''

def integral_betas(BeamSeg,Clamped,NumModes,NumBeams,Phi1,Phi1m,MPhi1,Phi2,CPhi2):

    beta2=np.zeros((NumModes,NumModes))
    beta1=np.zeros((NumModes,NumModes))

    for i in range(NumBeams):

        for j in range(0,BeamSeg[i].EnumNodes-1):

          bt2=np.zeros((NumModes,NumModes))
          bt1=np.zeros((NumModes,NumModes))

          for k1 in range(NumModes):
               for k2 in range(NumModes):


                    bt1[k1,k2] = Phi1m[i][k1][j+1,:].T.dot(Phi2[i][k2][j+1,:])-Phi1m[i][k1][j,:].T.dot(Phi2[i][k2][j+1,:])+BeamSeg[i].NodeDL[j]*Phi1m[i][k1][j,:].T.dot(V.EMAT.dot(Phi1m[i][k2][j,:]))
                    bt2[k1,k2] = Phi2[i][k1][j,:].T.dot(Phi1[i][k2][j+1,:]-Phi1[i][k2][j,:]+BeamSeg[i].NodeDL[j]*V.EMAT.T.dot(Phi1m[i][k2][j,:]))

          beta1 = beta1+bt1
          beta2 = beta2+bt2
    return beta1,beta2


def integral_etai(Fa,NumModes,NumBeams,BeamSeg,Phi1):
    eta = np.zeros(NumModes)
    for k in range(NumModes):
        for ii in i:
          for jj in j:

            eta[k] = eta[k] + Phi1[ii][k][jj].dot(Fa[ii][jj])

    return eta

def integral_etan(Fa,NumModes,NumBeams,BeamSeg,Phi1):
    eta = np.zeros(NumModes)
    for k in range(NumModes):
        for i in range(NumBeams):
          for j in range(BeamSeg[i].EnumNodes):

            eta[k] = eta[k] + Phi1[i][k][j].dot(Fa[i][j])

    return eta

def integral_eta(Fa,tn,NumModes,NumBeams,BeamSeg,Phi1):
    eta = np.zeros((tn,NumModes))
    for ti in range(tn):
      for k in range(NumModes):
        for i in range(NumBeams):
          for j in range(BeamSeg[i].EnumNodes):

            eta[ti][k] = eta[ti][k] + Phi1[i][k][j].dot(Fa[i][ti][j])
    return eta

#def var4multi(Phi1,Phi1m,MPhi1,Phi2,CPhi2):
#    return Phi1,Phi1m,MPhi1,Phi2,CPhi2

#Phi1y,Phi1my,MPhi1y,Phi2y,CPhi2y=var4multi(Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl)

def integral2_alphas(k):

    k1=k[0]; k2=k[1]
    alpha1=0.
    alpha2=0.

    for i in range(V.NumBeams):

        for j in range(0,BeamSeg[i].EnumNodes-1):

                alpha1 = alpha1+ Phi1y[i][k1][j+1,:].T.dot(MPhi1y[i][k2][j+1,:])
                if not V.Clamped and i==0 and j==0:
                 alpha1 = alpha1 + Phi1y[i][k1][0,:].T.dot(MPhi1y[i][k2][0,:])
                alpha2 = alpha2 + Phi2y[i][k1][j,:].T.dot(CPhi2y[i][k2][j,:])*BeamSeg[i].NodeDL[j]

    return alpha1,alpha2

def integral2_betas(k):

    k1=k[0]; k2=k[1]
    beta1=0.
    beta2=0.

    for i in range(V.NumBeams):

        for j in range(0,BeamSeg[i].EnumNodes-1):

                beta1 = beta1 + Phi1my[i][k1][j+1,:].T.dot(Phi2y[i][k2][j+1,:])-Phi1my[i][k1][j,:].T.dot(Phi2y[i][k2][j+1,:])+BeamSeg[i].NodeDL[j]*Phi1my[i][k1][j,:].T.dot(V.EMAT.dot(Phi1my[i][k2][j,:]))
                beta2 = beta2 + Phi2y[i][k1][j,:].T.dot(Phi1y[i][k2][j+1,:]-Phi1y[i][k2][j,:]+BeamSeg[i].NodeDL[j]*V.EMAT.T.dot(Phi1my[i][k2][j,:]))


    return beta1,beta2

if quadratic:
    def integral2_gammas(k):

        k1=k[0]; k2=k[1];  k3=k[2]
        gamma1=0.
        gamma2=0.

        for i in range(V.NumBeams):
            for j in range(0,BeamSeg[i].EnumNodes-1):
                quadterm=np.array([0., 1./12*(Phi1ly[i][k1][j,4] - Phi1ly[i][k1][j+1,4]), 1./12*(Phi1ly[i][k1][j,5] - Phi1ly[i][k1][j+1,5]), 0., 0., 0.])*BeamSeg[i].NodeDL[j]
                gamma1 = gamma1 + Phi1y[i][k1][j+1,:].T.dot(intrinsic.functions.L1fun(Phi1y[i][k2][j+1,:].T)).dot(MPhi1y[i][k3][j+1,:])
                if not V.Clamped and i==0 and j==0:
                    gamma1=gamma1 +  Phi1y[i][k1][0,:].T.dot(intrinsic.functions.L1fun(Phi1y[i][k2][0,:].T)).dot(MPhi1y[i][k3][0,:])
                gamma2 = gamma2 + (Phi1my[i][k1][j,:]+quadterm).T.dot(intrinsic.functions.L2fun(Phi2y[i][k2][j,:].T)).dot(CPhi2y[i][k3][j,:])*BeamSeg[i].NodeDL[j]

        return gamma1,gamma2
else:
    def integral2_gammas(k):

        k1=k[0]; k2=k[1];  k3=k[2]
        gamma1=0.
        gamma2=0.
        for i in range(V.NumBeams):
            for j in range(0,BeamSeg[i].EnumNodes-1):

                gamma1 = gamma1 + Phi1y[i][k1][j+1,:].T.dot(intrinsic.functions.L1fun(Phi1y[i][k2][j+1,:].T)).dot(MPhi1y[i][k3][j+1,:])
                if not V.Clamped and i==0 and j==0:
                    gamma1=gamma1 +  Phi1y[i][k1][0,:].T.dot(intrinsic.functions.L1fun(Phi1y[i][k2][0,:].T)).dot(MPhi1y[i][k3][0,:])
                gamma2 = gamma2 + Phi1my[i][k1][j,:].T.dot(intrinsic.functions.L2fun(Phi2y[i][k2][j,:].T)).dot(CPhi2y[i][k3][j,:])*BeamSeg[i].NodeDL[j]

        return gamma1,gamma2


def solve_integrals(multi,integral,NumModes,NumProcess=multiprocessing.cpu_count()-1):


    def permu4multiprocess(NumModes,integral):
        permu=[]
        for i in range(NumModes):
            for j in range(NumModes):

                if integral=='alphas' or integral=='betas':
                    permu.append([i,j])
                elif integral=='gammas':
                    for k in range(NumModes):
                       permu.append([i,j,k])
        return permu


    def recover_gamma(NumModes,multi,res=[]):
        gamma1x=np.zeros((NumModes,NumModes,NumModes))
        gamma2x=np.zeros((NumModes,NumModes,NumModes))
        for i in range(NumModes):
            for j in range(NumModes):
                for k in range(NumModes):

                    if multi:
                      gamma1x[i][j][k]=res[k+NumModes*j+NumModes*NumModes*i][0]
                      gamma2x[i][j][k]=res[k+NumModes*j+NumModes*NumModes*i][1]
                    else:
                      gamma1x[i][j][k],gamma2x[i][j][k]=integral2_gammas([i,j,k])

        return gamma1x,gamma2x

    def recover_al(NumModes,multi,res=[]):
            albe1x=np.zeros((NumModes,NumModes))
            albe2x=np.zeros((NumModes,NumModes))
            for i in range(NumModes):
                for j in range(NumModes):

                        if multi:
                          albe1x[i][j]=res[j+NumModes*i][0]
                          albe2x[i][j]=res[j+NumModes*i][1]
                        else:
                          albe1x[i][j],albe2x[i][j]=integral2_alphas([i,j])

            return albe1x,albe2x


    def recover_be(NumModes,multi,res=[]):
            albe1x=np.zeros((NumModes,NumModes))
            albe2x=np.zeros((NumModes,NumModes))
            for i in range(NumModes):
                for j in range(NumModes):

                        if multi:
                          albe1x[i][j]=res[j+NumModes*i][0]
                          albe2x[i][j]=res[j+NumModes*i][1]
                        else:
                          albe1x[i][j],albe2x[i][j]=integral2_betas([i,j])

            return albe1x,albe2x

    if multi:

        permu=permu4multiprocess(NumModes,integral)
        pool = multiprocessing.Pool(processes=NumProcess)

        if integral=='alphas':
            results=pool.map(integral2_alphas,permu)
            coeff1x,coeff2x=recover_al(NumModes,multi,results)

        elif integral=='betas':
            results=pool.map(integral2_betas,permu)
            coeff1x,coeff2x=recover_be(NumModes,multi,results)

        elif integral=='gammas':
            results=pool.map(integral2_gammas,permu)
            coeff1x,coeff2x=recover_gamma(NumModes,multi,results)
    else:

        if integral=='alphas':
            coeff1x,coeff2x=recover_al(NumModes,multi)

        if integral=='betas':
            coeff1x,coeff2x=recover_be(NumModes,multi)

        elif integral=='gammas':
            coeff1x,coeff2x=recover_gamma(NumModes,multi)

    return coeff1x,coeff2x



'''
alpha1=np.zeros((V.NumModes,V.NumModes))
alpha2=np.zeros((V.NumModes,V.NumModes))
beta2=np.zeros((V.NumModes,V.NumModes))
beta1=np.zeros((V.NumModes,V.NumModes))
gamma1=np.zeros((V.NumModes,V.NumModes,V.NumModes))
gamma2=np.zeros((V.NumModes,V.NumModes,V.NumModes))
for i in range(V.NumBeams):

    for j in range(0,BeamSeg[i].EnumNodes-1):

      ap1=np.zeros((V.NumModes,V.NumModes))
      ap2=np.zeros((V.NumModes,V.NumModes))
      bt2=np.zeros((V.NumModes,V.NumModes))
      ap1x=np.zeros((V.NumModes,V.NumModes))
      gm1=np.zeros((V.NumModes,V.NumModes,V.NumModes))
      gm1x=np.zeros((V.NumModes,V.NumModes,V.NumModes))
      gm2=np.zeros((V.NumModes,V.NumModes,V.NumModes))
      for k1 in range(V.NumModes):
           for k2 in range(V.NumModes):
                ap1[k1,k2] = Phi1[i][k1][j+1,:].T.dot(MPhi1[i][k2][j+1,:])
                if not V.Clamped and i==0 and j==0:
                 ap1x[k1,k2] = Phi1[i][k1][0,:].T.dot(MPhi1[i][k2][0,:])
                ap2[k1,k2] = Phi2l[i][k1][j,:].T.dot(CPhi2xl[i][k2][j,:])*BeamSeg[i].NodeDL[j]
                bt2[k1,k2] = Phi2l[i][k1][j,:].T.dot(Phi1l[i][k2][j+1,:]-Phi1l[i][k2][j,:]-BeamSeg[i].NodeDL[j]*V.EMAT.T.dot(Phi1ml[i][k2][j,:]))

                for k3 in range(V.NumModes):


                    gm1[k1,k2,k3] = Phi1[i][k1][j+1,:].T.dot(intrinsic.functions.L1fun(Phi1[i][k2][j+1,:].T)).dot(MPhi1[i][k3][j+1,:])
                    if not V.Clamped and i==0 and j==0:
                        gm1x[k1,k2,k3] = Phi1[i][k1][0,:].T.dot(intrinsic.functions.L1fun(Phi1[i][k2][0,:].T)).dot(MPhi1[i][k3][0,:])

                    gm2[k1,k2,k3] = Phi1ml[i][k1][j,:].T.dot(intrinsic.functions.L2fun(Phi2l[i][k2][j,:].T)).dot(CPhi2xl[i][k3][j,:])*BeamSeg[i].NodeDL[j]


      alpha1 = alpha1+ap1
      if not V.Clamped and i==0 and j==0:
          alpha1=alpha1 + ap1x
      alpha2 = alpha2+ap2
      beta2 = beta2+bt2
      gamma1 = gamma1 + gm1
      if not V.Clamped and i==0 and j==0:
          gamma1=gamma1 + gm1x
      gamma2 = gamma2 + gm2
'''


'''
gamma1l=[]
gamma2l=[]
for i in range(V.NumModes):
    for j in range(V.NumModes):
        for k in range(V.NumModes):

            if abs(gamma1[i][j][k])>1e-5:
                gamma1l.append([i,j,k,gamma1[i][j][k]])
            if abs(gamma2[i][j][k])>1e-5:
                gamma2l.append([i,j,k,gamma2[i][j][k]])
'''


def interpol4(p1,p2,ds):
  x1=p1[0:3]
  x2=p2[0:3]
  r1=p1[3:6]
  r2=p2[3:6]





def interpol1(x1,x2,ds):

  (x1+x2)/2*ds

if (__name__ == '__main__'):

#if 1:
    Phi1y=Phi1
    Phi1my=Phi1ml
    MPhi1y=MPhi1
    Phi2y=Phi2l
    CPhi2y=CPhi2xl
    if V.loading:
      Fa=np.load(V.feminas_dir+'/Runs/'+V.model+'/Fa.npy')
      eta=integral_eta(Fa,V.tn,V.NumModes,V.NumBeams,BeamSeg,Phi1l)

    start_time = time.time()
    gamma1i,gamma2i=integral_gammas(intrinsic.functions.L1fun,intrinsic.functions.L2fun,BeamSeg,V.Clamped,V.NumModes,V.NumBeams,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    gamma1, gamma2 = solve_integrals(0,'gammas',V.NumModes)
    print("--- %s seconds ---" % (time.time() - start_time))

#if 1:
    start_time = time.time()
    gamma1x,gamma2x=solve_integrals(1,'gammas',V.NumModes,NumProcess=multiprocessing.cpu_count()-1)
    print("--- %s seconds ---" % (time.time() - start_time))

#if 1:
    start_time = time.time()
    #pdb.set_trace()
    alpha1x,alpha2x=solve_integrals(1,'alphas',V.NumModes,NumProcess=multiprocessing.cpu_count()-1)
    print("--- %s seconds ---" % (time.time() - start_time))


#if 1:
    start_time = time.time()
    alpha1, alpha2 = solve_integrals(0,'alphas',V.NumModes)
    print("--- %s seconds ---" % (time.time() - start_time))
#if 1:
    start_time = time.time()
    alpha1i,alpha2i=integral_alphas(BeamSeg,V.Clamped,V.NumModes,V.NumBeams,Phi1,Phi1ml,MPhi1,Phi2l,CPhi2xl)
    print("--- %s seconds ---" % (time.time() - start_time))

#if 1:
    start_time = time.time()
    #pdb.set_trace()
    beta1x,beta2x=solve_integrals(1,'betas',V.NumModes,NumProcess=multiprocessing.cpu_count()-1)
    print("--- %s seconds ---" % (time.time() - start_time))


#if 1:
    start_time = time.time()
    #pdb.set_trace()
    beta1, beta2 = solve_integrals(0,'betas',V.NumModes)
    print("--- %s seconds ---" % (time.time() - start_time))
#if 1:
    start_time = time.time()
    beta1i,beta2i=integral_betas(BeamSeg,V.Clamped,V.NumModes,V.NumBeams,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Integrals')

#(BeamSeg,NumModes,NumBeams,Phi1,Phi1m,MPhi1,Phi2,CPhi2):
#np.isclose(alpha2i,np.eye(V.NumModes))
