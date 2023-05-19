import numpy as np
import Utils.FEM_Readers

def Qah2Qhh(Qah,Phi0,nx):
    
    Phi0ex = np.insert(Phi0,nx,np.zeros((6,(len(Phi0)))),axis=1)
    Qhh = Phi0ex.dot(Qah)
    return Qhh

def Qhh2Qh_hr(Qhh,Qhr):

    Qh_hr = np.concatenate((Qhr,Qhh),axis=1)
    return Qh_hr

def Q(Phi0V,Rigid_bodyAIC,ClampedAIC,r=1,nx=0,save=[]):

    Qahr,Qahi,Qah = Utils.FEM_Readers.read_complex_nastran_matrices2(Rigid_bodyAIC)
    Qhhr,Qhhi,Qhh = Utils.FEM_Readers.read_complex_nastran_matrices2(ClampedAIC)
    Q=[]
    QHH=[]
    
    for qi in range(len(Qhh)):
        Qhh_r = Qah2Qhh(Qah[qi],Phi0V,nx)
        Qh_hr = Qhh2Qh_hr(Qhh[qi],Qhh_r[:,:r])
        Q.append(Qh_hr)
        QHH.append(Qhh_r)
    Q = np.array(Q)
    QHH = np.array(QHH)
    if save:
        np.save(save[0],Q)
        np.save(save[1],QHH)
    return Q,QHH
