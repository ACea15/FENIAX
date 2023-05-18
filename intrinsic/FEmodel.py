import numpy as np
import pdb
#import pyNastran
try:
    from pyNastran.op2.op2 import OP2
    from pyNastran.op4.op4 import OP4
except:
    print("Print PyNastran not loading in FEmodel")
# ======================================================================================================
# Loading K and M matrices
#======================================================================================================

def fem(K_a,M_a,Nastran_modes,op2name,NumNode,NumModes):
    import scipy.linalg, scipy.sparse.linalg
    if K_a.split('.')[-1] == 'npy':
        Ka=np.load(K_a)
        Ma=np.load(M_a)
    elif  K_a.split('.')[-1] == 'op4':
        op4 = OP4()
        nameK = op4.read_op4(K_a)
        formK,Ka = nameK['KAA']
        nameM = op4.read_op4(M_a)
        formM,Ma = nameM['MAA']
        
    #Cg=np.load('CG.npy')
    #pdb.set_trace()
    if Nastran_modes:
      if op2name.split('.')[-1]=='npy':
          eigvalue = op2name.split('#')[0]
          eigvector = op2name.split('#')[1]
          Dreal = np.load(eigvalue)
          Vreal = np.load(eigvector)
      elif op2name.split('.')[-1]=='op2':

          Vreal=np.zeros((NumModes,6*NumNode))
          model= OP2()
          model.read_op2(op2name+'.op2')
          NumNode2 = np.shape(model.eigenvectors[1].data[0])[0]
          if NumNode == NumNode2:
              for i in range(NumModes):
                  Vreal[i]=np.reshape(model.eigenvectors[1].data[i],(1,6*NumNode))
          elif NumNode == NumNode2-1:
              for i in range(NumModes):
                  Vreal[i]=np.reshape(model.eigenvectors[1].data[i,1:],(1,6*NumNode))
          else:
              raise ValueError('Nastran modes from OP2 not the same size that the model.')

          Dreal=model.eigenvectors[1].eigns
          #Dreal=[]
          #for i in range(1,NumModes+1):
          #  Dreal.append(model.eigenvalues[model.eigenvalues.keys()[0]].eigenvalues[i])

    else:
      Dreal,Vreal = scipy.linalg.eigh(Ka,Ma)
      #Dreal=np.loadtxt("dreal.txt",delimiter=' ')
      #Vreal=np.loadtxt("vreal.txt",delimiter=' ')
      Vreal=Vreal.T

    return Ka,Ma,Dreal,Vreal

def fem2(K_a,M_a,op2name,NumModes,dic):
      #pdb.set_trace()
      if K_a.split('.')[-1] == 'npy':
        Ka=np.load(K_a)
        Ma=np.load(M_a)
      elif  K_a.split('.')[-1] == 'op4':
        op4 = OP4()
        nameK = op4.read_op4(K_a)
        formK,Ka = nameK['KAA']
        nameM = op4.read_op4(M_a)
        formM,Ma = nameM['MAA']
      if op2name.split('.')[-1]=='npy':
          eigvalue = op2name.split('#')[0]
          eigvector = op2name.split('#')[1]
          Dreal = np.load(eigvalue)
          Vreal = np.load(eigvector)
          #pdb.set_trace()
          return Ka,Ma,Dreal,Vreal
      elif op2name.split('.')[-1]=='op2':
          Vreal=[[] for i in range(NumModes)]#np.zeros((NumModes,6*NumNode))
          model= OP2()
          model.read_op2(op2name)
          NumNode2 = np.shape(model.eigenvectors[1].data[0])[0]
          if type(dic).__name__ == 'list':
              assert (model.eigenvectors[1].data[:,dic[0]] == np.zeros((np.shape(model.eigenvectors[1].data)[0],np.shape(model.eigenvectors[1].data)[2]))).all()
              Vreal = np.delete(model.eigenvectors[1].data,dic[0],0)
          if type(dic).__name__ == 'dict':
              for i in range(NumModes):
                  for j in dic.keys():
                      for k in range(len(dic[j])):
                          Vreal[i].append(model.eigenvectors[1].data[i][j][dic[j][k]])

          Dreal=model.eigenvectors[1].eigns[:NumModes]
          return Ka,Ma,Dreal,np.array(Vreal)


#======================================================================================================
# Calculation of  Centre of Mass
#======================================================================================================


def CentreofMass(Ma,Clamped,NumBeams,BeamSeg,cg=''):
  # XXXXXX Warning XXXXXX
  # BEWARE if there is mass offset from node location. Might need to double check this part of the code
  if cg:
      Xm=np.load(cg)
  else:
      xm=np.zeros(3)
      sm=0
      if Clamped != 1:
       xm=xm+BeamSeg[0].NodeX[0,:]*Ma[6*(BeamSeg[0].NodeOrder[0]),6*(BeamSeg[0].NodeOrder[0])]
       sm=sm+Ma[6*(BeamSeg[0].NodeOrder[0]),6*(BeamSeg[0].NodeOrder[0])]
      for i in range(NumBeams):
        for j in range(1,BeamSeg[i].EnumNodes):
          xm=xm+BeamSeg[i].NodeX[j,:]*Ma[6*(BeamSeg[i].NodeOrder[j]),6*(BeamSeg[i].NodeOrder[j])]
          sm=sm+Ma[6*(BeamSeg[i].NodeOrder[j]),6*(BeamSeg[i].NodeOrder[j])]

      Xm=xm/sm       # Center of mass whole structure
  return Xm

def CentreofMassX(Ma,ra,BeamSeg,V):

  sm = 0
  Xm = np.zeros((V.tn,3))
  for ti in range(V.tn):
      xm = np.zeros(3)
      if V.Clamped != 1:
          xm=xm+ra[0][0,ti,:]*Ma[6*(BeamSeg[0].NodeOrder[0]),6*(BeamSeg[0].NodeOrder[0])]
          if ti == 0:
              sm=sm+Ma[6*(BeamSeg[0].NodeOrder[0]),6*(BeamSeg[0].NodeOrder[0])]
      for i in range(V.NumBeams):
          for j in range(1,BeamSeg[i].EnumNodes):
              if ti == 0:
                  sm=sm+Ma[6*(BeamSeg[i].NodeOrder[j]),6*(BeamSeg[i].NodeOrder[j])]
              xm=xm+ra[i][j,ti,:]*Ma[6*(BeamSeg[i].NodeOrder[j]),6*(BeamSeg[i].NodeOrder[j])]

      Xm[ti]=xm/sm
  return Xm


if (__name__ == '__main__'):


   import importlib
   import Runs.Torun
   V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')

   import  intrinsic.geometry
   BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)
   Ka,Ma,Dreal,Vreal=fem(V.K_a,V.M_a,V.Nastran_modes,V.op2name,NumNode,V.NumModes)
   Xm=CentreofMass(Ma,V.Clamped,V.NumBeams,BeamSeg)
