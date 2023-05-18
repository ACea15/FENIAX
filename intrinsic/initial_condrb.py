import numpy as np
import pdb
import sys
import os
import copy
import time
import pickle
import importlib
import  intrinsic.functions
#import  intrinsic.beam_path
import  intrinsic.geometryrb
#import  intrinsic.FEmodel
from intrinsic.modesrb import Phi1,Phi1l,Phi2,Phi2l,Omega
from intrinsic.Tools.transformations import quaternion_from_matrix

#=================================================================================================================
import Runs.Torun
#Runs.Torun.torun ='Shell_Rafa30_lum'# 'Shell_Rafan'
#Runs.Torun.torun = 'Simo45_20'
#Runs.Torun.force = 'F'
#Runs.Torun.initcond = 'InitCond'
#pdb.set_trace()
V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
if Runs.Torun.force:
   F = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.force)
if Runs.Torun.initial_cond:
   L = importlib.import_module("Models"+'.'+Runs.Torun.torun+'.'+Runs.Torun.initial_cond)
#except ImportError or TypeError:
else:
   class initial:
      pass
   L = initial()
   L.init_x1 = 0
   L.init_x2 = 0
   L.init_q1 = 0
   L.init_q2 = 0
#pdb.set_trace()
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometryrb.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped,V.MBbeams)


def arrange_phi1():
   #V.NumModes = 50
   Beams =range(V.NumBeams) #range(12)#[2,3,4,5]
   x10 = np.hstack([np.zeros(6*len(BeamSeg[i].NodeX)) for i in Beams])
   x10x=np.reshape(x10,(len(x10)/6,6))
   k=0
   for i in Beams:
      for j in range(len(BeamSeg[i].NodeX)):

         x10[6*k:6*k+6] = L.fv(BeamSeg[i].NodeX[j])
         k=k+1

   # x02=np.zeros(NumNode*3)
   # lam=0.002
   # for i in range(NumNode):
   #    x02[3*i+1]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   #    x02[3*i+2]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   #V.NumModes=60
   #H=np.vstack([np.hstack(Phi1l[0][i][1:]) for i in range(V.NumModes)]).T
   H1=np.vstack([np.hstack(np.vstack([Phi1[i][k] for i in Beams])) for k in range(V.NumModes)]).T

   #H2=np.vstack([np.hstack(Phi1l[0][i][1:,0:3]) for i in range(V.NumModes)]).T
   return x10,H1

def arrange_phi2():
   #V.NumModes = 50
   Beams =range(V.NumBeams) #range(12)#[2,3,4,5]
   x20 = np.hstack([np.zeros(6*len(BeamSeg[i].NodeX)) for i in Beams])
   x20x=np.reshape(x20,(len(x20)/6,6))
   k=0
   for i in Beams:
      for j in range(len(BeamSeg[i].NodeX)):

         x20[6*k:6*k+6] = L.fs(BeamSeg[i].NodeX[j])
         k=k+1

   # x02=np.zeros(NumNode*3)
   # lam=0.002
   # for i in range(NumNode):
   #    x02[3*i+1]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   #    x02[3*i+2]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   #V.NumModes=60
   #H=np.vstack([np.hstack(Phi1l[0][i][1:]) for i in range(V.NumModes)]).T
   H2 = np.vstack([np.hstack(np.vstack([Phi2[i][k] for i in Beams])) for k in range(V.NumModes)]).T

   #H2=np.vstack([np.hstack(Phi1l[0][i][1:,0:3]) for i in range(V.NumModes)]).T
   return x20,H2
#q01=np.linalg.solve(H, x0)
#"""

#q012=np.linalg.solve(H2, x02)
#q012=np.linalg.lstsq(H2, x02)[0]


def define_q0(NumAeroStates=0,V=V,BeamSeg=BeamSeg,V0=V,nb=1):
   """ This function yields the initial conditions for the aeroelastic system 
       q0 = [q01,q02,q00,q0p,q0r] 
       -> [velocity component,force component, displacement component,aero states, rotation states]"""

   
   if L.init_x1:  # Initial velocities defined
      
      x10,H1 = arrange_phi1()
      try:
        q01 = np.linalg.solve(H1, x10)
      except:
        q01 = np.linalg.lstsq(H1, x10)[0]

   elif L.init_q1:   # Initial velocity modal-component defined
      
      q01 = L.q1
   else:
      q01 = np.zeros(V.NumModes - V.NumModes_res)

   if L.init_x2:   # Initial  strains defined
      x20,H2 = arrange_phi2()
      try:
        q02 = np.linalg.solve(H2, x20)
      except:
        q02 = np.linalg.lstsq(H2, x20)[0]

   elif L.init_q2:   # Initial strain modal-component defined
      q02 = L.q2
   else:
      q02 = np.zeros(V.NumModes - V.NumModes_res)

   q0 = np.hstack([q01,q02])

   if F.NumALoads:    # Aerodynamic loads

      try:
         q00 = L.q0       #  Initial displacement modal-component defined
      except:
         q00 = np.zeros(V.NumModes-V.NumModes_res)
      q0 = np.hstack([q0,q00])
      try:
         qp = L.qp         #  Initial aerodynamic-states defined
      except:
         qp = np.zeros((V.NumModes-V.NumModes_res)*NumAeroStates)
      q0 = np.hstack([q0,qp])

   try:
      qr = L.qr      
   except:
      qr=[]
      if F.NumDLoads>0 and not F.Gravity:   #  Rotational states (quaternions at the loads points )

         for i in range(F.NumDLoads):
           qr = np.hstack([qr,quaternion_from_matrix(BeamSeg[F.Dead_points_app[i][0]].GlobalAxes)])
         
      elif F.Gravity and V.rotation_quaternions: #  Rotational states (quaternions at all points)

           qr = np.hstack([qr,np.hstack(np.hstack([quaternion_from_matrix(BeamSeg[i].GlobalAxes) for j in range(BeamSeg[i].EnumNodes)]) for i in range(V.NumBeams))])

      elif F.Gravity and V.rotation_strains:   #  Rotational states (quaternion for rigid body)
         for i in range(V.NumBeams):
            if ((i in V.initialbeams and i not in V.BeamsClamped) or i in V.MBbeams):
                qr = np.hstack([qr,quaternion_from_matrix(BeamSeg[i].GlobalAxes)])
      if V0.NumBodies>1:
            for ci in range(V0.NumConstrains):
               qrx=[]
               if nb == V0.Constrains['c%s'%ci][0][0]:
                  qrx=quaternion_from_matrix(BeamSeg[V0.Constrains['c%s'%ci][1][0]].GlobalAxes)
               elif nb == V0.Constrains['c%s'%ci][0][1]:
                  qrx=quaternion_from_matrix(BeamSeg[V0.Constrains['c%s'%ci][1][1]].GlobalAxes)
               qr=np.hstack([qr,qrx])   
   q0 = np.hstack([q0,qr])
   return q0


def define_q0mb(Vmb,BeamSegmb,NumAeroStates=0):
   NumBodies = len(Vmb)
   q0=[]
   for i in range(NumBodies):
      q0.append(define_q0(NumAeroStates=NumAeroStates,V=Vmb[i],BeamSeg=BeamSegmb[i],V0=Vmb[0],nb=i))
   return np.hstack(q0),q0


def test_X1(q0):

   tn=V.tn
   V.tn=1
   #V.NumModes=150
   q1=[q0[0:V.NumModes]];q2=[np.zeros(V.NumModes)]
   #q1=[q012];q2=[np.zeros(V.NumModes)]
   import  intrinsic.solrb
   X1,X2 = intrinsic.solrb.solX(Phi1l,Phi2l,q1,q2,V,BeamSeg)

   V.tn=tn
   # sx=0.
   # k=0
   # for i in Beams:
   #    for j in range(BeamSeg[i].EnumNodes):
   #       for d in range(6):
   #          sx=sx+(x10[6*k+d]-X1[i][j][0][d])**2
   #       k=k+1

   # err=np.sqrt(sx/(6*k))

   # np.save(V.feminas_dir+'/Runs/'+V.model+'/'+V.q0_file,q0)

   # check_axes=0
   # if check_axes:
   #    k=6
   #    for i in range(len(BeamSeg[k].NodeX)-1):
   #       print BeamSeg[k].NodeX[i+1]-BeamSeg[k].NodeX[i]
   return X1,X2


'''
q1x=q0[0:NumNode*6]; q2x = q0[NumNode*6:NumNode*6+V.NumModes]
q1x=q01
q1x=[q1x];q2x=[q2x]
X1f=[[] for i in range(V.NumBeams)]
X1f2=[[] for i in range(V.NumBeams)]
X2f=[[] for i in range(V.NumBeams)]
'''

#X1l,X2l = intrinsic.sol.solX(Phi1l,Phi2l,q1x,q2x,BeamSeg,1,V.NumBeams,35)

'''
for i in range(V.NumBeams):
            X1f[i]=np.zeros((BeamSeg[i].EnumNodes,6))
            X1f2[i]=np.zeros((BeamSeg[i].EnumNodes,6))
            X2f[i]=np.zeros((BeamSeg[i].EnumNodes,6))

            #for j in range(BeamSeg[i].EnumNodes):
            for k in range(V.NumModes):

                            X1f[i]=Phi1l[i][k]*q1x[k]+X1f[i]
                            #X1f2[i]=H[:,k]*q1x[k]+X1f2[i]
                            #X2f[i]=Phi2[i][k]*q2x[k]+X2f[i]

t1=0.
for k in range(V.NumModes):

   t1=t1+H[-5,k]*q1x[k]
'''
