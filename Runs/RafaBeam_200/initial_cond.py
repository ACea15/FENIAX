import numpy as np
import pdb
import sys
import os
import copy
import time
import pickle

#import  intrinsic.functions
#import  intrinsic.beam_path
import  intrinsic.geometry
#import  intrinsic.FEmodel

import importlib
#=================================================================================================================
import Runs.Torun
Runs.Torun.torun = 'RafaBeam_25'
V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')

BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

from intrinsic.modes import Phi1,Phi1l,Phi2,Phi2l,Omega
x0=np.zeros(NumNode*6)
lam=0.002
for i in range(NumNode):
   x0[6*i+1]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   x0[6*i+2]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
x02=np.zeros(NumNode*3)
lam=0.002
for i in range(NumNode):
   x02[3*i+1]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   x02[3*i+2]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
H=np.vstack([np.hstack(Phi1l[0][i][1:]) for i in range(V.NumModes)]).T
H2=np.vstack([np.hstack(Phi1l[0][i][1:,0:3]) for i in range(V.NumModes)]).T

#q01=np.linalg.solve(H, x0)
#"""

q012=np.linalg.solve(H2, x02)
#q012=np.linalg.lstsq(H2, x02)[0]

try:
  q01=np.linalg.solve(H, x0)
except:
  q01=np.linalg.lstsq(H, x0)[0]
#"""
q0=np.zeros(2*V.NumModes)
q0[0:V.NumModes] = q01
q002=np.zeros(2*V.NumModes)
q002[0:V.NumModes] = q012
#np.save('q0_002.npy',q0)

q1=[q01];q2=[np.zeros(V.NumModes)]
q1=[q012];q2=[np.zeros(V.NumModes)]
import  intrinsic.sol
X1,X2 = intrinsic.sol.solX(Phi1l,Phi2l,q1,q2,V,BeamSeg)

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
