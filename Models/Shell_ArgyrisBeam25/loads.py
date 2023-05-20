
import numpy as np
#import pdb
import os
model=os.getcwd().split('/')[-1]
import importlib 
V=importlib.import_module("Runs"+'.'+model+'.'+'V')
#dt,time= functions.time_def(t0,tf,tn)

import IntrinsicSolver.geometry
import IntrinsicSolver.functions
model=os.getcwd().split('/')[-1]
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = IntrinsicSolver.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

#dt,time= functions.time_def(t0,tf,tn)
Fa_save='Fa_tip'
BeamForce=[0]
NodeForce=[[-1]]
load_max = [[120000]]
load_direction = [[np.array([0,1,0,0,0,0])]]
load_step = 1


#load_s
## Static Setting, load step in time.
#for i in range(NumBeams):

## Static Setting, load step in time.
Fa = [np.zeros((V.tn,BeamSeg[i].EnumNodes,6)) for i in range(V.NumBeams)]

specify=1
if specify:
 spec=10**3*np.asarray([1.,3.7,7.6,12.1,15.5,17.5,25.2,39.3,48.2,61.,80.,94.5,109.5,120.])
 for i in range(V.tn):
 
       Fa[0][i][-1][1]=spec[i]


pressure=0
if pressure:

 BeamForce=[0]
 pre=[1000*np.asarray([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.96])]
 direc=[np.array([0,1,0])]

 for i in range(len(BeamForce)):
   for ti in range(V.tn):
    for j in range(BeamSeg[i].EnumNodes):
     if j==BeamSeg[i].EnumNodes-1:
      Fa[i][ti][j][0:3]=0.5*pre[i][ti]*direc[i]*BeamSeg[i].L/BeamSeg[i].EnumNode*2
     else: 
      Fa[i][ti][j][0:3]=pre[i][ti]*direc[i]*BeamSeg[i].L/BeamSeg[i].EnumNode*2


ramp=0
if ramp:
      for i in range(len(BeamForce)):
        #if i in BeamsClamped:
            for j in range(len(NodeForce[i])):
              for ti in range(1,V.tn):
                Fa[BeamForce[i]][ti][NodeForce[i][j]] = load_max[i][j]*load_direction[i][j]

              for ti in range(int(round(load_step*(V.tn)))):
                if ti==0:
                 Fa[BeamForce[i]][ti][NodeForce[i][j]] = load_max[i][j]*load_direction[i][j]*(ti+1)/V.tn/load_step
                else:
                 Fa[BeamForce[i]][ti][NodeForce[i][j]] = load_max[i][j]*load_direction[i][j]*(ti+1)/V.tn/load_step



if (__name__ == '__main__'):

      saving=1
      if saving:
            for i in range(len(os.getcwd().split('/'))):
                    if os.getcwd().split('/')[-i-1]=='PyFem2NL2':
                      pyfem2nl_dir="/".join(os.getcwd().split('/')[0:-i])
            np.save(pyfem2nl_dir+'/Runs/'+model+'/'+Fa_save+'.npy',Fa)
            print('Force Saved')

      print('Loads')
