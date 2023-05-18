import numpy as np
import pdb
import sys
import os
import copy

import  intrinsic.functions
import  intrinsic.beam_path
import  intrinsic.geometry
import  intrinsic.FEmodel

import importlib
import Runs.Torun
#Runs.Torun.torun = 'HaleX1c'
#Runs.Torun.torun = 'GolandWing'
#Runs.Torun.torun = 'Tony_Flying_Wing'
#Runs.Torun.torun = 'Hingedfree3'
#Runs.Torun.torun = 'dpendulum'
#Runs.Torun.variables='V'
V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)

#V.Check_Phi2 = 1
#V.Path4Phi2 = 0
import  intrinsic.geometryrb

BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometryrb.geometry_def(V.Grid,
                                                  V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,
                                                  V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped,MBbeams=V.MBbeams)

if V.Nastran_modes:
  Ka,Ma,Dreal,Vreal=intrinsic.FEmodel.fem2(V.K_a,V.M_a,V.op2name,V.NumModes,V.Nastran_modes_dic)

else:
  Ka,Ma,Dreal,Vreal=intrinsic.FEmodel.fem(V.K_a,V.M_a,V.Nastran_modes,V.op2name,NumNode,V.NumModes)

Xm=intrinsic.FEmodel.CentreofMass(Ma,V.Clamped,V.NumBeams,BeamSeg,V.cg)
MdotV=np.zeros(np.shape(Vreal))
Mdotg0=np.zeros(np.shape(Vreal))
Phi0=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))    for i in range(V.NumBeams)]
Phi1=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))    for i in range(V.NumBeams)]
Phi1m=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))   for i in range(V.NumBeams)]
Phi2=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))    for i in range(V.NumBeams)]
CPhi2x=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))  for i in range(V.NumBeams)]
MPhi1=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))   for i in range(V.NumBeams)]
MPhi1x=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))  for i in range(V.NumBeams)]
Phig0=[np.zeros((BeamSeg[i].EnumNodes,6))   for i in range(V.NumBeams)]

Phi0l=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))    for i in range(V.NumBeams)]
Phi1l=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))    for i in range(V.NumBeams)]
Phi2l=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))    for i in range(V.NumBeams)]
Phi1ml=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))   for i in range(V.NumBeams)]
CPhi2xl=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))  for i in range(V.NumBeams)]
MPhi1l=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))   for i in range(V.NumBeams)]
MPhi1l2=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))  for i in range(V.NumBeams)]
MPhi1xl=[np.zeros((V.NumModes,BeamSeg[i].EnumNodes,6))  for i in range(V.NumBeams)]

if V.Check_Phi2:
  Phi2_contrary=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
  Phi2_total=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]

if V.RigidBody_Modes:

  if V.ReplaceRBmodes:
      for i in range(V.NumBeams):

         for j in range(BeamSeg[i].EnumNodes):
             for k in range(6):
                 Phi0[i][k][j][k]=1

             Phi0[i][3][j][0:3] = np.cross(np.array([1,0,0]),BeamSeg[i].NodeX[j,:]-Xm)
             Phi0[i][4][j][0:3] = np.cross(np.array([0,1,0]),BeamSeg[i].NodeX[j,:]-Xm)
             Phi0[i][5][j][0:3] = np.cross(np.array([0,0,1]),BeamSeg[i].NodeX[j,:]-Xm)

      # rewrite information back
         for k in range(6):
          for j in range(BeamSeg[i].EnumNodes):
            if   i!=0 and j==0:
              pass
            else:
              Vreal[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6] = Phi0[i][k][j]
#pdb.set_trace()
MdotV =  Vreal.dot(Ma.T)
Vreal2 = copy.copy(list(Vreal))
MdotV2 = copy.copy(list(MdotV))
Phi0V = copy.copy(list(Vreal))
#pdb.set_trace()
if V.MBbeams:
  try:
    Ma2=np.load(V.M_a2)
    Mdotg0 = Ma2.dot(np.hstack(V.g0 for i in range(np.shape(Ma2)[0]/6)))
  except:
    #Ma2=Ma
    try:
      Mdotg0 = Ma.dot(np.hstack([[V.g0[ki] for ki in V.MBdofree],np.hstack(V.g0 for i in range(np.shape(Ma)[0]/6))]))
    except:
      Mdotg0 = Ma.dot(np.hstack(V.g0 for i in range(np.shape(Ma)[0]/6)))
else:
  Mdotg0 = Ma.dot(np.hstack(V.g0 for i in range(np.shape(Ma)[0]/6)))
#pdb.set_trace()
for i in range(V.NumBeams):
  for j in range(BeamSeg[i].EnumNodes):
    if j==0:
      if i in V.initialbeams:      # Connected beams on the first node  
        if i in V.BeamsClamped:  # Clamped node: modes have a value of 0
          pass
        elif i in V.MBbeams:       # Multibody node
          Phig0[i][j] = Mdotg0[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
        else:                    # Free node in initial beams
          Phig0[i][j] = Mdotg0[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
      elif i in V.MBbeams:         # Multibody node in connecting beams
        Phig0[i][j] = Mdotg0[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]  
      else:                      # Normal node in connecting beams
        pass
    else:                        # Nodes in non-connecting beams
      Phig0[i][j] = Mdotg0[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]

#===============================================================================
# Phi0, Phi1, Phi2 Calculation
#===============================================================================
#pdb.set_trace()
Omega = np.zeros(V.NumModes)  # [in rad/s], extracted from diagonal eigenvalue matrix
for k in range(V.NumModes):
  #pdb.set_trace()
  #MdotV[k] = Ma.dot(Vreal[k])
  if Dreal[k] < 5e-5:
    Dreal[k]=0.; Omega[k] = 0.
  else:
    Omega[k]=np.sqrt(Dreal[k])
  assert (Dreal[k]>=0.)," Negative Frequency"
  # Nodal force in global frame [Compute overall inbalance of the force]
  #===================================================================================
  NodalForce=-Ma.dot(Vreal[k,:])*Dreal[k] # K*Vi[displacement eigenvector] = Force (This makes sure the nodal force is exactly 0 for RB modes)
  #if k==0:print(NodalForce)
  #pdb.set_trace()
  for m in V.MBbeams:
    for i2 in range(len(V.MBdof[m])):
      NodalForce = np.insert(NodalForce,V.MBnode2[m]*6+V.MBdof[m][i2],0.)
      #pdb.set_trace()
      Vreal2[k] = np.insert(Vreal2[k],V.MBnode2[m]*6+V.MBdof[m][i2],0.)
      Phi0V[k] = np.insert(Phi0V[k],V.MBnode2[m]*6+V.MBdof[m][i2],0.)
      #if k==0: print Vreal2[k]
      MdotV2[k] = np.insert(MdotV2[k],V.MBnode2[m]*6+V.MBdof[m][i2],0.)

  if V.Clamped:                         # Insert 0s in the clamped node of the clamped beams
       for i3 in range(len(V.BeamsClamped)):
            NodalForce=np.insert(NodalForce,(NumNode+i3)*6,np.zeros(6))

  for d in range(len(DupNodes)):
    NodalForce=np.insert(NodalForce,DupNodes[d,1]*6,np.zeros(6))
  
  for i in range(V.NumBeams):
    #pdb.set_trace()
      if V.Clamped or V.Path4Phi2:
             direction = [0 for ix in range(V.NumBeams)]
             totalchildseglist = intrinsic.beam_path.BeamTreeOld(i,BeamSeg)
             totalchildseglist_contrary,direction_contrary = intrinsic.beam_path.BeamTreeOposite(i,BeamSeg)
      else:
             totalchildseglist,direction = intrinsic.beam_path.TreeOpt2(intrinsic.beam_path.Treesec2(i,BeamSeg)[0],intrinsic.beam_path.Treesec2(i,BeamSeg)[1])
             #if V.Check_Phi2 and (not V.Path4Phi2):
             totalchildseglist_contrary,direction_contrary=intrinsic.beam_path.TreeOpt2_contrary(intrinsic.beam_path.Treesec2(i, BeamSeg)[0],intrinsic.beam_path.Treesec2(i,BeamSeg)[1])

      for j in range(BeamSeg[i].EnumNodes):

        if j==0:
          if i in V.initialbeams:      # Connected beams on the first node
            if i in V.BeamsClamped:  # Clamped node: modes have a value of 0
              pass
            elif i in V.MBbeams:       # Multibody node
              for di in range(len(V.MBdofree[i])):             # Free dof, copy from the node; the other dof are clamped
                Phi0[i][k][j][V.MBdofree[i][di]] =  Vreal2[k][6*BeamSeg[i].NodeOrder[j]+V.MBdofree[i][di]]
                Phi1[i][k][j][V.MBdofree[i][di]] =  Vreal2[k][6*BeamSeg[i].NodeOrder[j]+V.MBdofree[i][di]]
                MPhi1[i][k][j][V.MBdofree[i][di]] = MdotV2[k][6*BeamSeg[i].NodeOrder[j]+V.MBdofree[i][di]]
            else:                    # Free node in initial beams
              Phi0[i][k][j] =  Vreal2[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
              Phi1[i][k][j] =  Vreal2[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
              MPhi1[i][k][j] = MdotV2[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
              
          elif i in V.MBbeams:         # Multibody node in connecting beams
              for di in range(len(V.MBdof[i])):                # Fixed dof, duplicate
                Phi0[i][k][j][V.MBdof[i][di]] =  Vreal2[k][V.MBnode[i]*6+V.MBdof[i][di]]
                Phi1[i][k][j][V.MBdof[i][di]] =  Vreal2[k][V.MBnode[i]*6+V.MBdof[i][di]]
                #MPhi1[i][k][j][V.MBdofree[i][di]] = MdotV2[k][V.MBnode[i]*6+V.MBdof[i][di]]
              for di in range(len(V.MBdofree[i])):             # Free dof, copy from the node
                Phi0[i][k][j][V.MBdofree[i][di]] =  Vreal2[k][6*BeamSeg[i].NodeOrder[j]+V.MBdofree[i][di]]
                Phi1[i][k][j][V.MBdofree[i][di]] =  Vreal2[k][6*BeamSeg[i].NodeOrder[j]+V.MBdofree[i][di]]
                MPhi1[i][k][j][V.MBdofree[i][di]] = MdotV2[k][6*BeamSeg[i].NodeOrder[j]+V.MBdofree[i][di]]
          else:                      # Normal node in connecting beams
            Phi0[i][k][j] = Vreal2[k][BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6]
            Phi1[i][k][j] = Vreal2[k][BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6]
            MPhi1[i][k][j] = MdotV2[k][BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6]

        else:                        # Nodes in non-connecting beams
          Phi0[i][k][j] =  Vreal2[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
          Phi1[i][k][j] =  Vreal2[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
          MPhi1[i][k][j] = MdotV2[k][BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
          #Phig[i][j] = Mdotg0[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
            #MPhi1l[i][k][j] = intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(MPhi1[i][k][j])

      Phi1m[i][k][0:-1] = (Phi1[i][k][1:]+Phi1[i][k][0:-1])/2

      for j in range(BeamSeg[i].EnumNodes-1):

              if V.Check_Phi2 and (not V.Clamped):
        # Find the beams in the path that defines Phi2 as the sum of forces in the nodes of that path
        #============================================================================================

              # Sum all subsequent nodes on current segment
              #=============================================

                  for kk in range(V.NumBeams): # For all child segments
                       for ll in range(BeamSeg[kk].EnumNodes): # exclude last node on tree because it will be shared with parent
                          Phi2_total[i][k][j,0:3] = Phi2_total[i][k][j,0:3]+(NodalForce[(BeamSeg[kk].NodeOrder[ll])*6:(BeamSeg[kk].NodeOrder[ll])*6+3])
                          Phi2_total[i][k][j,3:6] = Phi2_total[i][k][j,3:6]+(NodalForce[(BeamSeg[kk].NodeOrder[ll])*6+3:(BeamSeg[kk].NodeOrder[ll])*6+6])
                          Phi2_total[i][k][j,3:6] = Phi2_total[i][k][j,3:6]+(np.cross(BeamSeg[kk].NodeX[ll,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2,NodalForce[(BeamSeg[kk].NodeOrder[ll]-1+1)*6:(BeamSeg[kk].NodeOrder[ll]-1+1)*6+3]))

                  if direction_contrary[i] == 1:  # Contrary path to the current beam
                    for jj in range(j+1):
                      Phi2_contrary[i][k][j,0:3] =  Phi2_contrary[i][k][j,0:3] + (NodalForce[(BeamSeg[i].NodeOrder[jj])*6:(BeamSeg[i].NodeOrder[jj])*6+3])
                      Phi2_contrary[i][k][j,3:6] = Phi2_contrary[i][k][j,3:6]+(NodalForce[(BeamSeg[i].NodeOrder[jj])*6+3:(BeamSeg[i].NodeOrder[jj])*6+6])  # and moments caused by forces
                      Phi2_contrary[i][k][j,3:6] = Phi2_contrary[i][k][j,3:6]+(np.cross((BeamSeg[i].NodeX[jj,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2),NodalForce[(BeamSeg[i].NodeOrder[jj]-1+1)*6:(BeamSeg[i].NodeOrder[jj]-1+1)*6+3]))

                  # plus all child segments
                  #========================
                    for kk in range(np.size(totalchildseglist_contrary)): # For all child segments
                       for ll in range(0,BeamSeg[totalchildseglist_contrary[kk]].EnumNodes): # exclude last node on tree because it will be shared with parent
                          Phi2_contrary[i][k][j,0:3]= Phi2_contrary[i][k][j,0:3]+(NodalForce[(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6:(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6+3])
                          Phi2_contrary[i][k][j,3:6]=Phi2_contrary[i][k][j,3:6]+(NodalForce[(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6+3:(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6+6])
                          Phi2_contrary[i][k][j,3:6]=Phi2_contrary[i][k][j,3:6]+(np.cross(BeamSeg[totalchildseglist_contrary[kk]].NodeX[ll,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2,NodalForce[(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll]-1+1)*6:(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll]-1+1)*6+3]))


                    Phi2_contrary[i][k][j,:]=-Phi2_contrary[i][k][j,:]

                  elif direction_contrary[i]==0:
                    for jj in range(j+1,BeamSeg[i].EnumNodes):
                      Phi2_contrary[i][k][j,0:3] =  Phi2_contrary[i][k][j,0:3] + (NodalForce[(BeamSeg[i].NodeOrder[jj])*6:(BeamSeg[i].NodeOrder[jj])*6+3])
                      Phi2_contrary[i][k][j,3:6] = Phi2_contrary[i][k][j,3:6]+(NodalForce[(BeamSeg[i].NodeOrder[jj])*6+3:(BeamSeg[i].NodeOrder[jj])*6+6])  # and moments caused by forces
                      Phi2_contrary[i][k][j,3:6] = Phi2_contrary[i][k][j,3:6]+(np.cross((BeamSeg[i].NodeX[jj,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2),NodalForce[(BeamSeg[i].NodeOrder[jj]-1+1)*6:(BeamSeg[i].NodeOrder[jj]-1+1)*6+3]))

                  # plus all child segments
                  #========================
                    for kk in range(np.size(totalchildseglist_contrary)): # For all child segments
                       for ll in range(0,BeamSeg[totalchildseglist_contrary[kk]].EnumNodes): # exclude last node on tree because it will be shared with parent
                          Phi2_contrary[i][k][j,0:3]= Phi2_contrary[i][k][j,0:3]+(NodalForce[(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6:(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6+3])
                          Phi2_contrary[i][k][j,3:6]=Phi2_contrary[i][k][j,3:6]+(NodalForce[(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6+3:(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll])*6+6])
                          Phi2_contrary[i][k][j,3:6]=Phi2_contrary[i][k][j,3:6]+(np.cross(BeamSeg[totalchildseglist_contrary[kk]].NodeX[ll,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2,NodalForce[(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll]-1+1)*6:(BeamSeg[totalchildseglist_contrary[kk]].NodeOrder[ll]-1+1)*6+3]))
        #============================================================================================#
        # Find the beams in the path that defines Phi2 as the sum of forces in the nodes of that path#
        #============================================================================================#

              # Sum all subsequent nodes on current segment
              #=============================================
              if direction[i]==1:  # Contrary path to the current beam
                for jj in range(j+1):
                  Phi2[i][k][j,0:3] =  Phi2[i][k][j,0:3] + (NodalForce[(BeamSeg[i].NodeOrder[jj])*6:(BeamSeg[i].NodeOrder[jj])*6+3])
                  Phi2[i][k][j,3:6]=Phi2[i][k][j,3:6]+(NodalForce[(BeamSeg[i].NodeOrder[jj])*6+3:(BeamSeg[i].NodeOrder[jj])*6+6])  # and moments caused by forces
                  Phi2[i][k][j,3:6]=Phi2[i][k][j,3:6]+(np.cross((BeamSeg[i].NodeX[jj,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2),NodalForce[(BeamSeg[i].NodeOrder[jj]-1+1)*6:(BeamSeg[i].NodeOrder[jj]-1+1)*6+3]))

              # plus all child segments
              #========================
                for kk in range(np.size(totalchildseglist)): # For all child segments
                   for ll in range(0,BeamSeg[totalchildseglist[kk]].EnumNodes): # exclude last node on tree because it will be shared with parent
                      Phi2[i][k][j,0:3]= Phi2[i][k][j,0:3]+(NodalForce[(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6:(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6+3])
                      Phi2[i][k][j,3:6]=Phi2[i][k][j,3:6]+(NodalForce[(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6+3:(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6+6])
                      Phi2[i][k][j,3:6]=Phi2[i][k][j,3:6]+(np.cross(BeamSeg[totalchildseglist[kk]].NodeX[ll,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2,NodalForce[(BeamSeg[totalchildseglist[kk]].NodeOrder[ll]-1+1)*6:(BeamSeg[totalchildseglist[kk]].NodeOrder[ll]-1+1)*6+3]))

                Phi2[i][k][j,:]=-Phi2[i][k][j,:]

              elif direction[i]==0:
                for jj in range(j+1,BeamSeg[i].EnumNodes):
                  Phi2[i][k][j,0:3] =  Phi2[i][k][j,0:3] + (NodalForce[(BeamSeg[i].NodeOrder[jj])*6:(BeamSeg[i].NodeOrder[jj])*6+3])
                  Phi2[i][k][j,3:6] = Phi2[i][k][j,3:6]+(NodalForce[(BeamSeg[i].NodeOrder[jj])*6+3:(BeamSeg[i].NodeOrder[jj])*6+6])  # and moments caused by forces
                  Phi2[i][k][j,3:6] = Phi2[i][k][j,3:6]+(np.cross((BeamSeg[i].NodeX[jj,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2),NodalForce[(BeamSeg[i].NodeOrder[jj]-1+1)*6:(BeamSeg[i].NodeOrder[jj]-1+1)*6+3]))

              # plus all child segments
              #========================
                for kk in range(np.size(totalchildseglist)): # For all child segments
                   for ll in range(0,BeamSeg[totalchildseglist[kk]].EnumNodes): # exclude last node on tree because it will be shared with parent
                      Phi2[i][k][j,0:3]= Phi2[i][k][j,0:3]+(NodalForce[(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6:(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6+3])
                      Phi2[i][k][j,3:6]=Phi2[i][k][j,3:6]+(NodalForce[(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6+3:(BeamSeg[totalchildseglist[kk]].NodeOrder[ll])*6+6])
                      Phi2[i][k][j,3:6]=Phi2[i][k][j,3:6]+(np.cross(BeamSeg[totalchildseglist[kk]].NodeX[ll,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2,NodalForce[(BeamSeg[totalchildseglist[kk]].NodeOrder[ll]-1+1)*6:(BeamSeg[totalchildseglist[kk]].NodeOrder[ll]-1+1)*6+3]))

      if V.Check_Phi2:
          if np.linalg.norm(Phi2[i][k])>1e-6:
            err = np.linalg.norm(Phi2[i][k] -  Phi2_contrary[i][k])/np.linalg.norm(Phi2[i][k])
            err_total = np.linalg.norm(Phi2_total[i][k])/np.linalg.norm(Phi2[i][k])
            #print err
            #pdb.set_trace()
            assert err < 5e-4,'Phi2 differs between paths in beams {} and mode {}. Err = {}'.format(i,k,err)
            assert err_total < 5e-4,'Phi2 differs between paths in beams {} and mode {}. Err_total = {}'.format(i,k,err_total)

for k in range(V.NumModes):
  for i in range(V.NumBeams):
    for j in range(BeamSeg[i].EnumNodes):
      Phi0l[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi0[i][k][j])
      Phi1l[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi1[i][k][j])
      Phi2l[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi2[i][k][j])
      Phi1ml[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi1m[i][k][j])
      #MPhi1l2[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(MPhi1[i][k][j])
      MPhi1l[i][k][j] = intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(MPhi1[i][k][j])

#===============================================================================
# Computation of CPhi2
#===============================================================================
for k in range(V.NumModes):

    if V.RigidBody_Modes and k<V.RigidBody_Modes:
        continue
    #pdb.set_trace()
    for i in range(V.NumBeams):
      for j in range(BeamSeg[i].EnumNodes-1):
        CPhi2x[i][k][j] = (-(Phi0[i][k][j+1]-Phi0[i][k][j])/BeamSeg[i].NodeDL[j] +
        V.EMAT.T.dot( Phi0[i][k][j+1]+Phi0[i][k][j])/2)
        CPhi2xl[i][k][j] = (-(Phi0l[i][k][j+1]-Phi0l[i][k][j])/BeamSeg[i].NodeDL[j] +
        V.EMAT.T.dot(Phi0l[i][k][j+1]+Phi0l[i][k][j])/2)

        #Mode[k].MPhi1x[k][j] = 1./Omega[k]*((Mode[k].Phi2[BeamSeg[segi].NodeOrder[j+1],:]-Mode[k].Phi2[BeamSeg[segi].NodeOrder[j],:])/BeamSeg[segi].NodeDL[j] +V.EMAT.dot(Mode[k].Phi2[BeamSeg[segi].NodeOrder[j],:]))


# Put rigib body modes at the end
#================================

"""
if V.RigidBody_Modes:
  m=[Mode[i] for i in range(6)]

  for i in range(V.NumModes-6):
    Mode[i]=Mode[i+6]
  for i in range(6):
    Mode[i+V.NumModes-6]=m[i]
"""
#Phi0V=np.array(Phi0V)
#Integral to compute intrinsic coefficients. Cycle through nodes.
#================================================================
scale = 1
if scale:

    Alpha1=np.zeros(V.NumModes)
    Alpha2=np.zeros(V.NumModes)
    for k1 in range(V.NumModes):
            alpha1 = 0.
            alpha2 = 0.
            for i in range(V.NumBeams):
                for j in range(0,BeamSeg[i].EnumNodes-1):

                        if i in V.initialbeams and j==0:
                               if i in V.BeamsClamped:
                                   pass
                               elif i in V.MBbeams:
                                   alpha1 = alpha1+ Phi1[i][k1][j,:].T.dot(MPhi1[i][k1][j,:])
                               else:
                                   if i == V.initialbeams[0]:
                                       alpha1 = alpha1+ Phi1[i][k1][j,:].T.dot(MPhi1[i][k1][j,:])
                        elif i in V.MBbeams and j==0:
                          alpha1 = alpha1+ Phi1[i][k1][j,:].T.dot(MPhi1[i][k1][j,:])
                        alpha1 = alpha1+ Phi1[i][k1][j+1,:].T.dot(MPhi1[i][k1][j+1,:])
                        alpha2 = alpha2 + Phi2l[i][k1][j,:].T.dot(CPhi2xl[i][k1][j,:])*BeamSeg[i].NodeDL[j]

            Alpha1[k1] = alpha1
            Alpha2[k1] = alpha2


    #pdb.set_trace()
    for k in range(V.NumModes):
      for i in range(V.NumBeams):
         assert Alpha1[k] > 1e-6,'Negative value of Alpha1 in mode {} '.format(k)
         Phi0[i][k] = Phi0[i][k]/np.sqrt(Alpha1[k])
         Phi1[i][k] = Phi1[i][k]/np.sqrt(Alpha1[k])
         Phi1m[i][k] = Phi1m[i][k]/np.sqrt(Alpha1[k])
         MPhi1[i][k] = MPhi1[i][k]/np.sqrt(Alpha1[k])

         Phi0l[i][k] = Phi0l[i][k]/np.sqrt(Alpha1[k])
         Phi1l[i][k] = Phi1l[i][k]/np.sqrt(Alpha1[k])
         Phi1ml[i][k] = Phi1ml[i][k]/np.sqrt(Alpha1[k])
         MPhi1l[i][k] = MPhi1l[i][k]/np.sqrt(Alpha1[k])

         if np.abs(Alpha2[k]) > 1e-6:
           #assert Alpha2[k] > 1e-6,'Negative value of Alpha2 in mode {} '.format(k)
           CPhi2x[i][k] = CPhi2x[i][k]/np.sqrt(Alpha2[k])
           Phi2l[i][k] = Phi2l[i][k]/np.sqrt(Alpha2[k])
           Phi2[i][k] = Phi2[i][k]/np.sqrt(Alpha2[k])
           CPhi2xl[i][k] = CPhi2xl[i][k]/np.sqrt(Alpha2[k])
      Phi0V[k] = Phi0V[k]/np.sqrt(Alpha1[k])
if (__name__ == '__main__'):

     plot=0
     if plot:
      #from plotting import mode_disp
      import Tools.plotting
      Tools.plotting.mode_disp(V.NumBeams,BeamSeg,[0,1,2,3,4],Phi0,0,0)
     #alpha1i,alpha2i=integral_alphas(BeamSeg,V.Clamped,V.NumModes,V.NumBeams,Phi1,Phi1ml,MPhi1,Phi2l,CPhi2xl)
     #if np.allclose(alpha1i,np.eye(V.NumModes)):
     #  print 'alpha1 ok'



     # if np.allclose(alpha2i,np.eye(V.NumModes)):
     #   print 'alpha2 ok'
     #from ProsProcessing.plotting import mode_disp
     #np.save(V.feminas_dir+'/Models/'+V.model+'/FEM/'+'Phi1.npy',Phi1)
     #np.save(V.feminas_dir+'/Models/'+V.model+'/FEM/'+'Phi0.npy',Phi0)
     #np.save(V.K_a+'Phi0',Phi0)
     print('Modes')
