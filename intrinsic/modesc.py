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
#Runs.Torun.torun = 'Tony_Flying_Wing'
#Runs.Torun.torun = 'Hesse_25n'
V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')
#V.Check_Phi2 =1
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)
Ka,Ma,Dreal,Vreal=intrinsic.FEmodel.fem(V.K_a,V.M_a,V.Nastran_modes,V.op2name,NumNode,V.NumModes)
Xm=intrinsic.FEmodel.CentreofMass(Ma,V.Clamped,V.NumBeams,BeamSeg)


Phi0=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
Phi1=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
Phi1m=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
Phi2=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
CPhi2x=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
MPhi1=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
MPhi1x=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]

Phi0l=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
Phi1l=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
Phi2l=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
Phi1ml=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
CPhi2xl=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
MPhi1l=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
MPhi1l2=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
MPhi1xl=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]

if V.Check_Phi2:
  Phi2_contrary=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]
  Phi2_total=[[np.zeros((BeamSeg[i].EnumNodes,6)) for j in range(V.NumModes)] for i in range(V.NumBeams)]

#pdb.set_trace()
if V.RigidBody_Modes:

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
#=================================================================================================================
# Phi0, Phi1, Phi2 Calculation
#=================================================================================================================

Omega=[0. for i in range(V.NumModes)]  # [in rad/s], extracted from diagonal eigenvalue matrix
#pdb.set_trace()
for i in range(V.NumBeams):

  if V.Clamped:
         direction=[0 for ix in range(V.NumBeams)]
         totalchildseglist=intrinsic.beam_path.BeamTreeOld(i,BeamSeg)
  else:
         totalchildseglist,direction=intrinsic.beam_path.TreeOpt2(intrinsic.beam_path.Treesec2(i,BeamSeg)[0],intrinsic.beam_path.Treesec2(i,BeamSeg)[1])
         if V.Check_Phi2:
           totalchildseglist_contrary,direction_contrary=intrinsic.beam_path.TreeOpt2_contrary(intrinsic.beam_path.Treesec2(i,BeamSeg)[0],intrinsic.beam_path.Treesec2(i,BeamSeg)[1])


         #direction=[0 for ix in range(V.NumBeams)]
         #totalchildseglist=intrinsic.beam_path.BeamTreeOld(i,BeamSeg)
  #pdb.set_trace()
  for k in range(V.NumModes):

      if V.RigidBody_Modes and k<6:

          #Omega[k]=0.
          Phi1[i][k]=copy.copy(Phi0[i][k])

          for j in range(BeamSeg[i].EnumNodes):

            if   i!=0 and j==0:
              pass
            else:
               MPhi1[i][k][j] = Ma[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6].dot(Vreal[k,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6])
               MPhi1l[i][k][j] = Ma[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6].dot(intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Vreal[k,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]))


          Phi1m[i][k][0:-1] = (Phi1[i][k][1:]+Phi1[i][k][0:-1])/2
          NodalForce=-Ma.dot(Vreal[k,:]) # M*Vi[displacement eigenvector] = Force/omega^2
          for d in range(len(DupNodes)):
            NodalForce=np.insert(NodalForce,DupNodes[d,1]*6,NodalForce[DupNodes[d,0]*6:DupNodes[d,0]*6+6])
      else:
        assert (Dreal[k]>0.)," Negative Frequency"
        Omega[k]=np.sqrt(Dreal[k])
        #pdb.set_trace()
        # Nodal force in global frame [Compute overall inbalance of the force]
        #===================================================================================
        NodalForce=-Ma.dot(Vreal[k,:])*Dreal[k] # K*Vi[displacement eigenvector] = Force
        #print(NodalForce)
        if V.Clamped:
             for i2 in range(len(V.BeamsClamped)):
                  NodalForce=np.insert(NodalForce,(NumNode+i2)*6,np.zeros(6))

        for d in range(len(DupNodes)):
          NodalForce=np.insert(NodalForce,DupNodes[d,1]*6,np.zeros(6))

        for j in range(BeamSeg[i].EnumNodes):

                if j==0 and  V.Clamped and i in V.BeamsClamped:
                    pass
                elif  i not in V.BeamsClamped and i!=0 and j==0:
                    Phi0[i][k][j] = Vreal[k,BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6]
                    MPhi1[i][k][j] = Omega[k]*Ma[BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6,BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6].dot(Vreal[k,BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6])
                    MPhi1l[i][k][j] = Omega[k]*Ma[BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6,BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6].dot(intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Vreal[k,BeamSeg[inverseconn[i]].NodeOrder[-1]*6:BeamSeg[inverseconn[i]].NodeOrder[-1]*6+6]))

                else:
                    #pdb.set_trace()
                    Phi0[i][k][j] =  Vreal[k,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]
                    MPhi1[i][k][j] = Omega[k]*Ma[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6].dot(Vreal[k,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6])

                    MPhi1l[i][k][j] = Omega[k]*Ma[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6].dot(intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Vreal[k,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6]))

        Phi1[i][k] = Phi0[i][k]*Omega[k] # Velocity mode = displacement mode * w
        Phi1m[i][k][0:-1] = (Phi1[i][k][1:]+Phi1[i][k][0:-1])/2
        for j in range(BeamSeg[i].EnumNodes-1):
          # This MPhi1 defined in GLOBAL axes
                if V.Check_Phi2:
          # Find the beams in the path that defines Phi2 as the sum of forces in the nodes of that path
          #============================================================================================

                # Sum all subsequent nodes on current segment
                #=============================================

                    for kk in range(V.NumBeams): # For all child segments
                         for ll in range(BeamSeg[kk].EnumNodes): # exclude last node on tree because it will be shared with parent
                            Phi2_total[i][k][j,0:3]= Phi2_total[i][k][j,0:3]+(NodalForce[(BeamSeg[kk].NodeOrder[ll])*6:(BeamSeg[kk].NodeOrder[ll])*6+3])
                            Phi2_total[i][k][j,3:6]=Phi2_total[i][k][j,3:6]+(NodalForce[(BeamSeg[kk].NodeOrder[ll])*6+3:(BeamSeg[kk].NodeOrder[ll])*6+6])
                            Phi2_total[i][k][j,3:6]=Phi2_total[i][k][j,3:6]+(np.cross(BeamSeg[kk].NodeX[ll,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2,NodalForce[(BeamSeg[kk].NodeOrder[ll]-1+1)*6:(BeamSeg[kk].NodeOrder[ll]-1+1)*6+3]))

                    if direction_contrary[i]==1:  # Contrary path to the current beam
                      for jj in range(j+1):
                        Phi2_contrary[i][k][j,0:3] =  Phi2_contrary[i][k][j,0:3] + (NodalForce[(BeamSeg[i].NodeOrder[jj])*6:(BeamSeg[i].NodeOrder[jj])*6+3])
                        Phi2_contrary[i][k][j,3:6]=Phi2_contrary[i][k][j,3:6]+(NodalForce[(BeamSeg[i].NodeOrder[jj])*6+3:(BeamSeg[i].NodeOrder[jj])*6+6])  # and moments caused by forces
                        Phi2_contrary[i][k][j,3:6]=Phi2_contrary[i][k][j,3:6]+(np.cross((BeamSeg[i].NodeX[jj,:]-(BeamSeg[i].NodeX[j,:]+BeamSeg[i].NodeX[j+1,:])/2),NodalForce[(BeamSeg[i].NodeOrder[jj]-1+1)*6:(BeamSeg[i].NodeOrder[jj]-1+1)*6+3]))

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



          # Find the beams in the path that defines Phi2 as the sum of forces in the nodes of that path
          #============================================================================================
                #pdb.set_trace()
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
          if k > 5:
            err = np.linalg.norm(Phi2[i][k] -  Phi2_contrary[i][k])/np.linalg.norm(Phi2[i][k])
            print err
            assert err < 5e-3,'Phi2 differs between paths in beams {} and mode {}. Err = {}'.format(i,k,err)

for k in range(V.NumModes):
  for i in range(V.NumBeams):
    for j in range(BeamSeg[i].EnumNodes):
      Phi0l[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi0[i][k][j])
      Phi1l[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi1[i][k][j])
      Phi2l[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi2[i][k][j])
      Phi1ml[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(Phi1m[i][k][j])
      MPhi1l2[i][k][j]=intrinsic.functions.Rot6(BeamSeg[i].GlobalAxes.T).dot(MPhi1[i][k][j])


#============================================================================================================
# Computation of CPhi2
#=============================================================================================================
for k in range(V.NumModes):

    if V.RigidBody_Modes and k<6:
        continue
    #pdb.set_trace()
    for i in range(V.NumBeams):
      for j in range(BeamSeg[i].EnumNodes-1):
        CPhi2x[i][k][j] = -(Phi0[i][k][j+1]-Phi0[i][k][j])/BeamSeg[i].NodeDL[j] +V.EMAT.T.dot( Phi0[i][k][j+1]+Phi0[i][k][j])/2
        CPhi2xl[i][k][j] = -(Phi0l[i][k][j+1]-Phi0l[i][k][j])/BeamSeg[i].NodeDL[j] +V.EMAT.T.dot(Phi0l[i][k][j+1]+Phi0l[i][k][j])/2



        #Mode[k].MPhi1x[k][j] = 1./Omega[k]*((Mode[k].Phi2[BeamSeg[segi].NodeOrder[j+1],:]-Mode[k].Phi2[BeamSeg[segi].NodeOrder[j],:])/BeamSeg[segi].NodeDL[j] +V.EMAT.dot(Mode[k].Phi2[BeamSeg[segi].NodeOrder[j],:]))
        #******ATTENTION******
        #  Phi0, Phi1 and MPhi1 defined in global
        # Phi0S, Phi1S, Phi2 and CPhi2 defined in local

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

#Integral to compute intrinsic coefficients. Cycle through nodes.
#================================================================
scale=1
if scale:

     alpha1=np.zeros(V.NumModes)
     alpha2=np.zeros(V.NumModes)
     ap1x=np.zeros(V.NumModes)
     for i in range(V.NumBeams):
         for j in range(0,BeamSeg[i].EnumNodes-1):

           ap1=np.zeros(V.NumModes)
           ap2=np.zeros(V.NumModes)

           for k1 in range(V.NumModes):

                     ap1[k1] = Phi1[i][k1][j+1,:].T.dot(MPhi1[i][k1][j+1,:])
                     if not V.Clamped and i==0 and j==0:
                      ap1x[k1] = Phi1[i][k1][0,:].T.dot(MPhi1[i][k1][0,:])
                     ap2[k1] = Phi2l[i][k1][j,:].T.dot(CPhi2xl[i][k1][j,:])*BeamSeg[i].NodeDL[j]

           alpha1 = alpha1+ap1
           if not V.Clamped and i==0 and j==0:
               alpha1=alpha1 + ap1x
           alpha2 = alpha2+ap2

     for i in range(V.NumBeams):
          for k in range(V.NumModes):

                 Phi0[i][k] = Phi0[i][k]/np.sqrt(alpha1[k])
                 Phi1[i][k] = Phi1[i][k]/np.sqrt(alpha1[k])
                 Phi1m[i][k] = Phi1m[i][k]/np.sqrt(alpha1[k])
                 MPhi1[i][k] = MPhi1[i][k]/np.sqrt(alpha1[k])

                 Phi0l[i][k] = Phi0l[i][k]/np.sqrt(alpha1[k])
                 Phi1l[i][k] = Phi1l[i][k]/np.sqrt(alpha1[k])
                 Phi1ml[i][k] = Phi1ml[i][k]/np.sqrt(alpha1[k])
                 MPhi1l[i][k] = MPhi1l[i][k]/np.sqrt(alpha1[k])


                 if V.Clamped or (V.RigidBody_Modes and k>5) or (V.RigidBody_Modes==0):
                   CPhi2x[i][k] = CPhi2x[i][k]/np.sqrt(alpha2[k])
                   Phi2l[i][k] = Phi2l[i][k]/np.sqrt(alpha2[k])
                   Phi2[i][k] = Phi2[i][k]/np.sqrt(alpha2[k])
                   CPhi2xl[i][k] = CPhi2xl[i][k]/np.sqrt(alpha2[k])



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


if (__name__ == '__main__'):

     plot=0
     if plot:
      #from plotting import mode_disp
      import Tools.plotting
      Tools.plotting.mode_disp(V.NumBeams,BeamSeg,[0,1,2,3,4],Phi0,0,0)
     alpha1i,alpha2i=integral_alphas(BeamSeg,V.Clamped,V.NumModes,V.NumBeams,Phi1,Phi1ml,MPhi1,Phi2l,CPhi2xl)
     #from ProsProcessing.plotting import mode_disp

     print('Modes')
