#!/usr/bin/env python2
import numpy as np
import pdb
import sys
import os
import copy
import time
import datetime
import multiprocessing
import pickle, gzip
import importlib
import argparse
import pdb

import  intrinsic.functions
import  intrinsic.beam_path
import  intrinsic.geometry
import  intrinsic.geometryrb
import  intrinsic.FEmodel
from Utils.common import class2dic
import Runs.Torun

terminal_run = 1
if terminal_run:
  parser = argparse.ArgumentParser(description='Main FEMINAS file')
  parser.add_argument('ModelToRun',type=str, help='Define model to be run')
  parser.add_argument('config_file',type=str, help='Define configuration file')
  args = parser.parse_args()
else:
  class arguments:
    pass
  args=arguments()
  args.ModelToRun = 'XRF1-FWT'
  #args.ModelToRun = 'DPendulum'
  #args.ModelToRun = 'HaleX1c'
  #args.ModelToRun = 'dpendulum2'
  #args.ModelToRun = 'Hesse_25'
  #args.config_file = 'Models.Hingetry3.confi_main'
  #args.ModelToRun = 'Simo_Moment'
  args.config_file = 'Models.XRF1-FWT.confi_clamped'
  #args.config_file = 'Models.DPendulum.confi_main'
  #args.config_file = 'Models.HaleX1c.confi'
  #args.config_file = 'Models.GolandWing.runs.1.confi_GW1'
  #args.config_file = 'Tests.Models.Hesse.confi2d'
  #args.config_file = 'Models.dpendulum2.confi_maing' 
confi = importlib.import_module(args.config_file)
confi.XNumProcess = eval(confi.NumProcess)
Runs.Torun.torun = args.ModelToRun
Runs.Torun.variables = confi.V
Runs.Torun.aero = confi.AeroToRun
Runs.Torun.force = confi.Fname
if confi.InitC is not '':
  Runs.Torun.initial_cond = confi.InitC
else:
  Runs.Torun.initial_cond = None
V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
#pdb.set_trace()
if confi.AeroToRun:
  A = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.aero)
  AICs = np.load(A.Amatrix)
  aerodynamics = 1
  rbd=A.rbd
else:
  aerodynamics = 0
  rbd=0
if confi.Fname:
  F = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.force)

import feminas_functions
nm = '_'+str(V.NumModes)
#NumStates = A.NumPoles*(V.NumModes-V.NumModes_res)

if confi.test_on:
  results=V.feminas_dir+V.model_name+'/Test'+'/'+confi.save_folder
  results_modes=V.feminas_dir+V.model_name+'/Test/'+confi.save_folder+'/Results_modes'
else:
  results = V.feminas_dir+V.model_name+'/'+confi.save_folder#+'/%s'%A.u_inf
  try:
    results_modes = V.feminas_dir+V.model_name+'/'+confi.save_folder_modes
  except:    
    results_modes = V.feminas_dir+V.model_name+'/Results_modes'
#pdb.set_trace()
if not os.path.exists(results):
  os.makedirs(results)
if not os.path.exists(results_modes):
  os.makedirs(results_modes)
#with open(V.feminas_dir+V.model_name+'/Test/Date.txt', "w") as f:
#  f.write(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
#pdb.set_trace()
#pdb.set_trace()
if confi.run_fem:
  Ka,Ma,Cg0 = feminas_functions.fem(results)
if confi.run_modes:
  #pdb.set_trace()
  Phi0,Phi1,Phi1m,MPhi1,Phi2,CPhi2x,Phi0l,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl,Omega,Phig0 = feminas_functions.modes(confi.load_modes,confi.save_modes,results_modes)
if confi.run_coefficients:
  gamma1,gamma2,alpha1,alpha2 = feminas_functions.coefficients(confi.load_gammas,confi.save_gammas,confi.multi,confi.XNumProcess,results_modes,Phi1,Phi1l,Phi1ml,MPhi1,Phi2l,CPhi2xl)
  #gamma1=gamma2=np.zeros((V.NumModes,V.NumModes,V.NumModes))
#pdb.set_trace()


if V.NumBodies>1 and (confi.run_modal_solution or confi.run_displacements):
    import intrinsic.Forcesmb
    dirmb = V.feminas_dir+V.model_name+'/'
    Vmb=[]; Fmb=[]; Amb=[]; Omegamb = []; Gamma1mb = []; Gamma2mb = []; Phi1mb =[]; Phigmb = []
    Phi2mb = []; CPhi2xmb=[]; BeamSegmb =[]; Inverseconnmb = []; Force1mb = []
    
    for nb in range(V.NumBodies):
      Vmb.append(importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+V.variablesmb[nb]))
      Fmb.append(importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+V.forcesmb[nb]))
    for nb in range(V.NumBodies):
      if nb==0:
        Vmb[0].rotation_states = []
        Vmb[0].total_states = []
      if V.aeromb[nb] is not None:
        Amb.append(importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+V.aeromb[nb]))
      else:
        Amb.append(None)
      Omegamb.append(np.load(dirmb+V.results_modesMB[nb]+'/Omega_%s.npy'%Vmb[nb].NumModes))
      Gamma1mb.append(np.load(dirmb+V.results_modesMB[nb]+'/gamma1_%s.npy'%Vmb[nb].NumModes))
      Gamma2mb.append(np.load(dirmb+V.results_modesMB[nb]+'/gamma2_%s.npy'%Vmb[nb].NumModes))
      with open (dirmb+V.results_modesMB[nb]+'/Phil_%s'%Vmb[nb].NumModes, 'rb') as fp:
        [Phi0l,Phi1l,Phi1ml,Phi2l,MPhi1l,CPhi2xl]  = pickle.load(fp)
      with open(dirmb+V.results_modesMB[nb]+'/Phig0_%s'%Vmb[nb].NumModes, 'rb') as fp:
        [Phig0] = pickle.load(fp)
      #print np.shape(Phi1l)
      #pdb.set_trace()
      Phi1mb.append(copy.deepcopy(Phi1l))
      #print np.shape(Phi1mb)
      Phi2mb.append(Phi2l[:])
      CPhi2xmb.append(CPhi2xl[:])
      Phigmb.append(Phig0[:])
      BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometryrb.geometry_def(Vmb[nb].Grid,
                                              Vmb[nb].NumBeams,Vmb[nb].BeamConn,Vmb[nb].start_reading,
                                              Vmb[nb].beam_start,Vmb[nb].nodeorder_start,Vmb[nb].node_start,
                                              Vmb[nb].Clamped,Vmb[nb].ClampX,Vmb[nb].BeamsClamped,Vmb[nb].MBbeams)
      BeamSegmb.append(BeamSeg)
      Inverseconnmb.append(inverseconn)
      Force1mb.append(intrinsic.Forcesmb.Force(Phi1mb[nb],Vmb[nb],Gravity=Fmb[nb].Gravity,Phig0=Phigmb[nb],BeamSeg=BeamSeg,NumFLoads=Fmb[nb].NumFLoads,
                   NumDLoads=Fmb[nb].NumDLoads,NumALoads=Fmb[nb].NumALoads,
                   Follower_points_app=Fmb[nb].Follower_points_app,Follower_interpol=Fmb[nb].Follower_interpol,
                    Dead_points_app=Fmb[nb].Dead_points_app,Dead_interpol=Fmb[nb].Dead_interpol))

      bodies = []
      for ci in range(Vmb[0].NumConstrains):
        bodies += Vmb[0].Constrains['c%s'%ci][0]
      rotation_states_mb = 4*bodies.count(nb)

      if V.rotation_quaternions:
          Vmb[0].rotation_states.append(4*sum([BeamSegmb[nb][i].EnumNodes for i in range(Vmb[nb].NumBeams)])+rotation_states_mb)
      elif V.rotation_strains:
          init_states = len(set(Vmb[nb].MBbeams+Vmb[nb].initialbeams)-set(Vmb[nb].BeamsClamped))
          Vmb[0].rotation_states.append(4*init_states + rotation_states_mb)
      elif max([Fmb[nbx].NumDLoads for nbx in range(V.NumBodies)])>0:
          Vmb[0].rotation_states.append(Fmb[nb].NumDLoads*4+4)  
      else:
          Vmb[0].rotation_states.append(rotation_states_mb)
      #pdb.set_trace()    
      Vmb[0].total_states.append(2*Vmb[nb].NumModes+Vmb[0].rotation_states[nb])
      if V.aeromb[nb] is not None:
        
        Vmb[0].total_states[nb] = Vmb[0].total_states[nb]+(Amb[nb].NumPoles+1)*Vmb[nb].NumModes
      #del Phi0l,Phi1l,Phi1ml,Phi2l,MPhi1l,CPhi2xl
else:
    BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometryrb.geometry_def(V.Grid,
                                                  V.NumBeams,V.BeamConn,V.start_reading,
                                                  V.beam_start,V.nodeorder_start,V.node_start,
                                                  V.Clamped,V.ClampX,V.BeamsClamped,V.MBbeams)

with open(results_modes+'/Geometry', 'wb') as fp:
  pickle.dump(([class2dic(BeamSeg[i]) for i in range(len(BeamSeg))],NumNode, NumNodes, DupNodes, inverseconn),fp)

if confi.run_modal_solution:
  if V.NumModes_res > 0:
    q,qh = feminas_functions.modal_solution_residualized(confi.load_qs,confi.save_qs,results,Omega,Phi1l,CPhi2xl,gamma1,gamma2,BeamSeg,Phig0)
  elif V.NumBodies>1:
    q = feminas_functions.modal_solution_multibody(confi.load_qs,confi.save_qs,results,Omegamb,Phi1mb,CPhi2xmb,Gamma1mb,Gamma2mb,Force1mb,BeamSegmb,Inverseconnmb,Vmb,Fmb,Amb)
  else:
    q = feminas_functions.modal_solution(confi.load_qs,confi.save_qs,results,Omega,Phi1l,CPhi2xl,gamma1,gamma2,BeamSeg,Phig0)
if  confi.run_displacements:
  if V.NumBodies>1:
    nbi = 0
    Rrv=[];Rrq=[];Rrs=[]
    X1b=[];X2b=[]
    for nb in range(V.NumBodies):
      q1 = q[:,nbi:nbi+Vmb[nb].NumModes]
      q2 = q[:,nbi+Vmb[nb].NumModes:nbi+2*Vmb[nb].NumModes]
      X1,X2 = intrinsic.solrb.solX(Phi1mb[nb],Phi2mb[nb],q1,q2,V=Vmb[nb],BeamSeg=BeamSegmb[nb])      
      ra0,ra_v,Rab_v = feminas_functions.displacements(confi.load_sol,0,results,q1,q2,X1,Phi1mb[nb],Phi2mb[nb],CPhi2xmb[nb],V=Vmb[nb],BeamSeg=BeamSegmb[nb],inverseconn=Inverseconnmb[nb])
      ra0,ra_q,Rab_q,Qq = feminas_functions.displacements(confi.load_sol,0,results,q1,q2,X1,Phi1mb[nb],Phi2mb[nb],CPhi2xmb[nb],2,V=Vmb[nb],BeamSeg=BeamSegmb[nb],inverseconn=Inverseconnmb[nb])
      ra0,ra_s,Rab_s,strain,kappa = feminas_functions.displacements(confi.load_sol,0,results,q1,q2,X1,Phi1mb[nb],Phi2mb[nb],CPhi2xmb[nb],3,V=Vmb[nb],BeamSeg=BeamSegmb[nb],inverseconn=Inverseconnmb[nb])
      nbi += Vmb[0].total_states[nb]
      X1b.append(X1); X2b.append(X2)
      Rrv.append([ra0,ra_v,Rab_v])
      Rrq.append([ra0,ra_q,Rab_q,Qq])
      Rrs.append([ra0,ra_s,Rab_s,strain,kappa])
      if confi.save_sol:
        np.save(results+'/ti%s.npy'%nm,V.ti)
        with open(results+'/Solv%s'%nm, 'wb') as fp:
            pickle.dump(Rrv, fp)
        with open(results+'/Solq%s'%nm, 'wb') as fp:
            pickle.dump(Rrq, fp)
        with open(results+'/Sols%s'%nm, 'wb') as fp:
            pickle.dump(Rrs, fp)
  elif V.static:
    q1=q[:,rbd:(V.NumModes-V.NumModes_res)+rbd]
    q2 = q[:,V.NumModes-V.NumModes_res+rbd:2*(V.NumModes-V.NumModes_res)+rbd]
    ra0,ra,Rab,strain,kappa = feminas_functions.displacements(confi.load_sol,confi.save_sol,results,q1,q2,[],Phi1l,Phi2l,CPhi2xl,3)
    if confi.run_cg:
      Cg = feminas_functions.Cg_t(results,ra,Ma)
  elif V.dynamic:
    import intrinsic.solrb
    q1=q[:,rbd:(V.NumModes-V.NumModes_res)+rbd]
    q2 = q[:,V.NumModes-V.NumModes_res+rbd:2*(V.NumModes-V.NumModes_res)+rbd]
    if confi.load_Xs:
      with open (results+'/Xs%s'%nm , 'rb') as fp:
        X1,X2  = pickle.load(fp)
    else:  
      X1,X2 = intrinsic.solrb.solX(Phi1l,Phi2l,q1,q2,V,BeamSeg)
      if confi.save_Xs:
         with open (results+'/Xs%s'%nm , 'wb') as fp:
           pickle.dump([X1,X2], fp)
    ra0,ra_v,Rab_v = feminas_functions.displacements(confi.load_sol,confi.save_sol,results,q1,q2,X1,Phi1l,Phi2l,CPhi2xl)
    ra0,ra_q,Rab_q,Qq = feminas_functions.displacements(confi.load_sol,confi.save_sol,results,q1,q2,X1,Phi1l,Phi2l,CPhi2xl,2)
    ra0,ra_s,Rab_s,strain,kappa = feminas_functions.displacements(confi.load_sol,confi.save_sol,results,q1,q2,X1,Phi1l,Phi2l,CPhi2xl,3)
    if confi.run_cg:
      Cg = feminas_functions.Cg_t(results,ra_v,Ma)
