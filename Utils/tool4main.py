import numpy as np
import pdb
import importlib
import argparse
import pickle
import os
import Runs.Torun

terminal_run=1
if terminal_run:
  parser = argparse.ArgumentParser(description='Main FEMINAS file')
  parser.add_argument('ModelToRun',type=str, help='Define model to be run')
  parser.add_argument('config_file',type=str, help='Define configuration file')
  args = parser.parse_args()
else:
  class arguments:
    pass
  args=arguments()
  args.ModelToRun = 'GolandWing'
  args.config_file = 'Models.GolandWing.runs.1.confi_GW1'

confi = importlib.import_module(args.config_file)
Runs.Torun.torun = args.ModelToRun
Runs.Torun.variables = confi.V
Runs.Torun.aero = confi.AeroToRun
Runs.Torun.force = confi.Fname

try:
  Runs.Torun.initial_cond = confi.InitC
except:
  Runs.Torun.initial_cond = None
V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
#pdb.set_trace()
if confi.AeroToRun:
  A = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.aero)
  AICs = np.load(A.Amatrix)
  aerodynamics = 1
else:
  aerodynamics = 0
if confi.Fname:
  F = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.force)

nm = '_'+str(V.NumModes)


if confi.test_on:

  results=V.feminas_dir+V.model_name+'/Test'+'/'+'Results_'+confi.Fname
  results_modes=V.feminas_dir+V.model_name+'/Test'+'/Results_modes'
  if not os.path.exists(V.feminas_dir+V.model_name+'/Test'):
    os.makedirs(V.feminas_dir+V.model_name+'/Test')
  with open(V.feminas_dir+V.model_name+'/Test/Date.txt', "w") as f:
     f.write(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
else:

    try:
      results = V.feminas_dir+V.model_name+'/'+confi.save_folder
    except:
      results = V.feminas_dir+V.model_name+'/Results'+'/'+confi.Fname+'_%s'%V.NumModes

    results_modes = V.feminas_dir+V.model_name+'/Results_modes'


import  intrinsic.geometry
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,
                                                  V.NumBeams,V.BeamConn,V.start_reading,
                                                  V.beam_start,V.nodeorder_start,V.node_start,
                                                  V.Clamped,V.ClampX,V.BeamsClamped)

def system_eigen(fun,t,**args1):

    n = len(args1['q0'])
    Ma = np.zeros((n,n))
    for i in range(n):
        zer = np.zeros(n)
        zer[i] = 1.
        Ma[:,i] = fun(t,zer,args1)

    w,v = np.linalg.eig(Ma)
    return w,Ma

def run_system_eigen(printout=1,save=1):
    import intrinsic.Forces
    import intrinsic.initial_cond
    import intrinsic.dq_new
    with open (results_modes+'/Phil%s'%nm , 'rb') as fp:
        Phi0l,Phi1l,Phi1ml,Phi2l,MPhi1l,CPhi2xl  = pickle.load(fp)
    Omega = np.load(results_modes+'/Omega%s.npy'%nm)
    gamma1 = np.zeros((V.NumModes,V.NumModes,V.NumModes))
    gamma2 = np.zeros((V.NumModes,V.NumModes,V.NumModes)) 
    force1 = intrinsic.Forces.Force(Phi1l,Gravity=F.Gravity,NumFLoads=F.NumFLoads,NumDLoads=F.NumDLoads,NumALoads=F.NumALoads,Follower_points_app=F.Follower_points_app,Follower_interpol=F.Follower_interpol,Dead_points_app=F.Dead_points_app,Dead_interpol=F.Dead_interpol)
    q0 = intrinsic.initial_cond.define_q0(NumAeroStates=A.NumPoles)  
    weigen, Meigen =  system_eigen(intrinsic.dq_new.dqa_lin,0,Phi1=Phi1l,q0=q0,BeamSeg=BeamSeg,NumModes=V.NumModes,NumPoles=A.NumPoles,Omega=Omega,gamma1=gamma1,gamma2=gamma2,t0=V.t0,tf=V.tf,dt=V.dt,tn=V.tn,printx=1,force1=force1,u_inf=A.u_inf,chord=A.c,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))    
    if printout:
        print Meigen
        print weigen
        np.save('/home/ac5015/Dropbox/Computations/FEM4INAS/Models/GolandWingClean/M600.npy',Meigen)
    if save:
        if not os.path.exists(results+'/linear_system'):
            os.makedirs(results+'/linear_system')
        if aerodynamics:
            np.save(results+'/linear_system'+'/Meigen%sm_%su.npy'%(nm,int(A.u_inf)),Meigen)
            np.save(results+'/linear_system'+'/weigen%sm_%su.npy'%(nm,int(A.u_inf)),weigen)
        else:
            np.save(results+'/linear_system'+'/Meigen%s.npy'%nm,Meigen)
            np.save(results+'/linear_system'+'/weigen%s.npy'%nm,weigen)

        
def system_energy(Nmodes,q1,q2):
    tn=len(q1)
    e0 = np.zeros(tn)
    e1 = np.zeros(tn)
    e2 = np.zeros(tn)
    for i in range(tn):
        for m in range(Nmodes):
            e0[i] = e0[i] + 0.5*(q1[i,m]**2+q2[i,m]**2)
            e1[i] = e1[i] + 0.5*(q1[i,m]**2)
            e2[i] = e2[i] + 0.5*(q2[i,m]**2)
    return e0,e1,e2

def run_system_energy(save=1):

    q = np.load(results+'/q%s.npy'%nm)
    q1=q[:,0:(V.NumModes-V.NumModes_res)]
    q2 = q[:,V.NumModes-V.NumModes_res:2*(V.NumModes-V.NumModes_res)]

    e0,e1,e2  = system_energy(V.NumModes,q1,q2)


#run_system_eigen()
