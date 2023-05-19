import numpy as np
import pdb
import sys
import os
import copy
import time
import multiprocessing
import pickle

import  intrinsic.functions
import  intrinsic.beam_path
import  intrinsic.geometry
import  intrinsic.FEmodel

import importlib
from Runs.Torun import torun
#=================================================================================================================

V=importlib.import_module("Runs"+'.'+torun+'.'+'V')
multi=1
NumProcess=multiprocessing.cpu_count()-1



results=V.feminas_dir+V.model_name+'/Results1'
results_modes=V.feminas_dir+V.model_name+'/Results1_modes'
save=0
nm='_'+str(V.NumModes)

Fa_name='Fa0'

BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

#k0=[np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)]
#k0=np.asarray([[np.array([0,0,1./50]) for j in range(BeamSeg[i].EnumNodes)] for i in range(V.NumBeams)])

#=================================================================================================================h
if not os.path.exists(results):
  os.makedirs(results)

if not os.path.exists(results_modes):
  os.makedirs(results_modes)

Ka=np.load(V.K_a)
Ma=np.load(V.M_a)

V.loading = 0
if V.loading:
  #Fa=np.load(V.feminas_dir+'/Runs/'+V.model+'/'+Fa_name+'.npy')  # BeamSeg,tn,NumNodes,6
  with open (V.feminas_dir+'/Runs/'+V.model+'/'+Fa_name, 'rb') as fp:
    Fa = pickle.load(fp)
try:
    Xm=np.load(results+'/Xm.npy')
except IOError:
    import intrinsic.FEmodel
    Xm=intrinsic.FEmodel.CentreofMass(Ma,V.Clamped,V.NumBeams,BeamSeg)
    if save:
      np.save(results+'/Xm.npy',Xm)

try:
   Phi0 =np.load(results_modes+'/Phi0%s.npy'%nm)
   Phi1 =np.load(results_modes+'/Phi1%s.npy'%nm)
   Phi1m =np.load(results_modes+'/Phi1m%s.npy'%nm)
   MPhi1 =np.load(results_modes+'/MPhi1%s.npy'%nm)
   Phi2 =np.load(results_modes+'/Phi2%s.npy'%nm)
   Phi0l =np.load(results_modes+'/Phi0%s.npy'%nm)
   Phi1l =np.load(results_modes+'/Phi1%s.npy'%nm)
   Phi1ml =np.load(results_modes+'/Phi1m%s.npy'%nm)
   MPhi1l =np.load(results_modes+'/MPhi1l%s.npy'%nm)
   Phi2l =np.load(results_modes+'/Phi2%s.npy'%nm)
   CPhi2x =np.load(results_modes+'/CPhi2x%s.npy'%nm)
   CPhi2xl =np.load(results_modes+'/CPhi2xl%s.npy'%nm)
   Omega =np.load(results_modes+'/Omega%s.npy'%nm)


except IOError:

   from intrinsic.modes import Phi0,Phi1,Phi1m,MPhi1,Phi2,CPhi2x,Phi0l,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl,Omega
   if save:

     np.save(results_modes+'/Phi0%s.npy'%nm,Phi0)
     np.save(results_modes+'/Phi1%s.npy'%nm,Phi1)
     np.save(results_modes+'/Phi1m%s.npy'%nm,Phi1m)
     np.save(results_modes+'/MPhi1%s.npy'%nm,Phi1m)
     np.save(results_modes+'/Phi2%s.npy'%nm,Phi2)
     np.save(results_modes+'/CPhi2x%s.npy'%nm,CPhi2x)
     np.save(results_modes+'/Phi0l%s.npy'%nm,Phi0l)
     np.save(results_modes+'/Phi1l%s.npy'%nm,Phi1l)
     np.save(results_modes+'/Phi1ml%s.npy'%nm,Phi1ml)
     np.save(results_modes+'/MPhi1l%s.npy'%nm,Phi1m)
     np.save(results_modes+'/Phi2l%s.npy'%nm,Phi2l)
     np.save(results_modes+'/CPhi2xl%s.npy'%nm,CPhi2xl)
     np.save(results_modes+'/Omega%s.npy'%nm,Omega)

'''
if V.loading:
    Fa=np.load(V.feminas_dir+'/Runs/'+V.model+'/Fa.npy')
    eta = np.zeros((V.tn,V.NumModes))
    for ti in range(V.tn):
      for k in range(V.NumModes):
        for i in range(V.NumBeams):
          for j in range(BeamSeg[i].EnumNodes):

            eta[ti][k] = eta[ti][k] + Phi1l[i][k][j].dot(Fa[i][ti][j])

np.save(results+'/eta%s.npy'%nm,eta)
'''

try:
   #raise ValueError
   gamma1 =np.load(results_modes+'/gamma1%s.npy'%nm)
   gamma2 =np.load(results_modes+'/gamma2%s.npy'%nm)

except:
#if 1:
   import intrinsic.integrals
   intrinsic.integrals.Phi1y=Phi1
   intrinsic.integrals.Phi1my=Phi1ml
   intrinsic.integrals.MPhi1y=MPhi1
   intrinsic.integrals.Phi2y=Phi2l
   intrinsic.integrals.CPhi2y=CPhi2xl

   gamma1,gamma2=intrinsic.integrals.solve_integrals(multi,'gammas',V.NumModes,NumProcess)
   alpha1,alpha2=intrinsic.integrals.solve_integrals(multi,'alphas',V.NumModes,NumProcess)

   if save:
       np.save(results_modes+'/gamma1%s.npy'%nm,gamma1)
       np.save(results_modes+'/gamma2%s.npy'%nm,gamma2)

if V.loading:

  try:
     raise ValueError
     eta=np.load(results+'/eta%s.npy'%nm)
  except:
     import intrinsic.integrals
     eta=intrinsic.integrals.integral_eta(Fa,V.tn,V.NumModes,V.NumBeams,BeamSeg,Phi1l)
     if save:
         np.save(results+'/eta%s.npy'%nm,eta)
    #print(qt.qstatic(q2,eta[-1]))
    #print(qt.qstatic_lin(q2_lin,eta[-1]))

eta=np.zeros(V.NumModes)
V.dt = float(V.tf-V.t0)/V.tn
#V.dt=(1./Omega[-1])/4
#V.NumModes =  10
try:
  raise ValueError
  q2_lin=np.load(results+'/q2_lin%s.npy'%nm)
  q2=np.load(results+'/q2%s.npy'%nm)
  #q2fix_lin=np.load(results+'/q2fix_lin%s.npy'%nm)
  #q2fix=np.load(results+'/q2fix%s.npy'%nm)
except:
  import intrinsic.qt
  from Tools.ODE import RK4

  x0=np.zeros(NumNode*6)
  lam=2
  for i in range(NumNode):
   x0[6*i+1]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
   x0[6*i+2]=lam*BeamSeg[0].NodeX[i+1][0]/BeamSeg[0].L
  H=np.vstack([np.hstack(Phi1[0][i][1:]) for i in range(V.NumModes)]).T
  q01=np.linalg.solve(H, x0)
  q0=np.zeros(2*V.NumModes)
  q0[0:V.NumModes] = q01[0:V.NumModes]
  param={}
  param['args']=[Omega[0:V.NumModes],gamma1,gamma2,eta]
  param['t0']=V.t0
  param['q0']=q0
  param['dt']=V.dt
  param['tf']=param['dt']*40
  param['integrator']= 'lsoda'#'dopri5'
  param['jacobian']=1
  #
  q2=intrinsic.qt.qsol(RK4,intrinsic.qt.dq12,param)


  pdb.set_trace()
  q2py=intrinsic.qt.qsolpy(intrinsic.qt.dq12,intrinsic.qt.dJq12,param)

  #q2 = intrinsic.qt.qstatic_sol(intrinsic.qt.qstatic,intrinsic.qt.Jqstatic,V.NumModes,Omega,gamma2,eta,V.tn)
  #q2=intrinsic.qt.qstatic_sol(intrinsic.qt.qstatic,eta)
  #q2_lin = intrinsic.qt.qstatic_sollin(eta,Omega,V.NumModes,V.tn)
  #q2fix,q2fix_lin=intrinsic.qt.qstatic_solfix(eta,Omega,gamma2)
  if save:

    np.save(results+'/q2_lin%s.npy'%nm,q2_lin)
    np.save(results+'/q2%s.npy'%nm,q2)
    #np.save(results+'/q2fix_lin%s.npy'%nm,q2fix_lin)
    #np.save(results+'/q2fix%s.npy'%nm,q2fix)

try:
  raise ValueError
  strain = np.load(results+'/strain%s.npy'%nm)
  kappa = np.load(results+'/kappa%s.npy'%nm)
  ra =  np.load(results+'/ra%s.npy'%nm)
  Rab =  np.load(results+'/Rab%s.npy'%nm)
  import intrinsic.sol
  strain0=np.asarray([np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)])
  ra0,Rab0=intrinsic.sol.integration_strains(strain0,strain0,k0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))
#import intrinsic.sol
#strain0=np.asarray([np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)])
#ra0,Rab0=intrinsic.sol.integration_strains(strain0,strain0,k0)

except:
    import intrinsic.sol
    #print 'ff'
    
    X1,X2=intrinsic.sol.solX(Phi1,Phi2,q1,q2,BeamSeg,V.tn,V.NumBeams,V.NumModes)
    strain=[[] for i in range(V.tn)]
    kappa=[[] for i in range(V.tn)]
    strain_lin=[[] for i in range(V.tn)]
    kappa_lin=[[] for i in range(V.tn)]
    ra=[[] for i in range(V.tn)]
    Rab=[[] for i in range(V.tn)]
    ra_lin=[[] for i in range(V.tn)]
    Rab_lin=[[] for i in range(V.tn)]
    ra2=[[] for i in range(V.tn)]
    Rab2=[[] for i in range(V.tn)]
    strain0=np.asarray([np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)])
    ra0,Rab0=intrinsic.sol.integration_strains(strain0,strain0,k0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))
    ra20,Rab20=intrinsic.sol.integration_strains2(BeamSeg,strain0,strain0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,V.ClampX)
    for ti in range(V.tn):

        strain[ti]=intrinsic.sol.strain_def(BeamSeg,V.NumModes,V.NumBeams,0,q2[ti],CPhi2xl)
        kappa[ti]=intrinsic.sol.strain_def(BeamSeg,V.NumModes,V.NumBeams,1,q2[ti],CPhi2xl)

        #strain_lin[ti]=intrinsic.sol.strain_def(BeamSeg,V.NumModes,V.NumBeams,0,q2_lin[ti],CPhi2xl)
        #kappa_lin[ti]=intrinsic.sol.strain_def(BeamSeg,V.NumModes,V.NumBeams,1,q2_lin[ti],CPhi2xl)
        ra[ti],Rab[ti]=intrinsic.sol.integration_strains(kappa[ti],strain[ti],k0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))
        #ra_lin[ti],Rab_lin[ti]=intrinsic.sol.integration_strains(kappa_lin[ti],strain_lin[ti],k0,V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,ra0=np.zeros(3),Rab0=np.eye(3))
        ra2[ti],Rab2[ti]=intrinsic.sol.integration_strains2(BeamSeg,kappa[ti],strain[ti],V.NumBeams,V.BeamsClamped,V.Clamped,inverseconn,intrinsic.functions.H0,intrinsic.functions.H1,V.ClampX)

    if save:

       np.save(results+'/strain%s.npy'%nm, strain)
       np.save(results+'/kappa%s.npy'%nm, kappa)
       np.save(results+'/ra%s.npy'%nm,ra)
       np.save(results+'/Rab%s.npy'%nm, Rab)

plo=1
if plo:
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  #from Tools.plotting import mode_disp,disp_sol
  #  mode_disp(V.NumBeams,BeamSeg,[0],Phi0,0,0)
  #disp_sol(BeamSeg,ra,rt,dis[])
  #disp_sol(BeamSeg,ra,rt,dis,axi)
  #disp_sol(BeamSeg,ra,[0,1,2,3,4,5,6,7,8,9],0,[-20,100,-80,0,0,1])

marker=['k','r','b','y','g','m','c','k','r','b','y','g','m','c','k','r','b','y','g','m','c']

discrete=[]
#disx=['0p05','0p1','0p2','0p3','0p4','0p5','0p6','0p7','0p8','0p9','0p96']
#disx=['0p05','0p1','0p2','0p3','0p4','0p5','0p6','0p7','0p8','0p9','0p96']
#disx=['0','0p2','0p6','1p2','1p8','2p2','2p6','3p2','3p6','4p2']
#disx=['0p2','0p4','0p6','0p8','1p0','1p2','1p4','1p6','2p0']
disx=['0p1','0p43','1p04','1p85','2p65','3p6','5p0','5p2','8p4']
#inid='cirTran'
#inid='cirVer'
#inid='beampre'
inid='frame'

for i in range(len(disx)):
 discrete.append([inid+disx[i]+'x',inid+disx[i]+'y'])

ra0=ra20
def disp_sol2D(ra,rt,discrete,dis=0,axi=[]):

    fig, ax = plt.subplots()
    j=0
    for ti in rt:

        for i in range(len(ra[ti])):
            if ti==rt[0]:
              figx = ax.plot(ra0[i][:,0],ra0[i][:,1],'k-o')

            if dis:
              figx = ax.plot(ra[ti][i][:,0]+ra[ti][i][:,0],ra[ti][i][:,1]+ra[ti][i][:,1])
            else:
              figx = ax.plot(ra[ti][i][:,0],ra[ti][i][:,1],marker[j])#,marker[i],markersize=5, linewidth=1,label=pix[i])
        j=j+1
    di=0
    #figx = ax.plot(argy[discrete[di][1]],argy[discrete[di][0]],'x'+marker[di])
    for di in range(0,len(discrete)):
       figx = ax.plot(np.asarray(argy[discrete[di][0]])/10,np.asarray(argy[discrete[di][1]])/10,'x'+marker[di])


    #figx = ax.plot(argy['c3p7x'],argy['c3p7y'],'x'+marker[0])
    #figx = ax.plot(argy['c12p1x'],argy['c12p1y'],'x'+marker[1])
    #figx = ax.plot(argy['c17p5x'],argy['c17p5y'],'x'+marker[2])
    #figx = ax.plot(argy['c39p3x'],argy['c39p3y'],'x'+marker[3])
    #figx = ax.plot(argy['c61x'],argy['c61y'],'x'+marker[4])
    #figx = ax.plot(argy['c94p5x'],argy['c94p5y'],'X'+marker[5])
    #figx = ax.plot(argy['c109p5x'],argy['c109p5y'],'x'+marker[5])
    #figx = ax.plot(argy['c120x'],argy['c120y'],'x'+marker[6])

    #plt.xlim([-2.5,10.5]);plt.ylim([-1,8])
    ax.legend(loc='upper right')
    #plt.legend(['Initial Configuration','F = 3.7KN','F = 12.1KN','F = 17.5KN','F = 39.3KN','F = 61KN','F = 109.5KN','F = 120KN','Argyris'], loc='best')

    #ax.legend(['Initial Configuration','F = 0.2KN','F = 0.4KN','F = 0.6KN','F = 0.8KN','F = 1KN','F = 1.2KN','F = 1.4KN','F = 1.6KN','F = 2KN'])
    plt.show()

import pickle
with open('argyris.pickle', 'rb') as handle:
    argy = pickle.load(handle)


#disp_sol2D(ra,range(9),discrete)
#disp_sol2D(ra,[1,3,5,7,9,12,13],0)
if (__name__ == '__main__'):

     print('Runned main file')
    # u1=[];u2=[];u3=[];fa=[]
    # for i in range(V.tn):

    #   u1.append((ra[i][-1]-ra0[-1])[1][0])
    #   u2.append((ra[i][-1]-ra0[-1])[1][1])
    #   u3.append((ra[i][-1]-ra0[-1])[1][2])
    #   fa.append(-Fa[-1][i][1][2])

import PostProcessing.plotting as plt
PostProcessing.plotting.disp_sol(BeamSeg,ra2,[0,1,2,3,4],0,[])
#Np.isclose(alpha1,np.eye(V.NumModes)).all()
ti=np.linspace(param['t0'],param['tf'],(param['tf']-param['t0'])/param['dt']+1)
u=np.asarray(ra)
plt.plot(ti,u[:,0,-1,0])
plt.show()
plt.plot(ti,X1[0][:,-1,1])
plt.show()
