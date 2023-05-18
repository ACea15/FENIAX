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

import  intrinsic.functions
import  intrinsic.beam_path
import  intrinsic.geometry
import  intrinsic.FEmodel
import Runs.Torun

V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
if Runs.Torun.aero:
  A = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.aero)
  aerodynamics = 1
  AICs = np.load(A.Amatrix)
else:
  aerodynamics = 0
  A = None
if Runs.Torun.force:
  F = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.force)

nm = '_'+str(V.NumModes)
#=================================================================================================================

##########################
# Geometry of the model  #
##########################
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometryrb.geometry_def(V.Grid,
                                                  V.NumBeams,V.BeamConn,V.start_reading,
                                                  V.beam_start,V.nodeorder_start,V.node_start,
                                                  V.Clamped,V.ClampX,V.BeamsClamped,V.MBbeams)
################################################################
# Finite-element matrices of the model and initial CG position #
################################################################
def fem(results):
    Ka=np.load(V.K_a)
    Ma=np.load(V.M_a)

    try:
        Cg0=np.load(results+'/Cg0.npy')
    except IOError:
        Cg0=intrinsic.FEmodel.CentreofMass(Ma,V.Clamped,V.NumBeams,BeamSeg)
        np.save(results+'/Cg0.npy',Cg0)
    return Ka,Ma,Cg0

##################################
# Calculation of intrinsic modes #
##################################
def modes(load_modes,save_modes,results_modes):

    try:

       if load_modes:
         with open (results_modes+'/Phi%s'%nm , 'rb') as fp:
            Phi0,Phi1,Phi1m,Phi2,MPhi1,CPhi2x  = pickle.load(fp)
         with open (results_modes+'/Phil%s'%nm , 'rb') as fp:
            Phi0l,Phi1l,Phi1ml,Phi2l,MPhi1l,CPhi2xl  = pickle.load(fp)
         with open(results_modes+'/Phig0%s'%nm, 'rb') as fp:
            Phig0 = pickle.load(fp)

         Omega =np.load(results_modes+'/Omega%s.npy'%nm)

       else:
         raise ValueError
    except:

       from intrinsic.modesrb import Phi0,Phi1,Phi1m,MPhi1,Phi2,CPhi2x,Phi0l,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl,Omega,Phig0

    if save_modes:

          with open(results_modes+'/Phi%s'%nm, 'wb') as fp:
              pickle.dump([Phi0,Phi1,Phi1m,Phi2,MPhi1,CPhi2x], fp)
          with open(results_modes+'/Phil%s'%nm, 'wb') as fp:
              pickle.dump([Phi0l,Phi1l,Phi1ml,Phi2l,MPhi1l,CPhi2xl], fp)
          with open(results_modes+'/Phig0%s'%nm, 'wb') as fp:
              pickle.dump([Phig0], fp)

          np.save(results_modes+'/Omega%s.npy'%nm,Omega)

    #pickle.dump( [Phi0,Phi1,Phi2], gzip.open( results_modes+'/Phi%s.gz'%nm,   'wb' ) )
    #Phi0x2,Phi1x2,Phi2x2  = pickle.load(gzip.open( results_modes+'/Phi%s.gz'%nm,   'rb' ))
    return Phi0,Phi1,Phi1m,MPhi1,Phi2,CPhi2x,Phi0l,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl,Omega,Phig0

###################################################
# Nonlinear coefficients for Galerkin projection  #
###################################################
def coefficients(load_gammas,save_gammas,multi,XNumProcess,results_modes,Phi1,Phi1l,Phi1ml,MPhi1,Phi2l,CPhi2xl):

    try:
       #pdb.set_trace()
       if load_gammas:

         gamma1 =np.load(results_modes+'/gamma1%s.npy'%nm)
         gamma2 =np.load(results_modes+'/gamma2%s.npy'%nm)
         alpha1 =np.load(results_modes+'/alpha1%s.npy'%nm)
         alpha2 =np.load(results_modes+'/alpha2%s.npy'%nm)
       else:
         raise ValueError

    except:
       import intrinsic.integralsrb
       intrinsic.integralsrb.Phi1y=Phi1
       intrinsic.integralsrb.Phi1ly=Phi1l
       intrinsic.integralsrb.Phi1my=Phi1ml
       intrinsic.integralsrb.MPhi1y=MPhi1
       intrinsic.integralsrb.Phi2y=Phi2l
       intrinsic.integralsrb.CPhi2y=CPhi2xl

       if V.linear:
         gamma1=np.zeros((V.NumModes,V.NumModes,V.NumModes))
         gamma2=np.zeros((V.NumModes,V.NumModes,V.NumModes))
         alpha1,alpha2=intrinsic.integralsrb.solve_integrals(multi,'alphas',V.NumModes,XNumProcess)
       else:
         gamma1,gamma2=intrinsic.integralsrb.solve_integrals(multi,'gammas',V.NumModes,XNumProcess)
         alpha1,alpha2=intrinsic.integralsrb.solve_integrals(multi,'alphas',V.NumModes,XNumProcess)

         if save_gammas:
             np.save(results_modes+'/gamma1%s.npy'%nm,gamma1)
             np.save(results_modes+'/gamma2%s.npy'%nm,gamma2)
             np.save(results_modes+'/alpha1%s.npy'%nm,alpha1)
             np.save(results_modes+'/alpha2%s.npy'%nm,alpha2)

    return gamma1,gamma2,alpha1,alpha2

############################################
# Calculation of temporal solution (qs(t)) #
############################################
def modal_solution(load_qs,save_qs,results,Omega,Phi1l,CPhi2xl,gamma1,gamma2,BeamSeg,Phig0):
    #pdb.set_trace()
    try:
      #pdb.set_trace()
      if load_qs:
        q=np.load(results+'/q%s.npy'%nm)
      else:
        raise ValueError

    except:
      import intrinsic.qsolvers
      #import intrinsic.dq_struct
      #import intrinsic.dq_new
      import intrinsic.dq
      import intrinsic.initial_condrb
      import intrinsic.Forces
      #reload(intrinsic.Forces)
      #from intrinsic.Forces import Force
      ##################
      # Static Problem #
      ##################
      if V.static:
       force1 = intrinsic.Forces.Force(Phi1l,Gravity=F.Gravity,Phig0=Phig0,BeamSeg=BeamSeg,NumFLoads=F.NumFLoads,NumDLoads=F.NumDLoads,NumALoads=F.NumALoads,
                     Follower_points_app=F.Follower_points_app,Follower_interpol=F.Follower_interpol,
                      Dead_points_app=F.Dead_points_app,Dead_interpol=F.Dead_interpol)
       solver1 = intrinsic.dq.StaticEqs(Omega,gamma2,force1,V,F)
       #q2fix,q2fix_lin=intrinsic.qt.qstatic_solfix(eta,Omega,gamma2)
       if F.NumDLoads > 0:
         if V.linear:
           q2 = intrinsic.qsolvers.qstatic_sol(solver1.qstatic_dead_lin,None,NumModes=V.NumModes,BeamSeg=BeamSeg,inverseconn=inverseconn,CPhi2=CPhi2xl,ti=V.ti)
           if save_qs:
             np.save(results+'/q_lin%s.npy'%nm,q)
         else:
           q2 = intrinsic.qsolvers.qstatic_sol(solver1.qstatic_dead,None,NumModes=V.NumModes,BeamSeg=BeamSeg,inverseconn=inverseconn,CPhi2=CPhi2xl,ti=V.ti)
       else:
         if V.linear:
           q2 = intrinsic.qsolvers.qstatic_sollin(force1=force1,Omega=Omega,NumModes=V.NumModes,ti=V.ti)
         else:
           q2 = intrinsic.qsolvers.qstatic_sol(solver1.qstatic_opt4,solver1.Jqstatic,NumModes=V.NumModes,ti=V.ti)

       q = np.zeros((V.tn,2*V.NumModes))
       q[:,V.NumModes:] = q2
       if save_qs:
        np.save(results+'/q%s.npy'%nm,q)
        np.save(results+'/ti%s.npy'%nm,V.ti)
       return q
      #####################
      # Dynamic Problem   #
      #####################
      if V.dynamic:
        if V.ODESolver[0:2] == 'Py':
          functionODE = V.ODESolver[2:]
        else:
          import intrinsic.Tools.ODE
          functionODE = getattr(intrinsic.Tools.ODE,V.ODESolver)
        ######################
        # Initial Conditions #
        ######################
        if aerodynamics:
          q0 = intrinsic.initial_condrb.define_q0(NumAeroStates=A.NumPoles)
        else:
          q0 = intrinsic.initial_condrb.define_q0()
        #pdb.set_trace()
        ######################
        # System solution    #
        ######################
        force1 = intrinsic.Forces.Force(Phi1l,Gravity=F.Gravity,Phig0=Phig0,BeamSeg=BeamSeg,NumFLoads=F.NumFLoads,NumDLoads=F.NumDLoads,NumALoads=F.NumALoads,
                     Follower_points_app=F.Follower_points_app,Follower_interpol=F.Follower_interpol,
                      Dead_points_app=F.Dead_points_app,Dead_interpol=F.Dead_interpol)
        solver1 = intrinsic.dq.DynamicODE(Omega,Phi1l,gamma1,gamma2,force1,V,F,A)
        if V.linear and aerodynamics  and not (F.NumDLoads or F.Gravity):  #Gravity or not, no rotations tracked     #110
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero_lin,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes+A.rbd) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))
        if (not V.linear) and aerodynamics  and (F.NumDLoads or F.Gravity):                #011
          if A.rbd:
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero_rot_rbd,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,BeamSeg=BeamSeg,CPhi2=CPhi2xl,inverseconn=inverseconn,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes+A.rbd) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))
          else:
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero_rot,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,BeamSeg=BeamSeg,CPhi2=CPhi2xl,inverseconn=inverseconn,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))
        if (V.linear) and aerodynamics  and (F.NumDLoads or F.Gravity):                    #111
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero_rot_lin,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,BeamSeg=BeamSeg,CPhi2=CPhi2xl,inverseconn=inverseconn,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))
        if not (V.linear) and (not aerodynamics)  and (F.NumDLoads or F.Gravity):          #001
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_12_rot,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,
                                        BeamSeg=BeamSeg,CPhi2=CPhi2xl,inverseconn=inverseconn)
        if (V.linear) and (not aerodynamics)  and not (F.NumDLoads or F.Gravity):          #100
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_12lin,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,force1=force1)
        if not (V.linear) and (aerodynamics)  and not (F.NumDLoads or F.Gravity):          #010
          if A.rbd:
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero_rbd,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes+A.rbd) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))
          else:
             q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,poles=np.load(A.LocPoles),Aqinv=np.linalg.inv(np.eye(V.NumModes) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:]))
        if not (V.linear) and (not aerodynamics)  and not (F.NumDLoads or F.Gravity):      #000
            q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_12,solver1.dJq_12,q0=q0,ti=V.ti,printx=V.print_timeSteps)

        if save_qs:
          np.save(results+'/q%s.npy'%nm,q)
          np.save(results+'/ti%s.npy'%nm,V.ti)
          if aerodynamics:
            if A.TrimOn:
              np.save(results+'/q_elevator%s.npy'%nm,force1.q_elevator)
              np.save(results+'/force_spring%s.npy'%nm,force1.force_spring)
              np.save(results+'/force_spring_dot%s.npy'%nm,force1.force_spring_dot)
              np.save(results+'/force_spring_int%s.npy'%nm,force1.force_spring_int)
        return q
def modal_solution_residualized():
    pass
#import intrinsic.qt_res
#if (not V.linear) and (not aerodynamics) and V.NumModes_res and not (F.NumDLoads or F.Gravity):#0010
# q,qh = IntrinsicSolver.qsolvers.Qsol_res(functionODE,IntrinsicSolver.qt_res.dq12res,
#              IntrinsicSolver.qt_res.dqh_fsol,Jdq=None,Jqh=IntrinsicSolver.qt_res.Jdqh_fsol,Phi1=Phi1l,
#             q0=q0,BeamSeg=BeamSeg,NumModes=V.NumModes,Omega=Omega,gamma1=gamma1,gamma2=gamma2,
#             t0=V.t0,tf=V.tf,dt=V.dt,tn=V.tn,printx=1,force1=force1,NumDLoads = F.NumDLoads,
#             DloadApp = F.Dead_points_app,Gravity = F.Gravity,NumModes_res=V.NumModes_res,fix_point = 0,
#             direct_dae = 1)
# q,qh = IntrinsicSolver.qsolvers.Qsol_res(RK4,IntrinsicSolver.qt_res.dq12res,IntrinsicSolver.qt_res.dqh_fix,Phi1=Phi1l,q0=q0,BeamSeg=BeamSeg,NumModes=V.NumModes,Omega=Omega,gamma1=gamma1,gamma2=gamma2,t0=V.t0,tf=V.tf,dt=V.dt,tn=V.tn,printx=1,force1=force1,NumDLoads = F.NumDLoads,DloadApp = F.Dead_points_app,Gravity = F.Gravity,NumModes_res=V.NumModes_res,fix_point = 1,direct_dae = 1,err_qh=0.0001,count_qh=100)

def modal_solution_multibody(load_qs,save_qs,results,Omegamb,Phi1mb,CPhi2mb,Gamma1mb,Gamma2mb,Force1mb,BeamSegmb,Inverseconnmb,Vmb,Fmb,Amb):

    try:
      #pdb.set_trace()
      if load_qs:
        q=np.load(results+'/q%s.npy'%nm)
      else:
        raise ValueError

    except:
      import intrinsic.qsolvers
      import intrinsic.dq
      import intrinsic.initial_condrb
      #import intrinsic.Forcesmb
      #reload(intrinsic.Forces)
      #from intrinsic.Forces import Force

      if V.ODESolver[0:2] == 'Py':
        functionODE = V.ODESolver[2:]
      else:
        import intrinsic.Tools.ODE
        functionODE = getattr(intrinsic.Tools.ODE,V.ODESolver)
      ######################
      # Initial Conditions #
      ######################
      #pdb.set_trace()
      if aerodynamics:
        q0,q0i = intrinsic.initial_condrb.define_q0mb(Vmb,BeamSegmb,NumAeroStates=A.NumPoles)
      else:
        q0,q0i  = intrinsic.initial_condrb.define_q0mb(Vmb,BeamSegmb)

      #pdb.set_trace()
      #q0=np.hstack([q0,np.zeros(6),q0])
      ######################
      # System solution    #
      ######################
      solver1 = intrinsic.dq.MultibodyODE(Omegamb,Phi1mb,CPhi2mb,Gamma1mb,Gamma2mb,Force1mb,BeamSegmb,Inverseconnmb,Vmb,Fmb,Amb)
      NumModesTotal = 0
      for bi in range(V.NumBodies):
         NumModesTotal += Vmb[bi].NumModes
      if aerodynamics:
        Aqinv=np.linalg.inv(np.eye(NumModesTotal+A.rbd) - 0.5*A.rho_inf*(A.c/2)**2*AICs[2,:,:])
      if V.linear and aerodynamics  and not (F.NumDLoads or F.Gravity):        #110
          pass
      if (not V.linear) and (not aerodynamics) and V.NumModes_res and (F.NumDLoads or F.Gravity):      #0011
        pass
      if (not V.linear) and aerodynamics and (F.NumDLoads or F.Gravity):      #011
          q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero_rbd,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,BeamSeg=BeamSeg,CPhi2=CPhi2xl,inverseconn=inverseconn,poles=np.load(A.LocPoles),Aqinv=Aqinv)
      if not (V.linear) and (not aerodynamics)  and (F.NumDLoads or F.Gravity):          #001
          q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_12_rot,None,q0=q0,ti=V.ti,printx=V.print_timeSteps)
      if (V.linear) and (not aerodynamics) and not (F.NumDLoads or F.Gravity):          #100
          q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_12lin,None,q0=q0,ti=V.ti,printx=1)
      if not (V.linear) and (aerodynamics)  and not (F.NumDLoads or F.Gravity):          #010
          q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_aero,None,q0=q0,ti=V.ti,printx=V.print_timeSteps,poles=np.load(A.LocPoles),Aqinv=Aqinv)
      if not (V.linear) and (not aerodynamics) and not (F.NumDLoads or F.Gravity):      #000
          q = intrinsic.qsolvers.Qsol(functionODE,solver1.dq_12,solver1.dJq_12,q0=q0,ti=V.ti,printx=V.print_timeSteps)

      if save_qs:
             np.save(results+'/q%s.npy'%nm,q)
             np.save(results+'/ti%s.npy'%nm,V.ti)
      return q

          # q,strain,kappa,ra,Rab = intrinsic.sol.Qsolstrains_im(RK4,intrinsic.dq_struct.dq12strain_im,None,Phi1=Phi1l,q0=q0,BeamSeg=BeamSeg,NumModes=V.NumModes,Omega=Omega,gamma1=gamma1,gamma2=gamma2,V=V,F=F,force1=force1,CPhi2=CPhi2xl,inverseconn=inverseconn,printx=1)

#Mbendxy,Mbendxz,Mtorsion,Maxial = intrinsic.functions.mode_classification(Phi0,BeamSeg,Omega,V.NumBeams,NumNodes,V.NumModes,0.001)
######################################
# Recover position and displacements #
######################################
def displacements(load_sol,save_sol,results,q1,q2,X1,Phi1l,Phi2l,CPhi2xl,method=1,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn):
    #pdb.set_trace()
    if method == 1:
      velocities=1;quaternions=0;strains=0
    elif method == 2:
      velocities=0;quaternions=1;strains=0
    elif method == 3:
      velocities=0;quaternions=0;strains=1

    if load_sol:
        if velocities:
          with open (results+'/Solv%s'%nm , 'rb') as fp:
            ra0,ra,Rab  = pickle.load(fp)
        elif quaternions:
          with open (results+'/Solq%s'%nm , 'rb') as fp:
            ra0,ra,Rab,Qq  = pickle.load(fp)
        elif strains:
          with open (results+'/Sols%s'%nm , 'rb') as fp:
            ra0,ra,Rab,strain,kappa  = pickle.load(fp)

    else:
        import intrinsic.sol
        import intrinsic.solrb
        strain0 = [np.zeros((BeamSeg[i].EnumNodes,3)) for i in range(V.NumBeams)]
        ra0,Rab0 = intrinsic.solrb.integration_strains(strain0,strain0,V=V,BeamSeg=BeamSeg,
                 inverseconn=inverseconn,rai=BeamSeg[0].NodeX[0], Rabi=BeamSeg[0].GlobalAxes)
        if aerodynamics:
           rbd = A.rbd
        else:
           rbd = 0
        #q1=q[:,rbd:rbd+(V.NumModes-V.NumModes_res)]
        #q2 = q[:,V.NumModes-V.NumModes_res+rbd:2*(V.NumModes-V.NumModes_res)+rbd]
        if V.static:
          strain,kappa,ra,Rab = intrinsic.solrb.integration_strain_time(q2,CPhi2xl,V,BeamSeg,inverseconn)
          if save_sol:
            np.save(results+'/ti%s.npy'%nm,V.ti)
            with open(results+'/Sols%s'%nm, 'wb') as fp:
              pickle.dump([ra0,ra,Rab,strain,kappa], fp)
          return ra0,ra,Rab,strain,kappa
        elif V.dynamic:

          #X1,X2 = intrinsic.solrb.solX(Phi1l,Phi2l,q1,q2,V,BeamSeg)
          if velocities:
            ra,Rab = intrinsic.solrb.integration_velocities(X1,V,BeamSeg) #
            if save_sol:
              np.save(results+'/ti%s.npy'%nm,V.ti)
              with open(results+'/Solv%s'%nm, 'wb') as fp:
                pickle.dump([ra0,ra,Rab], fp)
            return ra0,ra,Rab
          elif quaternions:
            ra,Rab,Qq = intrinsic.sol.Quatintegration_velocities(X1,
                            BeamSeg=BeamSeg,NumBeams=V.NumBeams,integrator='dopri',
                            t0=V.t0,tf=V.tf,dt=V.dt,tn=V.tn,printx=0)
            if save_sol:
              np.save(results+'/ti%s.npy'%nm,V.ti)
              with open(results+'/Solq%s'%nm, 'wb') as fp:
                pickle.dump([ra0,ra,Rab,Qq], fp)
            return ra0,ra,Rab,Qq
         # ra_vM,Rab_vM = intrinsic.sol.Matintegration_velocities(X1,
         #                 BeamSeg=BeamSeg,NumBeams=V.NumBeams,integrator='dopri',
         #                 t0=V.t0,tf=V.tf,dt=V.dt,tn=V.tn,printx=0)
          elif strains:
            if V.MBbeams:
               strain,kappa,ra,Rab = intrinsic.solrb.integration_strain_timerm(q2,
                                  CPhi2xl,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn,X1=X1)
            else:
               strain,kappa,ra,Rab = intrinsic.solrb.integration_strain_time(q2,
                                  CPhi2xl,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn,X1=X1)
            if save_sol:
              np.save(results+'/ti%s.npy'%nm,V.ti)
              with open(results+'/Sols%s'%nm, 'wb') as fp:
                pickle.dump([ra0,ra,Rab,strain,kappa], fp)
            return ra0,ra,Rab,strain,kappa
          #strainM,kappaM,ra_Ms,Rab_Ms = intrinsic.sol.Matintegration_strain_time(q2,CPhi2xl,
          #                           V=V,BeamSeg=BeamSeg,inverseconn=inverseconn)


def Cg_t(results,ra,Ma):
  Cg = intrinsic.FEmodel.CentreofMassX(Ma,ra,BeamSeg,V)
  np.save(results+'/Cg%s.npy'%nm,Cg)
  return Cg
