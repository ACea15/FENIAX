import pdb
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import ode
from intrinsic.functions import my_timer
import Tools.ODE

##################
# Static Solvers #
##################

def qstatic_sollin2(eta,Omega,NumModes,tn):
   q=np.zeros((args['tn',args['NumModes']]))
   for ti in range(args['tn']):
     for j in range(args['NumModes']):
       q[ti][j]=-eta[ti][j]/args['Omega'][j]
   return q

def qstatic_sollin(**args):
   q=np.zeros((len(args['ti']),args['NumModes']))
   for tix in range(len(args['ti'])):
     eta = args['force1'].eta(args['ti'][tix],'q')
     for j in range(args['NumModes']):
       q[tix][j] = -eta[j]/args['Omega'][j]
   return q

def qstatic_soln(qfun,jacobian,NumModes,Omega,gamma2,eta,q20):

    qsti=fsolve(qfun,q20,args=(NumModes,Omega,gamma2,eta),fprime=jacobian)

    return(qsti)

@my_timer
def qstatic_sol(qfun,jacobian,**args):#NumModes,Omega,gamma2,force1,ti):

    qst = [np.zeros(args['NumModes'])]
    for ix in args['ti']:
          x0=qst[-1]
          args['tix'] = ix
          qsti=fsolve(qfun,x0,args=(args),fprime=jacobian)
          qst.append(qsti)

    return(qst[1:])
 
#####################
#  Dynamic Solvers
#####################

def Qsollin(force1,Omega,NumModes,ti):
   q=np.zeros((len(ti),NumModes))
   for tix in range(len(ti)):
     eta = force1.eta(ti[tix],'q')
     for j in range(NumModes):
       q[tix][j] = -eta[j]/Omega[j]
   return q

@my_timer
def Qsol(solver,dq,Jdq,**kwargs):

   qsol=[]
   qsol.append(kwargs['q0'])
   ti=kwargs['ti']
   if type(solver).__name__ == 'function':
      for i in range(len(ti)-1):
        #pdb.set_trace()
        qsol.append(solver(dq,ti[i],qsol[-1],ti[i+1]-ti[i],kwargs))
        if kwargs['printx']:
         print(ti[i],i)

      return(np.array(qsol))

   elif type(solver).__name__ == 'str':

     qs=ode(dq,Jdq)
     #pdb.set_trace()
     qs.set_initial_value(kwargs['q0'],ti[0])
     qs.set_f_params(kwargs)
     if Jdq is not None:
        qs.set_integrator(solver,with_jacobian=True)
        qs.set_jac_params(kwargs)
     else:
        qs.set_integrator(solver)

     tni=0 
     while qs.successful() and qs.t+1e-8 < ti[-1]:

       qs.integrate(qs.t+ti[tni+1]-ti[tni])
       qsol.append(qs.y)
       tni=tni+1
       if kwargs['printx']:
        print(qs.t,tni)

     return(np.array(qsol))


def Qsol2(solver,dq,Jdq,kwargs):

   qsol=[]
   qsol.append(kwargs['q0'])
   tni = 1
   if type(solver).__name__ == 'function':
      t0=kwargs['t0'];q0=kwargs['q0'];dt=kwargs['dt'];tn=kwargs['tn']
      tfi = kwargs['t0']

      while tni<kwargs['tn']:
        qsol.append(solver(dq,tfi,qsol[-1],dt,kwargs))
        tni=tni+1
        tfi=tfi+dt
        if kwargs['printx']:
           print(tfi,tni)
           #print np.shape(qsol)
        #print tni

      return(np.array(qsol))


   elif type(solver).__name__ == 'str':

     qs=ode(dq,Jdq)

     qs.set_initial_value(kwargs['q0'],kwargs['t0'])
     qs.set_f_params(kwargs)
     if Jdq is not None:
      qs.set_integrator(solver,with_jacobian=True)
      qs.set_jac_params(kwargs)
     else:
      qs.set_integrator(solver)


     while qs.successful() and qs.t <= kwargs['tf']:

       if kwargs['printx']:
        print(qs.t,tni)
       qs.integrate(qs.t+kwargs['dt'])
       qsol.append(qs.y)
       tni=tni+1

     return(np.array(qsol))



@my_timer
def Qsol_res(solver,dq,dqh,Jdq=None,Jqh=None,**kwargs):

   qlsol=[]
   qhsol=[]
   qlsol.append(kwargs['q0'])
   tfi = kwargs['t0']
   kwargs['q'] = qlsol[-1]
   kwargs['tfi'] = tfi
   if kwargs['fix_point']:
     qh=dqh(qlsol[-1],np.zeros(2*kwargs['NumModes_res']),kwargs)
   else:
     qh=fsolve(dqh,np.zeros(2*kwargs['NumModes_res']),args=kwargs,fprime=Jqh)
   qhsol.append(qh)
   kwargs['qres']=qh

   tni = 1
   if type(solver).__name__ == 'function':
      t0=kwargs['t0'];q0=kwargs['q0'];dt=kwargs['dt'];tn=kwargs['tn']
      while tni<kwargs['tn']:

        if kwargs['direct_dae']:
          ql=solver(dq,tfi,qlsol[-1],dt,kwargs)
          qlsol.append(ql)
          kwargs['q'] = ql
          tfi=tfi+dt
          kwargs['tfi'] = tfi
          if kwargs['fix_point']:
              qh=dqh(qlsol[-1],qhsol[-1],kwargs)
          else:
              qh=fsolve(dqh,qhsol[-1],args=kwargs,fprime=Jqh)

          qhsol.append(qh)
          kwargs['qres']=qh

        tni=tni+1
        if kwargs['printx']:
         print(tfi,tni)
        #print tni

      return(np.array(qlsol),np.array(qhsol))


   elif type(solver).__name__ == 'str':

     qs=ode(dq,Jdq)

     qs.set_initial_value(kwargs['q0'],kwargs['t0'])
     qs.set_f_params(kwargs)
     if Jdq is not None:
      qs.set_integrator(solver,with_jacobian=True)
      qs.set_jac_params(kwargs)
     else:
      qs.set_integrator(solver)


     while qs.successful() and qs.t <= kwargs['tf']:

       if kwargs['printx']:
        print(qs.t,tni)
       qs.integrate(qs.t+kwargs['dt'])
       qsol.append(qs.y)
       tni=tni+1

     return(np.array(qsol))





def qsol(solver,dq,**kwargs):

      t0=kwargs['t0'];q0=kwargs['q0'];dt=kwargs['dt'];tn=kwargs['tn']
      qsol=[]
      tfi=t0
      tni=1
      qsol.append(q0)
      while tni<kwargs['tn']:
        if kwargs['printx']:
         print(tfi,(tfi-t0)/dt)
        qsol.append(solver(dq,tfi,qsol[-1],dt,kwargs))
        tni=tni+1
        tfi=tfi+dt
        #print tni

      return(np.array(qsol))

def qsolpy(dq,Jdq,**kwargs):

  qs=ode(dq,Jdq)

  qs.set_initial_value(kwargs['q0'],kwargs['t0'])
  qs.set_f_kwargss(kwargs)
  if Jdq is not None:
   qs.set_integrator(kwargs['integrator'],with_jacobian=True)
   qs.set_jac_kwargss(kwargs)
  else:
   qs.set_integrator(kwargs['integrator'])

  qsol=[]
  qsol.append(kwargs['q0'])
  while qs.successful() and qs.t <= kwargs['tf']:

    if kwargs['printx']:
     print(qs.t,qs.t/kwargs['dt'])
    qs.integrate(qs.t+kwargs['dt'])
    qsol.append(qs.y)

  return(np.array(qsol))




if (__name__ == '__main__'):

    import time
    import importlib
    from Runs.Torun import torun
    V=importlib.import_module("Runs"+'.'+torun+'.'+'V')

    results=V.feminas_dir+V.model_name+'/Results'
    save=1
    nm='_'+str(V.NumModes)


    import  intrinsic.geometry
    BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

    from intrinsic.modes import Phi0,Phi1,Phi1m,MPhi1,Phi2,CPhi2x,Phi0l,Phi1l,Phi1ml,MPhi1l,Phi2l,CPhi2xl,Omega
    Fa=np.load(V.feminas_dir+'/Runs/'+V.model+'/Fa.npy')  # BeamSeg,tn,NumNodes,6
    multi=1
#if 1:
    import intrinsic.integrals
    intrinsic.integrals.Phi1y=Phi1
    intrinsic.integrals.Phi1my=Phi1ml
    intrinsic.integrals.MPhi1y=MPhi1
    intrinsic.integrals.Phi2y=Phi2l
    intrinsic.integrals.CPhi2y=CPhi2xl

    from intrinsic.integrals import integral2_gammas,integral_eta,solve_integrals
    gamma1,gamma2=solve_integrals(multi,'gammas',V.NumModes)
    eta=integral_eta(Fa,V.tn,V.NumModes,V.NumBeams,BeamSeg,Phi1l)

    start_time = time.time()
    q2fix,q2fix_lin=qstatic_solfix(eta,Omega,gamma2)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    q2=qstatic_sol(qstatic,V.NumModes,Omega,gamma2,eta,V.tn)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    q2_lin=qstatic_sol(qstatic_lin,V.NumModes,Omega,gamma2,eta,V.tn)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    q2j=Jqstatic_sol(qstatic,Jqstatic,V.NumModes,Omega,gamma2,eta,V.tn)
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    q22_lin = qstatic_sollin(eta,Omega,V.NumModes,V.tn)
    print("--- %s seconds ---" % (time.time() - start_time))






    #print(timeit.timeit("qstatic_sol(qstatic,eta)", setup="from __main__ import qstatic_sol"))
    #q1,q2=qsol(ODE.RK4,dq12,time,0.)
    #q1_lin,q2_lin=qsol(ODE.RK4,dq12_lin,time,0.)
#max([max([max(gamma2[i][j]) for i  in range(V.NumModes)]) for j in range(V.NumModes)])
    #q2fix,q2fix_lin=qstatic_solfix(eta,Omega,gamma2)

    #q2=qstatic_sol(qstatic2,V.NumModes,Omega,gamma2,eta,V.tn)
    #q2j=Jqstatic_sol(qstatic2,Jqstatic2,V.NumModes,Omega,gamma2,eta,V.tn)
    #q2_lin = qstatic_sollin(eta,Omega,V.NumModes,V.tn):
    #print(qstatic(q22,eta[-1]))
    #print(qstatic_lin(q22_lin,eta[-1]))

    print('Reading Qs')



'''
q2=qstatic_sol(qstatic,eta)
q2j=Jqstatic_sol(qstatic,eta,Jqstatic)
q2_lin=qstatic_sol(qstatic_lin,eta)
'''

'''
def qstatic(q2st,args2):

   F=np.zeros(V.NumModes)
   for i in range(V.NumModes):
    for k in range(V.NumModes):
      for l in range(V.NumModes):
        F[i]=F[i]-gamma2[i,k,l]*(q2st[k]*q2st[l])
    F[i]=F[i]+Omega[i]*(q2st[i])+args2[i]
   return F


def Jqstatic(q2st,args2):

   Jf=np.zeros((V.NumModes,V.NumModes))
   for i in range(V.NumModes):
    for j in range(V.NumModes):
      for k in range(V.NumModes):
        Jf[i][j]=Jf[i][j]-(gamma2[i,j,k]+gamma2[i,k,j])*q2st[k]
      if i==j:
        Jf[i][j]=Jf[i][j]+Omega[i]
   return Jf


def qstatic_lin(q2st,args2):

   F=np.zeros(V.NumModes)
   for i in range(V.NumModes):
        F[i]=Omega[i]*(q2st[i])+args2[i]
   return F


def qstatic_sol(qfun,eta):

    qst=[np.zeros(V.NumModes)]
    for ti in range(V.tn):

          x0=qst[-1]
          qsti=fsolve(qfun,x0,eta[ti])
          qst.append(qsti)

    return(qst[1:])

def Jqstatic_sol(qfun,eta,jacobian):

    qst=[np.zeros(V.NumModes)]
    for ti in range(V.tn):

          x0=qst[-1]
          qsti=fsolve(qfun,x0,eta[ti],fprime=jacobian)
          qst.append(qsti)

    return(qst[1:])

'''
