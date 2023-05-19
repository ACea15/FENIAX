import numpy as np
import os

sol='103'
static=0
dynamic=1
pch=1
NumBeams=3
Clamped=0
BeamsClamped=[]
Node0=np.array([0.,0.,0.])
BeamConn=[0,0,1]


model='HesseUn_31'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])
femname='S_0.bdf'


L=[5.,20.,5.]
N=[6,20,5]
#W=[1. for i in range(NumBeams)]
#H=[0.1 for i in range(NumBeams)]
#TH=[0.01 for i in range(NumBeams)]
E=[7.e10 for i in range(NumBeams)]
J=[(0.1*0.05**3+0.1**3*0.05)/12 for i in range(NumBeams)]
Area=[0.1*0.05 for i in range(NumBeams)]
NU=[0.3 for i in range(NumBeams)]
I1=[0.1*0.005**3/12 for i in range(NumBeams)]#[0.1**3*0.005/12 for i in range(NumBeams)]
I2=[0.1**3*0.005/12 for i in range(NumBeams)]#[0.1*0.005**3/12 for i in range(NumBeams)]
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
Direc=[np.asarray([0,0,-1]),np.asarray([1,0,0]),np.asarray([0,0,1])]
PID=[1 for i in range(NumBeams)]
rho=[2700. for i in range(NumBeams)]


conm2=1
conm1=0
if conm1:
  m11=m22=m33= [[rho[i]*Area[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  m44=[[rho[i]*(I1[i]+I2[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  m21=m31=m32=m41=m42=m43=m51=m52=m53=m54=m55=m61=m62=m63=m64=m65=m66=[[0. for j in range(N[i])] for i in range(NumBeams)]
  eps=1e-6
  m55=[[rho[i]*(I1[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  m66=[[rho[i]*(I2[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]

if conm2:

  # mass=[]
  # for i in range(NumBeams):
  #   mass.append([])
  #   I11=I21=I21 = I31 = I32 =I22=I33=mass
  #   for j in range(N[i]):

  #     if i==0:
  #       mass[i].append(rho[i]*Area[i]*sum(L)/sum(N))
  #       I11[i].append(rho[i]*(I1[i]+I2[i])*sum(L)/sum(N))
  #       I22[i].append(rho[i]*(I1[i])*sum(L)/sum(N))
  #       I33[i].append(rho[i]*(I2[i])*sum(L)/sum(N))
  #     else:
  #       mass[i].append(rho[i]*Area[i]*L[i]/(N[i]))
  #       I11[i].append(rho[i]*(I1[i]+I2[i])*L[i]/(N[i]))
  #       I22[i].append(rho[i]*(I1[i])*L[i]/(N[i]))
  #       I33[i].append(rho[i]*(I2[i])*L[i]/(N[i]))

  #     I21[i].append(0.);I31[i].append(0.);I32[i].append(0.)
  #     X1[i].append(0.);X2[i].append(0.);X3[i].append(0.)


  mass = [[rho[i]*Area[i]*sum(L)/sum(N) for j in range(N[i])] for i in range(NumBeams)]
  I11 = [[rho[i]*(I1[i]+I2[i])*sum(L)/sum(N) for j in range(N[i])] for i in range(NumBeams)]
  I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  I22=[[rho[i]*(I1[i])*sum(L)/sum(N) for j in range(N[i])] for i in range(NumBeams)]
  I33=[[rho[i]*(I2[i])*sum(L)/sum(N) for j in range(N[i])] for i in range(NumBeams)]
  X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]


Velocity0 = 0
lamb=0.002
ti=7000

numLoads=0
numForce=0
Fl=[[1500.],[2000.],[2500.],[3000.]]
Fd=[[[0.,0.,1.]] for i in range(numLoads)]
gridF=[[[-1,-1]] for i in range(numLoads)]
