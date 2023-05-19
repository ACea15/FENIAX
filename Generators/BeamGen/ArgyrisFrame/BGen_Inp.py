import numpy as np
import os

sol='400'
static=1
dynamic=0
pch=1
NumBeams=2
Clamped=1
BeamsClamped=[0]
Node0=np.array([0.,0.,0.])
BeamConn=[0,0]

model='ArgyrisFrame_20'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])   
femname='S2_0.bdf'


L=[24., 24.]
N=[1+10,10]
#W=[1. for i in range(NumBeams)]
#H=[0.1 for i in range(NumBeams)]
#TH=[0.01 for i in range(NumBeams)]
E=[7.124e6 for i in range(NumBeams)]
J=[1e5 for i in range(NumBeams)]
Area=[0.18 for i in range(NumBeams)]
NU=[0.3 for i in range(NumBeams)]
I1=[0.135 for i in range(NumBeams)]
I2=[0.135 for i in range(NumBeams)]
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
Direc=[np.asarray([0.,1.,0.]),np.asarray([1.,0.,0.])]
PID=[1 for i in range(NumBeams)]
rho=[1. for i in range(NumBeams)]

conm2=1
conm1=0
if conm1:
  m11=m22=m33= [[rho[i]*Area[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  m44=[[rho[i]*(I1[i]+I2[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  m21=m31=m32=m41=m42=m43=m51=m52=m53=m54=m55=m61=m62=m63=m64=m65=m66=[[0. for j in range(N[i])] for i in range(NumBeams)]
  eps=1e-6
  m55=m66=[[m44[i][j]*eps for j in range(N[i])] for i in range(NumBeams)]

if conm2:
  mass = [[rho[i]*Area[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I11 = [[rho[i]*(I1[i]+I2[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  eps=1e-1
  I22 = I33 = [[I11[i][j]*eps for j in range(N[i])] for i in range(NumBeams)]
  X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]

#M=[[[m11,m12,m22,m31,m32,m33,m41,m42,m43,m44,m51,m52,m53,m54,m55,m61,m62,m63,m64,m65,m66] for j in range(N[i])]  for i in range(NumBeams)]

Velocity0 = 0
lamb=2
ti=25000

numLoads=6
numForce=1
Fl=[[0.05e3],[0.1e3],[0.3e3],[0.43e3],[0.7e3],[1.04e3]]#,[1.5e3],[1.85e3],[2.1e3]]#,[2.65e3],[3.e3],[3.6e3],[5.0e3],[5.2e3],[7.e3],[8.4e3]]
#Fl=[[0.1e3],[0.43e3],[1.04e3],[1.85e3],[2.65e3],[3.6e3],[5.2e3]]
Fd=[[[0.,-1.,0.]] for i in range(numLoads)]
gridF=[[[1,-1]] for i in range(numLoads)]

