import numpy as np
import os

sol='103'
static=0
dynamic=1
pch=1
NumBeams=1
Clamped=0
BeamsClamped=[]
Node0=np.array([0.,0.,0.])
BeamConn=[0]


model='Hesse_25op2'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])
femname='S_1.bdf'


L=[10. for i in range(NumBeams)]
N=[25]+[1 for i in range(NumBeams-1)]
#W=[1. for i in range(NumBeams)]
#H=[0.1 for i in range(NumBeams)]
#TH=[0.01 for i in range(NumBeams)]
E=[1.e5 for i in range(NumBeams)]
J=[1e-2 for i in range(NumBeams)]
Area=[0.1 for i in range(NumBeams)]
NU=[0. for i in range(NumBeams)]
I1=[0.5e-2 for i in range(NumBeams)]
I2=[0.5e-2 for i in range(NumBeams)]
Im1=[2. for i in range(NumBeams)]
Im2=[1. for i in range(NumBeams)]
K1=[2. for i in range(NumBeams)]
K2=[2. for i in range(NumBeams)]
Direc=[np.asarray([6,-8,0]) for i in range(NumBeams)]
PID=[1 for i in range(NumBeams)]
rho=[10. for i in range(NumBeams)]


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
  I11 = [[rho[i]*Im1[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  eps=1e-6
  I22 = I33 = [[rho[i]*Im2[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]


Velocity0 = 0
lamb=0.002
ti=7000

numLoads=0
numForce=0
Fl=[[1500.],[2000.],[2500.],[3000.]]
Fd=[[[0.,0.,1.]] for i in range(numLoads)]
gridF=[[[-1,-1]] for i in range(numLoads)]
