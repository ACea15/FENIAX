import numpy as np
import os
from Generators.beam_inp0 import *
sol='103'
static=1
dynamic=0
pch=1
NumBeams=1
Clamped=1
BeamsClamped=[0]
Node0=np.array([0.,0.,0.])
BeamConn=[0]

model='ArgyrisBeam_25'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])
femname='S_0.bdf'


L=[100 for i in range(NumBeams)]
N=[1+25]+[1 for i in range(NumBeams-1)]
#W=[1. for i in range(NumBeams)]
#H=[0.1 for i in range(NumBeams)]
#TH=[0.01 for i in range(NumBeams)]
E=[2.1e7 for i in range(NumBeams)]
J=[1e5 for i in range(NumBeams)]
Area=[20. for i in range(NumBeams)]
NU=[0.3 for i in range(NumBeams)]
I1=[15./9 for i in range(NumBeams)]
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
I2=[15./9 for i in range(NumBeams)]
Direc=[np.asarray([1,0,0]) for i in range(NumBeams)]
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
  I11 = [[I1[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  eps=1e-6
  I22 = I33 = [[I2[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]


Velocity0 = 0
lamb=0.002
ti=7000


numLoads=7
numForce=1
gridF=[[[0,-1]] for i in range(numLoads)]

#Fl=[[3700.],[7600.],[12100.],[15500.],[17500.],[25200.],[39300.],[48200.],[61000.],[80000.],[94500.],[109500.],[120000.]]
#Fl=[[3700.],[7600.],[12100.],[15500.],[17500.],[25200.],[39300.],[48200.],[61000.]]
Fl=[[0.6e3],[2e3],[3e3],[4e3],[6e3],[7e3],[8e3]]
Ml=[[3e4],[10e4],[15e4],[20e4],[30e4],[35e4],[40e4]]
Fd=[[[0.,-1.,0.]] for i in range(numLoads)]
Md=[[[0.,-1.,0.]] for i in range(numLoads)]

#Fp=1000*np.asarray([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.96])
