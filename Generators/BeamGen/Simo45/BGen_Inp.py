import numpy as np
import os

sol='103'
static=1
dynamic=0
pch=1
pc=5
R=100.
theta0=45*2*np.pi/360
dtheta=theta0/(pc)
dx=R*np.sin(dtheta)/np.cos(dtheta/2)

NumBeams=pc
Clamped=1
BeamsClamped=[0]
Node0=np.array([0.,0.,0.])
BeamConn=[0]+range(pc-1)

model='Simo45_5'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])
femname='S_0.bdf'


L=[dx for i in range(NumBeams)]
N=[1+1]+[1 for i in range(NumBeams-1)]
W=[1. for i in range(NumBeams)]
H=[1. for i in range(NumBeams)]
TH=[0.01 for i in range(NumBeams)]
E=[1e7 for i in range(NumBeams)]
J=[1./6 for i in range(NumBeams)]
Area=[1. for i in range(NumBeams)]
NU=[0. for i in range(NumBeams)]
I1=[1./12 for i in range(NumBeams)]
I2=[1./12 for i in range(NumBeams)]
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
#Direc=np.asarray([[np.sin(dtheta*(i+1))-np.sin(dtheta*i),+np.cos(dtheta*(i+1))-np.cos(dtheta*i),0] for i in range(pc)])
Direc=np.asarray([[-np.cos(dtheta*(i+1))+np.cos(dtheta*i),np.sin(dtheta*(i+1))-np.sin(dtheta*i),0] for i in range(pc)])
PID=[1 for i in range(NumBeams)]
rho=[1. for i in range(NumBeams)]


conm2=1
conm1=0
density=0
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
Fl=[[100.],[200.],[400.],[700.],[1000.],[1500.]]
Fd=[[[0.,0.,1.]] for i in range(numLoads)]
gridF=[[[-1,-1]] for i in range(numLoads)]
