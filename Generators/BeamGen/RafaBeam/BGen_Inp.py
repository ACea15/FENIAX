import numpy as np
import os

sol='400'
static=0
dynamic=1
pch=1
NumBeams=1
Clamped=1
BeamsClamped=[0]
Node0=np.array([0.,0.,0.])
BeamConn=[0]


model='RafaBeam30_lum'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])
femname='S_1.bdf'


L=[20. for i in range(NumBeams)]
N=[1+30]+[1 for i in range(NumBeams-1)]
W=[1. for i in range(NumBeams)]
H=[0.1 for i in range(NumBeams)]
TH=[0.01 for i in range(NumBeams)]
E=[1.e6 for i in range(NumBeams)]
J=[4*(W[0]*H[0])**2*TH[0]/(2*W[0]+2*H[0]) for i in range(NumBeams)]
Area=[2*TH[0]*W[0]+2*TH[0]*H[0] for i in range(NumBeams)]
NU=[0.3 for i in range(NumBeams)]
I2=[2*W[0]*TH[0]*(H[0]/2)**2+2*(1./12)*TH[0]*H[0]**3 for i in range(NumBeams)]
I1=[2*H[0]*TH[0]*(W[0]/2)**2+2*(1./12)*TH[0]*W[0]**3 for i in range(NumBeams)]
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
Direc=[np.asarray([1,0,0]) for i in range(NumBeams)]
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
  # mass = [[rho[i]*Area[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  # I11 = [[rho[i]*(I1[i]+I2[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  # I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  # eps=1e-6
  # I22 = I33 = [[I11[i][j]*eps for j in range(N[i])] for i in range(NumBeams)]
  # X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]

  mass = [[rho[i]*Area[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I11 = [[rho[i]*(I1[i]+I2[i])*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  eps=1.
  I22 =  [[rho[i]*(I1[i])*L[i]/(N[i])*eps for j in range(N[i])] for i in range(NumBeams)]
  I33 = [[rho[i]*(I2[i])*L[i]/(N[i])*eps for j in range(N[i])] for i in range(NumBeams)]
  X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]


#M=[[[m11,m12,m22,m31,m32,m33,m41,m42,m43,m44,m51,m52,m53,m54,m55,m61,m62,m63,m64,m65,m66] for j in range(N[i])]  for i in range(NumBeams)]

Velocity0 = 1
lamb=3.
def fv(x):
  x1=0.
  x2=lamb*(x[0]/L[0])**2
  x3=lamb*(x[0]/L[0])**2
  return np.array([x1,x2,x3])

ti=10000

numLoads=0
numForce=0
Fl=[[1500.],[2000.],[2500.],[3000.]]
Fd=[[[0.,0.,1.]] for i in range(numLoads)]
gridF=[[[-1,-1]] for i in range(numLoads)]
