import numpy as np
import os
# Main directory
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])
# Solution in Nastran
sol='103'
# Analysis type
static=1
dynamic=0
# Extract matrices in sol 103
pch=1
# Number of beams
NumBeams=1
# Clamp the model (needed in static solution)
Clamped=1
# Constraints
spc =[2]
spc_dimen = '123456'
# Nodes clamped: the first one of each beam clamped
BeamsClamped=[0]
# Coord. of the clamped node
Node0=np.array([0.,0.,0.])
# Connectivities of beams
BeamConn=[0]
# Model name
model=''
# Length of each beam
L=[1. for i in range(NumBeams)]
# Number of nodes in each beam
N=[1+1]+[1 for i in range(NumBeams-1)]
# Width of beam
W=[1. for i in range(NumBeams)]
# Height of beam
H=[1. for i in range(NumBeams)]
# Thickness of beam
TH=[1. for i in range(NumBeams)]
# Young modulus of beam
E=[1. for i in range(NumBeams)]
# Area of beam
Area=[1. for i in range(NumBeams)]
# Poisson's ratio of beam
NU=[1. for i in range(NumBeams)]
# Moment of inertia of beam, xx,zz,yy,zy
J=[1. for i in range(NumBeams)]
I1=[1. for i in range(NumBeams)]
I2=[1. for i in range(NumBeams)]
I12=[1. for i in range(NumBeams)]
# transverse shear stiffness
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
# x-direction of each beam
Direc=[np.asarray([1,0,0]) for i in range(NumBeams)]
# y-or-z cross-sectional direction
beamcross = [np.asarray([0.,1.,0.]) for i in range(NumBeams)]
# property id
PID=[1 for i in range(NumBeams)]
# density of each beam
rho=[1. for i in range(NumBeams)]
# Lumped masses, conm2 or conm1
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
numLoads=0
numForce=0
gridF=[[[]] for i in range(numLoads)]
#Fl=[[3700.],[7600.],[12100.],[15500.],[17500.],[25200.],[39300.],[48200.],[61000.],[80000.],[94500.],[109500.],[120000.]]
#Fl=[[3700.],[7600.],[12100.],[15500.],[17500.],[25200.],[39300.],[48200.],[61000.]]
Fl=[[]]
Ml=[[]]
Fd=[[[0.,-1.,0.]] for i in range(numLoads)]
Md=[[[0.,-1.,0.]] for i in range(numLoads)]
